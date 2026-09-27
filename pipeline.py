"""Source-independent segmentation, recognition and translation workers.

Each stage pulls from a queue, does its one job, and pushes to the next. They
run independently so a slow translation never blocks the next recognition.
"""
import queue
import threading
from contextlib import nullcontext
from dataclasses import replace
from typing import Callable, TypeVar

from audio import SpeechSegmenter
from config import ProcessingSettings
from engines import Recognizer, TextTranslator
from events import AudioChunk, AudioStreamEnded, PhraseMeta, SpeechSegment, Transcript, Translation

Event = TypeVar("Event", bound=SpeechSegment | Transcript | AudioStreamEnded)


def worker_segment(in_queue: queue.Queue[AudioChunk | AudioStreamEnded],
                   out_queue: queue.Queue[SpeechSegment | AudioStreamEnded],
                   stop: threading.Event, *,
                   get_settings: Callable[[], ProcessingSettings] = ProcessingSettings,
                   on_boundary: Callable[[AudioStreamEnded], None] | None = None,
                   on_stream: Callable[[str], None] | None = None, metrics=None):
    segmenter = None
    stream_id = None

    def publish(segment):
        settings = get_settings()
        job = replace(segment, settings=settings,
                      meta=replace(segment.meta, settings_version=settings.version))
        if metrics:
            metrics.enqueue("recognition", job)
        out_queue.put(job)

    while not stop.is_set():
        try:
            event = in_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        if isinstance(event, AudioStreamEnded):
            if on_boundary is not None:
                on_boundary(event)
            if stop.is_set():
                return
            if segmenter is not None and (event.stream_id == stream_id or event.source_finished):
                for segment in segmenter.finish():
                    publish(segment)
                segmenter = None
                stream_id = None
            out_queue.put(event)
            if event.source_finished:
                return
        else:
            if event.stream_id != stream_id:
                if segmenter is not None:
                    for segment in segmenter.finish():
                        publish(segment)
                    out_queue.put(AudioStreamEnded(stream_id, "stream_changed"))
                segmenter = SpeechSegmenter()
                stream_id = event.stream_id
                if on_stream is not None:
                    on_stream(stream_id)
            for segment in segmenter.push(event):
                publish(segment)


def drain(work_queue: queue.Queue[Event], first: Event, metrics=None, stage="") -> list[Event]:
    """Take everything queued, dropping drafts that a newer draft supersedes.

    Consecutive drafts with the same stream, phrase and settings version can
    be replaced by a strictly newer revision. Finals are never dropped.
    This reduces redundant work but does not impose a queue size limit.
    """
    items = [first]
    while True:
        try:
            items.append(work_queue.get_nowait())
        except queue.Empty:
            break

    kept = []
    for item in items:
        if isinstance(item, AudioStreamEnded):
            kept.append(item)
            continue
        previous = kept[-1].meta if kept and not isinstance(kept[-1], AudioStreamEnded) else None
        current = item.meta
        if (
            previous is not None
            and not previous.is_final
            and not current.is_final
            and previous.stream_id == current.stream_id
            and previous.phrase_id == current.phrase_id
            and previous.settings_version == current.settings_version
            and current.revision > previous.revision
        ):
            if metrics:
                metrics.discard(stage, kept[-1], "coalesced")
            kept[-1] = item  # newer draft replaces the stale one
        else:
            kept.append(item)
    return kept


def worker_transcribe(in_queue: queue.Queue[SpeechSegment | AudioStreamEnded], transcriber: Recognizer,
                      out_queue: queue.Queue[Transcript | AudioStreamEnded], stop: threading.Event, *,
                      is_current: Callable[[PhraseMeta], bool] = lambda meta: True,
                      on_transcript: Callable[[Transcript], None] | None = None, metrics=None):
    while not stop.is_set():
        try:
            first = in_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        for segment in drain(in_queue, first, metrics, "recognition"):
            if stop.is_set():
                return
            if isinstance(segment, AudioStreamEnded):
                out_queue.put(segment)
                if segment.source_finished:
                    return
                continue
            if not is_current(segment.meta):
                if metrics:
                    metrics.discard("recognition", segment, "obsolete")
                continue
            with metrics.measure("recognition", segment) if metrics else nullcontext():
                result = transcriber.transcribe(
                    segment.samples, partial=not segment.meta.is_final, settings=segment.settings
                )
            if stop.is_set() or not is_current(segment.meta):
                continue
            transcript = Transcript(segment.meta, result.text, result.language,
                                    segment.settings, result.detected_language)
            if on_transcript is not None:
                on_transcript(transcript)
            if result.text:
                print(f"[{result.language}]{'' if segment.meta.is_final else ' ~'} {result.text}")
                if metrics:
                    metrics.enqueue("translation", transcript)
                out_queue.put(transcript)
            elif metrics:
                metrics.count("recognition_empty")


def worker_translate(in_queue: queue.Queue[Transcript | AudioStreamEnded], translator: TextTranslator,
                     output, stop: threading.Event, *,
                     is_current: Callable[[PhraseMeta], bool] = lambda meta: True,
                     on_finished: Callable[[], None] | None = None, metrics=None):
    while not stop.is_set():
        try:
            first = in_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        for transcript in drain(in_queue, first, metrics, "translation"):
            if stop.is_set():
                return
            if isinstance(transcript, AudioStreamEnded):
                if transcript.source_finished:
                    print("[pipeline] source finished; all queued phrases processed")
                    if on_finished is not None:
                        on_finished()
                    return
                continue
            if not is_current(transcript.meta):
                if metrics:
                    metrics.discard("translation", transcript, "obsolete")
                continue
            target = transcript.settings.target_language
            with metrics.measure("translation", transcript) if metrics else nullcontext():
                text = translator.translate(transcript.text, transcript.language, target)
            if text and not stop.is_set() and is_current(transcript.meta):
                translation = Translation(
                    meta=transcript.meta,
                    source_text=transcript.text,
                    text=text,
                    source_language=transcript.language,
                    target_language=target,
                )
                print(f"  ->{'' if translation.meta.is_final else ' ~'} {text}")
                output.update(translation)
