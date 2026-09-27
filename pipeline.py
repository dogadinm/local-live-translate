"""Source-independent segmentation, recognition and translation workers.

Each stage pulls from a queue, does its one job, and pushes to the next. They
run independently so a slow translation never blocks the next recognition.
"""
import queue
import threading
from contextlib import nullcontext
from collections import OrderedDict
from dataclasses import replace
from typing import Callable, TypeVar

from audio import SpeechSegmenter
from config import ProcessingSettings
from engines import Recognizer, TextTranslator
from events import AudioChunk, AudioStreamEnded, PhraseMeta, SpeechSegment, Transcript, Translation
from scheduling import PendingQueue, phrase_key, supersedes

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

    PendingQueue already coalesces at insertion; do not pull its backlog into
    a private batch where a new final could no longer replace waiting drafts.
    Plain queues retain this helper for compatibility and isolated tests.
    """
    if isinstance(work_queue, PendingQueue):
        return [first]
    items = [first]
    while True:
        try:
            items.append(work_queue.get_nowait())
        except queue.Empty:
            break

    kept = []
    positions = {}
    for item in items:
        if isinstance(item, AudioStreamEnded):
            kept.append(item)
            positions.clear()
            continue
        key = phrase_key(item.meta)
        index = positions.get(key)
        if index is not None:
            old = kept[index]
            replace_old = supersedes(item.meta, old.meta)
            if metrics:
                metrics.discard(stage, old if replace_old else item,
                                "coalesced" if replace_old else "obsolete_revision")
            if replace_old:
                kept[index] = item
        else:
            positions[key] = len(kept)
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
            if isinstance(in_queue, PendingQueue) and in_queue.expired(segment):
                in_queue.report_drop(segment, "expired_after_inference")
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
                     on_finished: Callable[[], None] | None = None, metrics=None,
                     cache_size: int = 128):
    cache = OrderedDict()
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
            cache_key = (transcript.text, transcript.language, target)
            if cache_key in cache:
                text = cache[cache_key]
                cache.move_to_end(cache_key)
                if metrics:
                    metrics.reused("translation", transcript)
            else:
                with metrics.measure("translation", transcript) if metrics else nullcontext():
                    text = translator.translate(transcript.text, transcript.language, target)
                if text and cache_size > 0:
                    cache[cache_key] = text
                    if len(cache) > cache_size:
                        cache.popitem(last=False)
            if isinstance(in_queue, PendingQueue) and in_queue.expired(transcript):
                in_queue.report_drop(transcript, "expired_after_inference")
                continue
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
