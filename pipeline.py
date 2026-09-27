"""Source-independent segmentation, recognition and translation workers.

Each stage pulls from a queue, does its one job, and pushes to the next. They
run independently so a slow translation never blocks the next recognition.
"""
import queue
import threading
from typing import TypeVar

from audio import SpeechSegmenter
from events import AudioChunk, AudioStreamEnded, SpeechSegment, Transcript, Translation

Event = TypeVar("Event", bound=SpeechSegment | Transcript | AudioStreamEnded)


def worker_segment(in_queue: queue.Queue[AudioChunk | AudioStreamEnded],
                   out_queue: queue.Queue[SpeechSegment | AudioStreamEnded],
                   stop: threading.Event):
    segmenter = None
    stream_id = None
    while not stop.is_set():
        try:
            event = in_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        if isinstance(event, AudioStreamEnded):
            if segmenter is not None and (event.stream_id == stream_id or event.source_finished):
                for segment in segmenter.finish():
                    out_queue.put(segment)
                segmenter = None
                stream_id = None
            out_queue.put(event)
            if event.source_finished:
                return
        else:
            if event.stream_id != stream_id:
                if segmenter is not None:
                    for segment in segmenter.finish():
                        out_queue.put(segment)
                    out_queue.put(AudioStreamEnded(stream_id, "stream_changed"))
                segmenter = SpeechSegmenter()
                stream_id = event.stream_id
            for segment in segmenter.push(event):
                out_queue.put(segment)


def drain(work_queue: queue.Queue[Event], first: Event) -> list[Event]:
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
            kept[-1] = item  # newer draft replaces the stale one
        else:
            kept.append(item)
    return kept


def worker_transcribe(in_queue: queue.Queue[SpeechSegment | AudioStreamEnded], transcriber,
                      out_queue: queue.Queue[Transcript | AudioStreamEnded], stop: threading.Event):
    while not stop.is_set():
        try:
            first = in_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        for segment in drain(in_queue, first):
            if stop.is_set():
                return
            if isinstance(segment, AudioStreamEnded):
                out_queue.put(segment)
                if segment.source_finished:
                    return
                continue
            text, language = transcriber.transcribe(
                segment.samples, partial=not segment.meta.is_final
            )
            if text:
                print(f"[{language}]{'' if segment.meta.is_final else ' ~'} {text}")
                out_queue.put(Transcript(meta=segment.meta, text=text, language=language))


def worker_translate(in_queue: queue.Queue[Transcript | AudioStreamEnded], translator,
                     overlay, stop: threading.Event):
    while not stop.is_set():
        try:
            first = in_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        for transcript in drain(in_queue, first):
            if stop.is_set():
                return
            if isinstance(transcript, AudioStreamEnded):
                if transcript.source_finished:
                    print("[pipeline] source finished; all queued phrases processed")
                    return
                continue
            text, target = translator.translate_with_target(transcript.text, transcript.language)
            if text:
                translation = Translation(
                    meta=transcript.meta,
                    source_text=transcript.text,
                    text=text,
                    source_language=transcript.language,
                    target_language=target,
                )
                print(f"  ->{'' if translation.meta.is_final else ' ~'} {text}")
                overlay.update(translation)
