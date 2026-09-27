"""The two worker threads between the microphone queue and the screen.

Each stage pulls from a queue, does its one job, and pushes to the next. They
run independently so a slow translation never blocks the next recognition.
"""
import queue
import threading
from typing import TypeVar

from events import SpeechSegment, Transcript, Translation

Event = TypeVar("Event", SpeechSegment, Transcript)


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
        previous = kept[-1].meta if kept else None
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


def worker_transcribe(audio_cap, transcriber, out_queue: queue.Queue[Transcript], stop: threading.Event):
    while not stop.is_set():
        try:
            first = audio_cap.queue.get(timeout=1)
        except queue.Empty:
            continue
        for segment in drain(audio_cap.queue, first):
            text, language = transcriber.transcribe(
                segment.samples, partial=not segment.meta.is_final
            )
            if text:
                print(f"[{language}]{'' if segment.meta.is_final else ' ~'} {text}")
                out_queue.put(Transcript(meta=segment.meta, text=text, language=language))


def worker_translate(in_queue: queue.Queue[Transcript], translator, overlay, stop: threading.Event):
    while not stop.is_set():
        try:
            first = in_queue.get(timeout=1)
        except queue.Empty:
            continue
        for transcript in drain(in_queue, first):
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
