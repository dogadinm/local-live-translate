"""The two worker threads between the microphone queue and the screen.

Each stage pulls from a queue, does its one job, and pushes to the next. They
run independently so a slow translation never blocks the next recognition.
"""
import queue
import threading


def drain(work_queue: queue.Queue, first):
    """Take everything queued, dropping drafts that a newer draft supersedes.

    Each draft already contains all the audio of the drafts before it, so only
    the newest of a run matters. Without this the queue would grow forever
    whenever a stage runs slower than drafts arrive, and subtitles would fall
    further and further behind the sound. Finals are never dropped.
    """
    items = [first]
    while True:
        try:
            items.append(work_queue.get_nowait())
        except queue.Empty:
            break

    kept = []
    for item in items:
        if not item[-1] and kept and not kept[-1][-1]:
            kept[-1] = item  # newer draft replaces the stale one
        else:
            kept.append(item)
    return kept


def worker_transcribe(audio_cap, transcriber, out_queue: queue.Queue, stop: threading.Event):
    while not stop.is_set():
        try:
            first = audio_cap.queue.get(timeout=1)
        except queue.Empty:
            continue
        for audio, final in drain(audio_cap.queue, first):
            text, language = transcriber.transcribe(audio, partial=not final)
            if text:
                print(f"[{language}]{'' if final else ' ~'} {text}")
                out_queue.put((text, language, final))


def worker_translate(in_queue: queue.Queue, translator, overlay, stop: threading.Event):
    while not stop.is_set():
        try:
            first = in_queue.get(timeout=1)
        except queue.Empty:
            continue
        for text, language, final in drain(in_queue, first):
            translation = translator.translate(text, language)
            if translation:
                print(f"  ->{'' if final else ' ~'} {translation}")
                overlay.update(translation, final)
