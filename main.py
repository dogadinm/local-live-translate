import argparse
import queue
import sys
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


def main():
    parser = argparse.ArgumentParser(description="Live subtitles for any audio playing on this PC")
    parser.add_argument("--tgt", default="ru", help="Target language (default: ru)")
    parser.add_argument("--src", default=None, help="Source language (default: auto-detect)")
    parser.add_argument("--whisper", default="large-v3-turbo", help="Whisper model (default: large-v3-turbo)")
    parser.add_argument("--out-device", default=None, help="Pin capture to this output device instead of the default one")
    parser.add_argument("--list-devices", action="store_true", help="List output devices and exit")
    args = parser.parse_args()

    from audio import SystemAudio, list_devices

    if args.list_devices:
        list_devices()
        sys.exit(0)

    from langs import flores

    for name, code in (("--tgt", args.tgt), ("--src", args.src)):
        if code and flores(code) is None:
            print(f"Unsupported language for {name}: {code}")
            sys.exit(1)

    print("\nLoading models (first run downloads them)...")

    from overlay import SubtitleOverlay
    from transcriber import Transcriber
    from translator import Translator

    translator = Translator(target=args.tgt)
    overlay = SubtitleOverlay(source=args.src, target=args.tgt)
    transcriber = Transcriber(
        model_size=args.whisper,
        language=args.src,
        on_language=overlay.set_detected_language,
    )
    # wired after construction: the pickers drive the two models
    overlay.on_source_change = transcriber.set_source
    overlay.on_target_change = translator.set_target

    audio = SystemAudio(device_name=args.out_device)
    translate_queue: queue.Queue[tuple[str, str, bool]] = queue.Queue()
    stop = threading.Event()

    threading.Thread(
        target=worker_transcribe,
        args=(audio, transcriber, translate_queue, stop),
        daemon=True,
    ).start()
    threading.Thread(
        target=worker_translate,
        args=(translate_queue, translator, overlay, stop),
        daemon=True,
    ).start()

    print("Running — play anything, subtitles appear at the bottom. Esc or ✕ to quit.\n")
    audio.start()

    try:
        overlay.run()
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        audio.stop()
        print("Stopped.")


if __name__ == "__main__":
    main()
