import argparse
import queue
import sys
import threading

from pipeline import worker_transcribe, worker_translate
from events import Transcript


def parse_args():
    parser = argparse.ArgumentParser(description="Live subtitles for any audio playing on this PC")
    parser.add_argument("--tgt", default="ru", help="Target language (default: ru)")
    parser.add_argument("--src", default=None, help="Source language (default: auto-detect)")
    parser.add_argument("--whisper", default="large-v3-turbo", help="Whisper model (default: large-v3-turbo)")
    parser.add_argument("--out-device", default=None, help="Pin capture to this output device instead of the default one")
    parser.add_argument("--list-devices", action="store_true", help="List output devices and exit")
    return parser.parse_args()


def check_languages(args):
    from langs import flores

    for name, code in (("--tgt", args.tgt), ("--src", args.src)):
        if code and flores(code) is None:
            print(f"Unsupported language for {name}: {code}")
            sys.exit(1)


def build(args):
    """Construct the window and both models, wired to each other."""
    from audio import SystemAudio
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
    def change_target(iso: str):
        translator.set_target(iso)
        transcriber.set_target_language(iso)  # may switch Whisper to direct translation

    transcriber.set_target_language(args.tgt)

    # wired after construction: the pickers drive the two models, and the models
    # cannot be passed to the window that is built before them
    overlay.on_source_change = transcriber.set_source
    overlay.on_target_change = change_target

    return SystemAudio(device_name=args.out_device), transcriber, translator, overlay


def run(audio, transcriber, translator, overlay):
    """Start the workers, give the main thread to the window, clean up after."""
    translate_queue: queue.Queue[Transcript] = queue.Queue()
    stop = threading.Event()

    for target, worker_args in (
        (worker_transcribe, (audio, transcriber, translate_queue, stop)),
        (worker_translate, (translate_queue, translator, overlay, stop)),
    ):
        threading.Thread(target=target, args=worker_args, daemon=True).start()

    print("Running — play anything, subtitles appear at the bottom. Esc or ✕ to quit.\n")
    audio.start()

    try:
        overlay.run()  # tkinter owns the main thread until the window closes
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        audio.stop()
        print("Stopped.")


def main():
    args = parse_args()

    if args.list_devices:
        from audio import list_devices

        list_devices()
        sys.exit(0)

    check_languages(args)
    print("\nLoading models (first run downloads them)...")
    run(*build(args))


if __name__ == "__main__":
    main()
