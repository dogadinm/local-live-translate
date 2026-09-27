import argparse
import queue
import sys
import threading
import wave

from pipeline import worker_segment, worker_transcribe, worker_translate
from events import AudioStreamEnded, SpeechSegment, Transcript
from audio_sources import AudioSource, FileAudioSource, SystemLoopbackSource


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Live subtitles for any audio playing on this PC")
    parser.add_argument("--tgt", default="ru", help="Target language (default: ru)")
    parser.add_argument("--src", default=None, help="Source language (default: auto-detect)")
    parser.add_argument("--whisper", default="large-v3-turbo", help="Whisper model (default: large-v3-turbo)")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--out-device", default=None, help="Pin capture to this output device instead of the default one")
    source.add_argument("--audio-file", help="Read a PCM16 mono 16 kHz WAV at playback speed (no audio playback)")
    source.add_argument("--list-devices", action="store_true", help="List output devices and exit")
    return parser.parse_args(argv)


def check_languages(args):
    from langs import flores

    for name, code in (("--tgt", args.tgt), ("--src", args.src)):
        if code and flores(code) is None:
            print(f"Unsupported language for {name}: {code}")
            sys.exit(1)


def build(args, source: AudioSource):
    """Construct the window and both models, wired to each other."""
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

    return source, transcriber, translator, overlay


def run(audio: AudioSource, transcriber, translator, overlay):
    """Start the workers, give the main thread to the window, clean up after."""
    segment_queue: queue.Queue[SpeechSegment | AudioStreamEnded] = queue.Queue()
    translate_queue: queue.Queue[Transcript | AudioStreamEnded] = queue.Queue()
    stop = threading.Event()

    workers = []
    for target, worker_args in (
        (worker_segment, (audio.queue, segment_queue, stop)),
        (worker_transcribe, (segment_queue, transcriber, translate_queue, stop)),
        (worker_translate, (translate_queue, translator, overlay, stop)),
    ):
        worker = threading.Thread(target=target, args=worker_args, daemon=True)
        workers.append(worker)
        worker.start()

    print("Running — reading audio from the selected source. Esc or ✕ to quit.\n")

    try:
        audio.start()
        overlay.run()  # tkinter owns the main thread until the window closes
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        audio.stop()
        for worker in workers:
            worker.join(timeout=0.2)  # in-flight model calls cannot be cancelled
        print("Stopped.")


def main():
    args = parse_args()

    if args.list_devices:
        from audio_sources import list_devices

        list_devices()
        sys.exit(0)

    check_languages(args)
    try:
        source = (FileAudioSource(args.audio_file) if args.audio_file
                  else SystemLoopbackSource(device_name=args.out_device))
    except (OSError, ValueError, wave.Error, EOFError) as exc:
        print(f"Invalid audio source: {exc}", file=sys.stderr)
        sys.exit(1)
    print("\nLoading models (first run downloads them)...")
    run(*build(args, source))


if __name__ == "__main__":
    main()
