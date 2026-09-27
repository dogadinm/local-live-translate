import argparse
import sys
import wave

from app import ApplicationController
from config import ProcessingSettings
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
    """Compose the UI and controller; model loading belongs to the controller."""
    from overlay import SubtitleOverlay

    def load_recognizer():
        from transcriber import Transcriber
        return Transcriber(model_size=args.whisper)

    def load_translator():
        from translator import Translator
        return Translator()

    overlay = SubtitleOverlay(source=args.src, target=args.tgt)
    return ApplicationController(
        source, overlay,
        recognizer_factory=load_recognizer,
        translator_factory=load_translator,
        settings=ProcessingSettings(source_language=args.src, target_language=args.tgt),
    )

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
    build(args, source).run()


if __name__ == "__main__":
    main()
