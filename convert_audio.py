"""Extract benchmark audio using PyAV, without a separate ffmpeg executable."""
import argparse
import wave
from pathlib import Path

import av


def convert(source: Path, destination: Path, seconds: float = 180) -> None:
    rate = 16000
    limit = int(seconds * rate)
    if limit < 1:
        raise ValueError("Duration must be at least one audio sample")
    written = 0
    with av.open(str(source)) as container:
        if not container.streams.audio:
            raise ValueError("Input has no audio stream")
        resampler = av.AudioResampler(format="s16", layout="mono", rate=rate)
        # Exclusive creation protects an existing benchmark from replacement.
        with destination.open("xb") as output, wave.open(output, "wb") as writer:
            writer.setparams((1, 2, rate, 0, "NONE", "not compressed"))

            def write_frames(frames):
                nonlocal written
                for frame in frames:
                    samples = frame.to_ndarray().reshape(-1)[:limit - written]
                    writer.writeframesraw(samples.astype("<i2", copy=False).tobytes())
                    written += len(samples)

            for frame in container.decode(audio=0):
                write_frames(resampler.resample(frame))
                if written >= limit:
                    break
            if written < limit:
                write_frames(resampler.resample(None))
    with wave.open(str(destination), "rb") as result:
        assert (result.getnchannels(), result.getsampwidth(), result.getframerate()) == (1, 2, rate)
        assert result.getnframes() == written
    print(f"Created {destination}: {written / rate:.2f}s, mono, 16000 Hz, PCM16")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--seconds", type=float, default=180)
    args = parser.parse_args()
    convert(args.source, args.destination, args.seconds)
