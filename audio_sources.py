"""Audio producers. Every source emits contiguous mono float32 at 16 kHz."""
import queue
import sys
import threading
import time
import wave
from pathlib import Path
from typing import Protocol
from uuid import uuid4

import numpy as np

from events import AudioChunk, AudioStreamEnded

SAMPLE_RATE = 16000
BLOCK = 1600
AudioEvent = AudioChunk | AudioStreamEnded


class AudioSource(Protocol):
    queue: queue.Queue[AudioEvent]

    def start(self) -> None: ...
    def stop(self) -> None: ...


class _ThreadedSource:
    """One-shot producer with interruptible queue writes and bounded shutdown."""

    def __init__(self, queue_size: int = 20):
        if queue_size < 1:
            raise ValueError("queue_size must be positive")
        self.queue: queue.Queue[AudioEvent] = queue.Queue(maxsize=queue_size)
        self.done = threading.Event()
        self.error: str | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._stream_id = uuid4().hex

    def start(self) -> None:
        if self._thread is not None or self._stop.is_set():
            raise RuntimeError("Create a new audio source to restart capture")
        self._thread = threading.Thread(target=self._entry, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)

    def _publish(self, event: AudioEvent, *, live: bool = False) -> bool:
        while not self._stop.is_set():
            try:
                self.queue.put(event, timeout=0.1)
                return True
            except queue.Full:
                if live:
                    raise RuntimeError("Audio queue overflow; reopening capture as a new stream")
        return False

    def _entry(self) -> None:
        try:
            self._run()
        except Exception as exc:
            self.error = str(exc)
            print(f"[audio] {exc}")
            self._publish(AudioStreamEnded(self._stream_id, "error", True, self.error))
        finally:
            self.done.set()

    def _run(self) -> None:
        raise NotImplementedError


class FileAudioSource(_ThreadedSource):
    """Read PCM16 mono 16 kHz WAV, optionally paced like a live source.

    This reads audio for recognition; it does not play it through speakers.
    """

    def __init__(self, path: str | Path, realtime: bool = True, queue_size: int = 20):
        super().__init__(queue_size)
        self.path = Path(path)
        self.realtime = realtime
        # Validate before models and UI are loaded. Recheck when opening to read.
        with wave.open(str(self.path), "rb") as reader:
            self._validate(reader)

    @staticmethod
    def _validate(reader: wave.Wave_read) -> None:
        if (reader.getnchannels(), reader.getsampwidth(), reader.getframerate(), reader.getcomptype()) != (1, 2, SAMPLE_RATE, "NONE"):
            raise ValueError("Expected a PCM 16-bit mono WAV file at 16000 Hz")

    def _run(self) -> None:
        with wave.open(str(self.path), "rb") as reader:
            self._validate(reader)
            origin = time.monotonic()
            offset = 0
            expected_frames = reader.getnframes()
            while not self._stop.is_set():
                raw = reader.readframes(BLOCK)
                if not raw:
                    if offset != expected_frames:
                        raise ValueError("Truncated WAV audio data")
                    self._publish(AudioStreamEnded(self._stream_id, source_finished=True))
                    return
                samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
                chunk = AudioChunk(self._stream_id, origin + offset / SAMPLE_RATE, SAMPLE_RATE, samples)
                offset += len(samples)
                if self.realtime and self._stop.wait(max(0.0, origin + offset / SAMPLE_RATE - time.monotonic())):
                    return
                if not self._publish(chunk):
                    return


def _import_soundcard():
    """Import first, then initialize COM in the capture thread on Windows."""
    import soundcard as sc

    if sys.platform == "win32":
        import ctypes
        ctypes.windll.ole32.CoInitializeEx(None, 0)
    return sc


class SystemLoopbackSource(_ThreadedSource):
    """Capture the Windows output mix, following the default output device."""

    def __init__(self, device_name: str | None = None, queue_size: int = 20):
        super().__init__(queue_size)
        self._pinned = device_name
        self.device_name = device_name or "..."

    def _run(self) -> None:
        sc = _import_soundcard()
        while not self._stop.is_set():
            self._stream_id = uuid4().hex
            reason, error = "device_changed", None
            try:
                target = self._pinned or str(sc.default_speaker().name)
                mic = sc.get_microphone(target, include_loopback=True)
                self.device_name = target
                print(f"[audio] listening to: {target}")
                with mic.recorder(samplerate=SAMPLE_RATE, blocksize=BLOCK) as rec:
                    origin = checked = time.monotonic()
                    offset = 0
                    while not self._stop.is_set():
                        data = rec.record(numframes=BLOCK)
                        mono = data.mean(axis=1) if data.ndim > 1 else data
                        chunk = AudioChunk(self._stream_id, origin + offset / SAMPLE_RATE,
                                           SAMPLE_RATE, mono.astype(np.float32))
                        offset += len(chunk.samples)
                        if not self._publish(chunk, live=True):
                            return
                        if self._pinned is None and time.monotonic() - checked > 2:
                            checked = time.monotonic()
                            if str(sc.default_speaker().name) != target:
                                break
            except Exception as exc:
                reason, error = "error", str(exc)
                print(f"[audio] {exc} — retrying in 2s")
            # Recorder is closed before waiting for space to publish the boundary.
            if not self._publish(AudioStreamEnded(self._stream_id, reason, error=error)):
                return
            if error and self._stop.wait(2):
                return


def list_devices():
    sc = _import_soundcard()
    default = str(sc.default_speaker().name)
    print("Output devices (audio is captured from one of these):\n")
    for speaker in sc.all_speakers():
        mark = " <- default" if str(speaker.name) == default else ""
        print(f"  {speaker.name}{mark}")
