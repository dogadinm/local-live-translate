"""System audio capture (WASAPI loopback) + speech segmentation.

Taps the mix Windows sends to the current output device, so it hears whatever
plays on the machine — browser, player, Discord, games. No driver needed.
"""
import queue
import sys
import threading
import time
from collections import deque

import numpy as np

SAMPLE_RATE = 16000
BLOCK = 1600            # 100 ms per read
FRAME = 480             # 30 ms VAD frame
SILENCE_FLOOR = 0.004   # below this it is digital silence, never speech


def _import_soundcard():
    """Import soundcard, then make sure the calling thread has COM.

    soundcard initialises COM itself at module import, and its error check
    rejects S_FALSE — so initialising COM *before* that import crashes it.
    Importing first and topping up afterwards covers threads that were not the
    ones to import the module.
    """
    import soundcard as sc

    if sys.platform == "win32":
        import ctypes
        # return value ignored on purpose: S_FALSE (already initialised) and
        # RPC_E_CHANGED_MODE (initialised as STA elsewhere) are both fine here
        ctypes.windll.ole32.CoInitializeEx(None, 0)  # COINIT_MULTITHREADED
    return sc


class SpeechSegmenter:
    """Emits the speech buffer while it grows, and once more when it ends.

    Waiting for a pause before showing anything costs seconds of dead time, so
    the buffer is also handed out every `partial_ms` mid-phrase. Those partials
    are drafts of the same phrase and get rewritten on screen; the final one,
    cut on silence, is the clean version.
    """

    def __init__(
        self,
        silence_ms=400,
        min_speech_ms=350,
        max_speech_ms=6000,
        pad_ms=250,
        partial_ms=800,
        min_partial_ms=1000,
    ):
        self.silence_frames = silence_ms // 30
        self.partial_frames = partial_ms // 30
        self.min_samples = min_speech_ms * SAMPLE_RATE // 1000
        self.min_partial_samples = min_partial_ms * SAMPLE_RATE // 1000
        self.max_samples = max_speech_ms * SAMPLE_RATE // 1000
        self._pre = deque(maxlen=max(1, pad_ms // 30))
        self._tail = np.zeros(0, dtype=np.float32)
        self._buf: list[np.ndarray] = []
        self._speech = False
        self._speech_frames = 0
        self._silence_run = 0
        self._since_partial = 0
        self._noise = 0.003

    def push(self, block: np.ndarray) -> list[tuple[np.ndarray, bool]]:
        """Returns (audio, is_final) pairs."""
        self._tail = np.concatenate([self._tail, block])
        out = []
        while len(self._tail) >= FRAME:
            frame, self._tail = self._tail[:FRAME], self._tail[FRAME:]
            item = self._feed(frame)
            if item is not None:
                out.append(item)
        return out

    def _feed(self, frame: np.ndarray) -> tuple[np.ndarray, bool] | None:
        rms = float(np.sqrt(np.mean(frame**2)))
        # adaptive noise floor: falls fast, rises slowly, so steady background
        # hiss gets learned but speech never raises the bar mid-phrase
        self._noise += (0.10 if rms < self._noise else 0.001) * (rms - self._noise)
        threshold = max(self._noise * 3.0, SILENCE_FLOOR)

        if rms > threshold:
            if not self._speech:
                self._speech = True
                self._buf = list(self._pre)  # pre-roll so the first word survives
            self._buf.append(frame)
            self._speech_frames += 1
            self._silence_run = 0
            self._since_partial += 1
            if sum(len(f) for f in self._buf) >= self.max_samples:
                return self._flush()
            if (
                self._speech_frames * FRAME >= self.min_partial_samples
                and self._since_partial >= self.partial_frames
            ):
                self._since_partial = 0
                return np.concatenate(self._buf), False  # draft; buffer keeps growing
            return None

        self._pre.append(frame)
        if self._speech:
            self._buf.append(frame)
            self._silence_run += 1
            self._since_partial += 1
            if self._silence_run >= self.silence_frames:
                return self._flush()
        return None

    def _flush(self) -> tuple[np.ndarray, bool] | None:
        audio = np.concatenate(self._buf) if self._buf else None
        # measure the speech itself, not the pre-roll and trailing silence around
        # it, so clicks and notification blips get dropped
        speech_samples = self._speech_frames * FRAME
        self._buf = []
        self._speech = False
        self._speech_frames = 0
        self._silence_run = 0
        self._since_partial = 0
        if audio is None or speech_samples < self.min_samples:
            return None
        return audio, True


class SystemAudio:
    """Loopback capture that follows whatever Windows is playing to."""

    def __init__(self, device_name: str | None = None):
        self.queue: queue.Queue[tuple[np.ndarray, bool]] = queue.Queue()
        self._pinned = device_name
        self.device_name = device_name or "..."
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()

    def _run(self):
        sc = _import_soundcard()
        segmenter = SpeechSegmenter()
        while not self._stop.is_set():
            try:
                target = self._pinned or str(sc.default_speaker().name)
                mic = sc.get_microphone(target, include_loopback=True)
                self.device_name = target
                print(f"[audio] listening to: {target}")

                with mic.recorder(samplerate=SAMPLE_RATE, blocksize=BLOCK) as rec:
                    checked = time.monotonic()
                    while not self._stop.is_set():
                        data = rec.record(numframes=BLOCK)
                        mono = data.mean(axis=1) if data.ndim > 1 else data
                        for item in segmenter.push(mono.astype(np.float32)):
                            self.queue.put(item)

                        # follow the user switching headphones/speakers mid-session
                        if self._pinned is None and time.monotonic() - checked > 2:
                            checked = time.monotonic()
                            if str(sc.default_speaker().name) != target:
                                print("[audio] output device changed — reopening")
                                break
            except Exception as exc:
                if self._stop.is_set():
                    break
                print(f"[audio] {exc} — retrying in 2s")
                time.sleep(2)


def list_devices():
    sc = _import_soundcard()
    default = str(sc.default_speaker().name)
    print("Output devices (audio is captured from one of these):\n")
    for speaker in sc.all_speakers():
        mark = " <- default" if str(speaker.name) == default else ""
        print(f"  {speaker.name}{mark}")
