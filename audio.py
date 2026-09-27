"""Speech segmentation independent of the audio source."""
from collections import deque

import numpy as np

from events import AudioChunk, PhraseMeta, SpeechSegment

SAMPLE_RATE = 16000
FRAME = 480             # 30 ms frame
SILENCE_FLOOR = 0.004


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
        self._speech_samples = 0
        self._silence_run = 0
        self._since_partial = 0
        self._noise = 0.003
        self._phrase_id = 0
        self._revision = 0
        self._stream_id: str | None = None
        self._origin = 0.0
        self._processed_samples = 0
        self._phrase_start = 0.0
        self._finished = False

    def push(self, chunk: AudioChunk) -> list[SpeechSegment]:
        """Accept contiguous mono chunks from one 16 kHz capture stream."""
        if self._finished:
            raise ValueError("Cannot push audio after finish()")
        if chunk.sample_rate != SAMPLE_RATE or chunk.samples.ndim != 1:
            raise ValueError("Expected mono audio at 16000 Hz")
        if self._stream_id is None:
            self._stream_id = chunk.stream_id
            self._origin = chunk.started_at
        elif chunk.stream_id != self._stream_id:
            raise ValueError("Use a new SpeechSegmenter for a new capture stream")
        self._tail = np.concatenate([self._tail, chunk.samples])
        out = []
        while len(self._tail) >= FRAME:
            frame, self._tail = self._tail[:FRAME], self._tail[FRAME:]
            self._processed_samples += FRAME
            item = self._feed(frame)
            if item is not None:
                out.append(item)
        return out

    def finish(self) -> list[SpeechSegment]:
        """Finalize once, retaining the exact unpadded tail and its duration."""
        if self._finished:
            return []
        self._finished = True
        out = []
        if len(self._tail):
            tail, self._tail = self._tail, np.zeros(0, dtype=np.float32)
            self._processed_samples += len(tail)
            event = self._feed(tail, allow_partial=False)
            if event is not None:
                out.append(event)
        final = self._flush()
        if final is not None:
            out.append(final)
        self._pre.clear()
        return out

    def _feed(self, frame: np.ndarray, allow_partial: bool = True) -> SpeechSegment | None:
        rms = float(np.sqrt(np.mean(frame**2)))
        # adaptive noise floor: falls fast, rises slowly, so steady background
        # hiss gets learned but speech never raises the bar mid-phrase
        self._noise += (0.10 if rms < self._noise else 0.001) * (rms - self._noise)
        threshold = max(self._noise * 3.0, SILENCE_FLOOR)

        if rms > threshold:
            if not self._speech:
                self._speech = True
                self._buf = list(self._pre)  # pre-roll so the first word survives
                self._phrase_id += 1
                self._revision = 0
                self._phrase_start = self._origin + (
                    self._processed_samples - len(frame) - sum(len(f) for f in self._pre)
                ) / SAMPLE_RATE
            self._pre.clear()
            self._buf.append(frame)
            self._speech_samples += len(frame)
            self._silence_run = 0
            self._since_partial += 1
            if sum(len(f) for f in self._buf) >= self.max_samples:
                return self._flush()
            if (
                allow_partial
                and self._speech_samples >= self.min_partial_samples
                and self._since_partial >= self.partial_frames
            ):
                self._since_partial = 0
                return self._emit(False)  # draft; buffer keeps growing
            return None

        self._pre.append(frame)
        if self._speech:
            self._buf.append(frame)
            self._silence_run += 1
            self._since_partial += 1
            if self._silence_run >= self.silence_frames:
                return self._flush()
        return None

    def _emit(self, final: bool) -> SpeechSegment:
        self._revision += 1
        assert self._stream_id is not None
        return SpeechSegment(
            meta=PhraseMeta(
                stream_id=self._stream_id,
                phrase_id=self._phrase_id,
                revision=self._revision,
                started_at=self._phrase_start,
                ended_at=self._origin + self._processed_samples / SAMPLE_RATE,
                is_final=final,
            ),
            sample_rate=SAMPLE_RATE,
            samples=np.concatenate(self._buf),
        )

    def _flush(self) -> SpeechSegment | None:
        # measure the speech itself, not the pre-roll and trailing silence around
        # it, so clicks and notification blips get dropped
        speech_samples = self._speech_samples
        result = self._emit(True) if self._buf and speech_samples >= self.min_samples else None
        if not self._silence_run:
            self._pre.clear()  # a forced cut must not reuse old mid-phrase silence
        self._buf = []
        self._speech = False
        self._speech_samples = 0
        self._silence_run = 0
        self._since_partial = 0
        return result
