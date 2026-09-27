"""Messages passed between stages; published audio arrays must not be mutated.

Audio timestamps use the capture stream's monotonic clock and sample offsets.
Each job carries an immutable settings snapshot owned by the controller.
"""
from dataclasses import dataclass

import numpy as np
from config import ProcessingSettings


@dataclass(frozen=True, slots=True)
class AudioChunk:
    stream_id: str
    started_at: float
    sample_rate: int
    samples: np.ndarray


@dataclass(frozen=True, slots=True)
class AudioStreamEnded:
    """Ordered boundary; source_finished also ends downstream worker loops."""
    stream_id: str
    reason: str = "eof"
    source_finished: bool = False
    error: str | None = None


@dataclass(frozen=True, slots=True)
class PhraseMeta:
    stream_id: str
    phrase_id: int
    revision: int
    started_at: float
    ended_at: float
    is_final: bool
    settings_version: int = 0


@dataclass(frozen=True, slots=True)
class SpeechSegment:
    meta: PhraseMeta
    sample_rate: int
    samples: np.ndarray
    settings: ProcessingSettings = ProcessingSettings()


@dataclass(frozen=True, slots=True)
class Transcript:
    meta: PhraseMeta
    text: str
    language: str | None
    settings: ProcessingSettings = ProcessingSettings()
    detected_language: str | None = None


@dataclass(frozen=True, slots=True)
class Translation:
    meta: PhraseMeta
    source_text: str
    text: str
    source_language: str | None
    target_language: str


@dataclass(frozen=True, slots=True)
class RecognitionResult:
    text: str
    language: str | None
    detected_language: str | None


@dataclass(frozen=True, slots=True)
class ApplicationStatus:
    message: str
    is_error: bool = False
