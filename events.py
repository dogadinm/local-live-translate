"""Messages passed between stages; published audio arrays must not be mutated.

Audio timestamps use the capture stream's monotonic clock and sample offsets.
settings_version remains zero until the application controller owns settings.
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class AudioChunk:
    stream_id: str
    started_at: float
    sample_rate: int
    samples: np.ndarray


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


@dataclass(frozen=True, slots=True)
class Transcript:
    meta: PhraseMeta
    text: str
    language: str | None


@dataclass(frozen=True, slots=True)
class Translation:
    meta: PhraseMeta
    source_text: str
    text: str
    source_language: str | None
    target_language: str
