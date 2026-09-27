"""Immutable processing options shared by all stages of a job."""
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ProcessingSettings:
    source_language: str | None = None
    target_language: str = "ru"
    version: int = 0


@dataclass(frozen=True, slots=True)
class QueuePolicy:
    max_pending: int = 8
    max_lag_s: float = 8.0
    translation_cache_size: int = 128

    def __post_init__(self):
        if self.max_pending < 1 or self.max_lag_s <= 0 or self.translation_cache_size < 0:
            raise ValueError("Invalid queue policy")
