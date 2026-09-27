"""Immutable processing options shared by all stages of a job."""
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ProcessingSettings:
    source_language: str | None = None
    target_language: str = "ru"
    version: int = 0
