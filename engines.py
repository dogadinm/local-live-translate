"""Model interfaces: engines receive job options, never UI state."""
from typing import Protocol

import numpy as np

from config import ProcessingSettings
from events import RecognitionResult


class Recognizer(Protocol):
    def transcribe(self, audio: np.ndarray, partial: bool = False, *,
                   settings: ProcessingSettings) -> RecognitionResult: ...


class TextTranslator(Protocol):
    def translate(self, text: str, src_iso: str | None, tgt_iso: str) -> str: ...
