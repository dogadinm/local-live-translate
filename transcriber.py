from typing import Callable

import numpy as np


def _device() -> tuple[str, str]:
    import ctranslate2  # already a faster-whisper dependency; avoids importing torch

    try:
        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda", "float16"
    except Exception:
        pass
    return "cpu", "int8"


class Transcriber:
    """Whisper speech-to-text with a per-segment source language.

    Detecting the language once and locking it is worse than it sounds. Whisper
    was trained to translate as well as transcribe, so forcing the wrong
    language does not fail loudly — it quietly translates the audio into that
    language instead. Lock onto English early and every later phrase silently
    becomes English, which the translator then renders back, translating
    everything twice.

    So nothing is remembered: with `pinned` unset every segment is detected on
    its own, and the language travels back with the text instead of living in
    shared state. Setting `pinned` (the picker, or --src) overrides detection.
    """

    def __init__(
        self,
        model_size: str = "large-v3-turbo",
        language: str | None = None,
        on_language: Callable[[str], None] | None = None,
    ):
        from faster_whisper import WhisperModel

        device, compute_type = _device()
        print(f"[whisper] loading {model_size} on {device} ({compute_type})")
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        self.pinned = language  # None = detect every segment
        self._reported: str | None = None
        self.on_language = on_language

    def set_source(self, iso: str | None):
        """None puts detection back on auto."""
        self.pinned = iso
        self._reported = None
        print(f"[whisper] source language: {iso or 'auto'}")

    def transcribe(self, audio: np.ndarray, partial: bool = False) -> tuple[str, str | None]:
        """Returns (text, language of this segment).

        A partial is a phrase still being spoken: decoded cheaply, and without
        the VAD filter, which would trim the unfinished tail we are here for.
        """
        segments, info = self.model.transcribe(
            audio,
            language=self.pinned,
            vad_filter=not partial,
            vad_parameters={"min_silence_duration_ms": 400},
            beam_size=1 if partial else 2,
            condition_on_previous_text=False,  # segments are independent; stops repeat loops
        )
        text = " ".join(s.text for s in segments).strip()
        if not text:
            return "", None

        language = self.pinned or info.language
        if language != self._reported:
            self._reported = language
            if self.on_language:
                self.on_language(language)
        return text, language
