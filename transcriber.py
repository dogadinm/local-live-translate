from typing import Callable

import numpy as np

from device import best_device


def _can_translate(model_size: str) -> bool:
    """Whether this Whisper model was trained on the speech-translation task.

    large-v3-turbo is a pruned large-v3 retrained on transcription alone, and
    the distil models are English-only. Ask either of them to translate and it
    does not refuse — it silently transcribes instead, which looks like working
    code returning untranslated text.
    """
    name = model_size.lower()
    return not (name.endswith(".en") or "turbo" in name or "distil" in name)


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

        device, compute_type = best_device()
        print(f"[whisper] loading {model_size} on {device} ({compute_type})")
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        self.pinned = language  # None = detect every segment
        self.task = "transcribe"
        self._can_translate = _can_translate(model_size)
        self._reported: str | None = None
        self.on_language = on_language

    def set_source(self, iso: str | None):
        """None puts detection back on auto."""
        self.pinned = iso
        self._reported = None
        print(f"[whisper] source language: {iso or 'auto'}")

    def set_target_language(self, iso: str):
        """Take the English shortcut when this model actually supports it.

        Whisper translates into English by itself, so for an English target the
        separate translator is dead weight — one whole model and its latency
        drop out of the path. Models without the translation task fall back to
        the normal route rather than returning untranslated text.
        """
        wants_shortcut = iso == "en"
        self.task = "translate" if wants_shortcut and self._can_translate else "transcribe"
        if wants_shortcut:
            print(
                "[whisper] target is English — translating directly, NLLB skipped"
                if self._can_translate
                else "[whisper] this model cannot translate — going through NLLB"
            )

    def transcribe(self, audio: np.ndarray, partial: bool = False) -> tuple[str, str | None]:
        """Returns (text, language of this segment).

        A partial is a phrase still being spoken: decoded cheaply, and without
        the VAD filter, which would trim the unfinished tail we are here for.
        """
        task = self.task  # read once: the picker can change it mid-call
        segments, info = self.model.transcribe(
            audio,
            task=task,
            language=self.pinned,
            vad_filter=not partial,
            vad_parameters={"min_silence_duration_ms": 400},
            beam_size=1 if partial else 2,
            condition_on_previous_text=False,  # segments are independent; stops repeat loops
        )
        text = " ".join(s.text for s in segments).strip()
        if not text:
            return "", None

        detected = self.pinned or info.language
        if detected != self._reported:
            self._reported = detected
            if self.on_language:
                self.on_language(detected)  # the window shows what was heard

        # under task="translate" the text coming back is English whatever the
        # audio was, so English — not the audio's language — is what it is now
        return text, "en" if task == "translate" else detected
