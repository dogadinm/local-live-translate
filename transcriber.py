import numpy as np

from device import best_device
from config import ProcessingSettings
from events import RecognitionResult


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
    """Whisper engine with immutable per-call language and task options."""

    def __init__(self, model_size: str = "large-v3-turbo"):
        from faster_whisper import WhisperModel

        device, compute_type = best_device()
        print(f"[whisper] loading {model_size} on {device} ({compute_type})")
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        self._can_translate = _can_translate(model_size)

    def transcribe(self, audio: np.ndarray, partial: bool = False, *,
                   settings: ProcessingSettings) -> RecognitionResult:
        task = ("translate" if settings.target_language == "en" and self._can_translate
                else "transcribe")
        segments, info = self.model.transcribe(
            audio,
            task=task,
            language=settings.source_language,
            vad_filter=not partial,
            vad_parameters={"min_silence_duration_ms": 400},
            beam_size=1 if partial else 2,
            condition_on_previous_text=False,
        )
        text = " ".join(segment.text for segment in segments).strip()
        detected = settings.source_language or info.language
        return RecognitionResult(
            text=text,
            language="en" if task == "translate" else detected,
            detected_language=detected,
        )
