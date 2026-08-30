import threading
from pathlib import Path

import ctranslate2
from transformers import NllbTokenizer  # tokenizer only — no torch needed

from langs import display, flores

CACHE_DIR = Path.home() / ".cache" / "live_translate" / "nllb-ct2"


def _device() -> str:
    try:
        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda"
    except Exception:
        pass
    return "cpu"


class Translator:
    """NLLB-200: any of 200 languages to any other, in one model.

    Both directions are just tag tokens, so switching languages costs nothing —
    no reload, no second model. The source language is passed in per phrase
    rather than stored, because it is a property of that audio, not of the app.
    """

    def __init__(self, model_dir: Path = CACHE_DIR, target: str = "ru"):
        if not model_dir.exists():
            raise RuntimeError(
                f"Model not found: {model_dir}\nRun first:  python convert_model.py"
            )
        device = _device()
        print(f"[nllb] loading on {device}")
        self.translator = ctranslate2.Translator(
            str(model_dir),
            device=device,
            inter_threads=1 if device == "cuda" else 4,
        )
        # NllbTokenizer works without torch (pure sentencepiece)
        self.tokenizer = NllbTokenizer.from_pretrained(str(model_dir))
        self._lock = threading.Lock()
        self.tgt_iso = target

    def set_target(self, iso: str) -> bool:
        if flores(iso) is None:
            return False
        with self._lock:
            self.tgt_iso = iso
        print(f"[nllb] target language: {display(iso)}")
        return True

    def translate(self, text: str, src_iso: str | None) -> str:
        if not text.strip():
            return ""
        with self._lock:
            tgt_iso = self.tgt_iso

        if src_iso == tgt_iso:
            return text  # already in the target language — nothing to do
        src, tgt = flores(src_iso or ""), flores(tgt_iso)
        if src is None:
            return f"[{src_iso or '?'}: not in the NLLB language set]"

        # NLLB source format is [src_lang] tokens </s>; built explicitly so we do
        # not depend on the tokenizer's mutable src_lang state across threads
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        tokens = [src] + self.tokenizer.convert_ids_to_tokens(ids) + ["</s>"]

        result = self.translator.translate_batch(
            [tokens],
            target_prefix=[[tgt]],
            max_decoding_length=200,
            beam_size=2,  # 2 = good speed/quality balance for real-time
        )
        out = self.tokenizer.convert_tokens_to_ids(result[0].hypotheses[0])
        return self.tokenizer.decode(out, skip_special_tokens=True).strip()
