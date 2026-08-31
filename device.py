"""Where the models run.

Both engines ask the same question, so they ask it in one place — otherwise the
two copies drift and one of them silently keeps running on the CPU.
"""
import ctranslate2


def best_device() -> tuple[str, str]:
    """Returns (device, compute_type) for CTranslate2."""
    try:
        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda", "float16"
    except Exception:
        pass
    return "cpu", "int8"
