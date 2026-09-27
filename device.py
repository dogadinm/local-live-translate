"""Where the models run.

Both engines ask the same question, so they ask it in one place — otherwise the
two copies drift and one of them silently keeps running on the CPU.
"""
import os
from importlib.metadata import PackageNotFoundError, distribution

import ctranslate2


# Keep the directory handle alive: closing it removes the DLL search path.
_cublas_dll_directory = None
if os.name == "nt":
    try:
        _cublas_bin = distribution("nvidia-cublas-cu12").locate_file("nvidia/cublas/bin")
    except PackageNotFoundError:
        pass  # CPU-only setups or a system CUDA installation remain supported.
    else:
        if _cublas_bin.is_dir():
            _cublas_dll_directory = os.add_dll_directory(str(_cublas_bin))
            # CTranslate2 also loads libraries using the native Windows loader.
            os.environ["PATH"] = str(_cublas_bin) + os.pathsep + os.environ.get("PATH", "")


def best_device() -> tuple[str, str]:
    """Returns (device, compute_type) for CTranslate2."""
    try:
        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda", "float16"
    except Exception:
        pass
    return "cpu", "int8"
