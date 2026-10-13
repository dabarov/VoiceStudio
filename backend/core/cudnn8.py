"""CUDA 12 libraries for CTranslate2 4.8 alongside PyTorch's CUDA 13 runtime.

The historical module name is retained for callers. The managed runtime now
supplies cuDNN 9 and cuBLAS 12 in a separate directory. Both libraries must load
before entering CTranslate2, whose missing-library failures can abort Python.
This module remains stdlib-only for the isolated ASR sidecar.
"""
from __future__ import annotations

import glob
import logging
import os
import sys

logger = logging.getLogger(__name__)

# The library CTranslate2 names in its own failure message. Probing the exact
# one the error cites keeps the diagnosis honest — and because the probe runs
# in the same process, through the same OS loader, a failure here is the same
# failure CTranslate2 is about to hit.
_SENTINEL_LIB = (
    "cudnn64_9.dll" if sys.platform == "win32" else "libcudnn.so.9"
)
_BLAS_SENTINEL = "cublas64_12.dll" if sys.platform == "win32" else "libcublas.so.12"
_LIB_GLOB = "cudnn*64_9.dll" if sys.platform == "win32" else "libcudnn*.so.9"

_preloaded: bool = False
_status: tuple[bool, str] | None = None
# Handles from os.add_dll_directory (Windows). Held for the process lifetime —
# dropping one un-registers its directory. See preload().
_dll_dir_cookies: list = []


def _project_root() -> str:
    # backend/core/cudnn8.py → backend/core → backend → <project root>
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def compat_dirs() -> list[str]:
    """Every plausible ``ctranslate2_cuda12`` location, most specific first.

    The old inline preload hardcoded ``<project root>/.venv``. That is right
    for the managed desktop install and wrong everywhere else — a differently
    named venv, ``uv run`` from a checkout, Docker, or a system Python all
    resolve to a directory that does not exist, so the preload silently did
    nothing and the process died later anyway. ``sys.prefix`` is where the
    *running* interpreter actually lives, so it is checked too.
    """
    pyver = f"python{sys.version_info.major}.{sys.version_info.minor}"
    if sys.platform == "win32":
        tail = (("Lib", "site-packages"), ("nvidia", "cudnn", "bin"))
    else:
        tail = (("lib", pyver, "site-packages"), ("nvidia", "cudnn", "lib"))
    site_tail, lib_tail = tail

    roots = [os.path.join(_project_root(), ".venv"), sys.prefix]
    out: list[str] = []
    for root in roots:
        path = os.path.join(root, *site_tail, "ctranslate2_cuda12", *lib_tail)
        if path not in out:
            out.append(path)
    return out


def preload() -> None:
    """Load the side-loaded cuDNN 9 libraries into this process. Idempotent.

    Best-effort by design — a host with no side-load is not an error here, it
    is a fact :func:`ctranslate2_cudnn_status` reports later.
    """
    global _preloaded
    if _preloaded:
        return
    _preloaded = True
    if sys.platform == "darwin":  # no CUDA on macOS
        return

    import ctypes

    mode = 0 if sys.platform == "win32" else ctypes.RTLD_GLOBAL
    for lib_dir in compat_dirs():
        if not os.path.isdir(lib_dir):
            continue
        blas_dir = os.path.join(os.path.dirname(os.path.dirname(lib_dir)), "cublas", os.path.basename(lib_dir))
        blas_glob = "cublas*64_12.dll" if sys.platform == "win32" else "libcublas*.so.12"
        for directory, pattern in ((blas_dir, blas_glob), (lib_dir, _LIB_GLOB)):
            if not os.path.isdir(directory):
                continue
            if sys.platform == "win32":
                try:
                    _dll_dir_cookies.append(os.add_dll_directory(directory))
                except (OSError, AttributeError) as exc:
                    logger.debug("CUDA DLL directory not added: %s", exc)
            # cuBLASLt is a dependency of cuBLAS; load it first on Linux.
            for library in sorted(glob.glob(os.path.join(directory, pattern)), reverse=True):
                try:
                    ctypes.CDLL(library, mode=mode)
                except OSError as exc:
                    logger.debug("CUDA compatibility preload skipped %s: %s", library, exc)
        return  # first directory that exists wins


def _torch_wants_cudnn8() -> tuple[bool, str]:
    """Whether CTranslate2 will reach for cuDNN 9 *on this host at all*.

    cuDNN is a CUDA library: with no CUDA device, CTranslate2 runs on the CPU
    and never touches it, so a missing side-load is harmless and must not
    disqualify the engine. ROCm is excluded for the same reason and one more —
    ``torch.cuda.is_available()`` is True on a HIP build, but CTranslate2 has
    no ROCm backend, so it is CPU-only there regardless. Without this branch a
    Windows/CUDA fix would silently downgrade every ROCm user's ASR engine.
    Mirrors ``bootstrap.rs::classify_cuda_probe``, which likewise declines to
    install cuDNN 9 on HIP hosts (#124).
    """
    try:
        import torch
    except Exception as e:  # noqa: BLE001 — no torch: nothing will run anyway
        return False, f"torch unavailable ({type(e).__name__})"
    if getattr(getattr(torch, "version", None), "hip", None):
        return False, "ROCm build — CTranslate2 has no ROCm backend, runs on CPU"
    try:
        if not torch.cuda.is_available():
            return False, "no CUDA device — CTranslate2 runs on CPU"
    except Exception as e:  # noqa: BLE001
        return False, f"CUDA probe failed ({type(e).__name__})"
    return True, "CUDA device present"


def ctranslate2_cudnn_status() -> tuple[bool, str]:
    """``(usable, reason)`` — can CTranslate2 load cuDNN 9 in this process?

    Cached: the answer cannot change within a process, and it is consulted on
    every ASR engine selection.

    Conservative on purpose. A false negative costs a user WhisperX's forced
    alignment (and so lip-sync accuracy), so this returns True whenever cuDNN 9
    is not actually required, and only reports False for the one condition it
    can prove: the library the failure message names will not load, here, now,
    through the same loader CTranslate2 is about to use.
    """
    global _status
    if _status is not None:
        return _status
    _status = _compute_status()
    if not _status[0]:
        logger.warning("CTranslate2 ASR engines unavailable: %s", _status[1])
    return _status


def _try_load(name: str) -> None:
    """Load a shared library by *bare name*, raising OSError if it will not.

    Deliberately its own seam: this runs in the same process and through the
    same OS loader CTranslate2 is about to use, which is what makes the probe
    faithful — and it gives tests a way to simulate a missing library without
    replacing ``ctypes.CDLL`` process-wide (which would break torch's own load).
    """
    import ctypes

    ctypes.CDLL(name)


def _compute_status() -> tuple[bool, str]:
    # No separate macOS branch: there is no CUDA there, so `_torch_wants_cudnn8`
    # already answers "not required". One gate, testable on any platform.
    needed, why = _torch_wants_cudnn8()
    if not needed:
        return True, f"cuDNN 9 not required ({why})"

    preload()
    try:
        _try_load(_BLAS_SENTINEL)
        _try_load(_SENTINEL_LIB)
    except OSError as e:
        return False, (
            f"CUDA is active but cuBLAS 12 or {_SENTINEL_LIB} cannot be loaded ({e}). "
            f"WhisperX and faster-whisper are CTranslate2, which requires "
            f"cuDNN 9; loading it is not optional and its absence aborts the "
            f"backend process outright rather than raising (#1371). Reinstall "
            f"the compat libraries with "
            f"`uv pip install --target <venv>/{'Lib/site-packages' if sys.platform == 'win32' else 'lib/pythonX.Y/site-packages'}/ctranslate2_cuda12 "
            f"nvidia-cudnn-cu12==9.20.0.48 nvidia-cublas-cu12==12.9.1.4`, or pin a non-CTranslate2 engine "
            f"with OMNIVOICE_ASR_BACKEND=pytorch-whisper."
        )
    return True, "cuDNN 9 loadable"


def reset_cache_for_tests() -> None:
    """Clear the memoised probe. Tests only."""
    global _status, _preloaded
    _status = None
    _preloaded = False
