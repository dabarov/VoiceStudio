"""One native-runtime policy for managed and standalone engine bootstraps."""
from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parents[2]


def install_args(engine_id: str = "") -> list[str]:
    # File URIs preserve spaces in uv's constraints/overrides arguments.
    # Upstream source mappings (notably IndexTTS's cu128 index) cannot
    # supply this stack. Use only the host index chosen by our installer.
    args = ["--no-sources", "--override", (ROOT / "backend/config/native-overrides.txt").as_uri(),
            "--find-links", str(ROOT / "vendor/python/wheels"),
            "voicestudio-audio-compat==0.1.0"]
    family = {"indextts2": "452", "confucius4-tts": "452",
              "moss-tts-nano": "457", "moss-tts-v15": "500"}.get(engine_id)
    if family:
        args += ["--override", (ROOT / f"backend/config/transformers-{family}-overrides.txt").as_uri()]
    return args


def bootstrap_args(engine_id: str) -> list[str]:
    """Manual-clone bootstraps use the same host policy as managed installs."""
    args = install_args(engine_id)
    if sys.platform not in ("linux", "win32"):
        return args
    from core.torch_indexes import UV_PIP_CPU_ARGS, UV_PIP_CUDA_ARGS, PYTORCH_ROCM_INDEX_URL, ROCM_TORCH_PINS
    try:
        from core.device_caps import detect_host_caps
        family = str(detect_host_caps().family)
    except Exception:
        family = "cpu"
    rocm = family == "rocm" or os.environ.get("OMNIVOICE_TORCH_VARIANT", "").strip().lower() == "rocm"
    if sys.platform == "linux" and engine_id == "indextts2" and rocm:
        index = os.environ.get("OMNIVOICE_TORCH_INDEX") or PYTORCH_ROCM_INDEX_URL
        tail = index.rstrip("/").rsplit("/", 1)[-1]
        tag = f"+{tail}" if tail.startswith("rocm") else ""
        return args + [*(f"{pin}{tag}" for pin in ROCM_TORCH_PINS),
                       "--extra-index-url", index, "--index-strategy", "unsafe-best-match"]
    return args + list(UV_PIP_CUDA_ARGS if family == "cuda" else UV_PIP_CPU_ARGS)


PROBE_PREFIX = "from voicestudio_audio_compat import activate; activate(); "
