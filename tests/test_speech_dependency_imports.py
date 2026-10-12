"""The locked speech stack must import with the patched setuptools release."""
import os
import subprocess
import sys


def test_speech_libraries_import_without_legacy_pkg_resources(tmp_path):
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
           "HF_HOME": str(tmp_path / "hf"), "HF_HUB_CACHE": str(tmp_path / "hf" / "hub"),
           "XDG_DATA_HOME": str(tmp_path / "data"), "XDG_CACHE_HOME": str(tmp_path / "cache"),
           "TORCH_HOME": str(tmp_path / "torch")}
    result = subprocess.run([sys.executable, "-c", "\n".join([
        "import importlib.metadata as metadata",
        "assert int(metadata.version('setuptools').split('.')[0]) >= 83",
        "import faster_whisper, whisperx.asr, whisperx.alignment",
        "import argostranslate.translate, pyannote.audio",
    ])], env=env, capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stderr
