"""Immutable MOSS download identities, shared by parent and isolated sidecars.

No parent-process imports: this module also runs in the engine's own venv.
Pins prevent silent code changes; they are not a sandbox or a full code audit.
"""
from __future__ import annotations

import os
import re

NANO_REPO = "OpenMOSS-Team/MOSS-TTS-Nano"
NANO_REVISION = "44502f80dbf9743528fa921cc544d662c685ebec"
NANO_CODEC_REPO = "OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano"
NANO_CODEC_REVISION = "6aa02b01e445cc585582cf0ba480bc3ea6c8dd68"
V15_CODEC_REPO = "OpenMOSS-Team/MOSS-Audio-Tokenizer"
V15_CODEC_REVISION = "3cd226ba2947efa357ef453bcad111b6eafba782"
_SHA = re.compile(r"[0-9a-f]{40}\Z")


def nano_source(*, codec: bool = False) -> tuple[str, str]:
    """Custom executable code requires explicit trust and its own commit pin."""
    repo, revision, variable = (
        (NANO_CODEC_REPO, NANO_CODEC_REVISION, "OMNIVOICE_MOSS_TTS_TOKENIZER")
        if codec else (NANO_REPO, NANO_REVISION, "OMNIVOICE_MOSS_TTS_MODEL")
    )
    selected = os.environ.get(variable, repo)
    if selected == repo or (not codec and selected == "OpenMOSS-Team/MOSS-TTS-Nano-100M"):
        return selected, revision
    if os.environ.get("OMNIVOICE_MOSS_TTS_TRUST_REMOTE_CODE", "").lower() not in {"1", "true", "yes", "on"}:
        raise RuntimeError("Custom MOSS code requires OMNIVOICE_MOSS_TTS_TRUST_REMOTE_CODE=1 after review")
    pin_variable = variable + "_REVISION"
    revision = os.environ.get(pin_variable, "")
    if not _SHA.fullmatch(revision):
        raise RuntimeError(f"Custom MOSS code requires {pin_variable}=<40-character commit SHA>")
    # A local directory would ignore revision/code_revision entirely.
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", selected) or os.path.exists(selected):
        raise RuntimeError("Custom MOSS source must be a Hugging Face repository, not a local path")
    return selected, revision


def snapshot_path(repo: str, revision: str) -> str:
    """Resolve both weights and executable files at the same immutable commit."""
    if not _SHA.fullmatch(revision):
        raise ValueError("MOSS snapshot revision must be a 40-character commit SHA")
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    try:
        return snapshot_download(repo, revision=revision, local_files_only=True)
    except LocalEntryNotFoundError:
        # Called only during explicit model loading. HF_HUB_OFFLINE is still
        # respected; no floating main/tag fallback or alternative repository.
        return snapshot_download(repo, revision=revision)
