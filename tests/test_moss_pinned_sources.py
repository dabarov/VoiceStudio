"""Executable MOSS models and their nested codecs must not follow Hub main."""
import io
import json
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


def test_legacy_nano_passes_immutable_code_and_weight_revision(monkeypatch):
    from services import tts_backend as tb
    from engines import moss_sources

    factory = Mock()
    monkeypatch.setitem(sys.modules, "moss_tts_nano", SimpleNamespace(MossTTSNano=factory))
    monkeypatch.setattr(tb.MossTTSNanoBackend, "is_available", lambda self: (True, "ready"))
    snapshot = Mock(return_value="/snapshots/pinned-nano-codec")
    monkeypatch.setattr(moss_sources, "snapshot_path", snapshot)
    # A revision must be fixed before from_pretrained is allowed to execute code.
    tb.MossTTSNanoBackend()._ensure_loaded()
    kwargs = factory.from_pretrained.call_args.kwargs
    assert kwargs["revision"] == "44502f80dbf9743528fa921cc544d662c685ebec"
    assert kwargs["code_revision"] == kwargs["revision"]
    assert kwargs["audio_tokenizer_pretrained_name_or_path"] == "/snapshots/pinned-nano-codec"
    snapshot.assert_called_once_with(moss_sources.NANO_CODEC_REPO, moss_sources.NANO_CODEC_REVISION)


def test_v15_resolves_nested_codec_at_its_own_commit(monkeypatch):
    from engines.moss_tts_v15 import main

    processor_factory = Mock()
    processor_factory.from_pretrained.return_value.model_config.sampling_rate = 24000
    model_factory = Mock()
    snapshots = Mock(side_effect=lambda repo, **kw: "/snapshots/" + kw["revision"])
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshots)
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoModel=model_factory, AutoProcessor=processor_factory))
    monkeypatch.setattr(main, "_state", None)
    monkeypatch.delenv("OMNIVOICE_MOSS_TTS_V15_MODEL", raising=False)
    main._load_model(io.BytesIO())
    args = processor_factory.from_pretrained.call_args
    assert args.kwargs["codec_path"].endswith("3cd226ba2947efa357ef453bcad111b6eafba782")
    assert args.kwargs["local_files_only"] is True


@pytest.mark.parametrize("codec", [False, True])
@pytest.mark.parametrize("revision", ["", "main", "v1", "a" * 39, "../model"])
def test_custom_nano_rejects_mutable_revisions(monkeypatch, codec, revision):
    from engines.moss_sources import nano_source
    variable = "OMNIVOICE_MOSS_TTS_TOKENIZER" if codec else "OMNIVOICE_MOSS_TTS_MODEL"
    monkeypatch.setenv(variable, "someone/model")
    monkeypatch.setenv(variable + "_REVISION", revision)
    monkeypatch.setenv("OMNIVOICE_MOSS_TTS_TRUST_REMOTE_CODE", "1")
    with pytest.raises(RuntimeError, match="40-character"):
        nano_source(codec=codec)


def test_custom_nano_requires_explicit_code_trust(monkeypatch):
    from engines.moss_sources import nano_source
    monkeypatch.setenv("OMNIVOICE_MOSS_TTS_MODEL", "someone/model")
    monkeypatch.setenv("OMNIVOICE_MOSS_TTS_MODEL_REVISION", "a" * 40)
    monkeypatch.delenv("OMNIVOICE_MOSS_TTS_TRUST_REMOTE_CODE", raising=False)
    with pytest.raises(RuntimeError, match="after review"):
        nano_source()


def test_local_directory_cannot_bypass_revision(monkeypatch, tmp_path):
    from engines.moss_sources import nano_source
    monkeypatch.setenv("OMNIVOICE_MOSS_TTS_MODEL", str(tmp_path))
    monkeypatch.setenv("OMNIVOICE_MOSS_TTS_MODEL_REVISION", "a" * 40)
    monkeypatch.setenv("OMNIVOICE_MOSS_TTS_TRUST_REMOTE_CODE", "1")
    with pytest.raises(RuntimeError, match="not a local path"):
        nano_source()


def test_snapshot_reuses_cache_and_fails_closed_on_download_error(monkeypatch):
    import huggingface_hub
    from huggingface_hub.errors import LocalEntryNotFoundError
    from engines.moss_sources import snapshot_path, NANO_REPO, NANO_REVISION
    download = Mock(return_value="/cache/pinned")
    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    assert snapshot_path(NANO_REPO, NANO_REVISION) == "/cache/pinned"
    download.assert_called_once_with(NANO_REPO, revision=NANO_REVISION, local_files_only=True)
    download.reset_mock()
    download.side_effect = [LocalEntryNotFoundError("missing"), RuntimeError("offline")]
    with pytest.raises(RuntimeError, match="offline"):
        snapshot_path(NANO_REPO, NANO_REVISION)
    assert all(call.kwargs["revision"] == NANO_REVISION for call in download.call_args_list)
    assert len(download.call_args_list) == 2
