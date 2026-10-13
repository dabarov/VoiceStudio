"""The patched engine stack must not resolve against stale upstream sources."""
from types import SimpleNamespace
import pytest
from engines import native_compat


def test_engine_source_mapping_cannot_override_security_index():
    args = native_compat.install_args("indextts2")
    assert "--no-sources" in args
    assert "--find-links" in args and args.count("--override") == 2


@pytest.mark.parametrize(("platform", "family", "engine", "index"), [
    ("darwin", "mps", "indextts2", None),
    ("linux", "cpu", "indextts2", "cpu"),
    ("win32", "cuda", "indextts2", "cu130"),
    ("linux", "cuda", "moss-tts-v15", "cu130"),
    ("win32", "cpu", "moss-tts-v15", "cpu"),
    ("linux", "rocm", "indextts2", "rocm7.1"),
    ("linux", "rocm", "confucius4-tts", "cpu"),
])
def test_standalone_bootstraps_select_the_supported_host_index(monkeypatch, platform, family, engine, index):
    from core import device_caps
    monkeypatch.setattr(native_compat.sys, "platform", platform)
    monkeypatch.setattr(device_caps, "detect_host_caps", lambda: SimpleNamespace(family=family))
    monkeypatch.delenv("OMNIVOICE_TORCH_VARIANT", raising=False)
    monkeypatch.delenv("OMNIVOICE_TORCH_INDEX", raising=False)
    args = native_compat.bootstrap_args(engine)
    assert "--no-sources" in args
    if index is None:
        assert "--extra-index-url" not in args
    else:
        assert args.count("--extra-index-url") == 1
        assert args[args.index("--extra-index-url") + 1] == "https://download.pytorch.org/whl/" + index
