"""Locally patched versions remain visible to the upstream advisory database."""
import hashlib
import importlib.util
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location(
    "fork_audit", Path(__file__).resolve().parents[1] / "scripts/audit-fork-dependencies.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_local_versions_are_queried_at_upstream_version(tmp_path):
    wheel = tmp_path / "patched.whl"
    wheel.write_bytes(b"wheel bytes")
    packages = [
        {"name": "stanza", "version": "1.10.1+voicestudio.1", "source": {"path": wheel.name},
         "wheels": [{"hash": "sha256:" + hashlib.sha256(wheel.read_bytes()).hexdigest()}]},
        {"name": "torch", "version": "2.13.0+cpu", "source": {"registry": "https://download.pytorch.org/whl/cpu"}},
    ]
    queries, sources = audit.queries_for_lock({"package": packages}, tmp_path)
    assert [q["version"] for q in queries] == ["1.10.1", "2.13.0"]
    assert sources[0]["local_backport"] is True
    wheel.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="differs from uv.lock"):
        audit.queries_for_lock({"package": packages}, tmp_path)
