"""Exercise the real patched libraries, not replicas of their loaders."""
import json
import os
from pathlib import Path

import pytest
import torch


def _index(root, shard):
    root.mkdir(parents=True, exist_ok=True)
    (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"weight": shard}}))


@pytest.mark.parametrize("name", ["../outside.pt", "/tmp/outside.pt", "C:/outside.pt", "..\\outside.pt", "shards/../../outside.pt", None])
def test_accelerate_rejects_index_paths_before_opening(monkeypatch, tmp_path, name):
    from accelerate.utils import modeling
    root = tmp_path / "checkpoint"
    _index(root, name)
    def unexpected_open(*args, **kwargs):
        pytest.fail("Accelerate attempted to open an unvalidated shard")
    monkeypatch.setattr(modeling, "load_state_dict", unexpected_open)
    with pytest.raises(ValueError, match="Invalid checkpoint shard"):
        modeling.load_checkpoint_in_model(torch.nn.Linear(2, 2), str(root), device_map={"": "cpu"})


def test_accelerate_rejects_fifo_without_opening_it(monkeypatch, tmp_path):
    from accelerate.utils import modeling
    if not hasattr(os, "mkfifo"):
        pytest.skip("Named-pipe fixture uses the POSIX filesystem API")
    _index(tmp_path, "blocked.pt")
    os.mkfifo(tmp_path / "blocked.pt")
    monkeypatch.setattr(modeling, "load_state_dict", lambda *a, **kw: pytest.fail("attempted to read a FIFO"))
    with pytest.raises(ValueError, match="regular file"):
        modeling.load_checkpoint_in_model(torch.nn.Linear(2, 2), str(tmp_path), device_map={"": "cpu"})


@pytest.mark.parametrize("layout", ["nested", "hub", "outside"])
def test_accelerate_preserves_valid_weights_and_hub_cache_layout(tmp_path, layout):
    from accelerate.utils.modeling import load_checkpoint_in_model
    model = torch.nn.Linear(2, 2, bias=False)
    expected = torch.ones_like(model.weight)
    root = tmp_path / "models--owner--repo" / "snapshots" / ("a" * 40)
    _index(root, "shards/weights.pt")
    shard = root / "shards/weights.pt"
    shard.parent.mkdir()
    target = shard
    if layout != "nested":
        target = (root.parent.parent / "blobs" / ("b" * 64)) if layout == "hub" else (tmp_path / "outside.pt")
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            shard.symlink_to(target)
        except OSError:
            pytest.skip("Symlink creation unavailable on this host")
    torch.save({"weight": expected}, target)
    if layout == "outside":
        with pytest.raises(ValueError, match="escapes"):
            load_checkpoint_in_model(model, str(root), device_map={"": "cpu"})
    else:
        load_checkpoint_in_model(model, str(root), device_map={"": "cpu"})
        assert torch.equal(model.weight, expected)


class _MarkerPayload:
    def __init__(self, marker):
        self.marker = str(marker)

    def __reduce__(self):
        # Harmless local proof: the unsafe deserializer would create a directory.
        return os.mkdir, (self.marker,)


def test_stanza_does_not_retry_rejected_pickle_unsafely(tmp_path):
    from stanza.models.common.pretrain import Pretrain
    marker = tmp_path / "unsafe-deserialization-ran"
    checkpoint = tmp_path / "pretrain.pt"
    torch.save({"payload": _MarkerPayload(marker)}, checkpoint)
    with pytest.raises(Exception):
        Pretrain(filename=str(checkpoint)).load()
    assert not marker.exists(), "Stanza executed a rejected pickle payload"


def test_stanza_still_loads_tensor_pretrains(tmp_path):
    from stanza.models.common.pretrain import Pretrain, PretrainedWordVocab
    checkpoint = tmp_path / "pretrain.pt"
    vocab = PretrainedWordVocab(["hello"])
    expected = torch.zeros(len(vocab), 2)
    torch.save({"vocab": vocab.state_dict(), "emb": expected}, checkpoint)
    assert torch.equal(Pretrain(filename=str(checkpoint)).emb, expected)


@pytest.fixture
def nltk_sandbox(monkeypatch, tmp_path):
    from nltk import pathsec
    root = tmp_path / "allowed"
    root.mkdir(mode=0o700)
    monkeypatch.setattr(pathsec, "ENFORCE", True)
    monkeypatch.setattr(pathsec, "_get_allowed_roots", lambda: {root})
    return root


def test_nltk_perceptron_preserves_sandbox_for_reads_and_writes(nltk_sandbox, tmp_path):
    from nltk.tag.perceptron import AveragedPerceptron
    model = AveragedPerceptron()
    model.weights = {"bias": {"NN": 1.0}}
    valid = nltk_sandbox / "model.json"
    model.save(valid)
    restored = AveragedPerceptron()
    restored.load(valid)
    assert restored.weights == model.weights
    outside = tmp_path / "outside.json"
    with pytest.raises(PermissionError):
        model.save(outside)
    assert not outside.exists()
    outside.write_text('{}')
    with pytest.raises(PermissionError):
        model.load(outside)


def test_nltk_maxent_does_not_create_outside_directory(nltk_sandbox, tmp_path):
    import numpy as np
    from nltk.classify.maxent import save_maxent_params
    outside = tmp_path / "outside-dir"
    with pytest.raises(PermissionError):
        save_maxent_params(np.zeros(1), {}, [], {}, str(outside))
    assert not outside.exists()


def test_nltk_transition_parser_checks_path_before_read(nltk_sandbox, tmp_path, monkeypatch):
    from nltk.parse import transitionparser
    outside = tmp_path / "outside.pkl"
    outside.write_bytes(b"model")
    monkeypatch.setattr(transitionparser, "allowlisted_pickle_load", lambda *a, **kw: pytest.fail("opened outside model"))
    parser = transitionparser.TransitionParser("arc-standard")
    with pytest.raises(PermissionError):
        parser.parse([], str(outside))
