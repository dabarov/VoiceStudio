"""Run in each pinned optional-engine Transformers environment, offline."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

transformers = pytest.importorskip("transformers")
VERSION = transformers.__version__.split("+", 1)[0]
pytestmark = pytest.mark.skipif(VERSION not in {"4.52.4", "4.57.6", "5.0.0"},
    reason="Exercised separately in all three optional-engine API environments")


def test_hub_config_cannot_set_private_executable_kernel_fields():
    config = transformers.PretrainedConfig.from_dict({
        "_attn_implementation_internal": "untrusted/kernel",
        "_experts_implementation_internal": "untrusted/experts",
        "hidden_size": 8,
    })
    assert config._attn_implementation != "untrusted/kernel"
    assert getattr(config, "_experts_implementation_internal", None) != "untrusted/experts"
    assert config.hidden_size == 8


@pytest.mark.skipif(VERSION == "4.52.4", reason="Hub-kernel loader absent in this API family")
def test_kernel_namespace_is_restricted():
    from transformers.integrations.hub_kernels import is_kernel
    assert is_kernel("kernels-community/flash-attn")
    assert not is_kernel("untrusted/kernel")
    assert not is_kernel("paged|untrusted/kernel")
    assert not is_kernel(None)


@pytest.mark.skipif(VERSION == "4.52.4", reason="LightGlue absent in this API family")
def test_nested_lightglue_config_cannot_authorize_remote_code(monkeypatch):
    module = __import__('transformers.models.lightglue.configuration_lightglue', fromlist=['LightGlueConfig'])
    remote = Mock(side_effect=AssertionError("remote code resolution reached"))
    monkeypatch.setattr(module.AutoConfig, "from_pretrained", remote)
    with pytest.raises(KeyError):
        module.LightGlueConfig(keypoint_detector_config={"model_type": "untrusted-model",
            "_name_or_path": "untrusted/model"}, trust_remote_code=True)
    remote.assert_not_called()
    # The supported default detector is retained.
    assert module.LightGlueConfig().keypoint_detector_config.model_type == "superpoint"


def test_custom_generation_declined_before_cache_write(monkeypatch):
    from transformers.generation import utils
    import transformers.utils
    if hasattr(utils, "file_exists"):
        monkeypatch.setattr(utils, "file_exists", lambda *a, **k: True)
    monkeypatch.setattr(transformers.utils, "has_file", lambda *a, **k: True)
    fetch = Mock(return_value="fake/module.py")
    monkeypatch.setattr(utils, "get_cached_module_file", fetch)
    with pytest.raises(ValueError):
        utils.GenerationMixin.load_custom_generate(object(), "untrusted/model", trust_remote_code=False)
    fetch.assert_not_called()


def tokenizer(tmp_path):
    vocab = tmp_path / "vocab.json"
    vocab.write_text(json.dumps({"[PAD]": 0, "[UNK]": 1, "|": 2, "a": 3}))
    return transformers.Wav2Vec2CTCTokenizer(str(vocab), unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")


@pytest.mark.parametrize("name", ["../../escape", "../escape", "ABSOLUTE", "..\\escape"])
@pytest.mark.parametrize("kind", ["tokenizer", "processor"])
def test_named_templates_cannot_escape_destination(tmp_path, name, kind):
    target = tokenizer(tmp_path)
    if kind == "processor":
        target = transformers.Wav2Vec2Processor(transformers.Wav2Vec2FeatureExtractor(), target)
    if name == "ABSOLUTE":
        name = str(tmp_path / "escape")
    target.chat_template = {name: "untrusted content"}
    with pytest.raises(ValueError, match="Invalid chat template name"):
        target.save_pretrained(tmp_path / "saved")
    assert not (tmp_path / "escape.jinja").exists()


@pytest.mark.parametrize("kind", ["tokenizer", "processor"])
def test_regular_templates_still_save(tmp_path, kind):
    target = tokenizer(tmp_path)
    if kind == "processor":
        target = transformers.Wav2Vec2Processor(transformers.Wav2Vec2FeatureExtractor(), target)
    target.chat_template = {"default": "hello", "narration": "voice"}
    target.save_pretrained(tmp_path / "saved")
    assert (tmp_path / "saved/additional_chat_templates/narration.jinja").read_text() == "voice"


@pytest.mark.skipif(VERSION != "4.52.4", reason="Regression for the oldest engine API family")
def test_legacy_regex_inputs_remain_bounded():
    from transformers.models.marian.tokenization_marian import MarianTokenizer
    from transformers.models.clvp.number_normalizer import EnglishNormalizer
    from transformers.modeling_tf_pytorch_utils import convert_tf_weight_name_to_pt_weight_name
    instance = object.__new__(MarianTokenizer)
    assert instance.remove_language_code(">>en<<hello") == ([">>en<<"], "hello")
    text = ">>" + "x" * 100000
    assert instance.remove_language_code(text) == ([], text)
    with pytest.raises(ValueError):
        convert_tf_weight_name_to_pt_weight_name("/" + "a" * 5000 + "___b/")
    # Avoid huge numeric expansion; exercise the changed regexes with a
    # bounded replacement while retaining the real normalization path.
    normalizer = EnglishNormalizer()
    normalizer._expand_number = lambda match: "number"
    assert normalizer.normalize_numbers("1" * 20000) == "number"
