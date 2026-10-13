"""Real legacy audio operations on the patched native stack; no model download."""
import io
import subprocess
import sys

import pytest
import torch


@pytest.mark.parametrize("bits", [16, 24, 32])
def test_wav_metadata_read_crop_and_resampling(bits):
    from voicestudio_audio_compat import adapter
    from pyannote.audio.core.io import Audio
    from pyannote.core import Segment
    signal = torch.sin(torch.arange(16000, dtype=torch.float32) * 0.01)[None, :]
    wav = io.BytesIO()
    adapter.save(wav, signal, 16000, format="wav", encoding="PCM_S", bits_per_sample=bits)
    wav.seek(0)
    info = adapter.info(wav)
    assert (info.sample_rate, info.num_frames, info.num_channels, info.bits_per_sample) == (16000, 16000, 1, bits)
    wav.seek(0)
    data, rate = Audio(sample_rate=8000)(wav)
    assert data.shape == (1, 8000) and rate == 8000
    assert torch.isfinite(data).all() and data.abs().max() > 0.9
    wav.seek(0)
    cropped, rate = Audio(sample_rate=16000).crop(wav, Segment(0.2, 0.4))
    assert cropped.shape == (1, 3200) and rate == 16000
    assert torch.allclose(cropped, signal[:, 3200:6400], atol=1e-4)


def test_adapter_preserves_integer_wav_and_channel_order():
    from voicestudio_audio_compat import adapter
    signal = torch.tensor([[1, 2, -3], [4, -5, 6]], dtype=torch.int16)
    wav = io.BytesIO()
    adapter.save(wav, signal, 16000, format="wav", bits_per_sample=16)
    wav.seek(0)
    restored, rate = adapter.load(wav, normalize=False, channels_first=False)
    assert torch.equal(restored, signal.T) and restored.dtype == torch.int16


def test_activation_never_replaces_checkpoint_loading_or_native_operations():
    result = subprocess.run([sys.executable, "-c", "\n".join([
        "import torch, torchaudio, torchvision",
        "from voicestudio_audio_compat import activate",
        "load = torch.load; resample = torchaudio.functional.resample",
        "activate(); activate()",
        "assert torch.load is load and torchaudio.functional.resample is resample",
        "assert torchaudio.list_audio_backends() == ['soundfile']",
        "assert torch.nn.Conv1d(1, 2, 3)(torch.zeros(1, 1, 16)).shape == (1, 2, 14)",
        "assert torchvision.ops.nms(torch.tensor([[0., 0., 1., 1.]]), torch.tensor([1.]), .5).tolist() == [0]",
    ])], capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stderr


def test_unknown_soundfile_format_rewinds_before_native_decoder(monkeypatch):
    from voicestudio_audio_compat import adapter
    source = io.BytesIO(b"unknown container")
    def failed(*a, **kw):
        source.read()
        raise adapter._soundfile.soundfile.LibsndfileError(1)
    monkeypatch.setattr(adapter._soundfile, "load", failed)
    def native(stream, **kw):
        assert stream.tell() == 0
        return torch.zeros(1, 1), 8000
    monkeypatch.setattr(adapter, "_native_load", native)
    assert adapter.load(source)[1] == 8000
