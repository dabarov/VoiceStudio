"""Legacy SoundFile audio I/O over the current native TorchAudio runtime.

Only audio file I/O is adapted. Tensor operations, transforms and model loading
remain provided by the installed, patched PyTorch/TorchAudio distributions.
"""
import torchaudio as _native
from . import soundfile_backend as _soundfile
from .common import AudioMetaData

_native_load = _native.load
_native_save = _native.save


def __getattr__(name):
    return getattr(_native, name)


def list_audio_backends():
    return ["soundfile"] if _soundfile._IS_SOUNDFILE_AVAILABLE else []


def get_audio_backend():
    return "soundfile"


def set_audio_backend(backend):
    if backend not in (None, "soundfile"):
        raise ValueError("VoiceStudio's legacy audio adapter supports the soundfile backend")


def info(uri, format=None, buffer_size=4096, backend=None):
    if backend not in (None, "soundfile"):
        raise ValueError("Legacy metadata inspection requires the soundfile backend")
    return _soundfile.info(uri, format=format)


def load(uri, frame_offset=0, num_frames=-1, normalize=True, channels_first=True,
         format=None, buffer_size=4096, backend=None):
    if backend not in (None, "soundfile"):
        return _native_load(uri, frame_offset=frame_offset, num_frames=num_frames,
                            normalize=normalize, channels_first=channels_first, format=format)
    position = uri.tell() if hasattr(uri, "tell") and getattr(uri, "seekable", lambda: False)() else None
    try:
        return _soundfile.load(uri, frame_offset=frame_offset, num_frames=num_frames,
                               normalize=normalize, channels_first=channels_first, format=format)
    except _soundfile.soundfile.LibsndfileError:
        if backend == "soundfile":
            raise
        if position is not None:
            uri.seek(position)
        # Retain TorchCodec for formats SoundFile cannot decode (AAC/M4A).
        # If unavailable, the app's existing bounded ffmpeg fallback handles it.
        return _native_load(uri, frame_offset=frame_offset, num_frames=num_frames,
                            normalize=normalize, channels_first=channels_first, format=format)


def save(uri, src, sample_rate, channels_first=True, format=None, encoding=None,
         bits_per_sample=None, buffer_size=4096, backend=None, compression=None):
    if backend not in (None, "soundfile"):
        return _native_save(uri, src, sample_rate, channels_first=channels_first,
                            format=format, encoding=encoding, bits_per_sample=bits_per_sample,
                            compression=compression)
    # SoundFile accepts CPU tensors, just as the previous TorchAudio backend did.
    return _soundfile.save(uri, src, sample_rate, channels_first=channels_first,
                           format=format, encoding=encoding, bits_per_sample=bits_per_sample,
                           compression=compression)
