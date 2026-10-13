"""Explicit compatibility setup for legacy speech engines on patched PyTorch."""


def activate():
    """Restore the removed file I/O API before loading a legacy engine.

    This never alters torch.load, deserialization rules, native tensor kernels,
    GPU selection or user files. Older user-managed TorchAudio is left alone.
    """
    import torchaudio
    if hasattr(torchaudio, "info"):
        return
    from . import adapter
    for name in ("AudioMetaData", "info", "load", "save", "list_audio_backends",
                 "get_audio_backend", "set_audio_backend"):
        setattr(torchaudio, name, getattr(adapter, name))
