"""Build the pure-Python legacy audio adapter, checking upstream source bytes."""
import base64
import csv
import hashlib
import io
import json
from pathlib import Path
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "audio_compat"
NAME = "voicestudio_audio_compat"
VERSION = "0.1.0"


def build():
    spec = json.loads((SOURCE / "source.json").read_text())
    archive = urllib.request.urlopen(spec["url"], timeout=60).read()
    if hashlib.sha256(archive).hexdigest() != spec["sha256"]:
        raise ValueError("Upstream hash mismatch")
    wheel = zipfile.ZipFile(io.BytesIO(archive))
    for origin, filename in (
        ("torchaudio/_backend/soundfile_backend.py", "soundfile_backend.py"),
        ("torchaudio/_backend/common.py", "common.py"),
        ("torchaudio/_internal/module_utils.py", "module_utils.py"),
    ):
        expected = wheel.read(origin).replace(
            b"from torchaudio._internal import module_utils as _mod_utils",
            b"from . import module_utils as _mod_utils")
        if (SOURCE / NAME / filename).read_bytes() != expected:
            raise ValueError(f"Unexpected change to upstream audio I/O source: {filename}")
    license_path = next(n for n in wheel.namelist() if n.endswith("/LICENSE"))
    if (SOURCE / NAME / "LICENSE").read_bytes() != wheel.read(license_path):
        raise ValueError("Upstream licence must be preserved")
    files = {str(p.relative_to(SOURCE)): p.read_bytes()
             for p in (SOURCE / NAME).iterdir() if p.is_file()}
    info = f"{NAME}-{VERSION}.dist-info/"
    files[info + "METADATA"] = (
        f"Metadata-Version: 2.1\nName: voicestudio-audio-compat\nVersion: {VERSION}\n"
        "Summary: Legacy audio file I/O for VoiceStudio's patched native runtime\n"
        "License: BSD-2-Clause\nRequires-Python: >=3.10\n"
        "Requires-Dist: soundfile>=0.12\nRequires-Dist: torch>=2.13\n"
        "Requires-Dist: torchaudio>=2.11\n\n").encode()
    files[info + "WHEEL"] = b"Wheel-Version: 1.0\nGenerator: voicestudio\nRoot-Is-Purelib: true\nTag: py3-none-any\n"
    records = io.StringIO(newline="")
    writer = csv.writer(records, lineterminator="\n")
    for path, data in sorted(files.items()):
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
        writer.writerow((path, "sha256=" + digest, len(data)))
    writer.writerow((info + "RECORD", "", ""))
    files[info + "RECORD"] = records.getvalue().encode()
    destination = ROOT / "wheels" / f"{NAME}-{VERSION}-py3-none-any.whl"
    with zipfile.ZipFile(destination, "w") as result:
        for path, data in sorted(files.items()):
            entry = zipfile.ZipInfo(path, (2026, 10, 13, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            entry.external_attr = 0o644 << 16
            result.writestr(entry, data)
    print(destination.name, hashlib.sha256(destination.read_bytes()).hexdigest())


if __name__ == "__main__":
    build()
    from wheel_manifest import write
    write()
