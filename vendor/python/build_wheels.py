"""Reproduce the security backports from hash-verified upstream wheels.

Run from any directory with Python 3.11+. Only source bytes are edited: neither
upstream package is imported, and no setup/build scripts are executed.
Upstream licences and all unmodified files are retained in the wheel.
"""
from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
from pathlib import Path
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parent
SUFFIX = "+voicestudio.1"

SHARD_GUARD = '''

def _validated_checkpoint_shards(checkpoint_folder, index):
    """VoiceStudio backport: contain index paths and reject special files."""
    from pathlib import Path, PurePosixPath, PureWindowsPath

    root = Path(checkpoint_folder).resolve()
    if not isinstance(index, dict) or not index:
        raise ValueError("Checkpoint weight_map must be a non-empty object")
    paths = []
    for name in index.values():
        if (not isinstance(name, str) or not name or "\\x00" in name
                or "\\\\" in name or PureWindowsPath(name).drive
                or PurePosixPath(name).is_absolute() or ".." in name.split("/")):
            raise ValueError("Invalid checkpoint shard path")
        path = root / name
        target = path.resolve()
        # HF snapshots use symlinks into the same repository's blobs directory.
        # No other outside-root symlinks are allowed, including another repo.
        hub_blob = (
            re.fullmatch(r"[0-9a-f]{40}", root.name)
            and root.parent.name == "snapshots"
            and root.parent.parent.name.startswith("models--")
            and target.parent == root.parent.parent / "blobs"
            and re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", target.name)
        )
        if not target.is_relative_to(root) and not hub_blob:
            raise ValueError("Checkpoint shard escapes its model directory")
        if not target.is_file():
            raise ValueError("Checkpoint shard must be a regular file")
        paths.append(str(path))
    return sorted(set(paths))

'''


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError("Upstream source changed; review the backport again")
    return text.replace(old, new)


def backport(name, files, version=None):
    if name == "transformers":
        from transformers_backport import patch
        patch(files, version)
    elif name == "accelerate":
        path = "accelerate/utils/modeling.py"
        source = files[path].decode()
        source = replace_once(source, "def load_checkpoint_in_model(", SHARD_GUARD + "def load_checkpoint_in_model(")
        source = replace_once(source,
            "        checkpoint_files = sorted(list(set(index.values())))\n"
            "        checkpoint_files = [os.path.join(checkpoint_folder, f) for f in checkpoint_files]",
            "        checkpoint_files = _validated_checkpoint_shards(checkpoint_folder, index)")
        files[path] = source.encode()
    elif name == "stanza":
        # All five explicit unsafe loads in the original 1.10.1 wheel. Fail
        # closed on unsupported pickle globals, including fallback branches.
        expected = {
            "stanza/models/classifiers/trainer.py": 1,
            "stanza/models/common/pretrain.py": 1,
            "stanza/models/constituency/base_trainer.py": 1,
            "stanza/models/coref/model.py": 2,
        }
        for path, count in expected.items():
            source = files[path].decode()
            # The pretrain module also mentions the argument in a comment.
            lines = source.splitlines(keepends=True)
            changed = 0
            for i, line in enumerate(lines):
                if "torch.load(" in line and "weights_only=False" in line:
                    lines[i] = line.replace("weights_only=False", "weights_only=True")
                    changed += 1
            if changed != count:
                raise ValueError(f"Unexpected unsafe loader count in {path}")
            files[path] = "".join(lines).encode()
        path = "stanza/models/lemma_classifier/base_model.py"
        files[path] = replace_once(files[path].decode(),
            "torch.load(filename, lambda storage, loc: storage)",
            "torch.load(filename, lambda storage, loc: storage, weights_only=True)").encode()
    elif name == "nltk":
        path = "nltk/parse/transitionparser.py"
        source = files[path].decode()
        source = replace_once(source, "import pickle\n", "import pickle\nfrom nltk import pathsec\n")
        source = replace_once(source, '            pickle.dump(model, open(modelfile, "wb"))',
            '            with pathsec.open(modelfile, "wb") as output:\n                pickle.dump(model, output)')
        source = replace_once(source, 'with open(modelFile, "rb") as f:', 'with pathsec.open(modelFile, "rb") as f:')
        files[path] = source.encode()
        path = "nltk/tag/perceptron.py"
        source = files[path].decode()
        source = replace_once(source, "from nltk import jsontags", "from nltk import jsontags, pathsec")
        source = replace_once(source, 'with open(path, "w") as fout:', 'with pathsec.open(path, "w") as fout:')
        source = replace_once(source, 'with open(path) as fin:', 'with pathsec.open(path) as fin:')
        files[path] = source.encode()
        path = "nltk/classify/maxent.py"
        source = files[path].decode()
        source = replace_once(source, '    from nltk.tabdata import MaxentEncoder',
            '    from nltk import pathsec\n    from nltk.tabdata import MaxentEncoder\n\n'
            '    pathsec.validate_path(tab_dir, context="save_maxent_params")')
        for file in ("weights.txt", "mapping.tab", "labels.txt", "alwayson.tab"):
            source = replace_once(source, f'with open(f"{{tab_dir}}/{file}", "w") as f:',
                f'with pathsec.open(f"{{tab_dir}}/{file}", "w") as f:')
        files[path] = source.encode()
    elif name == "pyannote_audio":
        # Keep the 3.x model/pipeline contract while moving its audio I/O off
        # APIs removed in TorchAudio 2.9. Tensor operations remain native.
        for path in ("pyannote/audio/core/io.py", "pyannote/audio/utils/protocol.py"):
            files[path] = replace_once(files[path].decode(), "import torchaudio\n",
                "from voicestudio_audio_compat import adapter as torchaudio\n").encode()
        path = "pyannote/audio/tasks/segmentation/mixins.py"
        files[path] = replace_once(files[path].decode(), "from torchaudio import AudioMetaData",
            "from voicestudio_audio_compat.common import AudioMetaData").encode()
    elif name == "whisperx":
        path = "whisperx-3.7.4.dist-info/METADATA"
        source = files[path].decode()
        for old, new in (
            ("Requires-Dist: torch~=2.8.0", "Requires-Dist: torch>=2.13"),
            ("Requires-Dist: torchaudio~=2.8.0", "Requires-Dist: torchaudio>=2.11"),
            ('Requires-Dist: numpy<2.1.0,>=2.0.2;', 'Requires-Dist: numpy<2.3.0,>=2.0.2;'),
            ("Requires-Dist: av<16.0.0", "Requires-Dist: av>=14"),
            # WhisperX never imports Triton directly. CUDA torch supplies its
            # matched compiler; CPU installs must not pull this GPU compiler.
            ('Requires-Dist: triton>=3.3.0; sys_platform == "linux" and platform_machine == "x86_64"\n', ''),
        ):
            source = replace_once(source, old, new)
        files[path] = source.encode()
    else:
        raise ValueError(name)


def build(spec):
    with urllib.request.urlopen(spec["url"], timeout=60) as response:
        archive = response.read()
    if hashlib.sha256(archive).hexdigest() != spec["sha256"]:
        raise ValueError("Upstream wheel hash mismatch")
    wheel = zipfile.ZipFile(io.BytesIO(archive))
    files = {name: wheel.read(name) for name in wheel.namelist() if not name.endswith("/")}
    name, version = spec["name"], spec["version"]
    backport(name, files, version)
    new_version = version + SUFFIX
    old_info, new_info = f"{name}-{version}.dist-info/", f"{name}-{new_version}.dist-info/"
    metadata = old_info + "METADATA"
    files[metadata] = replace_once(files[metadata].decode(), f"\nVersion: {version}\n", f"\nVersion: {new_version}\n").encode()
    if name == "pyannote_audio":
        files[metadata] = files[metadata].replace(b"\n\n",
            b"\nRequires-Dist: voicestudio-audio-compat==0.1.0\n\n", 1)
    files = {key.replace(old_info, new_info): value for key, value in files.items()
             if key != old_info + "RECORD"}
    # The upstream source's public version must match distribution metadata.
    package = "pyannote/audio" if name == "pyannote_audio" else name
    for key in (f"{package}/__init__.py", f"{package}/_version.py", f"{package}/version.py"):
        if key in files:
            files[key] = files[key].replace(f'__version__ = "{version}"'.encode(), f'__version__ = "{new_version}"'.encode())
    if f"{name}/VERSION" in files:
        if files[f"{name}/VERSION"].decode().strip() != version:
            raise ValueError("Unexpected upstream VERSION file")
        files[f"{name}/VERSION"] = (new_version + "\n").encode()
    records = io.StringIO(newline="")
    writer = csv.writer(records, lineterminator="\n")
    for path, data in sorted(files.items()):
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
        writer.writerow((path, "sha256=" + digest, len(data)))
    writer.writerow((new_info + "RECORD", "", ""))
    files[new_info + "RECORD"] = records.getvalue().encode()
    destination = ROOT / "wheels" / f"{name}-{new_version}-py3-none-any.whl"
    destination.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as output:
        for path, data in sorted(files.items()):
            info = zipfile.ZipInfo(path, date_time=(2026, 10, 13, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            output.writestr(info, data)
    print(destination.name, hashlib.sha256(destination.read_bytes()).hexdigest())


if __name__ == "__main__":
    for source in json.loads((ROOT / "sources.json").read_text()):
        build(source)
    from wheel_manifest import write
    write()
