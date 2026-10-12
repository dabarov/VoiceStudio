"""The security backports must reach every supported installation path."""
import ast
import base64
import csv
import hashlib
import io
import json
from pathlib import Path
import tomllib
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def test_locked_backports_have_valid_wheel_records_and_no_unsafe_stanza_loads():
    config = tomllib.loads((ROOT / "pyproject.toml").read_text())
    lock = tomllib.loads((ROOT / "uv.lock").read_text())
    for name in ("accelerate", "stanza", "nltk"):
        source = config["tool"]["uv"]["sources"][name]["path"]
        package = next(p for p in lock["package"] if p["name"] == name)
        assert package["version"].endswith("+voicestudio.1")
        assert package["source"]["path"] == source
        assert package["wheels"][0]["hash"] == "sha256:" + hashlib.sha256((ROOT / source).read_bytes()).hexdigest()
        with zipfile.ZipFile(ROOT / source) as wheel:
            record = next(n for n in wheel.namelist() if n.endswith(".dist-info/RECORD"))
            for path, digest, size in csv.reader(io.StringIO(wheel.read(record).decode())):
                if not digest:
                    assert path == record
                    continue
                data = wheel.read(path)
                assert len(data) == int(size)
                actual = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
                assert digest == "sha256=" + actual
            if name == "stanza":
                for path in wheel.namelist():
                    if not path.endswith('.py'):
                        continue
                    for node in ast.walk(ast.parse(wheel.read(path))):
                        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                                and isinstance(node.func.value, ast.Name)
                                and node.func.value.id == "torch" and node.func.attr == "load"):
                            assert any(k.arg == "weights_only" and isinstance(k.value, ast.Constant)
                                       and k.value.value is True for k in node.keywords), path


def test_docker_and_colab_supply_local_wheels_and_javascript_patches():
    docker = (ROOT / "deploy/Dockerfile").read_text()
    assert docker.index("COPY patches/") < docker.index("RUN bun install")
    assert docker.index("COPY vendor/python/wheels/") < docker.index("RUN uv pip install")
    assert "--find-links vendor/python/wheels" in docker
    notebook = json.loads((ROOT / "notebooks/OmniVoice_Studio_Colab.ipynb").read_text())
    source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    assert '"--find-links", "vendor/python/wheels"' in source
