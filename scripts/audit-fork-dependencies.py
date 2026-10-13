"""Audit lock entries, including upstream versions of locally patched wheels.

Local version suffixes must not make vulnerabilities disappear from scanners.
This is a raw advisory report, not an exploitability decision or safety claim.
"""
import argparse
import hashlib
import json
from pathlib import Path
import tomllib
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def queries_for_lock(lock, root=ROOT):
    queries, sources = [], []
    for package in lock["package"]:
        source = package.get("source", {})
        if not source.get("registry") and not source.get("path", "").endswith(".whl"):
            continue
        if source.get("path"):
            wheel = root / source["path"]
            digest = "sha256:" + hashlib.sha256(wheel.read_bytes()).hexdigest()
            if digest not in {item.get("hash") for item in package.get("wheels", [])}:
                raise ValueError(f"Local wheel differs from uv.lock: {package['name']}")
        queries.append({"package": {"name": package["name"], "ecosystem": "PyPI"},
                        "version": package["version"].split("+", 1)[0]})
        sources.append({"installed_version": package["version"], "source": source,
                        "local_backport": bool(source.get("path"))})
    return queries, sources


def audit(lock):
    queries, sources = queries_for_lock(lock)
    # Optional engines have separate environments. Include every shipped
    # backport even when it is absent from the main application's lock.
    known = {(q["package"]["name"].replace("_", "-"), q["version"]) for q in queries}
    for wheel in json.loads((ROOT / "vendor/python/wheel-manifest.json").read_text()):
        path = ROOT / "vendor/python" / wheel["path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != wheel["sha256"]:
            raise ValueError(f"Shipped wheel differs from manifest: {wheel['name']}")
        name, version = wheel["name"].replace("_", "-"), wheel["version"].split("+", 1)[0]
        if (name, version) not in known:
            queries.append({"package": {"name": name, "ecosystem": "PyPI"}, "version": version})
            sources.append({"installed_version": wheel["version"],
                            "source": {"path": str(path.relative_to(ROOT))}, "local_backport": True})
    matches = []
    for offset in range(0, len(queries), 100):
        batch = queries[offset:offset + 100]
        request = urllib.request.Request("https://api.osv.dev/v1/querybatch",
            data=json.dumps({"queries": batch}).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=60) as response:
            outcomes = json.load(response)["results"]
        if len(outcomes) != len(batch):
            raise ValueError("Incomplete OSV response")
        for i, outcome in enumerate(outcomes):
            if outcome.get("vulns"):
                matches.append({**batch[i], **sources[offset + i],
                                "advisories": [v["id"] for v in outcome["vulns"]]})
    return {"queried_lock_entries": len(queries), "matches": matches,
            "note": "Backports are still reported at their upstream version; verify their regression tests separately."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(tomllib.loads((ROOT / "uv.lock").read_text()))
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered)
    else:
        print(rendered, end="")
    raise SystemExit(1 if result["matches"] else 0)
