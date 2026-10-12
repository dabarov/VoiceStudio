# Python security backports

These wheels retain upstream licences and dependency metadata, with the local
version suffix `+voicestudio.1`. They preserve the current Argos/WhisperX dependency
contracts instead of overriding conflicting requirements.

`build_wheels.py` downloads only the exact upstream URLs in `sources.json`,
verifies SHA-256 before opening them, edits source bytes without importing the
packages, and rebuilds wheel RECORD hashes. Fixed ZIP timestamps and ordering
make the result reproducible. Review the builder alongside the binary wheels.

| Package | Fix | Advisory |
| --- | --- | --- |
| Accelerate 1.14.0 | Reject traversal, absolute/Windows paths, escaping symlinks and non-regular shards; allow normal files and same-repository Hub blob links | GHSA-4j2p-28q2-5m79 |
| Stanza 1.10.1 | All torch.load calls explicitly use weights_only=True; rejected pickle globals cannot trigger an unsafe retry | GHSA-v5jw-96jm-7h2c |
| NLTK 3.10.3 | TransitionParser, AveragedPerceptron and Maxent persistence use pathsec checks, including validation before directory creation | GHSA-8mgp-746c-j5xp |

Stanza models that require arbitrary legacy pickle execution are deliberately
rejected. There is no automatic unsafe compatibility fallback.

Rebuild from the repository root:

```sh
python vendor/python/build_wheels.py
git diff --exit-code -- vendor/python/wheels
```

After an intentional backport change, regenerate uv.lock, then run:

```sh
uv run pytest tests/test_dependency_backports.py tests/test_fork_dependency_distribution.py
python scripts/audit-fork-dependencies.py
```

The audit queries the original upstream versions and checks local wheel hashes;
backported packages remain visible in the report. A future advisory must be
reviewed separately. Do not interpret the local suffix as a blanket exemption.
Use a new local version suffix for changes after this patch is released.

The desktop bundles the wheels and copies them into its managed environment.
Docker and Colab pass `--find-links vendor/python/wheels`; conventional pip/uv-pip
installs must do the same. The original Python project is not republished as a
standalone PyPI package by this fork.
