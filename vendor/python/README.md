# Python security backports

The backported wheels retain upstream licences and use the local version suffix
`+voicestudio.1`. The three security fixes retain dependency metadata. The
pyannote/WhisperX compatibility patches below deliberately change their dependency
contracts to support the tested native runtime.

`build_wheels.py` downloads only the exact upstream URLs in `sources.json`,
verifies SHA-256 before opening them, edits source bytes without importing the
packages, and rebuilds wheel RECORD hashes. Fixed ZIP timestamps and ordering
make the result reproducible. Review the builder alongside the binary wheels.

| Package | Fix | Advisory |
| --- | --- | --- |
| Accelerate 1.14.0 | Reject traversal, absolute/Windows paths, escaping symlinks and non-regular shards; allow normal files and same-repository Hub blob links | GHSA-4j2p-28q2-5m79 |
| Stanza 1.10.1 | All torch.load calls explicitly use weights_only=True; rejected pickle globals cannot trigger an unsafe retry | GHSA-v5jw-96jm-7h2c |
| NLTK 3.10.3 | TransitionParser, AveragedPerceptron and Maxent persistence use pathsec checks, including validation before directory creation | GHSA-8mgp-746c-j5xp |

| pyannote.audio 3.4.0 | Use the legacy audio adapter for metadata, reads and crops; retain the existing 3.x model API | Compatibility |
| WhisperX 3.7.4 | Require Torch >=2.13 and TorchAudio >=2.11; allow NumPy 2.2 and newer PyAV without changing the ASR/VAD/alignment code | Compatibility |

`voicestudio-audio-compat==0.1.0` is a small pure-Python adapter. Its SoundFile
implementation and metadata class come from the hash-verified TorchAudio 2.8
wheel recorded in `audio_compat/source.json`; its upstream BSD licence is
included. **No native library from TorchAudio/PyTorch 2.8 is copied.** Transforms,
tensor kernels and checkpoint deserialization still come from the patched native
runtime. `build_audio_compat.py` verifies the borrowed source before rebuilding
its wheel. There is no automatic startup hook: VoiceStudio explicitly activates
it before loading legacy engines, and the patched pyannote imports its adapter
directly. Unsupported SoundFile formats retain the native decoder/app ffmpeg
fallback.

Stanza models that require arbitrary legacy pickle execution are deliberately
rejected. There is no automatic unsafe compatibility fallback.

Rebuild from the repository root:

```sh
python vendor/python/build_audio_compat.py
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

## Optional-engine Transformers backports

Three API families remain available: 4.52.4 for IndexTTS/Confucius, 4.57.6 for
Nano, and 5.0.0 for MOSS v1.5. Each has `+voicestudio.1` metadata and is built
from the hash-verified official wheel in `sources.json`. No engine is removed.
The source changes are in `transformers_backport.py`:

- Private executable-kernel fields cannot be deserialized from model configs;
  Hub kernel names follow upstream's `kernels-community` restriction.
- LightGlue's nested config cannot grant remote-code trust.
- Custom generation resolves trust before caching executable code.
- Named chat templates cannot write outside their destination directory.
- Trainer RNG state requests restricted checkpoint deserialization explicitly.
- The 4.52 family includes upstream Marian, CLVP, TensorFlow weight-name and
  optimizer regular-expression denial-of-service fixes.

Upstream security changes used as references:
[a7f8e7f](https://github.com/huggingface/transformers/commit/a7f8e7ff37d87d1a1a0c8cf607971c607741452f),
[676559d](https://github.com/huggingface/transformers/commit/676559d5022b74aaa0cee1cee0842b7f27c5320e),
[cbc1651](https://github.com/huggingface/transformers/commit/cbc1651a032b923da7f4b44b3d0e6f68e6ba6b55),
[eaaaf84](https://github.com/huggingface/transformers/commit/eaaaf8494dd5386634ae37d1d122212fdc315be5),
[03c8082](https://github.com/huggingface/transformers/commit/03c8082ba4594c9b8d6fe190ca9bed0e5f8ca396),
[d37f751](https://github.com/huggingface/transformers/commit/d37f7517972f67e3f2194c000ed0f87f064e5099),
[701caef](https://github.com/huggingface/transformers/commit/701caef704e356dc2f9331cc3fd5df0eccb4720a),
[54a0216](https://github.com/huggingface/transformers/commit/54a02160eb030da9be18231c77791f2eb3a52216).

The 4.52 wheel has no LightGlue or Hub-kernel loader, and already checks custom
code trust before fetching. Version-only scanners still report these advisories;
they are retained in the audit output. Run `tests/test_transformers_backports.py`
in each engine environment. The main environment uses current Transformers and
skips these legacy-family tests.

WhisperX's unused direct Triton dependency is removed from its wheel metadata;
CUDA PyTorch still supplies its matching Triton compiler. CPU installs avoid it.
