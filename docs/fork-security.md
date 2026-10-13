# Security fork

This branch starts from upstream commit
`2fb9bf23e42da002de1fab61e661066e956365f1` (2026-10-12), rather than the older
v0.5.6 installer that was originally reviewed. This is a source hardening branch,
not a signed release or a claim that all vulnerabilities have been removed.

## Included upstream fixes

- The browser guard rejects foreign website requests and DNS rebinding against
  the local API, including protected WebSocket connections.
- Hugging Face account credentials are restricted to approved HTTPS Hugging Face
  endpoints. Mirror downloads explicitly disable implicit token lookup.
- Segmented model downloads validate repository filenames, commit identifiers,
  and blob identifiers before creating cache paths.
- Translation agents run in a temporary workspace with tool/config restrictions.
  Dialogue is treated as untrusted input and returned translations are validated.

## Additional changes on this branch

- A download range is checked before writing each chunk, so an oversized response
  cannot corrupt a completed neighboring range. A single-stream fallback also
  stops before writing beyond its known expected file size.
- Translation subprocesses no longer inherit `OMNIVOICE_*`, `VOICESTUDIO_*`,
  `HF_*`, either legacy Hugging Face token alias, `OPENCODE_CONFIG*`, or
  `OPENCODE_PERMISSION`. Matching is case-insensitive. This prevents inherited
  OpenCode inline configuration from overriding the generated restrictive config
  and reduces accidental exposure of studio credentials. Provider authentication
  and normal OS settings remain available. This is defense in depth: third-party
  CLIs still run as the user and are not an operating-system security sandbox.
- Upstream updates are disabled in the bundled configuration, runtime updater,
  release listing, package feed, and Linux AppImage metadata. An environment
  variable cannot enable them. Re-enabling requires an intentional source change,
  a review of the target release channel, and a new build.

## Model source and dependency hardening (2026-10-13)

- The legacy Nano loader pins both the model revision and executable-code
  revision. Its audio tokenizer uses a separately pinned local snapshot. The
  Nano sidecar also resolves both snapshots before handing them to the runtime.
- MOSS v1.5 resolves its model and codec independently and passes local-only paths
  to the processor/model. A mutable branch name cannot accidentally become a
  codec code revision. Pins prevent silent upstream changes; they do not turn
  downloaded Python into a sandbox.
- Custom Nano models/codecs require an explicit trust setting and an immutable
  SHA for each override. Custom local directories are rejected because a Hub
  revision does not authenticate their contents. v1.5 also rejects this bypass.
- Compatible JavaScript upgrades remove 15 of the 17 reported advisory findings.
  The two packages without fixed published releases have maintained patches:
  `braces` rejects excessive parser nesting before recursive AST traversal, and
  the used `sprintf-js` entrypoint clamps numeric precision to ECMAScript limits.
  Invalid brace patterns still raise a normal parsing error; callers must handle
  invalid input. The unused standalone browser distribution of sprintf-js is not
  imported by the app and is not covered by this patch.
- Setuptools is upgraded to 84.0.0. The locked WhisperX, faster-whisper and
  pyannote wheels no longer directly import `pkg_resources`.
- Accelerate, Stanza and NLTK use reproducible, locally patched wheels. Accelerate
  contains shard paths and rejects special files while preserving valid Hub cache
  links; Stanza always requests restricted checkpoint deserialization; NLTK model
  reads/writes use its existing path-security checks. The patches retain the
  dependency versions required by Argos and the speech libraries.

See [backport provenance and maintenance](../vendor/python/README.md). The
patched wheels ship with desktop resources and are staged during setup. Docker
and Colab explicitly include the local wheel directory. Source installs use
`uv sync --frozen`; a manual pip install must include
`--find-links vendor/python/wheels` and the existing torch constraints.

## Native runtime compatibility

The main lock now uses PyTorch 2.13.0, TorchAudio 2.11.0 and TorchVision 0.28.0.
CPU and CUDA locks remain separate; NVIDIA uses the official CUDA 13.0 index,
and the opt-in Linux ROCm recipe uses the official ROCm 7.1 index. Native macOS
wheels require Apple Silicon and macOS 14 or newer. Intel Macs retain remote
backend support. CUDA 13 also requires a compatible NVIDIA driver/GPU; preflight now requires
R580 or newer, following [NVIDIA's compatibility table](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html).

All engines remain in the catalogue. Managed torch-based sidecars and the four
standalone bootstrap paths apply a shared override file and install the audio
adapter. Existing managed engines receive a dependency repair marker; their
source checkouts and model weights are retained. Shared installer arguments
ignore upstream source mappings and select the patched host index, including
manual-clone bootstraps. User-owned environments are
not rewritten automatically and retain their own dependency/security status.
The adapter does not alter safe model deserialization or native tensor kernels.
Older Transformers API families receive security backports rather than an
incompatible major upgrade. Optional engine overrides also raise Datasets,
fsspec, orjson, Keras and Protobuf above their vulnerable versions. These are
maintained fork dependencies, not upstream releases; future advisories still
require review.

The locked CTranslate2 wheel still needs CUDA 12 cuBLAS and cuDNN 9. Desktop,
source and Docker setup supply those separately and probe them before inference.
The official GPU image tags were checked, but CUDA/ROCm execution cannot be
verified on this Apple Silicon laptop.

Advisory scanners still identify locally backported upstream versions. Run
`python scripts/audit-fork-dependencies.py` to include the upstream versions of
our local wheels in scans, and run the backport regressions to verify the fixes.
No advisory is silently suppressed. This is not a claim of zero vulnerabilities.

Keep the app local, leave optional agent features and network sharing off unless
needed, and do not supply personal tokens or voice samples during validation.
Use a read-only, narrowly scoped Hugging Face token only for a model that actually
requires one. No model weights are needed for the security regression tests.

## Validation

The added regression tests reproduce failures on the base commit and pass with
the patches. Run targeted tests with `HF_HUB_OFFLINE=1` and an empty
`HF_HUB_CACHE`:

```sh
uv run pytest tests/test_download_write_bounds.py tests/test_fdl_segmented_download.py tests/test_download_range_contract.py tests/test_browser_guard.py tests/test_hf_token_never_reaches_mirror.py tests/test_gated_mirror_install.py
cd electron
bun x vitest run src/main/updater.test.ts src/main/agent-run-lifecycle.test.ts src/main/repair-agents.test.ts src/main/llm-agent-bridge.test.ts
```

`node --test tests/frontend/fork-security-policy.test.mjs` checks the fork's
runtime/build configuration and Linux packaging entrypoints. These checks do
not replace full application tests, real model-generation tests, a Windows/Linux
validation run, or signing/notarization. Consult the accompanying validation
record for the checks actually completed on this laptop.
