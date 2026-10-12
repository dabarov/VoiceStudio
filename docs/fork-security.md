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

## Remaining work before installation

The Python lock still has advisory matches for Accelerate, NLTK, setuptools,
Stanza, and PyTorch. A database match alone does not prove that an application
exposes the affected function, but these entries are not resolved by this patch.
In particular:

- Accelerate 1.15.0 still joins untrusted checkpoint shard paths without a
  containment check; merely upgrading past the advisory's last listed version
  would not establish that the bug is fixed.
- Argos Translate 1.11.0 requires Stanza 1.10.1, while the unsafe checkpoint
  fallback is fixed in Stanza 1.12.2. Upgrading requires compatibility work rather
  than silently overriding that dependency requirement.
- PyTorch upgrades must be coordinated with torchaudio, torchvision and the
  platform-specific engine environments. The current pins remain unchanged.
- The setuptools advisory concerns source-distribution building on macOS; the
  app retains its older version for `pkg_resources` compatibility.
- Optional engines can execute downloaded Python code or deserialize checkpoints.
  Pinning a model is useful but does not establish that all its code and nested
  dependencies have been audited. These engines need separate review.

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
