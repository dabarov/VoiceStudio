import assert from 'node:assert/strict';
import { test } from 'node:test';
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { DESKTOP_UPDATES_ENABLED } from '../../config/app-config.mjs';
import builder from '../../electron/electron-builder.config.mjs';

test('fork runtime and packaging both disable upstream installers', () => {
  assert.equal(DESKTOP_UPDATES_ENABLED, false);
  assert.equal(builder.publish, null);
});

test('fork packaging skips Linux update embedding and feed checks without touching release files', () => {
  for (const script of ['scripts/embed-appimage-update.mjs', 'tests/update-package-contract.mjs']) {
    const path = fileURLToPath(new URL(`../../electron/${script}`, import.meta.url));
    const output = execFileSync(process.execPath, [path], { encoding: 'utf8' });
    assert.match(output, /Upstream updates disabled/);
  }
});
