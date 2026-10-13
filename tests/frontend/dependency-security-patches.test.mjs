import assert from 'node:assert/strict'
import { createRequire } from 'node:module'
import test from 'node:test'

const require = createRequire(new URL('../../electron/package.json', import.meta.url))
// Follow the real dependency chains, including Bun's isolated install layout.
function dependency(chain) {
  let loader = require
  for (const name of chain.slice(0, -1)) loader = createRequire(loader.resolve(name))
  return loader(chain.at(-1))
}
const braces = dependency(['shadcn', 'fast-glob', 'micromatch', 'braces'])
const { sprintf } = dependency(['electron-builder', 'app-builder-lib', '@electron/get', 'global-agent', 'roarr', 'sprintf-js'])

test('brace parser bounds both brace and parenthesis nesting before recursive walkers', () => {
  for (const [open, close] of [['{', '}'], ['(', ')'], ['{(', ')}']]) {
    const input = open.repeat(300) + 'a,b' + close.repeat(300)
    assert.throws(() => braces(input), { name: 'SyntaxError', message: 'Pattern nesting exceeds 128 levels' })
    assert.throws(() => braces.expand(input), { name: 'SyntaxError', message: 'Pattern nesting exceeds 128 levels' })
  }
  assert.deepEqual(braces.expand('voice-{en,fr}-{1..2}'), ['voice-en-1', 'voice-en-2', 'voice-fr-1', 'voice-fr-2'])
  assert.doesNotThrow(() => braces('\\{'.repeat(300)))
})

test('numeric formatting bounds precision at the ECMAScript limits', () => {
  for (const kind of ['f', 'e', 'g']) {
    assert.doesNotThrow(() => sprintf('%.' + '9'.repeat(200) + kind, 1.25))
    assert.equal(sprintf('%.99999' + kind, 1.25), sprintf('%.100' + kind, 1.25))
  }
  assert.equal(sprintf('%.0g', 1.25), sprintf('%.1g', 1.25))
  assert.equal(sprintf('%.0f', 1.75), '2')
  assert.equal(sprintf('%.2f', 1.25), '1.25')
  assert.equal(sprintf('%.150s', 'a'.repeat(200)), 'a'.repeat(150))
})
