// Sends sample prompts to the configured Jev endpoint with the mod's own
// question and prints what the mod would decide. Reads the same environment
// variables as the mod (OPENCODE_API_KEY, AUTO_EFFORT_*), and a .env file.
import { readFileSync } from 'node:fs'
import { buildRequest, decide, describe, resolveConfig } from '../hooks/policy.js'

try {
  for (const line of readFileSync(new URL('../.env', import.meta.url), 'utf8').split('\n')) {
    const m = line.match(/^\s*([A-Z0-9_]+)\s*=\s*(.*?)\s*$/)
    if (m && process.env[m[1]] === undefined) process.env[m[1]] = m[2].replace(/^(['"])(.*)\1$/, '$2')
  }
} catch {
  // No .env file.
}

const env = process.env
const config = resolveConfig({}, {
  apiKey: env.AUTO_EFFORT_API_KEY ?? env.OPENCODE_API_KEY,
  endpoint: env.AUTO_EFFORT_ENDPOINT,
  model: env.AUTO_EFFORT_MODEL,
  minConfidence: env.AUTO_EFFORT_MIN_CONFIDENCE,
  minEffort: env.AUTO_EFFORT_MIN_EFFORT,
  maxEffort: env.AUTO_EFFORT_MAX_EFFORT,
})
if (!config.apiKey) {
  console.error('Set OPENCODE_API_KEY (or AUTO_EFFORT_API_KEY), in the environment or in .env')
  process.exit(1)
}

// [prompt, previous assistant reply, level a person would pick]
const SAMPLES = [
  ['Fix the typo in the README title: "Instalation" should be "Installation"', undefined, 'low'],
  ['What does the `clip` function in policy.js do?', undefined, 'low'],
  ['thanks!', 'I renamed the variable and the tests pass.', 'low'],
  ['Add a --verbose flag to the CLI that prints each request', undefined, 'default'],
  ['The login button does nothing on Safari, can you fix it?', undefined, 'default'],
  ['yes, do it', 'Want me to migrate all 40 API handlers from callbacks to async/await and update their tests?', 'high'],
  ['Refactor the payment module into separate services, update every caller, and make sure the whole test suite passes', undefined, 'high'],
  ['We get a rare data race in the replication layer that corrupts writes under load. Find the root cause across the storage, network and consensus code, and verify the fix.', undefined, 'xhigh'],
  ['Do a full security audit of the auth system. Leave nothing unchecked, cost does not matter, use maximum effort.', undefined, 'max'],
]

console.log('Endpoint ' + config.endpoint + ' · model ' + config.model + '\n')
let tokens = 0
for (const [prompt, previousReply, expected] of SAMPLES) {
  const started = Date.now()
  const response = await fetch(config.endpoint, {
    method: 'POST',
    headers: { Authorization: 'Bearer ' + config.apiKey, 'Content-Type': 'application/json' },
    body: JSON.stringify(buildRequest({ prompt, previousReply }, config)),
  })
  const ms = Date.now() - started
  const text = await response.text()
  if (!response.ok) {
    console.log('HTTP ' + response.status + ' ' + text.slice(0, 200))
    continue
  }
  const body = JSON.parse(text)
  tokens += body.usage?.input_tokens ?? 0
  const answer = body.answers.effort
  const probs = Object.entries(answer.probabilities)
    .map(([k, v]) => k + ' ' + Math.round(v * 100))
    .join(', ')
  console.log((answer.choice === expected ? '✓' : '·') + ' ' + prompt.slice(0, 70))
  console.log('    expected ' + expected + ' · Jev ' + answer.choice + ' (' + probs + ') · ' + ms + 'ms')
  console.log('    mod → ' + describe(decide(body, config)))
}
console.log('\n' + tokens + ' input tokens in total')
