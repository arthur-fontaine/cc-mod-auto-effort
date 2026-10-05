import { expect, test } from 'claude-code/testing'
import { buildRequest, clamp, clip, decide, DEFAULTS, normalizeModel, resolveConfig } from '../hooks/policy.js'

test('endpoint, key, and model come only from the environment', () => {
  const config = resolveConfig({ endpoint: 'https://env.example', apiKey: 'k', model: 'jev-from-env' })
  expect(config.endpoint).toBe('https://env.example')
  expect(config.model).toBe('jev-from-env')
  expect(config.missing).toEqual([])
  expect(config.timeoutMs).toBe(DEFAULTS.timeoutMs)
  expect(resolveConfig({}).missing).toEqual(['AUTO_EFFORT_ENDPOINT', 'AUTO_EFFORT_API_KEY', 'AUTO_EFFORT_MODEL'])
})

test('bad levels and numbers fall back to defaults', () => {
  const config = resolveConfig({ maxEffort: 'huge', minConfidence: 'abc' })
  expect(config.maxEffort).toBe('xhigh')
  expect(config.minConfidence).toBe(0.5)
})

test('clamp keeps a level inside the range', () => {
  expect(clamp('max', 'low', 'high')).toBe('high')
  expect(clamp('low', 'medium', 'max')).toBe('medium')
  expect(clamp('high', 'low', 'xhigh')).toBe('high')
})

test('long prompts are truncated at both ends', () => {
  const prompt = 'A'.repeat(5000) + 'B'.repeat(5000)
  const body = buildRequest({ prompt }, resolveConfig({}))
  const sent = body.state.latest_user_message
  expect(sent.length < 6100).toBe(true)
  expect(sent.startsWith('AAA')).toBe(true)
  expect(sent.endsWith('BBB')).toBe(true)
})

test('include_context false sends the prompt alone', () => {
  const body = buildRequest({ prompt: 'go', previousReply: 'Shall I?' }, resolveConfig({ includeContext: 'false' }))
  expect(body.state.previous_assistant_reply).toBeUndefined()
})

test('malformed answers keep the session effort', () => {
  const config = resolveConfig({})
  expect(decide({}, config).effort).toBe(null)
  expect(decide({ answers: { effort: { type: 'choice', choice: 'turbo', confidence: 1 } } }, config).effort).toBe(null)
})

test('the environment wins over what setup stored, and a key is never read from the store', () => {
  const config = resolveConfig({ endpoint: 'https://env.example', model: 'jev-env' }, { provider: 'jev', endpoint: 'https://stored.example', model: 'jev-stored', apiKey: 'stored-key' })
  expect(config.endpoint).toBe('https://env.example')
  expect(config.model).toBe('jev-env')
  expect(config.apiKey).toBeUndefined()
  expect(resolveConfig({}, { provider: 'jev', endpoint: 'https://stored.example', model: 'm' }).endpoint).toBe('https://stored.example')
})

test('the Nisev provider needs no endpoint or key and clips like the training data', () => {
  const config = resolveConfig({ provider: 'nisev' })
  expect(config.missing).toEqual([])
  expect(config.endpoint).toBe('http://127.0.0.1:8765/v1/systemone')
  const body = buildRequest({ prompt: 'x'.repeat(5000), previousReply: 'y'.repeat(2000) }, config)
  expect(Array.from(body.state.latest_user_message).length).toBe(3000 - 20 + '\n[… truncated …]\n'.length)
  expect(Array.from(body.state.previous_assistant_reply).length).toBe(800 - 20 + '\n[… truncated …]\n'.length)
})

test('clip counts code points, as Python does', () => {
  const text = '😀'.repeat(50)
  expect(Array.from(clip(text, 40)).length).toBe(20 + '\n[… truncated …]\n'.length)
})

test('model names are normalized', () => {
  expect(normalizeModel('claude-opus-5-5[1m]')).toBe('claude-opus-5-5')
  expect(normalizeModel('us.anthropic.claude-opus-5-5-v1')).toBe('claude-opus-5-5')
  expect(normalizeModel('opus')).toBe('claude-opus-5-5')
})

test('Haiku keeps its own effort on the Nisev provider', () => {
  const config = resolveConfig({ provider: 'nisev' })
  const response = { answers: { effort: { type: 'choice', choice: 'hard', probabilities: { hard: 1 } } } }
  expect(decide(response, config, 'claude-haiku-4-5').effort).toBe(null)
  expect(decide(response, config, 'claude-opus-5-5').effort).toBe('xhigh')
})

test("Cloudflare Workers AI's response, wrapped in result, is read like any other", () => {
  const config = resolveConfig({ endpoint: 'https://api.cloudflare.com/x', apiKey: 'k', model: 'clef-flash' })
  const answer = { type: 'choice', choice: 'high', confidence: 0.8 }
  expect(decide({ result: { answers: { effort: answer } }, success: true }, config).effort).toBe('high')
})

test('an endpoint in the environment wins over a saved Nisev setup', () => {
  const config = resolveConfig({ endpoint: 'https://opencode.ai/zen/v1/systemone' }, { provider: 'nisev' })
  expect(config.provider).toBe('jev')
})

test('a saved cloud setup never leaks into Nisev', () => {
  const stored = { provider: 'jev', endpoint: 'https://opencode.ai/zen/v1/systemone', model: 'jev-1.13' }
  const config = resolveConfig({ provider: 'nisev' }, stored)
  expect(config.model).toBe('arthur-fontaine/nisev-1.7b-GGUF:Q8_0')
  expect(config.endpoint).toBe('http://127.0.0.1:8765/v1/systemone')
  expect(config.port).toBe(8765)
})

test('Nisev needs a local endpoint, whose port it serves on', () => {
  expect(resolveConfig({ provider: 'nisev', endpoint: 'http://127.0.0.1:9000/v1/systemone' }).port).toBe(9000)
  expect(resolveConfig({ provider: 'nisev', endpoint: 'https://example.com/v1/systemone' }).problems.length).toBe(1)
})

test('Nisev is asked under its alias, whatever file llama.cpp serves', () => {
  const config = resolveConfig({ provider: 'nisev', model: '/models/nisev.gguf' })
  expect(buildRequest({ prompt: 'hi' }, config).model).toBe('nisev')
})

test("a cloud endpoint's settings left in the environment are reported for Nisev, not served", () => {
  const env = { provider: 'nisev', endpoint: 'https://opencode.ai/zen/v1/systemone', model: 'jev-1.13' }
  const { problems } = resolveConfig(env)
  expect(problems.length).toBe(2)
  expect(problems[0]).toMatch(/AUTO_EFFORT_ENDPOINT is https:\/\/opencode\.ai/)
  expect(problems[1]).toMatch(/AUTO_EFFORT_MODEL is jev-1\.13/)
})
