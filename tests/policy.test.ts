import { expect, test } from 'claude-code/testing'
import { buildRequest, clamp, decide, DEFAULTS, resolveConfig } from '../hooks/policy.js'

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
