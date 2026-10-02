import { expect, mock, test } from 'claude-code/testing'

type Answer = { choice: string; confidence: number }

function jevReply({ choice, confidence }: Answer) {
  const body = {
    model: 'jev-1.13.0',
    answers: { effort: { type: 'choice', choice, probabilities: { [choice]: confidence }, confidence } },
    usage: { input_tokens: 300, output_tokens: 20 },
  }
  return { status: 200, ok: true, headers: {}, text: JSON.stringify(body) }
}

type Setup = {
  env?: Record<string, string>
  fetch?: (e: { url: string; init?: { headers?: Record<string, string>; body?: string } }) => unknown
  // What Claude Code answers for prompt.submit; by default the prompt enters
  submit?: (e: { text: string }) => unknown
}

const ENV = {
  AUTO_EFFORT_ENDPOINT: 'https://jev.example/v1/systemone',
  AUTO_EFFORT_API_KEY: 'test-key',
  AUTO_EFFORT_MODEL: 'jev-test',
}

// Stubs everything the mod reaches, and records what the model request carried.
function stub(on: any, { env = ENV, fetch, submit, clock = true }: Setup & { clock?: boolean }) {
  const seen = { requests: [] as any[], efforts: [] as unknown[], statuses: [] as unknown[] }
  if (clock) mock.clock(on)
  mock.env(on, env)
  mock.store(on, {})
  on('session.messages', () => ({ value: [{ role: 'assistant', text: 'Shall I refactor the parser?', toolUses: [] }] }))
  on('ui.status', ($: any, e: any) => {
    seen.statuses.push(e.text)
    return { value: undefined }
  })
  on('ui.log', () => ({ value: undefined }))
  on('http.fetch', ($: any, e: any) => {
    seen.requests.push({ url: e.url, headers: e.init?.headers, body: JSON.parse(e.init?.body ?? '{}') })
    return fetch ? fetch(e) : { value: jevReply({ choice: 'high', confidence: 0.9 }) }
  })
  on('prompt.submit', ($: any, e: any) => (submit ? submit(e) : { text: e.text }))
  on('turn.start', ($: any, e: any) => ({ turnId: e.turnId }))
  on('turn.complete', () => ({ text: '' }))
  on('turn.step', async function* ($: any, e: any) {
    seen.efforts.push(e.effort)
    return { turnId: e.turnId, index: e.index, answer: 'ok', toolUses: [], stopReason: 'end_turn', usage: null }
  })
  return seen
}

async function step($: any, fields: Record<string, unknown>) {
  const stream = $.turn.step({ turnId: 't1', index: 0, model: 'claude-opus-5-5', messageCount: 1, effort: 'medium', ...fields })
  let s = await stream.next()
  while (s.done !== true) s = await stream.next()
  return s.value
}

async function runTurn($: any, text: string, stepFields: Record<string, unknown> = {}) {
  await $.prompt.submit({ text, wait: false, origin: { kind: 'composer' } })
  await $.turn.start({ text, turnId: 't1' })
  await step($, stepFields)
}

test('a confident Jev answer sets the effort of the turn', async ($, on) => {
  const seen = stub(on, {})
  await runTurn($, 'Migrate every module from callbacks to async/await and make the tests pass')
  expect(seen.efforts).toEqual(['high'])
  expect(seen.statuses).toContain('effort high · 90%')
})

test('the request goes to the configured endpoint with the key and model', async ($, on) => {
  const seen = stub(on, {
    env: { AUTO_EFFORT_API_KEY: 'k-123', AUTO_EFFORT_ENDPOINT: 'https://other.example/v1/systemone', AUTO_EFFORT_MODEL: 'jev-1.13-free' },
  })
  await runTurn($, 'yes, do it')
  expect(seen.requests.length).toBe(1)
  const [req] = seen.requests
  expect(req.url).toBe('https://other.example/v1/systemone')
  expect(req.headers.Authorization).toBe('Bearer k-123')
  expect(req.body.model).toBe('jev-1.13-free')
  expect(req.body.state.latest_user_message).toBe('yes, do it')
  expect(req.body.state.previous_assistant_reply).toBe('Shall I refactor the parser?')
  expect(Object.keys(req.body.questions.effort.criteria)).toEqual(['low', 'default', 'high', 'xhigh', 'max'])
})

test('"default" keeps the session effort', async ($, on) => {
  const seen = stub(on, { fetch: () => ({ value: jevReply({ choice: 'default', confidence: 0.95 }) }) })
  await runTurn($, 'Add a --verbose flag to the CLI')
  expect(seen.efforts).toEqual(['medium'])
})

test('a low-confidence answer keeps the session effort', async ($, on) => {
  const seen = stub(on, { fetch: () => ({ value: jevReply({ choice: 'low', confidence: 0.3 }) }) })
  await runTurn($, 'hmm')
  expect(seen.efforts).toEqual(['medium'])
})

test('the pick is capped at max_effort', async ($, on) => {
  const seen = stub(on, { fetch: () => ({ value: jevReply({ choice: 'max', confidence: 0.99 }) }) })
  await runTurn($, 'Audit every endpoint, leave nothing unchecked')
  expect(seen.efforts).toEqual(['xhigh'])
})

test('an HTTP error keeps the session effort and lets the prompt through', async ($, on) => {
  const seen = stub(on, { fetch: () => ({ value: { status: 429, ok: false, headers: {}, text: 'rate limited' } }) })
  const entered = await $.prompt.submit({ text: 'Fix the flaky test', wait: false, origin: { kind: 'composer' } })
  expect(entered.text).toBe('Fix the flaky test')
  await $.turn.start({ text: 'Fix the flaky test', turnId: 't1' })
  await step($, {})
  expect(seen.efforts).toEqual(['medium'])
})

test('a failed fetch keeps the session effort', async ($, on) => {
  const seen = stub(on, { fetch: () => ({ deny: 'network refused' }) })
  await runTurn($, 'Fix the flaky test')
  expect(seen.efforts).toEqual(['medium'])
})

test('a slow Jev times out and keeps the session effort', async ($, on) => {
  const clock = mock.clock(on)
  const seen = stub(on, {
    clock: false,
    fetch: async () => {
      await clock.sleep(60_000)
      return { value: jevReply({ choice: 'max', confidence: 0.99 }) }
    },
  })
  const submitted = $.prompt.submit({ text: 'Audit everything', wait: false, origin: { kind: 'composer' } })
  await clock.advance(5_000)
  await submitted
  await $.turn.start({ text: 'Audit everything', turnId: 't1' })
  await step($, {})
  expect(seen.efforts).toEqual(['medium'])
})

test('with no environment Jev is not called, and the status names what to set', async ($, on) => {
  const seen = stub(on, { env: {} })
  await runTurn($, 'Refactor the parser')
  expect(seen.requests.length).toBe(0)
  expect(seen.efforts).toEqual(['medium'])
  expect(seen.statuses).toContain('auto-effort: set AUTO_EFFORT_ENDPOINT, AUTO_EFFORT_API_KEY, AUTO_EFFORT_MODEL')
})

test('there is no default endpoint or model', async ($, on) => {
  const seen = stub(on, { env: { AUTO_EFFORT_API_KEY: 'test-key' } })
  await runTurn($, 'Refactor the parser')
  expect(seen.requests.length).toBe(0)
  expect(seen.statuses).toContain('auto-effort: set AUTO_EFFORT_ENDPOINT, AUTO_EFFORT_MODEL')
})

test('OPENCODE_API_KEY is not read', async ($, on) => {
  const seen = stub(on, { env: { AUTO_EFFORT_ENDPOINT: ENV.AUTO_EFFORT_ENDPOINT, AUTO_EFFORT_MODEL: 'jev-test', OPENCODE_API_KEY: 'test-key' } })
  await runTurn($, 'Refactor the parser')
  expect(seen.requests.length).toBe(0)
})

test("a subagent's requests keep their own effort", async ($, on) => {
  const seen = stub(on, {})
  await runTurn($, 'Migrate the build to Vite', { agentId: 'agent-1', effort: 'low' })
  expect(seen.efforts).toEqual(['low'])
})

test('a model without effort gets none', async ($, on) => {
  const seen = stub(on, {})
  await runTurn($, 'Migrate the build to Vite', { effort: undefined })
  expect(seen.efforts).toEqual([undefined])
})

test('the decision does not leak into the next turn', async ($, on) => {
  const seen = stub(on, {})
  await runTurn($, 'Migrate the build to Vite')
  await $.turn.complete({ turnId: 't1', answer: 'done', durationMs: 10, isAborted: false, usage: null })
  await $.turn.start({ text: '', turnId: 't2' })
  await step($, { turnId: 't2' })
  expect(seen.efforts).toEqual(['high', 'medium'])
})

test('a prompt typed during a turn applies to its own turn, not the running one', async ($, on) => {
  const seen = stub(on, { fetch: () => ({ value: jevReply({ choice: 'xhigh', confidence: 0.9 }) }) })
  await $.turn.start({ text: '', turnId: 't1' })
  await $.prompt.submit({ text: 'Then audit the auth module', wait: false, origin: { kind: 'composer' }, turnId: 't1' })
  await step($, {})
  await $.turn.complete({ turnId: 't1', answer: 'done', durationMs: 10, isAborted: false, usage: null })
  await $.turn.start({ text: 'Then audit the auth module', turnId: 't2' })
  await step($, { turnId: 't2' })
  expect(seen.efforts).toEqual(['medium', 'xhigh'])
})

test('a message delivered into a running turn is not classified', async ($, on) => {
  const seen = stub(on, {})
  await runTurn($, 'Fix the flaky test')
  await $.prompt.submit({ text: 'Status? One line.', wait: false, origin: { kind: 'peer' }, turnId: 't1' })
  await step($, {})
  expect(seen.requests.length).toBe(1)
  expect(seen.efforts).toEqual(['high', 'high'])
})

test("a dropped prompt's decision does not reach the next turn", async ($, on) => {
  const seen = stub(on, { submit: () => ({ drop: 'blocked by another mod' }) })
  await $.prompt.submit({ text: 'Migrate the build to Vite', wait: false, origin: { kind: 'composer' } })
  await $.turn.start({ text: '', turnId: 't1' })
  await step($, {})
  expect(seen.efforts).toEqual(['medium'])
})

test('/auto-effort off stops the overrides', async ($, on) => {
  const seen = stub(on, {})
  const reply = await $.command.run({ command: 'auto-effort', args: 'off' })
  expect(reply.text).toBe('Jev effort picker turned off.')
  await runTurn($, 'Migrate the build to Vite')
  expect(seen.requests.length).toBe(0)
  expect(seen.efforts).toEqual(['medium'])
})

test('/auto-effort status reports the configuration', async ($, on) => {
  stub(on, {})
  const reply = await $.command.run({ command: 'auto-effort', args: '' })
  expect(reply.text).toMatch(/Endpoint: https:\/\/jev\.example\/v1\/systemone · model jev-test/)
  expect(reply.text).toMatch(/API key: set/)
  expect(reply.text).not.toMatch(/Missing/)
})

test('/auto-effort status lists unset variables', async ($, on) => {
  stub(on, { env: {} })
  const reply = await $.command.run({ command: 'auto-effort', args: 'status' })
  expect(reply.text).toMatch(/Endpoint: unset · model unset/)
  expect(reply.text).toMatch(/Missing: AUTO_EFFORT_ENDPOINT, AUTO_EFFORT_API_KEY, AUTO_EFFORT_MODEL/)
})
