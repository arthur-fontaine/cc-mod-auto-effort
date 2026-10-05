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
  // The session's main model
  model?: string
  // Records what the mod stores, in place of mock.store
  saved?: Record<string, unknown>
}

const ENV = {
  AUTO_EFFORT_ENDPOINT: 'https://jev.example/v1/systemone',
  AUTO_EFFORT_API_KEY: 'test-key',
  AUTO_EFFORT_MODEL: 'jev-test',
}

// Stubs everything the mod reaches, and records what the model request carried.
function stub(on: any, { env = ENV, fetch, submit, model = 'claude-opus-5-5', saved, clock = true }: Setup & { clock?: boolean }) {
  const seen = { requests: [] as any[], efforts: [] as unknown[], statuses: [] as unknown[] }
  if (clock) mock.clock(on)
  mock.env(on, env)
  if (saved) {
    on('store.get', ($: any, e: any) => ({ value: saved[e.key] }))
    on('store.set', ($: any, e: any) => {
      saved[e.key] = e.value
      return { value: undefined }
    })
  } else {
    mock.store(on, {})
  }
  on('session.messages', () => ({ value: [{ role: 'assistant', text: 'Shall I refactor the parser?', toolUses: [] }] }))
  on('session.model', () => ({ value: model }))
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
  expect(req.body.state.model).toBe('claude-opus-5-5')
  expect(Object.keys(req.body.questions.effort.criteria)).toEqual(['low', 'medium', 'default', 'high', 'xhigh', 'max'])
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
  expect(seen.statuses).toContain('auto-effort: run /auto-effort setup, or set AUTO_EFFORT_ENDPOINT, AUTO_EFFORT_API_KEY, AUTO_EFFORT_MODEL')
})

test('there is no default endpoint or model', async ($, on) => {
  const seen = stub(on, { env: { AUTO_EFFORT_API_KEY: 'test-key' } })
  await runTurn($, 'Refactor the parser')
  expect(seen.requests.length).toBe(0)
  expect(seen.statuses).toContain('auto-effort: run /auto-effort setup, or set AUTO_EFFORT_ENDPOINT, AUTO_EFFORT_MODEL')
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
  expect(reply.text).toBe('Effort picker turned off.')
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

// The local provider: our fine-tuned model in llama-server, asked the category question.

function categoryReply(probabilities: Record<string, number>) {
  const choice = Object.entries(probabilities).sort((a, b) => b[1] - a[1])[0][0]
  const body = { model: 'auto-effort', answers: { effort: { type: 'choice', choice, probabilities, confidence: 0.5 } } }
  return { status: 200, ok: true, headers: {}, text: JSON.stringify(body) }
}

const MODELS = { status: 200, ok: true, headers: {}, text: JSON.stringify({ data: [{ id: 'auto-effort' }] }) }
const LOCAL_ENV = { AUTO_EFFORT_PROVIDER: 'local' }
const MULTI_STEP = { trivial: 0.02, light: 0.03, ordinary: 0.1, multi_step: 0.8, hard: 0.04, exhaustive: 0.01 }

function localFetch(probabilities: Record<string, number>) {
  return (e: { url: string }) => ({ value: e.url.endsWith('/v1/models') ? MODELS : categoryReply(probabilities) })
}

test('the local model is asked the category question, and its answer is mapped for the model', async ($, on) => {
  const seen = stub(on, { env: LOCAL_ENV, fetch: localFetch(MULTI_STEP) })
  await runTurn($, 'Refactor the payment module and make the tests pass')
  const [req] = seen.requests.filter((r) => r.url.endsWith('/v1/systemone'))
  expect(req.url).toBe('http://127.0.0.1:8765/v1/systemone')
  expect(Object.keys(req.body.questions.effort.criteria)).toEqual(['trivial', 'light', 'ordinary', 'multi_step', 'hard', 'exhaustive'])
  // Opus 5.5 defaults to medium: verified multi-step work is raised to high.
  expect(seen.efforts).toEqual(['high'])
  expect(seen.statuses).toContain('effort high · 80%')
})

test("ordinary work on the local model keeps the model's default", async ($, on) => {
  const seen = stub(on, {
    env: LOCAL_ENV,
    fetch: localFetch({ trivial: 0.05, light: 0.15, ordinary: 0.6, multi_step: 0.15, hard: 0.04, exhaustive: 0.01 }),
  })
  await runTurn($, 'Add a --verbose flag')
  expect(seen.efforts).toEqual(['medium'])
})

test('the same answer means the default on a model whose default is high', async ($, on) => {
  const seen = stub(on, { env: LOCAL_ENV, fetch: localFetch(MULTI_STEP), model: 'claude-sonnet-5-5' })
  await runTurn($, 'Refactor the payment module and make the tests pass', { effort: 'high' })
  expect(seen.efforts).toEqual(['high'])
  expect(seen.statuses).toContain('effort default · 90%')
})

test('when llama-server is not running, the mod starts it and the prompt keeps its effort', async ($, on) => {
  const spawned: string[][] = []
  const seen = stub(on, {
    env: LOCAL_ENV,
    fetch: () => ({ value: { status: 502, ok: false, headers: {}, text: '' } }),
  })
  on('process.spawn', async function* ($: any, e: any) {
    spawned.push([...e.argv])
    return { code: 0, signal: null }
  })
  await runTurn($, 'Refactor the parser')
  expect(spawned.length).toBe(1)
  expect(spawned[0]).toEqual([
    'llama-server', '-hf', 'arthur-fontaine/auto-effort-qwen3-1.7b-GGUF:Q8_0',
    '--host', '127.0.0.1', '--port', '8765', '--alias', 'auto-effort', '-c', '8192', '-np', '2',
  ])
  expect(seen.efforts).toEqual(['medium'])
})

test('a local GGUF file is passed with -m', async ($, on) => {
  const spawned: string[][] = []
  stub(on, { env: { ...LOCAL_ENV, AUTO_EFFORT_LOCAL_MODEL: '/models/auto-effort-Q8_0.gguf' }, fetch: () => ({ deny: 'refused' }) })
  on('process.spawn', async function* ($: any, e: any) {
    spawned.push([...e.argv])
    return { code: 0, signal: null }
  })
  await runTurn($, 'Refactor the parser')
  expect(spawned[0].slice(0, 3)).toEqual(['llama-server', '-m', '/models/auto-effort-Q8_0.gguf'])
})

// The wizard. The engine answers $.ui.ask through the AskUserQuestion tool.
function answer(on: any, labels: string[]) {
  const queue = [...labels]
  on('tool.call', { tool: 'AskUserQuestion' }, ($: any, e: any) => {
    const label = queue.shift()
    return { result: { questions: e.questions, answers: { [e.questions[0].question]: label } } }
  })
}

test('/auto-effort setup saves a Jev preset, never a key', async ($, on) => {
  const saved: Record<string, any> = {}
  stub(on, { env: {}, saved })
  answer(on, ['OpenCode Zen'])
  const reply = await $.command.run({ command: 'auto-effort', args: 'setup' })
  expect(saved.config).toEqual({ provider: 'jev', endpoint: 'https://opencode.ai/zen/v1/systemone', model: 'jev-1.13' })
  expect(reply.text).toMatch(/set AUTO_EFFORT_API_KEY/)
})

test('/auto-effort setup refuses a llama-server without decision models', async ($, on) => {
  stub(on, { env: {} })
  on('process.run', () => ({ value: { exitCode: 0, stdout: 'version: 0.5.0 (build 11300, commit abc)', stderr: '' } }))
  answer(on, ['Local model'])
  const reply = await $.command.run({ command: 'auto-effort', args: 'setup' })
  expect(reply.text).toMatch(/llama-server is build 11300; decision models need build 11361 or later/)
})

test('/auto-effort setup falls back to the unified llama CLI', async ($, on) => {
  const spawned: string[][] = []
  const saved: Record<string, any> = {}
  stub(on, { env: {}, fetch: () => ({ deny: 'refused' }), saved })
  on('process.run', ($: any, e: any) =>
    e.argv[0] === 'llama'
      ? { value: { exitCode: 0, stdout: 'version: 0.5.0-dev (build 11406, commit 8216c84)', stderr: '' } }
      : { deny: 'not found' })
  on('process.spawn', async function* ($: any, e: any) {
    spawned.push([...e.argv])
    return { code: 0, signal: null }
  })
  answer(on, ['Local model', 'Download and start'])
  const reply = await $.command.run({ command: 'auto-effort', args: 'setup' })
  expect(saved.config.llamaServer).toBe('llama')
  expect(spawned[0].slice(0, 3)).toEqual(['llama', 'serve', '-hf'])
  expect(reply.text).toMatch(/served by llama serve \(build 11406\)/)
})

test('/auto-effort setup downloads and starts the local model', async ($, on) => {
  const spawned: string[][] = []
  const saved: Record<string, any> = {}
  stub(on, { env: {}, fetch: () => ({ deny: 'refused' }), saved })
  on('process.run', () => ({ value: { exitCode: 0, stdout: 'version: 0.5.0-dev (build 11408, commit 9f12cd4)', stderr: '' } }))
  on('process.spawn', async function* ($: any, e: any) {
    spawned.push([...e.argv])
    return { code: 0, signal: null }
  })
  answer(on, ['Local model', 'Download and start'])
  const reply = await $.command.run({ command: 'auto-effort', args: 'setup' })
  expect(saved.config.provider).toBe('local')
  expect(spawned.length).toBe(1)
  expect(reply.text).toMatch(/The first start downloads the model/)
})
