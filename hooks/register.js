import { atom, read, update } from 'claude-code'

import { buildRequest, decide, describe, JEV_PRESETS, LEVELS, NISEV, resolveConfig } from './policy.js'
import { parseBuild, parseServerOutput, serverArgs, serverCommand, servesOurModel } from './nisev.js'

const USER_ORIGINS = ['composer', 'bridge', 'sdk']

// Decision for the prompt whose turn hasn't started yet.
let pending = null
// turnId -> decision, for turns in flight.
const decisions = new Map()
let enabled = true
let lastDecision = null
let warnedMissing = false

// What the mod reports without asking anything of you (the last decision, Nisev starting) goes
// to the footer's mode labels; the status line, which the engine draws as a notice, is kept for
// what needs action.
const label = atom({ plugin: 'auto-effort', key: 'label' }, null)

function show($, text) {
  return update($, label, () => text).catch(() => {})
}
// The llama-server this module started, if any: { state: starting | downloading | ready | stopped }.
let server = null
// Set once a request reached Nisev, so a prompt doesn't probe the port every time.
let nisevConfirmed = false

function nisevState() {
  return server && server.state !== 'stopped' ? server.state : nisevConfirmed ? 'ready' : 'stopped'
}

async function llamaServerBuild($, binary) {
  try {
    const { stdout, stderr } = await $.process.run([binary, '--version'], { timeoutMs: 10_000 })
    return parseBuild(stdout + stderr)
  } catch {
    return null // Not installed.
  }
}

// The configured binary, else the first of llama-server and llama that runs: { binary, build }.
async function findServer($, config) {
  for (const binary of config.llamaServer ? [config.llamaServer] : NISEV.servers) {
    const build = await llamaServerBuild($, binary)
    if (build !== null) return { binary, build }
  }
  return { binary: config.llamaServer ?? NISEV.servers[0], build: null }
}

async function isServing($, config) {
  try {
    const response = await $.http.fetch(`http://127.0.0.1:${config.port}/v1/models`)
    return response.ok && servesOurModel(response.text)
  } catch {
    return false
  }
}

// Starts llama-server for the rest of the session unless our model is already served on the
// port; never waits. The first start downloads the model (-hf), reported on the status line.
async function startNisev($, config) {
  if (server && server.state !== 'stopped') return server.state
  if (await isServing($, config)) {
    nisevConfirmed = true
    return 'ready'
  }
  const current = { state: 'starting' }
  server = current
  const { binary } = await findServer($, config)
  // llama.cpp may print nothing while it downloads, so say up front that the first start can take minutes.
  void show($, 'Nisev starting · the first start downloads ' + NISEV.sizeLabel)
  void (async () => {
    try {
      // The loop is the child's life: it ends with the child or with this module.
      for await (const { text } of $.process.spawn({ argv: serverArgs(config, binary) })) {
        const seen = parseServerOutput(text)
        if (seen.ready && current.state !== 'ready') {
          current.state = 'ready'
          nisevConfirmed = true
          void show($, 'Nisev ready')
        } else if (seen.percent !== undefined && current.state !== 'ready') {
          current.state = 'downloading'
          void show($, 'Nisev downloading · ' + seen.percent + '%')
        }
        $.ui.log(text.trimEnd(), { to: 'debug' })
      }
    } catch (error) {
      $.ui.log('llama-server failed to start: ' + error.message, { to: 'debug' })
      void show($, null)
      $.ui.status('auto-effort: llama-server failed to start, see the debug log')
    }
    current.state = 'stopped'
    nisevConfirmed = false
  })()
  return current.state
}

async function storedConfig($) {
  try {
    const saved = await $.store.get('config')
    return saved && typeof saved === 'object' ? saved : {}
  } catch {
    return {}
  }
}

async function loadConfig($) {
  const env = {
    provider: await $.env.get('AUTO_EFFORT_PROVIDER'),
    apiKey: await $.env.get('AUTO_EFFORT_API_KEY'),
    endpoint: await $.env.get('AUTO_EFFORT_ENDPOINT'),
    model: await $.env.get('AUTO_EFFORT_MODEL'),
    minConfidence: await $.env.get('AUTO_EFFORT_MIN_CONFIDENCE'),
    timeoutMs: await $.env.get('AUTO_EFFORT_TIMEOUT_MS'),
    minEffort: await $.env.get('AUTO_EFFORT_MIN_EFFORT'),
    maxEffort: await $.env.get('AUTO_EFFORT_MAX_EFFORT'),
    includeContext: await $.env.get('AUTO_EFFORT_INCLUDE_CONTEXT'),
    llamaServer: await $.env.get('AUTO_EFFORT_LLAMA_SERVER'),
  }
  return resolveConfig(env, await storedConfig($))
}

async function previousReply($) {
  try {
    const messages = await $.session.messages()
    for (let i = messages.length - 1; i >= 0; i--) {
      if (messages[i].role === 'assistant' && messages[i].text) return messages[i].text
    }
  } catch {
    // No transcript to read; classify the prompt alone.
  }
  return undefined
}

async function sessionModel($) {
  try {
    return await $.session.model()
  } catch {
    return undefined
  }
}

// The same System One call for every provider: a hosted Jev, or llama-server on this machine.
async function askSystemOne($, config, body) {
  const request = $.http.fetch(config.endpoint, {
    method: 'POST',
    headers: { Authorization: 'Bearer ' + config.apiKey, 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  // $.http.fetch takes no timeout, and a slow classifier must never hold up the prompt.
  const timeout = $.clock.sleep(config.timeoutMs).then(() => null)
  const response = await Promise.race([request, timeout])
  if (response === null) throw new Error('timed out after ' + config.timeoutMs + 'ms')
  if (response.status === 404 && config.provider === 'nisev') {
    throw new Error('llama-server has no /v1/systemone; it needs build ' + NISEV.minBuild + ' or later')
  }
  if (!response.ok) throw new Error('HTTP ' + response.status + ' ' + response.text.slice(0, 200))
  return JSON.parse(response.text)
}

async function classify($, prompt) {
  const config = await loadConfig($)
  if (!config.provider) {
    if (!warnedMissing) {
      warnedMissing = true
      $.ui.status('auto-effort: run /auto-effort setup, or set ' + config.missing.join(', '))
    }
    return null
  }
  if (config.missing.length || config.problems.length) {
    if (!warnedMissing) {
      warnedMissing = true
      $.ui.status(config.problems.length
        ? 'auto-effort: Nisev is selected, but ' + config.problems.join('; ') + '. Unset them to use its defaults.'
        : 'auto-effort: set ' + config.missing.join(', '))
    }
    return null
  }
  if (config.provider === 'nisev' && nisevState() !== 'ready') {
    // Start it for the next prompts; this one keeps the session's effort.
    const state = await startNisev($, config)
    if (state !== 'ready') return { effort: null, reason: 'Nisev ' + state }
  }
  const model = await sessionModel($)
  const body = buildRequest({ prompt, previousReply: await previousReply($), model }, config)
  try {
    const decision = decide(await askSystemOne($, config, body), config, model)
    if (config.provider === 'nisev') nisevConfirmed = true
    return decision
  } catch (error) {
    $.ui.log('effort request failed: ' + error.message, { to: 'debug' })
    if (config.provider === 'nisev') nisevConfirmed = false
    return { effort: null, reason: config.provider === 'nisev' ? 'Nisev unavailable' : 'endpoint unavailable' }
  }
}

async function saveConfig($, config) {
  try {
    await $.store.set('config', config)
    return true
  } catch {
    return false
  }
}

async function ask($, question, options) {
  try {
    return await $.ui.ask(question, options)
  } catch {
    return null // Dismissed, or no one to ask (a -p run).
  }
}

async function setupJev($, endpoint, model) {
  const saved = await saveConfig($, { provider: 'jev', endpoint, model })
  const key = await $.env.get('AUTO_EFFORT_API_KEY')
  return {
    text: [
      saved ? 'Saved: ' + endpoint + ', model ' + model + '.' : 'Could not save the setting for later sessions.',
      key
        ? 'AUTO_EFFORT_API_KEY is set, so the next prompt uses it.'
        : 'Now set AUTO_EFFORT_API_KEY to the key for that endpoint, in your shell or in the env block of a ' +
          'settings file that is not committed. The key is never stored by the mod.',
      'AUTO_EFFORT_* environment variables, when set, take precedence over this setup.',
    ].join('\n'),
  }
}

async function setupNisev($) {
  const env = { provider: 'nisev', llamaServer: await $.env.get('AUTO_EFFORT_LLAMA_SERVER') }
  const config = resolveConfig(env, await storedConfig($))
  const { binary, build } = await findServer($, config)
  if (build === null) {
    return {
      text:
        'llama.cpp was not found. Install build ' + NISEV.minBuild + ' or later, from ' +
        'https://github.com/ggml-org/llama.cpp/releases, so that llama-server or llama is on your PATH, or ' +
        'set AUTO_EFFORT_LLAMA_SERVER to its path. Then run /auto-effort setup again.',
    }
  }
  if (build < NISEV.minBuild) {
    return {
      text:
        binary + ' is build ' + build + '; decision models need build ' + NISEV.minBuild + ' or later. ' +
        'Update it from https://github.com/ggml-org/llama.cpp/releases and run /auto-effort setup again.',
    }
  }
  config.llamaServer = binary
  const serving = await isServing($, config)
  if (!serving) {
    const go = await ask($, 'Download ' + config.model + ' (' + NISEV.sizeLabel + ') and run it with llama.cpp?', {
      header: 'Download',
      options: ['Download and start', 'Cancel'],
    })
    if (go !== 'Download and start') return { text: 'Setup cancelled; nothing changed.' }
  }
  await saveConfig($, { provider: 'nisev', model: config.model, endpoint: config.endpoint, llamaServer: config.llamaServer })
  await startNisev($, config)
  return {
    text: [
      'Saved: Nisev, served by ' + serverCommand(binary).join(' ') + ' (build ' + build + ') at ' + config.endpoint + '.',
      serving
        ? 'It is already running.'
        : 'The first start downloads the model; the status line shows the progress. Until it is ready, ' +
          'prompts keep the session effort.',
      'llama-server runs while this session does; the next session starts it again from the cache.',
      'AUTO_EFFORT_* environment variables, when set, take precedence over this setup.',
    ].join('\n'),
  }
}

async function setup($) {
  const choice = await ask($, 'Which classifier should pick the effort of each prompt?', {
    header: 'Classifier',
    options: ['Nisev (local)', ...Object.keys(JEV_PRESETS)],
  })
  if (choice === null) return { text: 'Setup cancelled; nothing changed.' }
  if (choice === 'Nisev (local)') return setupNisev($)
  if (JEV_PRESETS[choice]) return setupJev($, JEV_PRESETS[choice].endpoint, JEV_PRESETS[choice].model)
  // "Other": any System One endpoint.
  if (!/^https?:\/\/\S+$/.test(choice.trim())) {
    return { text: 'Under Other, enter the URL of a System One endpoint, such as https://example.com/v1/systemone.' }
  }
  const model = await ask($, 'Which model name does that endpoint expect?', { header: 'Model', options: ['jev-latest', 'jev-1.13'] })
  if (model === null) return { text: 'Setup cancelled; nothing changed.' }
  return setupJev($, choice.trim(), model.trim())
}

async function status($) {
  const config = await loadConfig($)
  const lines = ['Effort picker: ' + (enabled ? 'on' : 'off')]
  if (config.provider === 'nisev') {
    lines.push('Provider: Nisev ' + config.model + ' · llama.cpp at ' + config.endpoint + ' (' + nisevState() + ')')
    for (const problem of config.problems) lines.push('Problem: ' + problem)
  } else {
    lines.push('Provider: ' + (config.provider ? 'System One endpoint' : 'not set, run /auto-effort setup'))
    lines.push('Endpoint: ' + (config.endpoint ?? 'unset') + ' · model ' + (config.model ?? 'unset'))
    lines.push('API key: ' + (config.apiKey ? 'set' : 'unset'))
    if (config.missing.length) lines.push('Missing: ' + config.missing.join(', '))
  }
  lines.push(
    'Range: ' + config.minEffort + '–' + config.maxEffort + ' of ' + LEVELS.join(', ') +
      ' · min confidence ' + config.minConfidence,
    'Last decision: ' + (lastDecision ? describe(lastDecision) : 'none yet'),
  )
  return { text: lines.join('\n') }
}

export function register(on) {
  on('session.start', async ($, e, next) => {
    try {
      const saved = await $.store.get('enabled')
      if (typeof saved === 'boolean') enabled = saved
    } catch {
      // Keep the default.
    }
    try {
      await $.command.register({
        name: 'auto-effort',
        description: 'Set up, show or toggle the effort picker',
        argumentHint: '[setup|on|off|status]',
        immediate: true,
      })
    } catch (error) {
      $.ui.log('could not register /auto-effort: ' + error.message, { to: 'debug' })
    }
    const result = await next(e)
    // Warm Nisev up before the first prompt needs it.
    const config = await loadConfig($)
    if (enabled && config.provider === 'nisev' && !config.problems.length) await startNisev($, config)
    return result
  })

  on('prompt.submit', async ($, e, next) => {
    // A delivery into a running turn (a peer's message, a notification) isn't a
    // new request from the user, so the running turn keeps its decision. Prompts
    // the user types mid-turn are queued and get a turn of their own.
    const intoRunningTurn = e.turnId && !USER_ORIGINS.includes(e.origin?.kind)
    if (!enabled || intoRunningTurn || !e.text.trim()) return next(e)
    const decision = await classify($, e.text)
    if (decision) {
      lastDecision = decision
      await show($, describe(decision))
      $.ui.status(undefined)
      pending = decision
    }
    const result = await next(e)
    // A later hook dropped the prompt: don't let its decision reach another turn.
    if (result.drop) pending = null
    return result
  })

  on('turn.start', async ($, e, next) => {
    if (pending) {
      decisions.set(e.turnId, pending)
      pending = null
    }
    return next(e)
  })

  on('ui.render', { component: 'SessionMode' }, async ($, e, next) => {
    const text = await read($, label)
    return text ? next({ ...e, props: { ...e.props, modes: [...e.props.modes, text] } }) : next(e)
  })

  on('turn.step', async function* ($, e, next) {
    const decision = decisions.get(e.turnId)
    // Subagents keep their own effort; a model without effort, or a numeric
    // budget set by hand, is left alone.
    if (!enabled || e.agentId || !decision?.effort || typeof e.effort !== 'string') {
      return yield* next(e)
    }
    if (e.index === 0) $.ui.log('effort ' + e.effort + ' → ' + decision.effort, { to: 'debug' })
    return yield* next({ ...e, effort: decision.effort })
  })

  on('turn.complete', async ($, e, next) => {
    decisions.delete(e.turnId)
    return next(e)
  })

  on('command.run', { command: 'auto-effort' }, async ($, e) => {
    const arg = e.args.trim().toLowerCase()
    if (arg === 'setup') return setup($)
    if (arg === 'on' || arg === 'off') {
      enabled = arg === 'on'
      try {
        await $.store.set('enabled', enabled)
      } catch {
        // Applies to this session only.
      }
      $.ui.status(undefined)
      await show($, enabled ? null : 'effort picker off')
      return { text: 'Effort picker turned ' + arg + '.' }
    }
    if (arg && arg !== 'status') return { text: 'Usage: /auto-effort [setup|on|off|status]' }
    return status($)
  })
}

