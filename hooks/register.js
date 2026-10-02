import { buildRequest, decide, describe, LEVELS, resolveConfig } from './policy.js'

const USER_ORIGINS = ['composer', 'bridge', 'sdk']

// Decision for the prompt whose turn hasn't started yet.
let pending = null
// turnId -> decision, for turns in flight.
const decisions = new Map()
let enabled = true
let lastDecision = null
let warnedMissing = false

async function loadConfig($) {
  const env = {
    apiKey: await $.env.get('AUTO_EFFORT_API_KEY'),
    endpoint: await $.env.get('AUTO_EFFORT_ENDPOINT'),
    model: await $.env.get('AUTO_EFFORT_MODEL'),
    minConfidence: await $.env.get('AUTO_EFFORT_MIN_CONFIDENCE'),
    timeoutMs: await $.env.get('AUTO_EFFORT_TIMEOUT_MS'),
    minEffort: await $.env.get('AUTO_EFFORT_MIN_EFFORT'),
    maxEffort: await $.env.get('AUTO_EFFORT_MAX_EFFORT'),
    includeContext: await $.env.get('AUTO_EFFORT_INCLUDE_CONTEXT'),
  }
  return resolveConfig(env)
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

async function askJev($, config, body) {
  const request = $.http.fetch(config.endpoint, {
    method: 'POST',
    headers: { Authorization: 'Bearer ' + config.apiKey, 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  // $.http.fetch takes no timeout, and a slow classifier must never hold up the prompt.
  const timeout = $.clock.sleep(config.timeoutMs).then(() => null)
  const response = await Promise.race([request, timeout])
  if (response === null) throw new Error('timed out after ' + config.timeoutMs + 'ms')
  if (!response.ok) throw new Error('HTTP ' + response.status + ' ' + response.text.slice(0, 200))
  return JSON.parse(response.text)
}

async function classify($, prompt) {
  const config = await loadConfig($)
  if (config.missing.length) {
    if (!warnedMissing) {
      warnedMissing = true
      $.ui.status('auto-effort: set ' + config.missing.join(', '))
    }
    return null
  }
  const body = buildRequest({ prompt, previousReply: await previousReply($) }, config)
  try {
    return decide(await askJev($, config, body), config)
  } catch (error) {
    $.ui.log('Jev request failed: ' + error.message, { to: 'debug' })
    return { effort: null, reason: 'Jev unavailable' }
  }
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
        description: 'Show or toggle the Jev effort picker',
        argumentHint: '[on|off|status]',
        immediate: true,
      })
    } catch (error) {
      $.ui.log('could not register /auto-effort: ' + error.message, { to: 'debug' })
    }
    return next(e)
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
      $.ui.status(describe(decision))
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
    if (arg === 'on' || arg === 'off') {
      enabled = arg === 'on'
      try {
        await $.store.set('enabled', enabled)
      } catch {
        // Applies to this session only.
      }
      $.ui.status(enabled ? 'auto-effort on' : undefined)
      return { text: 'Jev effort picker turned ' + arg + '.' }
    }
    if (arg && arg !== 'status') return { text: 'Usage: /auto-effort [on|off|status]' }
    const config = await loadConfig($)
    return {
      text: [
        'Jev effort picker: ' + (enabled ? 'on' : 'off'),
        'Endpoint: ' + (config.endpoint ?? 'unset') + ' · model ' + (config.model ?? 'unset'),
        'API key: ' + (config.apiKey ? 'set' : 'unset'),
        ...(config.missing.length ? ['Missing: ' + config.missing.join(', ')] : []),
        'Range: ' + config.minEffort + '–' + config.maxEffort + ' of ' + LEVELS.join(', ') +
          ' · min confidence ' + config.minConfidence,
        'Last decision: ' + (lastDecision ? describe(lastDecision) : 'none yet'),
      ].join('\n'),
    }
  })
}
