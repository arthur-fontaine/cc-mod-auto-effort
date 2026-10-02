// Pure decision logic: no `$`, so it can be unit tested on its own.

export const LEVELS = ['low', 'medium', 'high', 'xhigh', 'max']

export const DEFAULTS = {
  endpoint: 'https://opencode.ai/zen/v1/systemone',
  model: 'jev-1.13',
  minConfidence: 0.5,
  timeoutMs: 4000,
  minEffort: 'low',
  maxEffort: 'xhigh',
  includeContext: true,
}

// Keeps the request well under Jev's 32k-token budget for state plus question.
const MAX_PROMPT_CHARS = 6000
const MAX_CONTEXT_CHARS = 1500

// Worded after https://claude.com/blog/claude-model-and-effort-level-in-claude-code:
// the model's default is right for most tasks, effort controls how thorough
// Claude is (files read, verification, how far it pushes before checking in),
// and deviating should need a clear reason.
export const EFFORT_QUESTION = {
  type: 'choice',
  instructions: {
    task:
      'A developer sent `latest_user_message` to Claude, an AI coding agent working in their repository. ' +
      'Pick how much effort Claude should spend on it. Effort controls how many files Claude reads, ' +
      'how much it verifies (running tests, double-checking), and how far it pushes through a multi-step ' +
      'task before checking back in. It is not about how capable Claude is. Most requests should get ' +
      '`default`. Pick another level only when the request clearly calls for less or more thoroughness. ' +
      '`previous_assistant_reply`, when present, is what Claude last said; use it to understand short ' +
      'follow-ups such as "yes, do it".',
  },
  criteria: {
    low:
      'Routine work that needs no investigation: a precisely described edit, a rename, a typo, a one-line ' +
      'change, a question about code already in context, a quick lookup or shell command, small talk.',
    default:
      'A typical coding request: an ordinary feature, a normal bug fix, a focused explanation, or anything ' +
      'unclear. The model default already scales the work to the task.',
    high:
      'Multi-step work that must be verified: changes across several files, a bug that needs several ' +
      'hypotheses checked, a refactor that has to be finished completely, work where skipping a file or ' +
      'not running the tests would make the result wrong.',
    xhigh:
      'Long, hard, high-stakes work: a large migration or refactor, a subtle bug across systems, an ' +
      'architecture decision, a security or correctness audit, or a request that explicitly asks Claude to ' +
      'be thorough and verify everything.',
    max:
      'Exhaustive work where cost does not matter: the request explicitly demands maximum effort or ' +
      'leaving nothing unchecked, on a critical, very large task.',
  },
}

function clip(text, max) {
  if (text.length <= max) return text
  // Keep both ends: the ask is usually at the start, pasted output at the end.
  const half = Math.floor((max - 20) / 2)
  return text.slice(0, half) + '\n[… truncated …]\n' + text.slice(-half)
}

function pick(...values) {
  return values.find((v) => v !== undefined && v !== null && v !== '')
}

function toNumber(value, fallback) {
  const n = typeof value === 'number' ? value : Number(value)
  return Number.isFinite(n) ? n : fallback
}

function toLevel(value, fallback) {
  return LEVELS.includes(value) ? value : fallback
}

function toBool(value, fallback) {
  if (typeof value === 'boolean') return value
  if (value === 'true' || value === '1') return true
  if (value === 'false' || value === '0') return false
  return fallback
}

// `options` are the userConfig values; `env` the environment fallbacks.
export function resolveConfig(options = {}, env = {}) {
  return {
    apiKey: pick(options.api_key, env.apiKey),
    endpoint: pick(options.endpoint, env.endpoint, DEFAULTS.endpoint),
    model: pick(options.model, env.model, DEFAULTS.model),
    minConfidence: toNumber(pick(options.min_confidence, env.minConfidence), DEFAULTS.minConfidence),
    timeoutMs: toNumber(pick(options.timeout_ms, env.timeoutMs), DEFAULTS.timeoutMs),
    minEffort: toLevel(pick(options.min_effort, env.minEffort), DEFAULTS.minEffort),
    maxEffort: toLevel(pick(options.max_effort, env.maxEffort), DEFAULTS.maxEffort),
    includeContext: toBool(pick(options.include_context, env.includeContext), DEFAULTS.includeContext),
  }
}

export function buildRequest({ prompt, previousReply }, config) {
  const state = { latest_user_message: clip(prompt, MAX_PROMPT_CHARS) }
  if (config.includeContext && previousReply) {
    state.previous_assistant_reply = clip(previousReply, MAX_CONTEXT_CHARS)
  }
  return { model: config.model, state, questions: { effort: EFFORT_QUESTION } }
}

export function clamp(level, min, max) {
  const lo = LEVELS.indexOf(min)
  const hi = Math.max(lo, LEVELS.indexOf(max))
  const i = LEVELS.indexOf(level)
  return LEVELS[Math.min(Math.max(i, lo), hi)]
}

// Returns `{ effort, choice, confidence, reason }`. `effort` is null when the
// session's own effort should stand.
export function decide(response, config) {
  const answer = response?.answers?.effort
  if (!answer || answer.type !== 'choice' || typeof answer.choice !== 'string') {
    return { effort: null, reason: 'Jev returned no effort answer' }
  }
  const { choice, confidence } = answer
  if (choice === 'default') return { effort: null, choice, confidence, reason: 'default' }
  if (!LEVELS.includes(choice)) return { effort: null, choice, confidence, reason: 'unknown choice ' + choice }
  if (typeof confidence !== 'number' || confidence < config.minConfidence) {
    return { effort: null, choice, confidence, reason: 'low confidence' }
  }
  return { effort: clamp(choice, config.minEffort, config.maxEffort), choice, confidence, reason: 'jev' }
}

export function describe(decision) {
  const pct = typeof decision.confidence === 'number' ? ' · ' + Math.round(decision.confidence * 100) + '%' : ''
  if (decision.effort) {
    const capped = decision.choice && decision.choice !== decision.effort ? ' (Jev said ' + decision.choice + ')' : ''
    return 'effort ' + decision.effort + capped + pct
  }
  if (decision.reason === 'default') return 'effort default' + pct
  if (decision.reason === 'low confidence') return 'effort default (Jev unsure: ' + decision.choice + pct + ')'
  return 'effort default (' + decision.reason + ')'
}
