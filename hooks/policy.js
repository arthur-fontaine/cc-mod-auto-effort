// Pure decision logic: no `$`, so it can be unit tested on its own.

export const LEVELS = ['low', 'medium', 'high', 'xhigh', 'max']

export const DEFAULTS = {
  minConfidence: 0.5,
  timeoutMs: 4000,
  minEffort: 'low',
  maxEffort: 'xhigh',
  includeContext: true,
}

// The fine-tuned classifier, served by llama.cpp's llama-server as a decision model.
export const NISEV = {
  // A Hugging Face repo for `llama-server -hf`, or a path to a .gguf file.
  model: 'arthur-fontaine/nisev-1.7b-GGUF:Q8_0',
  sizeLabel: 'about 1.9 GB',
  endpoint: 'http://127.0.0.1:8765/v1/systemone',
  // Tried in order when none is configured: the server binary, or the unified CLI (`llama serve`).
  servers: ['llama-server', 'llama'],
  alias: 'nisev',
  // The first llama.cpp build with /v1/systemone (ggml-org/llama.cpp#29818).
  minBuild: 11361,
}

export const JEV_PRESETS = {
  'OpenCode Zen': { endpoint: 'https://opencode.ai/zen/v1/systemone', model: 'jev-1.13' },
  TypeSafe: { endpoint: 'https://api.typesafe.ai/v1/systemone', model: 'jev-latest' },
}

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
      'follow-ups such as "yes, do it". `model`, when present, is the Claude model; `default` keeps that ' +
      "model's own default effort.",
  },
  criteria: {
    low:
      'Routine work that needs no investigation: a precisely described edit, a rename, a typo, a one-line ' +
      'change, a question about code already in context, a quick lookup or shell command, small talk.',
    medium:
      'Light work: a small, well-scoped change or a focused answer that needs a little reading, but no ' +
      'multi-file investigation.',
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

// The question Nisev was trained on, word for word (training/pipeline/task.py): it
// sizes the request, and the model's table below turns the size into a level.
export const CATEGORIES = ['trivial', 'light', 'ordinary', 'multi_step', 'hard', 'exhaustive']
export const CATEGORY_QUESTION = {
  type: 'choice',
  instructions:
    'A developer sent latest_user_message to an AI coding agent working in their repository. ' +
    'How much thoroughness does it need: how many files to read, how much to verify, and how far ' +
    'to push before checking back in? previous_assistant_reply, when present, is the agent\'s last ' +
    'reply; use it to size short follow-ups such as "yes, do it".',
  criteria: {
    trivial: 'Routine: a precise small edit, a lookup, a question about code in context, an acknowledgement.',
    light: 'A small, well-scoped change or focused answer that needs a little reading.',
    ordinary: 'A typical feature, bug fix or explanation, or anything unclear.',
    multi_step: 'Several files, several hypotheses, or work that must be verified by running tests.',
    hard: 'A large migration, a subtle cross-system bug, an audit, or an explicit ask to be thorough.',
    exhaustive: 'An explicit demand for maximum effort on a critical, very large task.',
  },
}

// The level each kind of request gets on each model (training/pipeline/models.py). Levels are
// calibrated per model: Opus 5.5 defaults to medium, so verified multi-step work is where it
// pays to raise effort there, while models that default to high already run it at high.
const HIGH = { trivial: 'low', light: 'medium', ordinary: 'high', multi_step: 'high', hard: 'xhigh', exhaustive: 'max' }
const MEDIUM = { ...HIGH, ordinary: 'medium' }
const NO_XHIGH = { ...HIGH, hard: 'high' }
const NO_MAX = { ...HIGH, hard: 'high', exhaustive: 'high' }
export const PROFILES = {
  'claude-opus-5-5': { default: 'medium', table: MEDIUM },
  'claude-sonnet-5-5': { default: 'high', table: HIGH },
  'claude-fable-5-1': { default: 'high', table: HIGH },
  'claude-opus-5': { default: 'high', table: HIGH },
  'claude-sonnet-5': { default: 'high', table: HIGH },
  'claude-fable-5': { default: 'high', table: HIGH },
  'claude-opus-4-8': { default: 'high', table: HIGH },
  'claude-opus-4-7': { default: 'high', table: HIGH },
  'claude-opus-4-6': { default: 'high', table: NO_XHIGH },
  'claude-sonnet-4-6': { default: 'high', table: NO_XHIGH },
  'claude-opus-4-5': { default: 'high', table: NO_MAX },
}
const FALLBACK = { default: 'high', table: HIGH }
const ALIASES = { opus: 'claude-opus-5-5', sonnet: 'claude-sonnet-5-5', fable: 'claude-fable-5-1' }

// `claude-opus-5-5[1m]`, `us.anthropic.claude-opus-5-5-v1`, `opus` -> `claude-opus-5-5`
export function normalizeModel(model) {
  if (!model) return undefined
  const m = String(model).toLowerCase().split('/').pop()
    .replace(/^((us|eu|apac|global)\.)?anthropic\./, '')
    .replace(/\[.*?\]$|:.*$|-v\d+$|-\d{8}$|-fast$/g, '')
    .replaceAll('.', '-')
  return ALIASES[m] ?? m
}

export function profileOf(model) {
  return PROFILES[normalizeModel(model)] ?? FALLBACK
}

// Haiku 4.5 takes no effort; an unknown or missing model gets the common profile.
export function takesEffort(model) {
  return !(normalizeModel(model) ?? '').includes('haiku')
}

// Keeps the request well under the classifier's context.
const CLIP = {
  jev: { prompt: 6000, context: 1500 },
  // What Nisev was trained with (training/pipeline/task.py).
  nisev: { prompt: 3000, context: 800 },
}

// Counts code points, as the Python that built the training data does, so a long prompt is
// cut at the same place.
export function clip(text, max) {
  const chars = Array.from(text)
  if (chars.length <= max) return text
  // Keep both ends: the ask is usually at the start, pasted output at the end.
  const half = Math.floor((max - 20) / 2)
  return chars.slice(0, half).join('') + '\n[… truncated …]\n' + chars.slice(-half).join('')
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

// The endpoint, key, and model of a Jev provider have no default, so the mod never calls
// a provider the user didn't choose.
export const REQUIRED = {
  endpoint: 'AUTO_EFFORT_ENDPOINT',
  apiKey: 'AUTO_EFFORT_API_KEY',
  model: 'AUTO_EFFORT_MODEL',
}

// `env` holds the AUTO_EFFORT_* values, keyed as in the returned config. `stored` holds what
// `/auto-effort setup` saved: never a key, and the environment wins over it.
// The port Nisev serves on, from its endpoint; null unless the endpoint is on this machine.
export function localPort(endpoint) {
  const match = /^http:\/\/(?:127\.0\.0\.1|localhost):(\d+)\/v1\/systemone$/.exec(endpoint ?? '')
  return match ? Number(match[1]) : null
}

export function resolveConfig(env = {}, stored = {}) {
  // The environment wins: an endpoint there means a cloud endpoint unless AUTO_EFFORT_PROVIDER says otherwise.
  const provider = [env.provider, env.endpoint && 'jev', stored.provider].find((p) => ['jev', 'nisev'].includes(p))
  // Saved values belong to the provider they were saved for: a cloud model name is no GGUF to serve.
  const saved = stored.provider === provider ? stored : {}
  const nisev = provider === 'nisev'
  const config = {
    provider,
    minConfidence: toNumber(pick(env.minConfidence), DEFAULTS.minConfidence),
    timeoutMs: toNumber(pick(env.timeoutMs), DEFAULTS.timeoutMs),
    minEffort: toLevel(pick(env.minEffort), DEFAULTS.minEffort),
    maxEffort: toLevel(pick(env.maxEffort), DEFAULTS.maxEffort),
    includeContext: toBool(pick(env.includeContext), DEFAULTS.includeContext),
    // For Nisev, the model is what llama.cpp serves: a Hugging Face repo:quant or a .gguf path.
    endpoint: pick(env.endpoint, saved.endpoint, nisev ? NISEV.endpoint : undefined),
    model: pick(env.model, saved.model, nisev ? NISEV.model : undefined),
    apiKey: nisev ? 'none' : pick(env.apiKey),
    llamaServer: nisev ? pick(env.llamaServer, saved.llamaServer) : undefined,
  }
  if (nisev) {
    config.port = localPort(config.endpoint)
    config.missing = config.port ? [] : ['AUTO_EFFORT_ENDPOINT (for Nisev, a local URL such as ' + NISEV.endpoint + ')']
  } else {
    config.missing = Object.keys(REQUIRED).filter((key) => !config[key]).map((key) => REQUIRED[key])
  }
  return config
}

export function buildRequest({ prompt, previousReply, model }, config) {
  const limits = CLIP[config.provider === 'nisev' ? 'nisev' : 'jev']
  const state = { latest_user_message: clip(prompt, limits.prompt) }
  if (config.includeContext && previousReply) {
    state.previous_assistant_reply = clip(previousReply, limits.context)
  }
  // Effort levels are calibrated per model, so the classifier needs to know which one runs.
  if (model) state.model = model
  const question = config.provider === 'nisev' ? CATEGORY_QUESTION : EFFORT_QUESTION
  // llama.cpp answers to the alias it serves Nisev under, whatever file it loaded.
  return { model: config.provider === 'nisev' ? NISEV.alias : config.model, state, questions: { effort: question } }
}

export function clamp(level, min, max) {
  const lo = LEVELS.indexOf(min)
  const hi = Math.max(lo, LEVELS.indexOf(max))
  const i = LEVELS.indexOf(level)
  return LEVELS[Math.min(Math.max(i, lo), hi)]
}

// Nisev answers with category probabilities: sum them into levels through the
// model's table, so the confidence is the probability of the level it picks.
export function levelProbabilities(categoryProbabilities, model) {
  const table = profileOf(model).table
  const levels = Object.fromEntries(LEVELS.map((level) => [level, 0]))
  for (const category of CATEGORIES) levels[table[category]] += categoryProbabilities?.[category] ?? 0
  return levels
}

function decideNisev(answer, config, model) {
  if (!takesEffort(model)) return { effort: null, reason: 'model takes no effort' }
  const levels = levelProbabilities(answer.probabilities, model)
  const level = LEVELS.reduce((best, l) => (levels[l] > levels[best] ? l : best), LEVELS[0])
  const confidence = levels[level]
  const category = answer.choice
  // The model's default is what `default` keeps: the session's own effort stands.
  if (level === profileOf(model).default) return { effort: null, choice: 'default', confidence, category, reason: 'default' }
  if (confidence < config.minConfidence) return { effort: null, choice: level, confidence, category, reason: 'low confidence' }
  return { effort: clamp(level, config.minEffort, config.maxEffort), choice: level, confidence, category, reason: 'jev' }
}

// Returns `{ effort, choice, confidence, reason }`. `effort` is null when the
// session's own effort should stand.
export function decide(response, config, model) {
  // Cloudflare Workers AI wraps the System One response in `result`.
  const answer = (response?.answers ?? response?.result?.answers)?.effort
  if (!answer || answer.type !== 'choice' || typeof answer.choice !== 'string') {
    return { effort: null, reason: 'no effort answer' }
  }
  if (config.provider === 'nisev') return decideNisev(answer, config, model)
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
    const capped = decision.choice && decision.choice !== decision.effort ? ' (picked ' + decision.choice + ')' : ''
    return 'effort ' + decision.effort + capped + pct
  }
  if (decision.reason === 'default') return 'effort default' + pct
  if (decision.reason === 'low confidence') return 'effort default (unsure: ' + decision.choice + pct + ')'
  return 'effort default (' + decision.reason + ')'
}
