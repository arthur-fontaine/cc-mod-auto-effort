// The local provider: the fine-tuned classifier served by llama.cpp's llama-server.
// Pure helpers; register.js does the calls ($ is only passed within a file).

import { LOCAL } from './policy.js'

// `llama` is llama.cpp's unified CLI, whose server is `llama serve`; it takes the same flags.
export function serverCommand(binary) {
  return /(^|\/)llama$/.test(binary) ? [binary, 'serve'] : [binary]
}

export function serverArgs(config, binary) {
  const source = /\.gguf$/i.test(config.localModel) ? ['-m', config.localModel] : ['-hf', config.localModel]
  return [
    ...serverCommand(binary), ...source,
    '--host', '127.0.0.1', '--port', String(config.localPort),
    '--alias', LOCAL.alias,
    // Prompts are at most about 1,300 tokens; two slots let two prompts run at once.
    '-c', '8192', '-np', '2',
  ]
}

// `llama-server --version` prints e.g. "version: 0.5.0-dev (build 11408, commit 9f12cd4a4)".
export function parseBuild(output) {
  const match = output.match(/build (\d+)/)
  return match ? Number(match[1]) : null
}

// What a piece of llama-server's output says about its start: { ready }, { percent } or {}.
export function parseServerOutput(text) {
  if (/listening on/i.test(text)) return { ready: true }
  const percent = /download/i.test(text) && text.match(/(\d{1,3}(?:\.\d+)?)%/)
  return percent ? { percent: Math.round(Number(percent[1])) } : {}
}

// True when /v1/models lists our alias, whoever started the server.
export function servesOurModel(text) {
  try {
    return (JSON.parse(text).data ?? []).some((m) => m.id === LOCAL.alias)
  } catch {
    return false
  }
}
