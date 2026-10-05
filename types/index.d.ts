// The values the mod keeps in $.state: the footer label it draws.
export type AutoEffortLabel = string | null

declare module 'claude-code' {
  interface PluginState {
    'auto-effort': { label: AutoEffortLabel }
  }
}
