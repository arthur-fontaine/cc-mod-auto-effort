# auto-effort

A Claude Code [mod](https://code.claude.com/docs/en/plugins/mods/overview) that lets
[Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev), TypeSafe's System One
model, pick Claude's effort level for each prompt. By default it calls Jev through the
[OpenCode Zen](https://opencode.ai/docs/zen/) API.

Tested with Claude Code v2.1.287.

## How it decides

[Anthropic's guidance](https://claude.com/blog/claude-model-and-effort-level-in-claude-code):
the model's default effort is right for most tasks. Effort controls how thorough Claude is
(how many files it reads, how much it verifies, how far it pushes before checking in), so
it should change only for a clear reason. The mod follows that:

1. On `prompt.submit`, it sends your prompt to Jev, truncated, with Claude's previous reply
   so that follow-ups like "yes, do it" make sense. It asks one `choice` question with
   five options: `low`, `default`, `high`, `xhigh`, and `max`. Each option's rubric is taken
   from the blog post.
2. On each `turn.step` of that turn, it rewrites the request's `effort` to Jev's pick.
3. The session's own effort stands, from `/effort`, `--effort`, `effortLevel`, or the
   model's default, whenever any of these happens:
   - Jev answers `default`.
   - Jev's confidence is below `min_confidence`.
   - Jev times out or errors, or no API key is set.
   - The model takes no effort.
   - The request comes from a subagent.
4. Picks are clamped to `min_effort`–`max_effort`. `max_effort` defaults to `xhigh`, so a
   `max` pick never happens unless you allow it.

The status line under the prompt shows the last decision, for example `effort high · 82%`.
Each prompt waits for Jev before its turn starts. That is usually well under a second,
and never longer than `timeout_ms`.

## Install

Load it for one session:

```sh
claude --plugin-dir /path/to/cc-mod-auto-effort
```

To load it in every session, add the directory to `CLAUDE_CODE_PLUGIN_DIRS` in the `env`
block of `~/.claude/settings.json`.

## Configure

The endpoint, key, and model are configurable. Each value is read from the plugin's
`userConfig` first, then from the environment, then from the default.

| userConfig | Environment | Default |
| :- | :- | :- |
| `api_key` (stored in the OS keychain) | `AUTO_EFFORT_API_KEY`, then `OPENCODE_API_KEY` | none: the mod does nothing |
| `endpoint` | `AUTO_EFFORT_ENDPOINT` | `https://opencode.ai/zen/v1/systemone` |
| `model` | `AUTO_EFFORT_MODEL` | `jev-1.13` (`jev-1.13-free` while OpenCode offers it) |
| `min_confidence` | `AUTO_EFFORT_MIN_CONFIDENCE` | `0.5` |
| `timeout_ms` | `AUTO_EFFORT_TIMEOUT_MS` | `4000` |
| `min_effort` / `max_effort` | `AUTO_EFFORT_MIN_EFFORT` / `AUTO_EFFORT_MAX_EFFORT` | `low` / `xhigh` |
| `include_context` | `AUTO_EFFORT_INCLUDE_CONTEXT` | `true` |

For a mod loaded with `--plugin-dir`, values go under
`pluginConfigs["auto-effort@inline"].options` in `~/.claude/settings.json`, or in a file
passed with `--settings`. That path is tested, `api_key` included. The keychain path
for `api_key`, used by an installed plugin's configuration dialog, hasn't been tested
yet; the environment variables always work.

Use an `https://` endpoint. The key is sent as a bearer token.

To call TypeSafe directly instead of OpenCode, set the endpoint to
`https://api.typesafe.ai/v1/systemone`, the model to `jev-latest`, and use a TypeSafe key.

### Get an OpenCode API key

1. Sign in at <https://opencode.ai/auth>.
2. In your workspace, open **API Keys** and create a key, for example `cc-mod-auto-effort`.
3. Export it as `OPENCODE_API_KEY` in the shell that starts Claude Code, or put it in a
   gitignored `.env` for `pnpm eval`.

## Commands

`/auto-effort` shows the configuration and the last decision. `/auto-effort off` stops
the overrides, and `/auto-effort on` turns them back on. The setting is saved across
sessions.

## Privacy

Each prompt goes to the configured endpoint, along with up to 1,500 characters of
Claude's previous reply. The prompt itself is truncated to 6,000 characters. Set
`include_context` to `false` to send the prompt alone. OpenCode says Jev inputs aren't
used for training.

## Develop

```sh
pnpm test        # claude plugin test: hooks with stubbed Jev, no network
pnpm validate    # claude plugin validate --strict
pnpm eval        # sample prompts against the live endpoint (needs OPENCODE_API_KEY)
```

`claude --plugin-dir . --debug-file /tmp/cc.log` logs each override as
`[auto-effort] … effort low → high`.
