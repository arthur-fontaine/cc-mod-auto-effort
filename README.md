# auto-effort

A Claude Code [mod](https://code.claude.com/docs/en/plugins/mods/overview) that lets
[Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev), TypeSafe's System One
model, pick Claude's effort level for each prompt.

It works with any provider that serves the System One API, such as
[OpenCode Zen](https://opencode.ai/docs/zen/) or [TypeSafe](https://docs.typesafe.ai/api).
There is no default provider. The mod is configured entirely through environment
variables, and it calls nothing until they are set.

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
   - Jev's confidence is below `AUTO_EFFORT_MIN_CONFIDENCE`.
   - Jev times out or errors.
   - A required variable is unset.
   - The model takes no effort.
   - The request comes from a subagent.
4. Picks are clamped to `AUTO_EFFORT_MIN_EFFORT`–`AUTO_EFFORT_MAX_EFFORT`. The maximum
   defaults to `xhigh`, so a `max` pick never happens unless you allow it.

The status line under the prompt shows the last decision, for example `effort high · 82%`.
Each prompt waits for Jev before its turn starts. That is usually well under a second,
and never longer than `AUTO_EFFORT_TIMEOUT_MS`.

## Install

The repository is its own plugin marketplace. Install the mod from GitHub:

```sh
claude plugin marketplace add arthur-fontaine/cc-mod-auto-effort
claude plugin install auto-effort@auto-effort-dev
```

Then [set the environment variables](#configure) and start a new session, or run
`/reload-plugins` in an open one. `claude plugin list` shows whether it's enabled, and
`/auto-effort` shows what the mod sees.

From inside a session, `/plugin` does the same: add the marketplace
`arthur-fontaine/cc-mod-auto-effort`, then install `auto-effort`.

### Try it without installing

Load a local checkout for one session. The mod reloads when you save a file:

```sh
git clone https://github.com/arthur-fontaine/cc-mod-auto-effort
claude --plugin-dir ./cc-mod-auto-effort
```

### Enable it for one repository

Add this to the repository's `.claude/settings.local.json`, and make sure that file is
gitignored. Use `.claude/settings.json` instead to share the plugin with everyone who
works in the repository, but keep the key out of that file.

```json
{
  "extraKnownMarketplaces": {
    "auto-effort-dev": { "source": { "source": "github", "repo": "arthur-fontaine/cc-mod-auto-effort" } }
  },
  "enabledPlugins": { "auto-effort@auto-effort-dev": true }
}
```

To develop against a local checkout, use
`{ "source": "directory", "path": "/path/to/cc-mod-auto-effort" }` as the source. A
`directory` marketplace loads the plugin in place, so edits apply after
`/reload-plugins`. Claude Code honors these entries only after you trust the folder.

## Configure

The mod reads environment variables only. Three of them are required:

| Variable | Required | What it is |
| :- | :- | :- |
| `AUTO_EFFORT_ENDPOINT` | Yes | The System One endpoint URL. Use `https://`, since the key is sent as a bearer token. |
| `AUTO_EFFORT_API_KEY` | Yes | The key for that endpoint |
| `AUTO_EFFORT_MODEL` | Yes | The Jev model name the endpoint expects |
| `AUTO_EFFORT_MIN_CONFIDENCE` | No | Below this confidence, from 0 to 1, the session's effort stands. Defaults to `0.5`. |
| `AUTO_EFFORT_TIMEOUT_MS` | No | How long to wait for Jev. Defaults to `4000`. |
| `AUTO_EFFORT_MIN_EFFORT` / `AUTO_EFFORT_MAX_EFFORT` | No | The range of levels the mod may pick, from `low`, `medium`, `high`, `xhigh`, `max`. Defaults to `low` / `xhigh`. |
| `AUTO_EFFORT_INCLUDE_CONTEXT` | No | `false` sends the prompt without Claude's previous reply. Defaults to `true`. |

While a required variable is unset, the mod changes nothing. The status line names the
missing variables, and so does `/auto-effort`.

### Where to set them

Claude Code passes its own environment to the mod, so set the variables in either of
these places:

- **Your shell**: `export` them before you start `claude`.
- **The `env` block of a settings file**: `~/.claude/settings.json` for every session, or a
  repository's `.claude/settings.local.json` for that repository only. Don't put the key in
  a settings file that's committed.

```json
{
  "env": {
    "AUTO_EFFORT_ENDPOINT": "https://opencode.ai/zen/v1/systemone",
    "AUTO_EFFORT_API_KEY": "oc_…",
    "AUTO_EFFORT_MODEL": "jev-1.13"
  }
}
```

The mod reads the variables on each prompt. If you change a settings file, start a new
session or run `/reload-plugins`.

### Providers

| Provider | `AUTO_EFFORT_ENDPOINT` | `AUTO_EFFORT_MODEL` | Key |
| :- | :- | :- | :- |
| OpenCode Zen | `https://opencode.ai/zen/v1/systemone` | `jev-1.13` or `jev-1.13-free` | An OpenCode API key, see below |
| TypeSafe | `https://api.typesafe.ai/v1/systemone` | `jev-latest` | A key from <https://console.typesafe.ai/keys> |

To get an OpenCode API key:

1. Sign in at <https://opencode.ai/auth> and open your workspace.
2. Open **Keys**, which lists service accounts, and click **Add Service Account**. Name it,
   for example `cc-mod-auto-effort`.
3. On the service account, click **Add API Key** and set **Permissions** to
   **Inference only**. The expiry date is optional.
4. Copy the key, since it is shown once, and set it as `AUTO_EFFORT_API_KEY`.

## Commands

`/auto-effort` shows the configuration, any missing variables, and the last decision.
`/auto-effort off` stops the overrides, and `/auto-effort on` turns them back on. The
setting is saved across sessions.

## Privacy

Each prompt goes to `AUTO_EFFORT_ENDPOINT`, along with up to 1,500 characters of
Claude's previous reply. The prompt itself is truncated to 6,000 characters. Set
`AUTO_EFFORT_INCLUDE_CONTEXT=false` to send the prompt alone. Your provider's data
policy applies: OpenCode says Jev inputs aren't used for training.

## Develop

```sh
pnpm test        # claude plugin test: hooks with stubbed Jev, no network
pnpm validate    # claude plugin validate --strict
pnpm eval        # sample prompts against the live endpoint, reading .env
```

`pnpm eval` needs the three required variables, in the environment or in a gitignored
`.env` (see `.env.example`).

On 2026-10-02, `pnpm eval` against `jev-1.13` on OpenCode Zen matched the expected level
on 6 of the 9 sample prompts. The other 3 were off by one level. Each call took
0.4–1 s, and the whole run used about 6k input tokens (about $0.0003). `jev-1.13-free`
returned `429 FreeUsageLimitError` at the time.

`claude --plugin-dir . --debug-file /tmp/cc.log` logs each override as
`[auto-effort] … effort low → high`.
