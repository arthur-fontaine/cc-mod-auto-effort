# auto-effort

A Claude Code [mod](https://code.claude.com/docs/en/plugins/mods/overview) that picks
Claude's effort level for each prompt, using a System One decision model such as
[Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev), TypeSafe's.

It works with two kinds of provider, through the same System One API:

- **A Jev endpoint**, such as [OpenCode Zen](https://opencode.ai/docs/zen/) or
  [TypeSafe](https://docs.typesafe.ai/api), or any other server of the API.
- **The local model**: a Qwen3-1.7B fine-tuned for this one question
  ([`training/`](training/README.md)), run on your machine by
  [llama.cpp](https://github.com/ggml-org/llama.cpp). It isn't bundled: setup downloads it
  (about 1.9 GB) only if you choose it.

There is no default provider, and the mod calls nothing until you pick one with
`/auto-effort setup` or the environment variables.

Tested with Claude Code v2.1.287.

## How it decides

[Anthropic's guidance](https://claude.com/blog/claude-model-and-effort-level-in-claude-code):
the model's default effort is right for most tasks. Effort controls how thorough Claude is
(how many files it reads, how much it verifies, how far it pushes before checking in), so
it should change only for a clear reason. The mod follows that:

1. On `prompt.submit`, it sends your prompt to the provider, truncated, with Claude's
   previous reply so that follow-ups like "yes, do it" make sense, and the session's model,
   since effort levels are calibrated per model (Opus 5.5 defaults to `medium`, most others
   to `high`).
   - A Jev endpoint gets one `choice` question with six options: `low`, `medium`, `default`,
     `high`, `xhigh`, and `max`. Each option's rubric is taken from the blog post.
   - The local model gets the question it was trained on, which sorts the request into one
     of six kinds of task, from `trivial` to `exhaustive`. A table in
     [`hooks/policy.js`](hooks/policy.js) turns the kind into a level for the session's
     model: an ordinary request keeps the model's default, verified multi-step work gets
     `high`, and models without `xhigh` keep hard work at `high`.
2. On each `turn.step` of that turn, it rewrites the request's `effort` to the pick.
3. The session's own effort stands, from `/effort`, `--effort`, `effortLevel`, or the
   model's default, whenever any of these happens:
   - The pick is the model's default.
   - The pick's confidence is below `AUTO_EFFORT_MIN_CONFIDENCE`.
   - The provider times out or errors, or the local model is still starting.
   - No provider is set up.
   - The model takes no effort.
   - The request comes from a subagent.
4. Picks are clamped to `AUTO_EFFORT_MIN_EFFORT`–`AUTO_EFFORT_MAX_EFFORT`. The maximum
   defaults to `xhigh`, so a `max` pick never happens unless you allow it.

The status line under the prompt shows the last decision, for example `effort high · 82%`.
Each prompt waits for the provider before its turn starts. That is usually well under a
second (about 140 ms for the local model on an M4 Pro, 600 ms for Jev), and never longer
than `AUTO_EFFORT_TIMEOUT_MS`.

## Install

The repository is its own plugin marketplace. Install the mod from GitHub:

```sh
claude plugin marketplace add arthur-fontaine/cc-mod-auto-effort
claude plugin install auto-effort@auto-effort-dev
```

Then start a new session, or run `/reload-plugins` in an open one, and
[choose a provider](#configure) with `/auto-effort setup`. `claude plugin list` shows
whether it's enabled, and `/auto-effort` shows what the mod sees.

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

Run `/auto-effort setup` and choose a provider:

- **Local model**: the mod checks that llama.cpp build b11361 or later is installed, as
  `llama-server` or as the unified `llama` CLI (`llama serve`), and asks before
  downloading. It then starts the server on `127.0.0.1:8765` and keeps it running for the
  session; it stops with the session. llama.cpp downloads the model on the first start and
  caches it, and the status line shows the progress. Prompts keep the session's effort
  until the model is ready. Install llama.cpp from
  [its releases](https://github.com/ggml-org/llama.cpp/releases) or a package manager.
- **OpenCode Zen** or **TypeSafe**: the mod saves the endpoint and model. Set the key as
  `AUTO_EFFORT_API_KEY` in your environment ([where](#where-to-set-them)): the mod never
  stores it.
- **Other**: type the URL of any System One endpoint, then the model name it expects.

The choice is saved across sessions. Environment variables, when set, take precedence over
it, so you can also configure the mod with them alone:

| Variable | What it is |
| :- | :- |
| `AUTO_EFFORT_PROVIDER` | `local` or `jev`. Setting `AUTO_EFFORT_ENDPOINT` implies `jev`. |
| `AUTO_EFFORT_ENDPOINT` | Jev: the System One endpoint URL. Use `https://`, since the key is sent as a bearer token. |
| `AUTO_EFFORT_API_KEY` | Jev: the key for that endpoint. Only ever read from the environment. |
| `AUTO_EFFORT_MODEL` | Jev: the model name the endpoint expects |
| `AUTO_EFFORT_LOCAL_MODEL` | Local: a Hugging Face `repo:quant` for llama.cpp's `-hf`, or the path to a `.gguf` file. Defaults to `arthur-fontaine/auto-effort-qwen3-1.7b-GGUF:Q8_0`. |
| `AUTO_EFFORT_LOCAL_PORT` | Local: the port to serve on. Defaults to `8765`. If a server already answers there with the `auto-effort` model, the mod uses it instead of starting one. |
| `AUTO_EFFORT_LLAMA_SERVER` | Local: the llama.cpp binary, `llama-server` or `llama`. Defaults to the first of the two on your `PATH`. |
| `AUTO_EFFORT_MIN_CONFIDENCE` | Below this confidence, from 0 to 1, the session's effort stands. Defaults to `0.5`. |
| `AUTO_EFFORT_TIMEOUT_MS` | How long to wait for the provider. Defaults to `4000`. |
| `AUTO_EFFORT_MIN_EFFORT` / `AUTO_EFFORT_MAX_EFFORT` | The range of levels the mod may pick, from `low`, `medium`, `high`, `xhigh`, `max`. Defaults to `low` / `xhigh`. |
| `AUTO_EFFORT_INCLUDE_CONTEXT` | `false` sends the prompt without Claude's previous reply. Defaults to `true`. |

While a Jev endpoint lacks its endpoint, key or model, the mod changes nothing, and the
status line and `/auto-effort` name what's missing.

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

| Provider | Endpoint | Model | Key |
| :- | :- | :- | :- |
| Local model | started by the mod | `auto-effort` | none |
| OpenCode Zen | `https://opencode.ai/zen/v1/systemone` | `jev-1.13` or `jev-1.13-free` | An OpenCode API key, see below |
| TypeSafe | `https://api.typesafe.ai/v1/systemone` | `jev-latest` | A key from <https://console.typesafe.ai/keys> |

On 136 held-out turns from real sessions, the local model picks the right level more often
than `jev-1.13` (65% against 60%, 74% against 70% on Opus 5.5) and answers in about 140 ms
instead of 625 ms, with nothing leaving your machine. It needs about 2 GB of memory while
the session runs. See [`training/`](training/README.md#results) for how it was built and
measured.

To get an OpenCode API key:

1. Sign in at <https://opencode.ai/auth> and open your workspace.
2. Open **Keys**, which lists service accounts, and click **Add Service Account**. Name it,
   for example `cc-mod-auto-effort`.
3. On the service account, click **Add API Key** and set **Permissions** to
   **Inference only**. The expiry date is optional.
4. Copy the key, since it is shown once, and set it as `AUTO_EFFORT_API_KEY`.

## Commands

- `/auto-effort` (or `/auto-effort status`) shows the provider, what's missing, the local
  server's state, and the last decision.
- `/auto-effort setup` chooses the provider.
- `/auto-effort off` stops the overrides, and `/auto-effort on` turns them back on. The
  setting is saved across sessions.

## Privacy

With the local model, nothing leaves your machine except llama.cpp's one-time download of
the model from Hugging Face.

With a Jev endpoint, each prompt goes to that endpoint, along with up to 1,500 characters
of Claude's previous reply and the name of the session's model. The prompt itself is
truncated to 6,000 characters. Set `AUTO_EFFORT_INCLUDE_CONTEXT=false` to send the prompt
alone. Your provider's data policy applies: OpenCode says Jev inputs aren't used for
training.

## Develop

```sh
pnpm test        # claude plugin test: hooks with stubbed providers and processes, no network
pnpm validate    # claude plugin validate --strict
pnpm eval        # sample prompts against the live provider, reading .env
```

`pnpm eval` needs a Jev endpoint's three variables, in the environment or in a gitignored
`.env` (see `.env.example`), or `AUTO_EFFORT_PROVIDER=local` with the local model already
served on its port. It reports the session model as `claude-opus-5-5`; set
`AUTO_EFFORT_EVAL_CLAUDE_MODEL` to try another.

On 2026-10-03, `pnpm eval` against `jev-1.13` on OpenCode Zen matched the expected level
on 6 of the 9 sample prompts, and so did the local model. Each Jev call took 0.4–1 s, and
the whole run used about 7k input tokens (about $0.0003). `jev-1.13-free` returned
`429 FreeUsageLimitError` at the time.

`claude --plugin-dir . --debug-file /tmp/cc.log` logs each override as
`[auto-effort] … effort low → high`.
