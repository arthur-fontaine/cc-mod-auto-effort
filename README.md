# auto-effort

A Claude Code [mod](https://code.claude.com/docs/en/plugins/mods/overview) that picks
Claude's effort level for each prompt, using a System One decision model such as
[Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev), TypeSafe's.

It works with two kinds of provider, through the same System One API:

- **A Jev endpoint**, such as [OpenCode Zen](https://opencode.ai/docs/zen/) or
  [TypeSafe](https://docs.typesafe.ai/api), or any other server of the API.
- **[Nisev](https://huggingface.co/arthur-fontaine/nisev-1.7b-GGUF)**, our own decision
  model: a Qwen3-1.7B fine-tuned for this one question ([`training/`](training/README.md)),
  run on your machine by [llama.cpp](https://github.com/ggml-org/llama.cpp). It isn't
  bundled: setup downloads it (about 1.9 GB) only if you choose it.

Other decision models you run yourself, such as Kev or Clef-Flash in llama.cpp, work as a
Jev endpoint at their local URL. See the [benchmark](#benchmark) for how they compare.

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
   - Nisev gets the question it was trained on, which sorts the request into one
     of six kinds of task, from `trivial` to `exhaustive`. A table in
     [`hooks/policy.js`](hooks/policy.js) turns the kind into a level for the session's
     model: an ordinary request keeps the model's default, verified multi-step work gets
     `high`, and models without `xhigh` keep hard work at `high`.
2. On each `turn.step` of that turn, it rewrites the request's `effort` to the pick.
3. The session's own effort stands, from `/effort`, `--effort`, `effortLevel`, or the
   model's default, whenever any of these happens:
   - The pick is the model's default.
   - The pick's confidence is below `AUTO_EFFORT_MIN_CONFIDENCE`.
   - The provider times out or errors, or Nisev is still starting.
   - No provider is set up.
   - The model takes no effort.
   - The request comes from a subagent.
4. Picks are clamped to `AUTO_EFFORT_MIN_EFFORT`–`AUTO_EFFORT_MAX_EFFORT`. The maximum
   defaults to `xhigh`, so a `max` pick never happens unless you allow it.

The status line under the prompt shows the last decision, for example `effort high · 82%`.
Each prompt waits for the provider before its turn starts. That is usually well under a
second (about 130 ms for Nisev on an M4 Pro, 600 ms for Jev), and never longer
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

- **Nisev (local)**: the mod checks that llama.cpp build b11361 or later is installed, as
  `llama-server` or as the unified `llama` CLI (`llama serve`), and asks before
  downloading. It then starts the server on `127.0.0.1:8765` and keeps it running for the
  session; it stops with the session. llama.cpp downloads the model on the first start and
  caches it (about 3 minutes on a fast connection); the status line says when it's ready. Prompts keep the session's effort
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
| `AUTO_EFFORT_PROVIDER` | `nisev` or `jev`. Setting `AUTO_EFFORT_ENDPOINT` implies `jev`. |
| `AUTO_EFFORT_ENDPOINT` | Jev: the System One endpoint URL. Use `https://`, since the key is sent as a bearer token. |
| `AUTO_EFFORT_API_KEY` | Jev: the key for that endpoint. Only ever read from the environment. |
| `AUTO_EFFORT_MODEL` | Jev: the model name the endpoint expects |
| `AUTO_EFFORT_NISEV_MODEL` | Nisev: a Hugging Face `repo:quant` for llama.cpp's `-hf`, or the path to a `.gguf` file. Defaults to `arthur-fontaine/nisev-1.7b-GGUF:Q8_0`. |
| `AUTO_EFFORT_NISEV_PORT` | Nisev: the port to serve on. Defaults to `8765`. If a server already answers there with the `nisev` model, the mod uses it instead of starting one. |
| `AUTO_EFFORT_LLAMA_SERVER` | Nisev: the llama.cpp binary, `llama-server` or `llama`. Defaults to the first of the two on your `PATH`. |
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
| Nisev | started by the mod | `nisev` | none |
| OpenCode Zen | `https://opencode.ai/zen/v1/systemone` | `jev-1.13` or `jev-1.13-free` | An OpenCode API key, see below |
| TypeSafe | `https://api.typesafe.ai/v1/systemone` | `jev-latest` | A key from <https://console.typesafe.ai/keys> |
| Cloudflare Workers AI (Clef) | `https://api.cloudflare.com/client/v4/accounts/<account id>/ai/run/@cf/cloudflare/clef` (or `…/clef-flash`) | `clef` (or `clef-flash`) | A token with the Workers AI template |

Nisev needs about 2 GB of memory while the session runs. To use another decision model
in llama.cpp, start it yourself (`llama-server -hf ggml-org/Kev-4B-GGUF:Q8_0 --port 8080`,
say) and choose **Other** with `http://127.0.0.1:8080/v1/systemone`.

To get an OpenCode API key:

1. Sign in at <https://opencode.ai/auth> and open your workspace.
2. Open **Keys**, which lists service accounts, and click **Add Service Account**. Name it,
   for example `cc-mod-auto-effort`.
3. On the service account, click **Add API Key** and set **Permissions** to
   **Inference only**. The expiry date is optional.
4. Copy the key, since it is shown once, and set it as `AUTO_EFFORT_API_KEY`.

## Benchmark

How often each model picks the effort level that two Claude teachers (Sonnet 5.5 and Opus
5.5) agreed on, with hindsight, for 136 held-out turns from 35 real coding sessions. Always
keeping the model's default scores 52.9%. Each model got the request the mod sends it, one
request at a time. The local runs used the same llama.cpp on the same Mac; cloud latency
includes the round trip from that Mac to the provider. Clef (27B) and Clef-Flash (9B) ran
on Cloudflare Workers AI, since neither runs well on a 24 GB Mac: Clef-Flash in llama.cpp
took 3.4 s per prompt at the median, at 4-bit.

| Model | Where | Served by | Level | Level, as Opus 5.5 | Mod applies right level | Latency p50 / p95 |
| :- | :- | :- | -: | -: | -: | -: |
| **Nisev 1.7B** (this repo) | Local, Apple M4 Pro, 24 GB | llama.cpp b11406, `nisev-1.7b-Q8_0.gguf` (1.8 GB) | 65.4% | 73.5% | 66.9% | 129 / 364 ms |
| Kev-4B | Local, Apple M4 Pro, 24 GB | llama.cpp b11406, `Kev-4B-Q8_0.gguf` (4.5 GB) | 55.1% | 60.3% | 52.2% | 1750 / 3236 ms |
| Clef-Flash 9B | Cloud | Cloudflare Workers AI, network round trip included | 56.6% | 64.7% | 51.5% | 309 / 909 ms |
| Clef 27B | Cloud | Cloudflare Workers AI, network round trip included | 51.5% | 65.4% | 51.5% | 495 / 880 ms |
| Jev 1.13 | Cloud | OpenCode Zen, network round trip included | 61.8% | 70.6% | 62.5% | 588 / 770 ms |

- **Level**: the pick, for the Claude model that ran the turn.
- **Level, as Opus 5.5**: the same turns, as if they ran on Opus 5.5, whose default is
  `medium`.
- **Mod applies right level**: what the mod ends up doing with its default confidence
  threshold of 0.5: the pick when confident enough, otherwise the session's own effort.

Notes:

- Both Clef models answer `medium` most often (40% of turns), which suits Opus 5.5, whose
  default is `medium`, and is a step too low on models that default to `high`. Clef also
  answers `max` on 12 of the 136 turns.
- Kev and both Clef models report a confidence of 0.5 or more on at most 2 of the 136
  turns (median 0.12 to 0.18), so with the default threshold the mod almost never applies
  their picks: "applied" stays near the always-default 52.9%. Jev clears it on half the
  turns.
- With 136 turns, gaps of a few points are within noise. Kev, Clef and Jev answer
  zero-shot; Nisev was trained on this question, from the same kind of sessions.

Rerun it with `uv run python pipeline/benchmark.py` in [`training/`](training/README.md),
which also explains how Nisev was built.

## Commands

- `/auto-effort` (or `/auto-effort status`) shows the provider, what's missing, the local
  server's state, and the last decision.
- `/auto-effort setup` chooses the provider.
- `/auto-effort off` stops the overrides, and `/auto-effort on` turns them back on. The
  setting is saved across sessions.

## Privacy

With Nisev, nothing leaves your machine except llama.cpp's one-time download of
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
`.env` (see `.env.example`), or `AUTO_EFFORT_PROVIDER=nisev` with Nisev already served on
its port. It reports the session model as `claude-opus-5-5`; set
`AUTO_EFFORT_EVAL_CLAUDE_MODEL` to try another.

On 2026-10-03, `pnpm eval` against `jev-1.13` on OpenCode Zen matched the expected level
on 6 of the 9 sample prompts, and so did Nisev. Each Jev call took 0.4–1 s, and
the whole run used about 7k input tokens (about $0.0003). `jev-1.13-free` returned
`429 FreeUsageLimitError` at the time.

`claude --plugin-dir . --debug-file /tmp/cc.log` logs each override as
`[auto-effort] … effort low → high`.
