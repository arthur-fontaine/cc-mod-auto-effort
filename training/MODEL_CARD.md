---
base_model: Qwen/Qwen3-1.7B
library_name: gguf
license: apache-2.0
tags:
  - llama.cpp
  - decision-model
  - system-one
  - claude-code
---

# Nisev 1.7B

A small decision model that sizes a coding agent's request, made for the
[auto-effort](https://github.com/arthur-fontaine/cc-mod-auto-effort) Claude Code mod, which
picks Claude's effort level for each prompt. Given a prompt and the previous assistant
reply, Nisev picks one of six kinds of task: `trivial`, `light`, `ordinary`, `multi_step`,
`hard` or `exhaustive`. The mod turns that into an effort level for the Claude model in
use.

Nisev is a llama.cpp decision model: llama.cpp b11361 or later serves it at
`/v1/systemone`, the System One API, with no extra code.

```sh
llama-server -hf {{repo}}:Q8_0 --alias nisev     # or: llama serve -hf …
```

In the mod, `/auto-effort setup` → **Nisev (local)** does this for you.

The GGUF sets `qwen3.decision.type = openjev` (one letter per option, read from the
next-token logits), a `systemone` chat template that renders a request into the exact
training prompt, and `qwen3.decision.temperature.choice`, fitted on held-out data so its
probabilities are calibrated. It expects the mod's question; other questions get poor
answers.

## Benchmark

136 held-out turns from 35 real coding sessions. The answer key is the category on which
two Claude teachers (Sonnet 5.5 and Opus 5.5) agree after seeing what happened in the
turn. "Level" maps it to the effort level for the Claude model that ran the turn; always
keeping the model's default scores 52.9%. Every model got the request the mod sends it;
latency is end to end on an Apple M4 Pro (24 GB), one request at a time.

{{benchmark}}

With 136 turns, gaps of a few points are within noise. Method and limits:
[training/README.md](https://github.com/arthur-fontaine/cc-mod-auto-effort/blob/main/training/README.md).

## Training

LoRA (rank 16, all linear layers) on Qwen3-1.7B with MLX, the loss on the answer letter
only, 3 epochs, best epoch on dev, then merged and converted to GGUF Q8_0, following
Together's tev1 recipe. The code is in
[`training/`](https://github.com/arthur-fontaine/cc-mod-auto-effort/tree/main/training).

## Data

- Prompts: human prompts from public `format:agent-traces` datasets on the Hugging Face
  Hub (Claude Code and Pi sessions), deduplicated, filtered and scrubbed of secrets. The
  source datasets are listed in `training/data/sources.json`, under their own licenses.
- Labels: written by Claude (Sonnet 5.5 and Opus 5.5) acting as teachers.
- No traces or labels are distributed with the model.

## Limits

- `hard` has 78 training examples and `exhaustive` 2, so Nisev rarely answers `hard` and
  never `exhaustive`.
- 2,441 training turns from 367 sessions. The labels come from Claude teachers, not people.
