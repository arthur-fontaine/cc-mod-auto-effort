# Local effort classifier

A small Jev-like model that sizes each prompt for the auto-effort mod and runs on an Apple
silicon Mac. It follows the recipe in Together's
[How to train your own Jev for $17](https://x.com/nutlope/status/2102881280115249597)
([togethercomputer/tev1](https://github.com/togethercomputer/tev1)): a JSON decision task,
a fixed system prompt, a one-letter answer, and LoRA fine-tuning. The training data comes
from public Claude Code and Pi session traces on Hugging Face
([`format:agent-traces`](https://huggingface.co/datasets?format=format:agent-traces)).

Results: see [Results](#results).

## How it decides

The classifier never sees the model name. It puts the request in one of six task
categories, from what the mod sends: the prompt and Claude's previous reply.

| Category | Meaning |
| :- | :- |
| `trivial` | A precise small edit, a lookup, a question about code in context, an acknowledgement |
| `light` | A small, well-scoped change or a focused answer that needs a little reading |
| `ordinary` | A typical feature, bug fix or explanation |
| `multi_step` | Changes across several files, several hypotheses, work that must be verified |
| `hard` | A large migration, a subtle cross-system bug, an audit, an explicit request for thoroughness |
| `exhaustive` | An explicit demand for maximum effort on a critical task |

[`pipeline/models.py`](pipeline/models.py) then turns the category into a level for the
Claude model in use. Effort levels are calibrated per model, and the defaults differ:
Opus 5.5 defaults to `medium` and most other models to `high`
([effort docs](https://platform.claude.com/docs/en/build-with-claude/effort)). So an
ordinary request runs at each model's default, and verified multi-step work gets `high`,
which is a step up on Opus 5.5 and the default elsewhere. Models without `xhigh` keep
hard work at `high`. Haiku 4.5 takes no effort and is left alone.

One model for every Claude model, rather than one per model: the public traces hold 3
real Opus 5.5 turns, 1 Sonnet 5.5 turn and 4 Fable 5.1 turns after cleaning, far too few
to train on. What changes between models is the level each kind of task deserves, which
Anthropic documents, so it lives in an editable table instead of in the weights. A first
attempt that labeled levels directly, since discarded, lost the
`ordinary` / `multi_step` distinction on models whose default is `high`, which is exactly
the distinction Opus 5.5 needs.

## Pipeline

```sh
uv sync
uv run python pipeline/fetch.py      # find and download Claude trace datasets (set HF_TOKEN to avoid rate limits)
uv run python pipeline/extract.py    # one record per human prompt, with hindsight about the turn
uv run python pipeline/clean.py      # dedupe, drop scripted and junk turns, scrub secrets
uv run python pipeline/label.py --prompt-only --out data/labels-prompt-only.jsonl   # training labels
uv run python pipeline/label.py                                                     # hindsight labels (test gold)
uv run python pipeline/label.py --teacher opus --ids data/split/test-candidates.ids --out data/labels-opus.jsonl
uv run python pipeline/build.py      # split by session, render the training files
uv run python pipeline/train.py --base Qwen/Qwen3-1.7B --out runs/qwen3-1.7b --epochs 3
uv run python pipeline/select_checkpoint.py runs/qwen3-1.7b     # best epoch on dev
uv run python pipeline/export.py runs/qwen3-1.7b                # fuse, then 8-bit and 4-bit copies
uv run python pipeline/calibrate.py runs/qwen3-1.7b/fused-8bit  # temperature fitted on dev
uv run python pipeline/evaluate.py --model runs/qwen3-1.7b/fused-8bit --name qwen3-1.7b
uv run --group merge python pipeline/export_gguf.py runs/qwen3-1.7b --llama-cpp ~/src/llama.cpp   # see Serve it
uv run python pipeline/report.py
uv run python -m unittest discover -s tests
```

### Data

[`fetch.py`](pipeline/fetch.py) lists every `format:agent-traces` dataset, reads the start
of each, and keeps the ones with Claude sessions in Claude Code or Pi format: 113 of 503,
about 2.4 GB.

[`extract.py`](pipeline/extract.py) keeps only real human prompts in the main thread. Tool
results, meta records, subagent messages, local commands, compaction summaries, task
notifications, interruption markers and IDE context blocks are dropped. Turns from
sessions run by other agents (GPT, GLM, Gemini…) in the same datasets are kept too: the
classifier never sees the model, and their prompts are the same developers' requests.
Each record keeps what the mod sees (the prompt, Claude's previous reply) and, separately,
what happened in that turn: tools called, files read and edited, test or build runs,
subagents, errors, output tokens, duration, the end of the final reply, and the user's
next message.

[`clean.py`](pipeline/clean.py) removes what would teach the wrong thing:

- sessions republished in several repos (most "datasets" are forks of a few releases);
- transcripts flattened into a single "user" message (`…[earlier truncated]…`,
  `ASSISTANT (tool call)`, `TOOL RESULT:`), which one release does throughout. The
  teacher's `pertinent` flag did not catch them: 377 of 521 passed it, and they were
  found by reading samples. Every filter here came from reading random samples;
- scripted benchmarks that replay a question list, and harness prompts: a prompt template
  is kept at most 3 times, and long prompts that share their first eight words count as
  one generator;
- turns with nothing to judge: interrupted, or no reply. A turn cut off when the export
  stopped is kept but marked, so it gets a prompt-only label and no hindsight label;
- secrets (API keys, tokens, private keys), scrubbed before anything is written.

The effort recorded in a trace is not used as a label. About 97% of turns record none, and
the rest are a session-wide setting that almost nobody changes: across 43k raw turns only
26 follow an explicit `/effort` or thinking-level change.

### Labels

[`label.py`](pipeline/label.py) asks a Claude teacher what each request needed, in two modes.

**Hindsight (test gold).** The teacher also sees what happened: how much work the agent
did, whether it ran tests, and whether the user's next message corrected it ("still
failing", "you missed…") or moved on. Sonnet 5.5 labels every Claude turn this way, Opus
5.5 labels the test split independently, and the test set keeps the turns where both agree.
They agree on 75.5% of categories and 81.2% of levels (208 turns).

**Prompt only (training labels).** The teacher sees exactly what the classifier will see.
Hindsight carries information the prompt doesn't (how big the codebase is, how deep the bug
went), which is noise for a student that only reads the prompt. Prompt-only labels are a
function of the student's input, and they still match the hindsight gold on 71.3% of
categories and 75.0% of levels, against 40% for a bag-of-words model trained on hindsight
labels. The training set uses them for that reason. An early run on hindsight labels did
stall at the class prior, but it also lacked gradient clipping (see Training), so it does
not show which labels train better; that comparison was not rerun.

### Training

[`train.py`](pipeline/train.py) runs LoRA (rank 16, scale 2, all linear layers of all
blocks, no dropout) with AdamW at 1e-4, cosine decay, and the loss on the answer letter
only. Two details matter:

- The default loss projects every position onto a ~150k-token vocabulary: tens of
  gigabytes per batch, for one scored token per example.
- Gradients are clipped to norm 1, as in tev1. mlx-lm does not clip by default, and the
  first steps see norms around 60. Unclipped, at 2e-4, they knock the adapter into
  predicting the label prior, where it stays: such a run could not even memorize 32
  examples. With clipping at 1e-4 it memorizes them in 100 steps.

The base is dense Qwen3, not the Qwen3.5 of the tev1 recipe. mlx-lm trains Qwen3.5's
Gated DeltaNet layers through a per-token recurrence that keeps a float32 state per token
for backprop, which runs out of memory even at 0.8B on 24 GB. Training the exact tev1
recipe on Together (Qwen3.5-4B, about $17) and converting it with `mlx_lm.convert` would
also work, since only training is the problem, not inference.

Splits are by session, and each session contributes at most 40 training turns and 12
held-out turns, so one long conversation can't dominate. The prompt is clipped to 3,000
characters and the previous reply to 800, keeping both ends, the same way in training and
serving: prefill is compute-bound, so the longest prompts set the latency tail.

## Train Qwen3.5 on a free Colab GPU

mlx-lm can't train Qwen3.5 on this Mac (see Training), but Unsloth can on a free Colab T4.
Everything stays in this repository; only the computation runs remotely, through the
[Colab CLI](https://pypi.org/project/google-colab-cli/):

```sh
uv tool install google-colab-cli
uv run python cloud/pack.py                                   # tokenized train/dev + trainer, 1.2 MB
uv run python cloud/colab_train.py --base unsloth/Qwen3.5-2B --name qwen3.5-2b
uv run --group merge python cloud/import_adapter.py runs/cloud/qwen3.5-2b --base Qwen/Qwen3.5-2B
uv run python pipeline/calibrate.py runs/cloud/qwen3.5-2b/fused-8bit
uv run python pipeline/evaluate.py --model runs/cloud/qwen3.5-2b/fused-8bit --name qwen3.5-2b
```

- [`cloud/pack.py`](cloud/pack.py) tokenizes the prompts here and ships token ids, so the VM
  trains on exactly the tokens the local server will feed the model.
- [`cloud/colab_train.py`](cloud/colab_train.py) creates the VM, installs Unsloth with
  [`cloud/setup_vm.py`](cloud/setup_vm.py) (Unsloth's own Colab install), and starts
  [`cloud/train_unsloth.py`](cloud/train_unsloth.py) in the background. Every few minutes it
  checks the run, and whenever a checkpoint appears (every 50 optimizer steps) it downloads it
  with the epoch adapters and dev metrics to `runs/cloud/NAME/`. Free sessions end without
  warning: rerun the same command and it creates a VM, uploads the last checkpoint, and
  training resumes from it.
- [`cloud/import_adapter.py`](cloud/import_adapter.py) merges the best adapter into the base
  on the CPU and converts the result to MLX. It stops if the adapter attaches to no layer.

The recipe matches the MLX one: LoRA rank 16 / alpha 32 on the attention and MLP projections,
loss on the answer letter, AdamW 1e-4 with cosine decay, clipping at 1, 3 epochs, best epoch
on dev. On a T4 the 2B takes about 6.5 s per optimizer step, 1 h 40 min for 3 epochs.

Results for Qwen3.5-2B on a free T4:

- With Unsloth's suggested targets (attention and MLP projections), 3 epochs, 1 h 40 min:
  61.9% on dev, but 51.5% category and 60.3% level on test. Those targets miss the five
  projections of the 18 linear-attention layers (379M of the 1.37B weights), which have
  other names, so most of the model's token mixing was never adapted.
- With every linear layer adapted (pass all twelve names to `--targets`), 4 epochs, 2 h
  15 min: 64.4% on dev at epoch 3 (epoch 4 overfit), and on test 56.6% category, 64.0%
  level, 61.8% applied, at 170 / 472 ms.

That ties Qwen3-1.7B on categories and trails it by 2 to 4 points on levels, with worse
calibration and slightly higher latency. With 136 held-out turns that is within noise: a
newer, larger base does not buy a measurable gain at this data size. Qwen3-1.7B stays the
served model.

Unsloth's fused MLP LoRA path trains the MLP adapters at half the scaling the saved adapter
declares. Loaded with standard PEFT, the Qwen3.5-2B adapter dropped from 61.9% to 54.4% on
dev; the base model's logits match between Unsloth on the T4 and MLX here, and halving the
MLP scaling restores 60.6%. `import_adapter.py` applies that and refuses to export a model
whose dev accuracy is more than 3 points from the one measured during training.

[`cloud/colab`](cloud/colab) runs the CLI with only the Colab scope (plus profile and email).
The CLI asks for full Google Cloud and Drive access by default and refreshes its token with
that list, which fails if you declined them at consent. Note that the CLI writes its OAuth
tokens in clear to `~/.config/colab-cli/colab.log` at debug level.

## Serve it

The mod runs the model with [llama.cpp](https://github.com/ggml-org/llama.cpp), which serves
"decision models" at `/v1/systemone`, the System One API the mod already speaks, from build
b11361. [`pipeline/export_gguf.py`](pipeline/export_gguf.py) converts the fused model to a
GGUF and adds what llama.cpp needs to answer decisions:

- `qwen3.decision.type = openjev`: answer with one letter per option, read from the
  next-token logits;
- a `systemone` chat template that renders a request into exactly the prompt the model
  was trained on (system prompt, JSON task, the two state fields it saw);
- `qwen3.decision.temperature.choice`: the temperature fitted on dev, so confidences stay
  calibrated.

It needs a llama.cpp source tree (for `convert_hf_to_gguf.py` and `gguf-py`) and writes
`runs/qwen3-1.7b/auto-effort-Q8_0.gguf`, 1.8 GB. To serve it by hand:

```sh
llama-server -m runs/qwen3-1.7b/auto-effort-Q8_0.gguf --port 8765 --alias auto-effort   # or: llama serve …
```

Users don't need any of this: `/auto-effort setup` in the mod starts `llama-server` (or
`llama serve`) with `-hf` on the published GGUF, which llama.cpp downloads on first start.
The mod sends the local model the question it was trained on and maps the category it
picks through the table for the session's model (`hooks/policy.js`, checked against
[`pipeline/task.py`](pipeline/task.py) and [`pipeline/models.py`](pipeline/models.py) by
`tests/test_pipeline.py`).

On the 136 test turns, the GGUF picks the same letter as the MLX model on 132 and scores
within a point of it (table below).

## Results

The test set is 136 held-out turns from 35 Claude sessions. Its answer key is the category
that Sonnet 5.5 and Opus 5.5 each chose independently after reading what happened in the
turn (the work done, tests run, the user's next message); only turns where they agree are
kept. Every percentage below is the share of those 136 turns where a classifier matches
that key, so 100% means agreeing with both teachers' hindsight judgment every time. Always
picking the model's default is the floor (52.9% of levels). No test session is in training. "Level"
maps the category through the table for the model that ran the turn; "as Opus 5.5" maps
the same turns as if they ran on Opus 5.5. "Mod applies right level" is what the mod ends
up doing with its default confidence threshold of 0.5: the pick when confident enough,
otherwise the session's own effort. Latency is measured end to end on an M4 Pro (24 GB),
one request at a time; Jev's includes the network round trip to OpenCode Zen. The MLX rows
ran through an in-process server; the llama.cpp row through `llama-server` b11408.

| Classifier | Category | Level | Level, as Opus 5.5 | Mod applies right level | Latency p50 / p95 |
| :- | -: | -: | -: | -: | -: |
| Always the model's default | 30.9% | 52.9% | 52.9% | 52.9% |  |
| TF-IDF + logistic regression | 41.2% | 53.7% | 59.6% | 55.1% |  |
| Qwen3-1.7B, not fine-tuned | 25.7% | 52.2% | 31.6% | 52.9% | 140 / 393 ms |
| Qwen3-4B-Instruct, not fine-tuned | 41.2% | 55.1% | 56.6% | 55.1% | 321 / 847 ms |
| Jev 1.13 (OpenCode Zen) |  | 59.6% | 69.9% | 62.5% | 625 / 749 ms |
| Kev-0.8B, local (MLX) |  | 50.7% | 52.2% | 51.5% | 157 / 257 ms |
| Kev-4B, local (MLX) |  | 55.1% | 60.3% | 52.2% | 858 / 1527 ms |
| Kev-4B GGUF Q8_0, local (ggmlc `laya`, Metal) |  | 52.9% | 58.1% | 51.5% | 3542 / 3893 ms |
| Clef-flash 9B 4-bit, local (MLX) |  | 44.9% | 61.0% | 58.8% | 2177 / 3389 ms |
| Qwen3-1.7B fine-tuned, 8-bit (MLX) | 56.6% | 66.2% | 73.5% | 65.4% | 145 / 406 ms |
| **Qwen3-1.7B fine-tuned, GGUF Q8_0 (llama.cpp)** | 56.6% | 65.4% | 73.5% | 66.9% | 137 / 391 ms |
| Qwen3.5-2B on Colab (Unsloth), attention + MLP adapters, 8-bit | 51.5% | 60.3% | 66.9% | 61.8% | 166 / 435 ms |
| Qwen3.5-2B on Colab (Unsloth), all linear layers, 8-bit | 56.6% | 64.0% | 69.1% | 61.8% | 170 / 472 ms |
| Sonnet 5.5 from the prompt alone (teacher) | 71.3% | 75.0% | | | |

Through MLX, the fine-tuned Qwen3-1.7B, 1.8 GB at 8-bit, gets the level right on 66.2% of turns against
59.6% for Jev, and 73.5% against 69.9% as Opus 5.5. It moves effort in the right direction
(raise, keep or lower) on 75.0% of turns against 72.1%, and is within one level on 97.1%
against 91.9%. It answers in 145 ms at the median and 406 ms at p95 (max 827 ms), against
625 / 749 ms for Jev. Jev was asked through the mod's own request, model included. With
n = 136 the accuracy gaps are a few points of standard error, so read them as "at least as
good as Jev, locally, at a quarter of the latency" rather than as precise margins.
Lowering `AUTO_EFFORT_MIN_CONFIDENCE` to 0.45, the best value on dev, raises "applied" to
66.9%. The student agrees with its prompt-only teacher on 63.2% of test categories, so
most of the remaining gap to the teacher is distillation, which more data would help.

Served as a GGUF by llama.cpp, the shipped form, it scores the same on categories, 65.4%
on levels and 66.9% applied, at 137 / 391 ms. In a real Claude Code session on Opus 5.5
(`--plugin-dir .`, the mod starting `llama-server` itself), it raised a multi-file refactor
to `high` in 298 ms and lowered a typo fix to `low` in 118 ms. The first prompt of the
session kept its effort while the server started.

The teacher row is Sonnet 5.5 answering the same question from the prompt alone, as the
classifier does. It is the realistic ceiling for a prompt-only classifier, and an
optimistic one: the answer key includes Sonnet's own hindsight label, so Sonnet partly
agrees with itself. For reference, the two teachers agree with each other, both with
hindsight, on 75.5% of categories and 81.2% of levels.

Kev and Clef-flash are open Jev-like decision models with their own output head. Neither
was trained for this question; they answer the mod's request zero-shot, exactly as Jev
does, through their System One servers on this Mac:

- [Kev](https://github.com/jaredpalmer/kev) 0.8B and 4B, served by `python -m kev.serve`
  (MLX, bf16);
- Kev-4B as the [`mys/kev-4b-GGUF`](https://huggingface.co/mys/kev-4b-GGUF) Q8_0 file,
  served by `laya`, built from [ggmlc](https://github.com/monatis/ggmlc) with Metal;
- [Clef-flash](https://huggingface.co/Cloudflare/clef-flash) through
  [`mlx-community/clef-flash-4bit`](https://huggingface.co/mlx-community/clef-flash-4bit)
  and its `clef_mlx.py serve`.

These runs predate llama.cpp's decision models; recent builds also define `kev` and
`clef` decision types, which were not benchmarked here.

None beats Jev here. Kev answers `default` on most turns (88% for the 4B GGUF), and its
`confidence` field is not the chosen option's probability (median 0.16 for Kev-4B), so with
the mod's 0.5 threshold it almost never changes effort. Clef-flash prefers `medium`, which is
right for Opus 5.5 (61% as Opus 5.5) and too low for models that default to `high`. On this
M4 Pro, Clef-flash 9B takes 2.2 s at the median and Kev-4B GGUF 3.5 s, with some requests
past the mod's 4 s timeout. Run the comparison with
`evaluate.py --baseline systemone --endpoint <url> --api-model <name>`.

Other findings:

- 4-bit is no faster than 8-bit (prefill is compute-bound) and loses about a point, so
  the 8-bit copy is the one to serve.
- The not-fine-tuned 4B-Instruct reaches 41% on categories; the fine-tuned 1.7B reaches
  57%, at less than half the 4B's latency. A fine-tuned 4B was not trained: it would take
  about 5 hours here, and its p95 latency (847 ms untuned) is already above Jev's.
- The fine-tuned model is overconfident after three epochs. A softmax temperature of 2.25
  fitted on dev brings the calibration error from 0.25 to 0.06, which matters because the
  mod acts on confidence.
- Latency outliers of several seconds came from MLX's buffer cache growing with every new
  prompt length until macOS paged. The classifier caps it at 512 MB.

Limits:

- `hard` has 78 training examples and is the weakest class; `exhaustive` has 2, so the
  model never answers `max`. The mod caps effort at `xhigh` by default anyway.
- The data is 2,441 training turns from 367 sessions, about a third of them from other
  agents. Public traces hold almost no Opus 5.5, Sonnet 5.5 or Fable 5.1 human prompts;
  what's specific to those models comes from the table, not the data.
- Labels come from Claude teachers, not from people. Teacher agreement (75.5% category,
  81.2% level) bounds what any score here can mean.
- Teacher labeling ran through `claude -p` on a Claude subscription, about $36 at list
  price in total: $9 for prompt-only labels, $18 for Sonnet hindsight labels, $4 for the
  Opus test labels, and $5 for a first pass that labeled levels instead of categories and
  was discarded.

Next steps, if wanted: synthetic prompts for the `hard` and `exhaustive` classes (the
thinnest), and the exact tev1 recipe (Qwen3.5-4B) trained on Together and exported with
`export_gguf.py` for local serving.
