"""LoRA fine-tune a dense Qwen3 model on the effort task with MLX, on Apple silicon.

    uv run python pipeline/train.py --base Qwen/Qwen3-1.7B --out runs/qwen3-1.7b
    uv run python pipeline/train.py --base Qwen/Qwen3-4B-Instruct-2507 --out runs/qwen3-4b --batch-size 2 --grad-accum 4

Recipe after togethercomputer/tev1 (LoRA on all linear layers, loss on the answer letter
only, cosine schedule with warmup), scaled to this smaller dataset. Prompts are rendered
with the base model's chat template in non-thinking mode, exactly as the server renders
them. The adapter directory loads with `mlx_lm.load(base, adapter_path=out)`.

tev1 starts from Qwen3.5-4B. Its Gated DeltaNet layers train in mlx-lm through a
per-token Python recurrence that keeps a float32 state per token for backprop (about
1 MB x batch per token per layer), which runs out of memory even at 0.8B on a 24 GB Mac.
Dense Qwen3 trains through the fused attention kernel instead.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx_lm import load
from mlx_lm.tuner.trainer import TrainingArgs, train
from mlx_lm.tuner.utils import linear_to_lora_layers, print_trainable_parameters

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))


def encode(tokenizer, example, max_len):
    prompt = tokenizer.apply_chat_template(example["prompt"], add_generation_prompt=True, enable_thinking=False)
    if hasattr(prompt, "input_ids"):
        prompt = prompt["input_ids"]
    end = tokenizer.encode("<|im_end|>", add_special_tokens=False)
    completion = tokenizer.encode(example["completion"], add_special_tokens=False) + end
    if len(prompt) + len(completion) > max_len:
        return None  # Truncating would cut the answer position; drop the rare outlier.
    return (list(prompt) + completion, len(prompt))


def answer_loss(model, batch, lengths):
    """Cross-entropy on the answer letter only.

    mlx-lm's default loss projects every position onto the vocabulary before masking. With
    Qwen3.5's ~248k-token vocabulary that is ~24 GB of float32 logits for a batch of 8
    long prompts, for a single scored token per example. Here the LM head runs on that one
    position: the token right after the prompt, `lengths[:, 0]`.
    """
    lm = getattr(model, "language_model", model)
    inputs, targets = batch[:, :-1], batch[:, 1:]
    hidden = lm.model(inputs)
    pos = (lengths[:, 0] - 1)[:, None]
    h = mx.take_along_axis(hidden, mx.broadcast_to(pos[..., None], (pos.shape[0], 1, hidden.shape[-1])), axis=1)
    logits = lm.model.embed_tokens.as_linear(h) if lm.args.tie_word_embeddings else lm.lm_head(h)
    target = mx.take_along_axis(targets, pos, axis=1)
    ce = nn.losses.cross_entropy(logits.astype(mx.float32), target)
    return ce.sum() / ce.size, mx.array(ce.size)


class ClippedAdamW(optim.AdamW):
    """AdamW with global gradient-norm clipping, as in tev1 (max_grad_norm 1).

    The first steps see gradient norms around 60: the base model puts the answer letter far
    from where the labels are. Unclipped, at 2e-4, they throw the adapter into predicting
    the label prior, where it stays (it then cannot even memorize 32 examples).
    """

    def __init__(self, *args, max_norm=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_norm = max_norm

    def update(self, model, gradients):
        gradients, _ = optim.clip_grad_norm(gradients, self.max_norm)
        return super().update(model, gradients)


class Dataset:
    def __init__(self, items):
        self.items = items

    def __getitem__(self, i):
        return self.items[i]

    def __len__(self):
        return len(self.items)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "split")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=float, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=32)
    # tev1 uses 0; dropout masks would also be re-sampled when checkpointed layers recompute.
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--layers", type=int, default=-1, help="Blocks to adapt from the top; -1 for all")
    parser.add_argument("--max-seq", type=int, default=3072)
    parser.add_argument("--warmup", type=float, default=0.05)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--grad-checkpoint", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--grad-accum", type=int, default=2, help="Micro-batches per optimizer step")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    mx.random.seed(args.seed)
    # Keep freed buffers from piling up: this runs next to the user's other apps on 24 GB.
    mx.set_cache_limit(2 * 1024**3)
    np.random.seed(args.seed)
    model, tokenizer = load(args.base)
    read = lambda name: [json.loads(line) for line in (args.data / f"{name}.jsonl").open()]
    train_set = Dataset([x for x in (encode(tokenizer, ex, args.max_seq) for ex in read("train")) if x])
    dev_set = Dataset([x for x in (encode(tokenizer, ex, args.max_seq) for ex in read("dev")) if x])
    lengths = [len(t) for t, _ in train_set.items]
    print(f"train {len(train_set)} · dev {len(dev_set)} · tokens/example p50 {int(np.median(lengths))} "
          f"p95 {int(np.percentile(lengths, 95))} max {max(lengths)}")

    model.freeze()
    num_layers = len(model.layers) if args.layers < 0 else args.layers
    lora = {"rank": args.rank, "scale": args.alpha / args.rank, "dropout": args.dropout}
    linear_to_lora_layers(model, num_layers, lora)
    print_trainable_parameters(model)

    # mlx-lm counts every micro-batch as an iteration and steps the optimizer every grad_accum.
    iters = math.ceil(len(train_set) * args.epochs / args.batch_size)
    updates = math.ceil(iters / args.grad_accum)
    warmup = max(1, int(updates * args.warmup))
    schedule = optim.join_schedules(
        [optim.linear_schedule(1e-7, args.lr, warmup), optim.cosine_decay(args.lr, updates - warmup, 0.0)], [warmup]
    )
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "adapter_config.json").write_text(json.dumps({
        "model": args.base,
        "fine_tune_type": "lora",
        "num_layers": num_layers,
        "lora_parameters": lora,
        "training": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "iters": iters,
        "train_examples": len(train_set),
    }, indent=1))

    started = time.time()
    train(
        model=model,
        optimizer=ClippedAdamW(learning_rate=schedule, weight_decay=0.0, max_norm=args.max_grad_norm),
        train_dataset=train_set,
        val_dataset=dev_set,
        loss=answer_loss,
        args=TrainingArgs(
            batch_size=args.batch_size,
            iters=iters,
            val_batches=-1,
            steps_per_report=20,
            steps_per_eval=max(50, iters // 6),
            steps_per_save=max(100, iters // 3),
            max_seq_length=args.max_seq,
            adapter_file=str(args.out / "adapters.safetensors"),
            grad_checkpoint=args.grad_checkpoint,
            grad_accumulation_steps=args.grad_accum,
        ),
    )
    print(f"trained {iters} iters in {(time.time() - started) / 60:.1f} min -> {args.out}")


if __name__ == "__main__":
    main()
