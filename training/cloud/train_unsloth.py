"""Fine-tune Qwen3.5 on the effort task with Unsloth, on one CUDA GPU (a free Colab T4 works).

    python train_unsloth.py --base unsloth/Qwen3.5-2B --out /content/drive/MyDrive/auto-effort/runs/qwen3.5-2b

Same recipe as pipeline/train.py: LoRA rank 16 / alpha 32, loss on the answer letter only,
AdamW at 1e-4 with cosine decay and 5% warmup, gradients clipped to norm 1, 3 epochs.

Resumable. Checkpoints (adapter, optimizer, scheduler, RNG) go to OUT/checkpoints every
--save-steps optimizer steps; rerun the same command after a disconnect and it continues
from the latest one. At each epoch end the dev accuracy goes to OUT/dev_metrics.jsonl and
the adapter to OUT/epoch-N. When training ends, the best epoch on dev is copied to
OUT/best, which is what to download.
"""
import argparse
import json
import os
import shutil
from pathlib import Path

from unsloth import FastLanguageModel, is_bf16_supported  # Unsloth must be imported before transformers

import torch
from transformers import Trainer, TrainerCallback, TrainingArguments

CATEGORIES = ["trivial", "light", "ordinary", "multi_step", "hard", "exhaustive"]


class Examples(torch.utils.data.Dataset):
    def __init__(self, rows, im_end_id):
        self.rows, self.im_end_id = rows, im_end_id

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        # The loss is on the answer letter alone: it is all the server reads.
        return {"input_ids": r["prompt_ids"] + [r["answer_id"], self.im_end_id],
                "labels": [-100] * len(r["prompt_ids"]) + [r["answer_id"], -100]}


def collate(pad_id):
    def fn(batch):
        n = max(len(b["input_ids"]) for b in batch)
        ids = torch.full((len(batch), n), pad_id, dtype=torch.long)
        labels = torch.full((len(batch), n), -100, dtype=torch.long)
        mask = torch.zeros((len(batch), n), dtype=torch.long)
        for i, b in enumerate(batch):
            # Right padding: the model is causal, so padding after the answer changes nothing before it.
            ids[i, :len(b["input_ids"])] = torch.tensor(b["input_ids"])
            labels[i, :len(b["labels"])] = torch.tensor(b["labels"])
            mask[i, :len(b["input_ids"])] = 1
        return {"input_ids": ids, "labels": labels, "attention_mask": mask}
    return fn


@torch.no_grad()
def dev_scores(model, rows, letter_ids):
    was_training = model.training
    model.eval()
    head = model.get_output_embeddings()
    letters = torch.tensor(letter_ids, device=model.device)
    correct, nll = 0, 0.0
    for r in rows:
        out = model(input_ids=torch.tensor([r["prompt_ids"]], device=model.device), output_hidden_states=True)
        logits = getattr(out, "logits", None)
        if logits is None or logits.numel() == 0:
            # Unsloth may skip materializing logits; project the last hidden state ourselves.
            logits = head(out.hidden_states[-1][:, -1:])
        scores = logits[0, -1, letters].float()
        gold = CATEGORIES.index(r["category"])
        correct += int(scores.argmax().item() == gold)
        nll -= torch.log_softmax(scores, -1)[gold].item()
    if was_training:
        model.train()
    return correct / len(rows), nll / len(rows)


class EpochEval(TrainerCallback):
    def __init__(self, out, model, dev, letter_ids):
        self.out, self.model, self.dev, self.letter_ids = out, model, dev, letter_ids

    def on_epoch_end(self, args, state, control, **kwargs):
        epoch = round(state.epoch)
        accuracy, nll = dev_scores(self.model, self.dev, self.letter_ids)
        self.model.save_pretrained(self.out / f"epoch-{epoch}")
        with (self.out / "dev_metrics.jsonl").open("a") as f:
            f.write(json.dumps({"epoch": epoch, "step": state.global_step, "dev_accuracy": accuracy,
                                "dev_nll": nll}) + "\n")
        print(f"epoch {epoch}: dev accuracy {accuracy:.3f}, log-loss {nll:.3f}", flush=True)


def latest_checkpoint(directory):
    found = sorted(directory.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
    return str(found[-1]) if found else None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", default="unsloth/Qwen3.5-2B")
    parser.add_argument("--data", type=Path, default=Path(__file__).resolve().parent / "data")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--max-len", type=int, default=3072)
    parser.add_argument("--save-steps", type=int, default=50, help="Optimizer steps between checkpoints")
    parser.add_argument("--targets", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    args = parser.parse_args()

    manifest = json.loads((args.data.parent / "manifest.json").read_text())
    read = lambda name: [json.loads(line) for line in (args.data / f"{name}.jsonl").open()]
    train_rows = [r for r in read("train") if len(r["prompt_ids"]) + 2 <= args.max_len]
    dev_rows = read("dev")
    args.out.mkdir(parents=True, exist_ok=True)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base, max_seq_length=args.max_len, load_in_4bit=False, load_in_16bit=True,
        full_finetuning=False,
    )
    tok = getattr(tokenizer, "tokenizer", tokenizer)
    letters = [tok.encode(c, add_special_tokens=False)[0] for c in "ABCDEF"]
    # The bundle was tokenized locally; a different vocabulary would silently scramble it.
    assert letters == manifest["letter_ids"], f"tokenizer mismatch: {letters} vs {manifest['letter_ids']}"

    model = FastLanguageModel.get_peft_model(
        model, r=args.rank, lora_alpha=args.alpha, lora_dropout=0, bias="none",
        target_modules=args.targets.split(","), use_gradient_checkpointing="unsloth",
        random_state=42, max_seq_length=args.max_len,
    )
    model.print_trainable_parameters()

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(args.out / "checkpoints"),
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            lr_scheduler_type="cosine",
            warmup_ratio=0.05,
            weight_decay=0.0,
            max_grad_norm=1.0,
            fp16=not is_bf16_supported(),
            bf16=is_bf16_supported(),
            logging_steps=10,
            save_strategy="steps",
            save_steps=args.save_steps,
            save_total_limit=2,
            train_sampling_strategy="group_by_length",  # transformers 5 name for group_by_length
            remove_unused_columns=False,
            report_to="none",
            seed=42,
        ),
        train_dataset=Examples(train_rows, manifest["im_end_id"]),
        data_collator=collate(tok.pad_token_id if tok.pad_token_id is not None else manifest["im_end_id"]),
        callbacks=[EpochEval(args.out, model, dev_rows, letters)],
    )
    resume = latest_checkpoint(args.out / "checkpoints")
    print(f"train {len(train_rows)} · dev {len(dev_rows)} · resume from {resume or 'scratch'}", flush=True)
    trainer.train(resume_from_checkpoint=resume)

    metrics = [json.loads(line) for line in (args.out / "dev_metrics.jsonl").open()]
    best = max(metrics, key=lambda m: (m["dev_accuracy"], -m["dev_nll"]))
    shutil.copytree(args.out / f"epoch-{best['epoch']}", args.out / "best", dirs_exist_ok=True)
    tok.save_pretrained(args.out / "best")
    (args.out / "summary.json").write_text(json.dumps({"base": args.base, "best": best, "epochs": metrics,
                                                         "args": {k: str(v) for k, v in vars(args).items()}},
                                                        indent=1))
    print(f"best: epoch {best['epoch']} (dev accuracy {best['dev_accuracy']:.3f}) -> {args.out / 'best'}")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
