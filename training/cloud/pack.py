"""Bundle the training data and script for a run on a cloud GPU (Colab).

    uv run python cloud/pack.py      # -> cloud/dist/auto-effort-bundle.tar.gz

The prompts are tokenized here, with the same chat template call the local server uses
(pipeline/infer.py), and shipped as token ids. The cloud side never renders a prompt, so
training sees exactly the tokens the served model will see. Qwen3.5 0.8B to 9B share one
tokenizer; the bundle records which one was used and the trainer checks it.
"""
import argparse
import hashlib
import json
import shutil
import tarfile
from pathlib import Path

from mlx_lm.utils import load_tokenizer
from huggingface_hub import snapshot_download

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SPLIT = ROOT / "data" / "split"


def encode(tokenizer, example):
    ids = tokenizer.apply_chat_template(example["prompt"], add_generation_prompt=True, enable_thinking=False)
    ids = list(ids["input_ids"] if hasattr(ids, "input_ids") else ids)
    answer = tokenizer.encode(example["completion"], add_special_tokens=False)
    assert len(answer) == 1, f"answer letter {example['completion']!r} is not one token"
    return {"id": example["id"], "category": example["category"], "prompt_ids": ids, "answer_id": answer[0]}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tokenizer", default="Qwen/Qwen3.5-2B")
    args = parser.parse_args()

    tokenizer = load_tokenizer(Path(snapshot_download(args.tokenizer, allow_patterns=["*.json", "*.jinja", "*.txt"])))
    letters = [tokenizer.encode(c, add_special_tokens=False)[0] for c in "ABCDEF"]
    stage = HERE / "dist" / "auto-effort"
    if stage.exists():
        shutil.rmtree(stage)
    (stage / "data").mkdir(parents=True)
    counts = {}
    for name in ("train", "dev"):
        rows = [encode(tokenizer, json.loads(line)) for line in (SPLIT / f"{name}.jsonl").open()]
        with (stage / "data" / f"{name}.jsonl").open("w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        counts[name] = len(rows)
    for script in ("train_unsloth.py", "vm_packer.py"):
        shutil.copy(HERE / script, stage / script)
    manifest = {
        "tokenizer": args.tokenizer,
        "letter_ids": letters,
        "im_end_id": tokenizer.encode("<|im_end|>", add_special_tokens=False)[0],
        "examples": counts,
        "max_prompt_tokens": max(len(json.loads(l)["prompt_ids"]) for l in (stage / "data" / "train.jsonl").open()),
    }
    (stage / "manifest.json").write_text(json.dumps(manifest, indent=1))
    out = HERE / "dist" / "auto-effort-bundle.tar.gz"
    with tarfile.open(out, "w:gz") as tar:
        tar.add(stage, arcname="auto-effort")
    digest = hashlib.sha256(out.read_bytes()).hexdigest()[:12]
    print(json.dumps(manifest), f"\n{out} ({out.stat().st_size / 1e6:.1f} MB, sha256 {digest})")


if __name__ == "__main__":
    main()
