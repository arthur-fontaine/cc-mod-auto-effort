"""Merge a LoRA adapter trained in the cloud into its base model and convert it to MLX.

    uv run --group merge python cloud/import_adapter.py runs/cloud/qwen3.5-2b --base Qwen/Qwen3.5-2B

Reads RUN/synced/best (the PEFT adapter train_unsloth.py picked on dev), writes RUN/merged
(Hugging Face format, bf16) and RUN/fused-8bit (MLX, what pipeline/evaluate.py loads). The merge runs on
the CPU with PyTorch. It stops if the adapter attaches to no layer, and it checks that the
merged model reproduces the dev accuracy measured during training.

The MLP adapters are applied at half the scaling the adapter declares. Unsloth's fused MLP
LoRA path trains them that way, so a standard PEFT load (lora_alpha / r on every module)
over-applies them: on Qwen3.5-2B the saved adapter scored 54.4% on dev loaded as declared,
and 60.6% with the MLP at half scale, against 61.9% measured in training. Attention adapters
match as declared.
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor

ROOT = Path(__file__).resolve().parent.parent
CATEGORIES = ["trivial", "light", "ordinary", "multi_step", "hard", "exhaustive"]


@torch.no_grad()
def dev_accuracy(model, device):
    data = ROOT / "cloud" / "dist" / "auto-effort"
    letters = json.loads((data / "manifest.json").read_text())["letter_ids"]
    rows = [json.loads(line) for line in (data / "data" / "dev.jsonl").open()]
    correct = 0
    for r in rows:
        logits = model(input_ids=torch.tensor([r["prompt_ids"]], device=device)).logits[0, -1, letters]
        correct += int(logits.argmax().item() == CATEGORIES.index(r["category"]))
    return correct / len(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path)
    parser.add_argument("--base", required=True)
    parser.add_argument("--bits", type=int, default=8)
    parser.add_argument("--mlp-scale", type=float, default=0.5, help="Factor on the MLP adapters' declared scaling")
    parser.add_argument("--max-gap", type=float, default=0.03, help="Allowed dev accuracy gap with training")
    args = parser.parse_args()

    adapter = args.run / "synced" / "best"
    merged_dir, mlx_dir = args.run / "merged", args.run / f"fused-{args.bits}bit"
    base_path = snapshot_download(args.base)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = AutoModelForImageTextToText.from_pretrained(base_path, dtype=torch.bfloat16).to(device)
    model = PeftModel.from_pretrained(model, adapter).eval()
    layers = [(name, m) for name, m in model.named_modules() if isinstance(getattr(m, "scaling", None), dict)]
    print(f"LoRA attached to {len(layers)} modules")
    if not layers:
        sys.exit("The adapter attached to no module: module names differ between the cloud and local model.")
    for name, m in layers:
        if ".mlp." in name:
            m.scaling = {k: v * args.mlp_scale for k, v in m.scaling.items()}

    trained = json.loads((args.run / "synced" / "summary.json").read_text())["best"]["dev_accuracy"]
    measured = dev_accuracy(model, device)
    print(f"dev accuracy: {measured:.3f} here, {trained:.3f} in training")
    if abs(measured - trained) > args.max_gap:
        sys.exit("The merged model does not reproduce the training-time dev accuracy; not exporting it.")

    model = model.merge_and_unload().to("cpu")
    if merged_dir.exists():
        shutil.rmtree(merged_dir)
    model.save_pretrained(merged_dir)
    AutoProcessor.from_pretrained(base_path).save_pretrained(merged_dir)
    if (Path(base_path) / "chat_template.jinja").exists():
        shutil.copy(Path(base_path) / "chat_template.jinja", merged_dir / "chat_template.jinja")
    if mlx_dir.exists():
        shutil.rmtree(mlx_dir)
    convert = Path(sys.executable).parent / "mlx_lm.convert"
    subprocess.run([str(convert), "--hf-path", str(merged_dir), "--mlx-path", str(mlx_dir), "-q", "--q-bits",
                    str(args.bits)], check=True)
    print(f"-> {mlx_dir}")


if __name__ == "__main__":
    main()
