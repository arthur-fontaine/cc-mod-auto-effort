"""Fuse a trained adapter into its base model and write quantized copies for serving.

    uv run python pipeline/export.py runs/qwen3.5-4b          # -> runs/qwen3.5-4b/fused, fused-8bit, fused-4bit

The base model is read from the run's adapter_config.json.
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

BIN = Path(sys.executable).parent


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path)
    parser.add_argument("--bits", type=int, nargs="*", default=[8, 4])
    args = parser.parse_args()

    base = json.loads((args.run / "adapter_config.json").read_text())["model"]
    fused = args.run / "fused"
    if fused.exists():
        shutil.rmtree(fused)
    subprocess.run([BIN / "mlx_lm.fuse", "--model", base, "--adapter-path", args.run, "--save-path", fused], check=True)
    for bits in args.bits:
        out = args.run / f"fused-{bits}bit"
        if out.exists():
            shutil.rmtree(out)
        subprocess.run([BIN / "mlx_lm.convert", "--hf-path", fused, "--mlx-path", out, "-q", "--q-bits", str(bits)],
                       check=True)
    for path in [fused] + [args.run / f"fused-{b}bit" for b in args.bits]:
        size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
        print(f"{path}: {size / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
