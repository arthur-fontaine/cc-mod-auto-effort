"""Pick a run's best checkpoint on the dev split and make it the run's adapter.

    uv run python pipeline/select_checkpoint.py runs/qwen3-1.7b

Scores every saved checkpoint (NNNNNNN_adapters.safetensors) and the final adapter on dev
by category accuracy, then mapped-level accuracy. The final adapter is kept as
adapters-final.safetensors; the winner is copied to adapters.safetensors. The test split
is never used here.
"""
import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from infer import Classifier  # noqa: E402
from models import level_for, level_probabilities  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def score(base, run, weights, examples):
    with tempfile.TemporaryDirectory() as tmp:
        shutil.copy(run / "adapter_config.json", tmp)
        shutil.copy(weights, Path(tmp) / "adapters.safetensors")
        clf = Classifier(base, tmp)
        cat = lvl = 0
        for e in examples:
            p = clf.predict(**e["input"])
            cat += p["choice"] == e["category"]
            probs = level_probabilities(p["probabilities"], e["model"])
            lvl += max(probs, key=probs.get) == level_for(e["category"], e["model"])
        return cat / len(examples), lvl / len(examples)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()

    base = json.loads((args.run / "adapter_config.json").read_text())["model"]
    examples = [json.loads(line) for line in (ROOT / "data" / "split" / "dev.jsonl").open()]
    final = args.run / "adapters-final.safetensors"
    if not final.exists():
        shutil.copy(args.run / "adapters.safetensors", final)
    candidates = sorted(args.run.glob("*_adapters.safetensors")) + [final]
    results = {}
    for weights in candidates:
        results[weights.name] = score(base, args.run, weights, examples)
        print(f"{weights.name}: category {results[weights.name][0]:.3f} · level {results[weights.name][1]:.3f}",
              flush=True)
    best = max(results, key=results.get)
    shutil.copy(args.run / best, args.run / "adapters.safetensors")
    (args.run / "selection.json").write_text(json.dumps({"best": best, "dev": results}, indent=1))
    print(f"best: {best}")


if __name__ == "__main__":
    main()
