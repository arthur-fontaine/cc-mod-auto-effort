"""Fit a softmax temperature on the dev split and store it next to the model.

    uv run python pipeline/calibrate.py runs/qwen3-1.7b/fused-8bit

The mod acts only when confidence clears AUTO_EFFORT_MIN_CONFIDENCE, so the probabilities
need to mean what they say. One scalar, fitted by minimizing dev log-loss; it never changes
which option wins.
"""
import argparse
import json
import sys
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from infer import Classifier  # noqa: E402
from models import CATEGORIES  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", type=Path)
    args = parser.parse_args()

    clf = Classifier(args.model)
    dev = [json.loads(line) for line in (ROOT / "data" / "split" / "dev.jsonl").open()]
    logits = mx.stack([clf.letter_logits(**e["input"])[0] for e in dev])
    gold = mx.array([CATEGORIES.index(e["category"]) for e in dev])

    def nll(t):
        return -mx.take_along_axis((logits / t - mx.logsumexp(logits / t, axis=-1, keepdims=True)), gold[:, None], axis=1).mean().item()

    grid = [0.5 + 0.05 * i for i in range(91)]  # 0.5 .. 5.0
    best = min(grid, key=nll)
    (args.model / "calibration.json").write_text(json.dumps({"temperature": round(best, 2), "dev_nll": nll(best),
                                                             "dev_nll_uncalibrated": nll(1.0)}, indent=1))
    print(f"temperature {best:.2f}: dev log-loss {nll(1.0):.3f} -> {nll(best):.3f}")


if __name__ == "__main__":
    main()
