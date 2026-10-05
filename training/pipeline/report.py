"""Print the results table for the README from results/*.json.

    uv run python pipeline/report.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ROWS = [
    ("baseline-default", "Always the model's default"),
    ("baseline-tfidf", "TF-IDF + logistic regression"),
    ("zeroshot-qwen3-1.7b", "Qwen3-1.7B, not fine-tuned"),
    ("zeroshot-qwen3-4b-instruct-2507", "Qwen3-4B-Instruct, not fine-tuned"),
    ("baseline-jev", "Jev 1.13 (OpenCode Zen)"),
    ("kev-0.8b", "Kev-0.8B, local (MLX)"),
    ("kev-4b", "Kev-4B, local (MLX)"),
    ("kev-4b-gguf-q8", "Kev-4B GGUF Q8_0, local (ggmlc `laya`, Metal)"),
    ("clef-flash-4bit", "Clef-flash 9B 4-bit, local (MLX)"),
    ("qwen3-1.7b", "Qwen3-1.7B fine-tuned, 8-bit (MLX)"),
    ("qwen3-1.7b-llama-cpp", "**Qwen3-1.7B fine-tuned, GGUF Q8_0 (llama.cpp)**"),
    ("qwen3.5-2b", "Qwen3.5-2B on Colab (Unsloth), attention + MLP adapters, 8-bit"),
    ("qwen3.5-2b-all", "Qwen3.5-2B on Colab (Unsloth), all linear layers, 8-bit"),
    ("reference-sonnet-prompt-only", "Sonnet 5.5 from the prompt alone (teacher)"),
]


def pct(x):
    return "" if x is None else f"{100 * x:.1f}%"


def main():
    print("| Classifier | Category | Level | Level, as Opus 5.5 | Mod applies right level | Latency p50 / p95 |")
    print("| :- | -: | -: | -: | -: | -: |")
    for name, label in ROWS:
        path = ROOT / "results" / f"{name}.json"
        if not path.exists():
            continue
        r = json.loads(path.read_text())
        if "category_accuracy" in r:  # the teacher reference
            print(f"| {label} | {pct(r['category_accuracy'])} | {pct(r['level_accuracy'])} | | | |")
            continue
        lat = r.get("latency_ms")
        latency = f"{lat['p50']:.0f} / {lat['p95']:.0f} ms" if lat else ""
        print(f"| {label} | {pct(r.get('category', {}).get('accuracy'))} | {pct(r['level_actual_model']['accuracy'])} "
              f"| {pct(r['level_as_claude-opus-5-5']['accuracy'])} | {pct(r['level_actual_model']['applied_accuracy'])} "
              f"| {latency} |")


if __name__ == "__main__":
    main()
