"""Join labels to turns, split by session, and render training files.

    uv run python pipeline/build.py                                   # train/dev on prompt-only labels
    uv run python pipeline/build.py --train-labels data/labels.jsonl  # train/dev on hindsight labels

Each example is the exact request the classifier gets at inference: the system prompt and a
JSON decision task whose state is what the mod sends (the prompt and Claude's previous
reply, clipped like the mod clips them). The completion is one option letter, the task
category. Hindsight never reaches the prompt; `assert_no_leak` checks it.

Splits are by session, so no conversation has turns on both sides, and each session
contributes at most SESSION_CAP turns.

Train and dev use the labels in --train-labels: by default Sonnet's judgment from the prompt
alone, which is a function of exactly what the classifier sees. The test gold is stricter:
Claude sessions only, labeled with hindsight by Sonnet and independently by Opus
(data/labels-opus.jsonl), kept only where both agree on the category.
"""
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models import CATEGORIES, level_for  # noqa: E402
from task import LETTER_OF, messages  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
SPLIT = DATA / "split"
MIN_CONFIDENCE = 0.4
TEST_FRACTION, DEV_FRACTION = 0.12, 0.08
# Long sessions would otherwise dominate: one 200-turn conversation is one user's style.
# Held-out splits are capped harder so no single session drives the metrics.
SESSION_CAP = {"train": 40, "dev": 12, "test": 12}


def bucket(key):
    return int(hashlib.sha1(key.encode()).hexdigest(), 16) % 10_000 / 10_000


def split_of(r):
    b = bucket(f"{r['source']}/{r.get('session') or r['file']}")
    return "test" if b < TEST_FRACTION else "dev" if b < TEST_FRACTION + DEV_FRACTION else "train"


def example(r, category):
    return {
        "id": r["id"], "category": category, "model": r["model"], "source": r["source"],
        "input": {"prompt": r["prompt"], "previous_reply": r.get("previous_reply")},
        "prompt": messages(prompt=r["prompt"], previous_reply=r.get("previous_reply")),
        "completion": LETTER_OF[category],
    }


def assert_no_leak(ex, r):
    """The classifier may only see what the mod sends: no outcome, no later turn, no model."""
    state = json.loads(ex["prompt"][1]["content"])["state"]
    allowed = {"latest_user_message", "previous_assistant_reply"}
    assert set(state) <= allowed, f"unexpected state keys {set(state) - allowed} in {r['id']}"
    seen = state["latest_user_message"] + (state.get("previous_assistant_reply") or "")
    visible = r["prompt"] + (r.get("previous_reply") or "")
    for field in ("final_reply", "next_prompt"):
        text = (r.get(field) or "").strip()
        if len(text) > 80 and text[:80] in seen and text[:80] not in visible:
            raise AssertionError(f"{field} leaked into the prompt of {r['id']}")


def read_labels(path):
    labels = {}
    if path.exists():
        for line in path.open():
            lab = json.loads(line)
            labels[lab["id"]] = lab
    return labels


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-labels", type=Path, default=DATA / "labels-prompt-only.jsonl")
    args = parser.parse_args()

    turns = {json.loads(line)["id"]: json.loads(line) for line in (DATA / "clean.jsonl").open()}
    train_labels = read_labels(args.train_labels)
    hindsight = read_labels(DATA / "labels.jsonl")
    second = read_labels(DATA / "labels-opus.jsonl")
    report = Counter()
    splits = {"train": [], "dev": [], "test": []}
    agreement = {"category": [], "level": []}
    for tid, r in turns.items():
        name = split_of(r)
        if name == "test":
            lab, other = hindsight.get(tid), second.get(tid)
            if not lab or not other or not r["model"].startswith("claude-"):
                report["test_without_both_hindsight_labels"] += 1
                continue
            if not (lab["pertinent"] and other["pertinent"]):
                report["test_not_pertinent"] += 1
                continue
            agreement["category"].append(other["category"] == lab["category"])
            agreement["level"].append(level_for(other["category"], r["model"]) == level_for(lab["category"], r["model"]))
            if other["category"] != lab["category"]:
                report["test_teacher_disagreement"] += 1
                continue
        else:
            lab = train_labels.get(tid)
            if not lab:
                report[f"{name}_unlabeled"] += 1
                continue
            if not lab["pertinent"]:
                report[f"{name}_not_pertinent"] += 1
                continue
            if lab["confidence"] < MIN_CONFIDENCE:
                report[f"{name}_low_confidence"] += 1
                continue
        ex = example(r, lab["category"])
        assert_no_leak(ex, r)
        splits[name].append(ex)

    rng = random.Random(42)
    for name, exs in splits.items():
        rng.shuffle(exs)
        per_session, kept = Counter(), []
        for ex in exs:
            key = (ex["source"], turns[ex["id"]].get("session") or turns[ex["id"]]["file"])
            if per_session[key] < SESSION_CAP[name]:
                per_session[key] += 1
                kept.append(ex)
            else:
                report[f"{name}_session_cap"] += 1
        splits[name] = kept
    SPLIT.mkdir(parents=True, exist_ok=True)
    for name, exs in splits.items():
        with (SPLIT / f"{name}.jsonl").open("w") as f:
            for ex in exs:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    summary = {
        "filtered": dict(report),
        "sizes": {k: len(v) for k, v in splits.items()},
        "sessions": {k: len({(e["source"], turns[e["id"]].get("session")) for e in v}) for k, v in splits.items()},
        "categories": {k: {c: sum(e["category"] == c for e in v) for c in CATEGORIES} for k, v in splits.items()},
        "models": {k: dict(Counter(e["model"] for e in v).most_common()) for k, v in splits.items()},
    }
    if agreement["category"]:
        summary["teacher_agreement"] = {k: round(sum(v) / len(v), 3) for k, v in agreement.items()}
        summary["teacher_agreement"]["n"] = len(agreement["category"])
    (DATA / "build-report.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
