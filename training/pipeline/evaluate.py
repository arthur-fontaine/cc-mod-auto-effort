"""Evaluate an effort classifier, or a baseline, on held-out real turns.

    uv run python pipeline/evaluate.py --model runs/qwen3.5-4b/fused --name qwen4b
    uv run python pipeline/evaluate.py --model Qwen/Qwen3.5-4B --name base-4b      # zero-shot
    uv run python pipeline/evaluate.py --baseline default --name default           # always `ordinary`
    uv run python pipeline/evaluate.py --baseline jev --name jev                   # Jev, through the mod's request
    uv run python pipeline/evaluate.py --baseline systemone --endpoint http://127.0.0.1:8009/v1/systemone \
        --api-model kev-latest --name kev-4b                                       # any System One server
    uv run python pipeline/evaluate.py --baseline systemone --question category \
        --endpoint http://127.0.0.1:8090/v1/systemone --name llama-cpp             # our model in llama-server
    uv run python pipeline/evaluate.py --baseline tfidf --name tfidf               # bag of words on the train split

Categories are scored directly. Levels are scored after the per-model table, for the model
that ran the turn and, since the table is deterministic, as if the same turns ran on
Opus 5.5 and Sonnet 5.5. For the mod, what matters is whether effort moves in the right
direction from the model's default, and what effort it ends up applying once its
confidence threshold is taken into account.
"""
import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models import CATEGORIES, LEVELS, level_for, level_probabilities, profile  # noqa: E402
from task import MAX_CONTEXT_CHARS, MAX_PROMPT_CHARS, OPTIONS, QUESTION, clip  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
REPO = ROOT.parent
TARGETS = ["claude-opus-5-5", "claude-sonnet-5-5"]


def direction(level, model):
    d = LEVELS.index(level) - LEVELS.index(profile(model)["default"])
    return (d > 0) - (d < 0)


def classification(gold, pred, labels):
    n = len(gold)
    out = {"n": n, "accuracy": sum(g == p for g, p in zip(gold, pred)) / n}
    idx = {c: i for i, c in enumerate(labels)}
    out["within_one"] = sum(abs(idx[g] - idx[p]) <= 1 for g, p in zip(gold, pred)) / n
    f1s = []
    for c in labels:
        tp = sum(g == c and p == c for g, p in zip(gold, pred))
        fp = sum(g != c and p == c for g, p in zip(gold, pred))
        fn = sum(g == c and p != c for g, p in zip(gold, pred))
        if tp + fn:
            f1s.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
    out["macro_f1"] = sum(f1s) / len(f1s)
    out["confusion"] = {g: dict(Counter(p for gg, p in zip(gold, pred) if gg == g)) for g in labels if g in gold}
    return out


def level_metrics(rows, model_of, min_confidence):
    """Level accuracy and mod behaviour, with each row's level taken for model_of(row)."""
    gold, pred, applied = [], [], []
    for r in rows:
        m = model_of(r)
        g = level_for(r["gold"], m)
        if r.get("level_probs") is not None:
            probs = r["level_probs"](m)
            p = max(probs, key=probs.get)
            conf = probs[p]
        else:
            p, conf = r["pred_level"](m), r.get("confidence", 1.0)
        gold.append(g)
        pred.append(p)
        applied.append(p if conf >= min_confidence else profile(m)["default"])
    models = [model_of(r) for r in rows]
    out = classification(gold, pred, LEVELS)
    out["direction_accuracy"] = sum(direction(g, m) == direction(p, m) for g, p, m in zip(gold, pred, models)) / len(rows)
    moves = [(g, p, m) for g, p, m in zip(gold, pred, models) if direction(p, m) != 0]
    out["moves"] = len(moves) / len(rows)
    out["move_precision"] = sum(direction(g, m) == direction(p, m) for g, p, m in moves) / len(moves) if moves else None
    out["applied_accuracy"] = sum(a == g for a, g in zip(applied, gold)) / len(rows)
    out["applied_within_one"] = sum(abs(LEVELS.index(a) - LEVELS.index(g)) <= 1 for a, g in zip(applied, gold)) / len(rows)
    return out


def calibration(rows):
    n = len(rows)
    bins = defaultdict(list)
    for r in rows:
        bins[min(int(r["confidence"] * 10), 9)].append(r)
    return sum(len(b) / n * abs(sum(x["gold"] == x["pred"] for x in b) / len(b)
                                - sum(x["confidence"] for x in b) / len(b)) for b in bins.values())


def tfidf_predict(examples):
    """Word and bigram TF-IDF of the prompt and previous reply, logistic regression."""
    from scipy.sparse import hstack
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    train = [json.loads(line) for line in (ROOT / "data" / "split" / "train.jsonl").open()]
    prompt = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=50_000)
    reply = TfidfVectorizer(min_df=2, sublinear_tf=True, max_features=20_000)

    def features(rows, fit=False):
        p = [clip(r["input"]["prompt"], MAX_PROMPT_CHARS) for r in rows]
        c = [clip(r["input"].get("previous_reply") or "", MAX_CONTEXT_CHARS) for r in rows]
        if fit:
            return hstack([prompt.fit_transform(p), reply.fit_transform(c)]).tocsr()
        return hstack([prompt.transform(p), reply.transform(c)]).tocsr()

    clf = LogisticRegression(C=4, max_iter=5000).fit(features(train, fit=True), [r["category"] for r in train])
    out = []
    for probs in clf.predict_proba(features(examples)):
        full = {c: 0.0 for c in CATEGORIES}
        full.update({c: float(p) for c, p in zip(clf.classes_, probs)})
        out.append((max(full, key=full.get), full))
    return out


def jev_question():
    # The exact question hooks/policy.js sends, so this is the mod's real setup.
    code = "import('./hooks/policy.js').then(m => console.log(JSON.stringify(m.EFFORT_QUESTION)))"
    return json.loads(subprocess.run(["node", "-e", code], cwd=REPO, capture_output=True, text=True, check=True).stdout)


def category_question():
    """The question our model was trained on, as hooks/policy.js sends it to the local provider."""
    return {"type": "choice", "instructions": QUESTION, "criteria": {c: OPTIONS[c] for c in CATEGORIES}}


def ask_systemone(question, ex, env, clip_limits=False):
    # The mod's limits: 6000 / 1500 for Jev, the training limits for our model.
    limits = (MAX_PROMPT_CHARS, MAX_CONTEXT_CHARS) if clip_limits else (6000, 1500)
    state = {"latest_user_message": clip(ex["input"]["prompt"], limits[0])}
    if ex["input"].get("previous_reply"):
        state["previous_assistant_reply"] = clip(ex["input"]["previous_reply"], limits[1])
    state["model"] = ex["model"]  # as hooks/policy.js sends it
    body = json.dumps({"model": env.get("AUTO_EFFORT_MODEL", "jev-1.13"), "state": state,
                       "questions": {"effort": question}}).encode()
    req = urllib.request.Request(env["AUTO_EFFORT_ENDPOINT"], data=body, headers={
        "Authorization": "Bearer " + env["AUTO_EFFORT_API_KEY"], "Content-Type": "application/json",
        # Cloudflare in front of OpenCode Zen rejects urllib's default agent (error 1010).
        "User-Agent": "auto-effort-eval/1.0"})
    started = time.perf_counter()
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                body = json.load(r)
                return body["answers"]["effort"], (time.perf_counter() - started) * 1000, body.get("usage") or {}
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)


def load_env():
    env = dict(os.environ)
    try:
        for line in (REPO / ".env").read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                env.setdefault(k.strip(), v.strip().strip("'\""))
    except FileNotFoundError:
        pass
    return env


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", help="Model path or HF id (fused, or base with --adapter)")
    parser.add_argument("--adapter")
    parser.add_argument("--baseline", choices=["default", "jev", "systemone", "tfidf"])
    parser.add_argument("--endpoint", help="With --baseline systemone: the server's /v1/systemone URL")
    parser.add_argument("--api-model", default="local", help="With --baseline systemone: the request's model")
    parser.add_argument("--question", choices=["effort", "category"], default="effort",
                        help="What to ask: the mod's effort question (Jev), or the category question our model "
                             "was trained on, mapped to levels as the mod does")
    parser.add_argument("--split", default="test")
    parser.add_argument("--name", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--min-confidence", type=float, default=0.5, help="The mod's AUTO_EFFORT_MIN_CONFIDENCE")
    args = parser.parse_args()

    examples = [json.loads(line) for line in (ROOT / "data" / "split" / f"{args.split}.jsonl").open()][: args.limit]
    rows = []
    if args.baseline == "default":
        for e in examples:
            rows.append({"id": e["id"], "gold": e["category"], "pred": "ordinary", "confidence": 1.0, "model": e["model"],
                         "pred_level": lambda m: profile(m)["default"]})
    elif args.baseline == "tfidf":
        for e, (pred, probs) in zip(examples, tfidf_predict(examples)):
            rows.append({"id": e["id"], "gold": e["category"], "pred": pred, "confidence": probs[pred],
                         "model": e["model"], "probabilities": probs,
                         "level_probs": (lambda pr: lambda m: level_probabilities(pr, m))(probs)})
    elif args.baseline in ("jev", "systemone"):
        env = load_env()
        question = category_question() if args.question == "category" else jev_question()
        if args.baseline == "systemone":
            env.update(AUTO_EFFORT_ENDPOINT=args.endpoint, AUTO_EFFORT_API_KEY="local", AUTO_EFFORT_MODEL=args.api_model)
        for i, e in enumerate(examples):
            answer, ms, usage = ask_systemone(question, e, env, clip_limits=args.question == "category")
            choice = answer["choice"]
            if args.question == "category":
                probs = {c: answer["probabilities"].get(c, 0.0) for c in CATEGORIES}
                rows.append({"id": e["id"], "gold": e["category"], "pred": choice, "confidence": probs[choice],
                             "model": e["model"], "ms": ms, "probabilities": probs,
                             "input_tokens": usage.get("input_tokens"),
                             "level_probs": (lambda pr: lambda m: level_probabilities(pr, m))(probs)})
                continue
            confidence = answer.get("confidence")
            if confidence is None:
                confidence = (answer.get("probabilities") or {}).get(choice, 0)
            rows.append({"id": e["id"], "gold": e["category"], "pred": None, "confidence": confidence,
                         "model": e["model"], "ms": ms, "jev": choice,
                         # Jev picks absolute levels, with `default` meaning the session's own.
                         "pred_level": (lambda c: lambda m: profile(m)["default"] if c == "default" else c)(choice)})
            if i % 50 == 0:
                print(f"{i}/{len(examples)}", flush=True)
    else:
        from infer import Classifier
        clf = Classifier(args.model, args.adapter)
        clf.predict("warm up")
        for e in examples:
            p = clf.predict(**e["input"])
            probs = p["probabilities"]
            rows.append({"id": e["id"], "gold": e["category"], "pred": p["choice"], "confidence": probs[p["choice"]],
                         "model": e["model"], "ms": p["ms"], "tokens": p["tokens"], "probabilities": probs,
                         "level_probs": (lambda pr: lambda m: level_probabilities(pr, m))(probs)})

    result = {"name": args.name, "split": args.split, "n": len(rows)}
    if rows[0]["pred"] is not None:
        result["category"] = classification([r["gold"] for r in rows], [r["pred"] for r in rows], CATEGORIES)
        if args.baseline in (None, "tfidf") or args.question == "category":
            result["category"]["ece"] = calibration(rows)
    result["level_actual_model"] = level_metrics(rows, lambda r: r["model"], args.min_confidence)
    for target in TARGETS:
        result[f"level_as_{target}"] = level_metrics(rows, lambda r, t=target: t, args.min_confidence)
    if all(r.get("ms") is not None for r in rows):
        ms = sorted(r["ms"] for r in rows)
        result["latency_ms"] = {"p50": ms[len(ms) // 2], "p95": ms[int(len(ms) * 0.95)], "max": ms[-1]}

    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    (out / f"{args.name}.json").write_text(json.dumps(result, indent=1))
    with (out / f"{args.name}.predictions.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps({k: v for k, v in r.items() if not callable(v)}) + "\n")

    def short(d):
        return {k: round(v, 3) if isinstance(v, float) else v for k, v in d.items() if k != "confusion"}

    print(json.dumps({k: short(v) if isinstance(v, dict) else v for k, v in result.items()}, indent=1))


if __name__ == "__main__":
    main()
