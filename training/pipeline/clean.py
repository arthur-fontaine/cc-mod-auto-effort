"""Filter, deduplicate and scrub extracted turns.

    uv run python pipeline/clean.py      # data/turns.jsonl -> data/clean.jsonl + data/clean-report.json

Public trace datasets are dominated by forks of the same release and by scripted runs (one
benchmark prompt replayed hundreds of times, paper-reproduction harnesses). Left in, they
would teach the classifier those few templates. So:

- a session that appears in several repos is kept once;
- a prompt template (prompt text with numbers, paths, ids and URLs normalized away) is kept
  at most once per model and TEMPLATE_CAP times overall, and long prompts sharing their
  first eight words count as one generator, kept PREFIX_CAP times;
- benchmark sources that replay a scripted question list are dropped;
- no source contributes more than SOURCE_CAP turns;
- turns with nothing to judge are dropped: interrupted, or no assistant reply; the last
  turn of a session cut off before the agent finished is kept but marked
  `outcome_complete: false`, so it only gets a prompt-only label;
- secrets are scrubbed before anything is written.
"""
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IN = ROOT / "data" / "turns.jsonl"
OUT = ROOT / "data" / "clean.jsonl"
REPORT = ROOT / "data" / "clean-report.json"

# Benchmark runs that replay a scripted question list: not requests a developer wrote.
SCRIPTED_SOURCES = re.compile(r"hf-coding-tools|mentionsanalysis")
TEMPLATE_CAP = 3
PREFIX_CAP = 6
SOURCE_CAP = 1500
MIN_CHARS = 2

NOISE = re.compile(r"^\[(task_started|task_notification|task_progress|init)\]|^<task-|^\s*$")
# Some releases flatten a whole transcript into one "user" message.
FLATTENED = re.compile(r"^…\[earlier truncated\]…|ASSISTANT \(tool call\)|\bTOOL RESULT:|^USER: ")
SECRETS = [
    (re.compile(r"\b(sk-(ant-)?[A-Za-z0-9_\-]{20,})"), "[SECRET]"),
    (re.compile(r"\b(oc_sk_[A-Za-z0-9_\-]{10,})"), "[SECRET]"),
    (re.compile(r"\b(gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})"), "[SECRET]"),
    (re.compile(r"\b(hf_[A-Za-z0-9]{30,})"), "[SECRET]"),
    (re.compile(r"\b(AKIA[0-9A-Z]{16})"), "[SECRET]"),
    (re.compile(r"\b(xox[abpr]-[A-Za-z0-9\-]{10,})"), "[SECRET]"),
    (re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._\-]{20,}"), r"\1[SECRET]"),
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.S), "[SECRET]"),
    (re.compile(r"(?i)\b(password|passwd|api[_-]?key|secret|token)(\s*[:=]\s*)['\"]?[^\s'\"]{8,}"), r"\1\2[SECRET]"),
]


def scrub(text):
    if not text:
        return text
    for pattern, repl in SECRETS:
        text = pattern.sub(repl, text)
    return text


def template(text):
    t = text.lower()
    t = re.sub(r"https?://\S+", "<url>", t)
    t = re.sub(r"(~|/|\.\.?/)[\w.\-/<>\[\]]+", "<path>", t)
    t = re.sub(r"\b[0-9a-f]{8}-[0-9a-f-]{27,}\b|\b[0-9a-f]{12,}\b", "<id>", t)
    t = re.sub(r"\d+", "0", t)
    t = re.sub(r"[^\w<>]+", " ", t).strip()
    return t


def digest(text):
    return hashlib.sha1(text.encode()).hexdigest()


def main():
    rows = [json.loads(line) for line in IN.open()]
    report = {"input": len(rows), "dropped": Counter(), "cut_off_kept": 0}
    rng = random.Random(0)
    rng.shuffle(rows)  # So caps keep a random sample, not the first files on disk.

    kept, sessions_seen = [], {}
    per_template, per_template_model, per_prefix, per_source = Counter(), set(), Counter(), Counter()
    session_owner = {}
    for r in rows:
        prompt = r["prompt"].strip()
        out = r["outcome"]
        if SCRIPTED_SOURCES.search(r["source"]):
            report["dropped"]["scripted_source"] += 1
            continue
        if FLATTENED.search(prompt):
            report["dropped"]["flattened_transcript"] += 1
            continue
        if NOISE.search(prompt) or len(prompt) < MIN_CHARS:
            report["dropped"]["noise"] += 1
            continue
        if r.get("interrupted"):
            report["dropped"]["interrupted"] += 1
            continue
        if out["assistant_messages"] == 0:
            report["dropped"]["no_reply"] += 1
            continue
        # The export stopped mid-turn: the outcome is unknown, so no hindsight label, but
        # the prompt can still be labeled from what the classifier sees.
        r["outcome_complete"] = not (r["next_prompt"] is None and r["turn"] == r["session_turns"] - 1
                                     and not r["final_reply"])
        if not r["outcome_complete"]:
            report["cut_off_kept"] += 1
        # The same session re-published in several repos: keep the first repo seen.
        skey = r.get("session") or r["file"]
        owner = session_owner.setdefault(skey, r["source"])
        if owner != r["source"]:
            report["dropped"]["fork_duplicate"] += 1
            continue
        tkey = digest(template(prompt))
        exact = digest(f"{tkey}|{r['model']}|{template(r['previous_reply'] or '')[:300]}")
        if exact in sessions_seen:
            report["dropped"]["exact_duplicate"] += 1
            continue
        sessions_seen[exact] = True
        if (tkey, r["model"]) in per_template_model or per_template[tkey] >= TEMPLATE_CAP:
            report["dropped"]["template_cap"] += 1
            continue
        # Long prompts that only differ after a shared opening come from one harness.
        # Generators vary a topic after a fixed opening, so compare the first eight words.
        pkey = digest(" ".join(template(prompt).split()[:8])) if len(prompt) > 150 else None
        if pkey and per_prefix[pkey] >= PREFIX_CAP:
            report["dropped"]["prefix_cap"] += 1
            continue
        if per_source[r["source"]] >= SOURCE_CAP:
            report["dropped"]["source_cap"] += 1
            continue
        per_template[tkey] += 1
        per_template_model.add((tkey, r["model"]))
        if pkey:
            per_prefix[pkey] += 1
        per_source[r["source"]] += 1
        for field in ("prompt", "previous_reply", "final_reply", "next_prompt"):
            r[field] = scrub(r[field])
        r["template"] = tkey[:12]
        kept.append(r)

    kept.sort(key=lambda r: (r["source"], r["file"], r["turn"]))
    with OUT.open("w") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    report["kept"] = len(kept)
    report["dropped"] = dict(report["dropped"])
    report["by_model"] = dict(Counter(r["model"] for r in kept).most_common())
    report["by_source"] = dict(Counter(r["source"] for r in kept).most_common())
    report["sessions"] = len({(r["source"], r["file"]) for r in kept})
    REPORT.write_text(json.dumps(report, indent=1))
    print(json.dumps({k: report[k] for k in ("input", "kept", "cut_off_kept", "sessions", "dropped", "by_model")}, indent=1))


if __name__ == "__main__":
    main()
