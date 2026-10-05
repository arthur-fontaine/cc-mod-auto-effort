"""Label each clean turn with the effort it needed, using a Claude teacher that sees hindsight.

    uv run python pipeline/label.py                         # Sonnet labels data/clean.jsonl
    uv run python pipeline/label.py --teacher opus --ids data/split/test.ids --out data/labels-opus.jsonl

The setting recorded in a trace is no label: almost every session runs at the model's
default, whatever the prompt. What a turn *needed* is visible after the fact: how many
files the agent read and edited, whether it ran tests, how long it worked, and whether
the user's next message corrected it ("you missed…", "still failing") or moved on. The
teacher reads that summary and puts the request in a task category (trivial … exhaustive);
pipeline/models.py turns a category into a level for each model. It also flags turns that are not a pertinent developer request
(automated harness prompts, pasted logs with no ask, chit-chat already in the data).

The teacher runs through `claude -p` with tools, MCP servers, settings and the default
system prompt stripped, so each call carries only this file's prompt. Results append to
the output file, so an interrupted run resumes where it stopped.
"""
import argparse
import concurrent.futures as cf
import json
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models import CATEGORIES, CATEGORY_DESCRIPTIONS  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
BATCH = 12

SYSTEM = """You label training data for an effort classifier used with Claude Code, Anthropic's coding agent.

Claude Code runs each request at an effort level. Effort controls how thorough Claude is: how many files it reads, how much it verifies (running tests, double-checking), and how far it pushes through a multi-step task before checking back in. It is not about how capable the model is. Anthropic's guidance: the model's default effort is right for most tasks; raise it when Claude would get the task wrong by skipping a file, not running the tests, or not double-checking; lower it for routine work that needs no investigation.

Your job is to put each request in one category describing how much thoroughness it needs. The category describes the request, not the model that ran it; a separate table turns it into a level for each model.

""" + "\n".join(f"- {c}: {d}" for c, d in CATEGORY_DESCRIPTIONS.items()) + """

Most requests are `ordinary`. Use `exhaustive` only when the user explicitly demands maximum effort.

For each item you get the request (`prompt`), Claude's previous reply for context, and hindsight: which model ran it and at what effort if known, what Claude did during that turn, the end of its final reply, and the user's next message. Use the hindsight to judge what the request really needed:
- Much necessary work (many files, tests, long debugging) means the request needed more than a quick glance suggests. Weigh it against the effort the session ran at: 30 tool calls at xhigh is weaker evidence of a hard task than 30 at medium.
- A next message showing Claude was not thorough enough (missed files, broken tests, "still failing", "you forgot", the same ask again) means the request needed MORE than it got.
- A lot of work for a trivial ask, or the user simply moving on, means it needed less.
- Short follow-ups ("yes", "do it", "continue") take the size of what they approve, which the previous reply shows.
But label the request as a careful person reading it with the previous reply would size it: the classifier will only see the prompt and the previous reply.

Set `pertinent` to false when the item is not a real developer request to a coding agent: a machine-generated harness or benchmark prompt, a prompt that is only a pasted log or file with no ask, system or tool output that leaked into the user turn, or unintelligible text. Genuine short replies ("thanks", "yes") are pertinent.

`confidence` is your probability that the category is right (0 to 1). Keep `reason` under 20 words."""

# For reference only: the same judgment from the prompt alone, i.e. with what the
# classifier sees. It measures how far a strong model gets without hindsight.
SYSTEM_PROMPT_ONLY = SYSTEM.split("For each item you get")[0] + """For each item you get only the request (`prompt`) and Claude's previous reply for context. Size the request as a careful person reading it would.

`pertinent`: false only for machine-generated harness prompts or unintelligible text. `confidence` is your probability that the category is right (0 to 1). Keep `reason` under 20 words."""

SCHEMA = {
    "type": "object",
    "properties": {
        "labels": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "category": {"type": "string", "enum": CATEGORIES},
                    "confidence": {"type": "number"},
                    "pertinent": {"type": "boolean"},
                    "reason": {"type": "string"},
                },
                "required": ["id", "category", "confidence", "pertinent", "reason"],
            },
        }
    },
    "required": ["labels"],
}


def clip(text, n):
    text = text or ""
    if len(text) <= n:
        return text
    half = (n - 20) // 2
    return text[:half] + "\n[… truncated …]\n" + text[-half:]


def item(r):
    o = r["outcome"]
    tools = ", ".join(f"{k}×{v}" for k, v in sorted(o["tools"].items(), key=lambda kv: -kv[1])[:8])
    return {
        "id": r["id"],
        "prompt": clip(r["prompt"], 2500),
        "previous_reply": clip(r.get("previous_reply"), 700) or None,
        "hindsight": {
            "model": r["model"],
            "session_effort": r.get("effort_setting") or "unknown (probably the model default)",
            "tool_calls": o["tool_calls"],
            "tools": tools,
            "files_read": o["files_read"],
            "files_edited": o["files_edited"],
            "test_or_build_runs": o["verify_commands"],
            "subagents": o["subagents"],
            "tool_errors": o["tool_errors"],
            "output_tokens": o["output_tokens"],
            "minutes": round(o["duration_s"] / 60, 1) if o["duration_s"] is not None else None,
            "final_reply_end": clip(r.get("final_reply"), 500),
            "next_user_message": clip(r.get("next_prompt"), 500) or "(none: session ended)",
        },
    }


def call_teacher(model, batch, prompt_only=False):
    items = [item(r) for r in batch]
    if prompt_only:
        items = [{k: v for k, v in it.items() if k != "hindsight"} for it in items]
    payload = json.dumps(items, ensure_ascii=False, indent=1)
    prompt = f"Label these {len(batch)} items. Return one label per id.\n\n{payload}"
    with tempfile.TemporaryDirectory() as cwd:
        proc = subprocess.run(
            ["claude", "-p", "--model", model, "--strict-mcp-config", "--mcp-config", '{"mcpServers":{}}',
             "--setting-sources", "project", "--disable-slash-commands",
             "--exclude-dynamic-system-prompt-sections", "--no-session-persistence", "--tools", "",
             "--output-format", "json", "--system-prompt", SYSTEM_PROMPT_ONLY if prompt_only else SYSTEM,
             "--json-schema", json.dumps(SCHEMA)],
            input=prompt, capture_output=True, text=True, cwd=cwd, timeout=600,
        )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr[-500:] or proc.stdout[-500:])
    out = json.loads(proc.stdout)
    result = out.get("structured_output") or json.loads(out["result"])
    return result["labels"], out.get("total_cost_usd") or 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--teacher", default="sonnet")
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "clean.jsonl")
    parser.add_argument("--ids", type=Path, help="Only label the ids listed in this file")
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "labels.jsonl")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--prompt-only", action="store_true", help="Hide hindsight: a reference, not training labels")
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.data.open()]
    if args.ids:
        wanted = set(args.ids.read_text().split())
        rows = [r for r in rows if r["id"] in wanted]
    done = set()
    if args.out.exists():
        done = {json.loads(line)["id"] for line in args.out.open()}
    todo = [r for r in rows if r["id"] not in done][: args.limit]
    # Keep a session's turns together so the teacher sees related items side by side.
    batches = [todo[i:i + BATCH] for i in range(0, len(todo), BATCH)]
    print(f"{len(done)} already labeled, {len(todo)} to go in {len(batches)} batches with {args.teacher}")

    lock, cost = threading.Lock(), [0.0]

    def run(batch):
        for attempt in range(3):
            try:
                labels, usd = call_teacher(args.teacher, batch, args.prompt_only)
                ids = {r["id"] for r in batch}
                good = [lab for lab in labels if lab["id"] in ids and lab["category"] in CATEGORIES]
                with lock, args.out.open("a") as f:
                    for lab in good:
                        lab["teacher"] = args.teacher
                        f.write(json.dumps(lab, ensure_ascii=False) + "\n")
                    cost[0] += usd
                return len(good)
            except Exception as e:
                err = e
        print(f"batch failed: {err}", file=sys.stderr)
        return 0

    n = 0
    with cf.ThreadPoolExecutor(args.workers) as ex:
        for i, got in enumerate(ex.map(run, batches), 1):
            n += got
            if i % 10 == 0 or i == len(batches):
                print(f"[{i}/{len(batches)}] {n} labels, ${cost[0]:.2f} at list price", flush=True)


if __name__ == "__main__":
    main()
