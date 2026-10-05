"""Turn raw Claude Code and Pi session files into one record per human prompt.

    uv run python pipeline/extract.py      # data/raw/** -> data/turns.jsonl

A record holds what the mod sees at inference (the prompt, Claude's previous reply, the
model) and, separately under `outcome`, what happened during that turn. The outcome is
hindsight: it is used to label the turn and never shown to the classifier.

Only real human prompts in the main thread count. Tool results, meta records, subagent
(sidechain) messages, local commands, compaction summaries, task notifications and
interruption markers are dropped. A `/effort` command, or a Pi thinking-level change,
is kept as an explicit effort choice for the next prompt.
"""
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "turns.jsonl"

SKIP_PREFIXES = (
    "<local-command-", "<command-name>", "<command-message>", "<bash-input>", "<bash-stdout>",
    "<bash-stderr>", "<task-notification>", "[Request interrupted", "Caveat:", "<system-reminder>",
    "This session is being continued", "[REDACTED_CONFIG_DUMP", "<ci_monitor_event>", "<ci-monitor-event>",
)
READ_TOOLS = {"read", "grep", "glob", "ls", "find", "notebookread", "lsp"}
EDIT_TOOLS = {"edit", "write", "multiedit", "notebookedit", "apply_patch"}
SHELL_TOOLS = {"bash", "shell", "exec_command", "powershell"}
AGENT_TOOLS = {"task", "agent"}
WEB_TOOLS = {"webfetch", "websearch", "web_search", "fetch"}
VERIFY = re.compile(
    r"\b(pytest|unittest|jest|vitest|mocha|rspec|go test|cargo (test|check|build|clippy)|"
    r"(npm|pnpm|yarn|bun) (run )?(test|build|lint|typecheck|check)|make( \w+)?|tsc|mypy|ruff|eslint|"
    r"gradle|mvn|xcodebuild|swift (test|build)|dotnet (test|build))\b"
)
EFFORT_ALIASES = {"off": "low", "minimal": "low", "low": "low", "medium": "medium", "high": "high",
                  "xhigh": "xhigh", "max": "max", "ultrathink": "max"}


def normalize_model(name):
    if not name or name == "<synthetic>":
        return None
    m = name.lower().split("/")[-1]
    m = re.sub(r"^(us|eu|apac|global)\.anthropic\.", "", m)
    m = re.sub(r"^anthropic\.", "", m)
    m = re.sub(r"\[.*?\]$", "", m)
    m = re.sub(r":.*$", "", m)
    m = re.sub(r"-fast$", "", m)
    m = re.sub(r"-v\d+(:\d+)?$", "", m)
    m = re.sub(r"-\d{8}$", "", m)
    m = m.replace(".", "-")
    # Old-style names like claude-4-5-sonnet -> claude-sonnet-4-5.
    old = re.match(r"claude-(\d+(?:-\d+)?)-(opus|sonnet|haiku|fable)$", m)
    if old:
        m = f"claude-{old.group(2)}-{old.group(1)}"
    if m.startswith(("opus", "sonnet", "haiku", "fable")):
        m = "claude-" + m
    return m if m.startswith("claude-") else None


def model_name(name):
    """Claude names normalized; other agents' models kept as-is. The classifier never sees
    the model, so their human prompts are training data too."""
    if not name or name == "<synthetic>":
        return None
    return normalize_model(name) or name.lower().split("/")[-1]


def block_text(content):
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    return "\n".join(b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text")


def has_tool_result(content):
    return isinstance(content, list) and any(
        isinstance(b, dict) and b.get("type") in ("tool_result", "toolResult") for b in content
    )


def clean_prompt(text):
    """Unwrap the envelopes some exporters add; return None if nothing human is left."""
    text = text.strip()
    m = re.search(r"<user_query>\s*(.*?)\s*</user_query>", text, re.S)
    if m:
        text = m.group(1)
    text = re.sub(r"<timestamp>.*?</timestamp>", "", text, flags=re.S)
    text = re.sub(r"<system-reminder>.*?</system-reminder>", "", text, flags=re.S)
    text = re.sub(r"<image_files>.*?</image_files>", "", text, flags=re.S)
    # IDE extensions prepend editor context; the mod only ever sees what the user typed.
    text = re.sub(r"<ide_(opened_file|selection|diagnostics)>.*?</ide_\1>", "", text, flags=re.S)
    text = re.sub(r"^(\[Image[^\]]*\]\s*)+", "", text)
    text = text.strip()
    if not text or text.startswith(SKIP_PREFIXES):
        return None
    return text


def effort_command(text):
    """`/effort high` typed by the user, as Claude Code records it."""
    if "<command-name>/effort</command-name>" not in text:
        return None
    m = re.search(r"<command-args>\s*([a-z]+)", text)
    return EFFORT_ALIASES.get(m.group(1)) if m else None


class Turn:
    def __init__(self, prompt, ts, model, effort, explicit, previous_reply, index):
        self.prompt, self.start, self.end = prompt, ts, ts
        self.model, self.effort, self.explicit = model, effort, explicit
        self.previous_reply, self.index = previous_reply, index
        self.tools = Counter()
        self.files_read, self.files_edited = set(), set()
        self.shell, self.verify, self.errors = 0, 0, 0
        self.output_tokens = {}
        self.assistant_messages = set()
        self.last_text = ""
        self.models = Counter()
        self.efforts = Counter()

    def tool(self, name, args):
        name = (name or "?").lower()
        self.tools[name] += 1
        args = args if isinstance(args, dict) else {}
        path = args.get("file_path") or args.get("path") or args.get("notebook_path")
        if name in READ_TOOLS and isinstance(path, str):
            self.files_read.add(path)
        if name in EDIT_TOOLS and isinstance(path, str):
            self.files_edited.add(path)
        if name in SHELL_TOOLS:
            self.shell += 1
            cmd = args.get("command") or args.get("cmd") or ""
            if isinstance(cmd, list):
                cmd = " ".join(map(str, cmd))
            if isinstance(cmd, str) and VERIFY.search(cmd):
                self.verify += 1

    def outcome(self):
        from datetime import datetime

        def parse(ts):
            try:
                return datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
            except Exception:
                return None

        a, b = parse(self.start or ""), parse(self.end or "")
        return {
            "assistant_messages": len(self.assistant_messages),
            "tool_calls": sum(self.tools.values()),
            "tools": dict(self.tools),
            "files_read": len(self.files_read),
            "files_edited": len(self.files_edited),
            "shell_commands": self.shell,
            "verify_commands": self.verify,
            "subagents": sum(self.tools[t] for t in AGENT_TOOLS),
            "web_calls": sum(self.tools[t] for t in WEB_TOOLS),
            "tool_errors": self.errors,
            "output_tokens": sum(self.output_tokens.values()),
            "duration_s": round(b - a) if a and b and b >= a else None,
            "final_reply_chars": len(self.last_text),
        }


def finish(turns, out, source, path, session):
    for i, t in enumerate(turns):
        nxt = turns[i + 1] if i + 1 < len(turns) else None
        model = t.models.most_common(1)[0][0] if t.models else t.model
        effort = t.efforts.most_common(1)[0][0] if t.efforts else t.effort
        if not model:
            continue
        out.append({
            "id": hashlib.sha1(f"{source}/{path.name}/{t.index}".encode()).hexdigest()[:16],
            "source": source,
            "file": path.name,
            "session": session,
            "turn": t.index,
            "session_turns": len(turns),
            "model": model,
            "effort_setting": effort,
            "effort_explicit": t.explicit,
            "prompt": t.prompt,
            "previous_reply": t.previous_reply,
            "outcome": t.outcome(),
            "final_reply": t.last_text[-1500:],
            "next_prompt": nxt.prompt[:1500] if nxt else None,
            "interrupted": getattr(t, "interrupted", False),
        })


def parse_claude_code(path, source, out):
    turns, current, seen_ids = [], None, set()
    model, effort, explicit, last_reply, session = None, None, None, None, None
    for line in path.open(errors="ignore"):
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if not isinstance(r, dict) or r.get("isSidechain"):
            continue
        session = session or r.get("sessionId")
        uid = r.get("uuid")
        if uid:
            if uid in seen_ids:
                continue
            seen_ids.add(uid)
        msg = r.get("message") if isinstance(r.get("message"), dict) else {}
        if r.get("type") == "user":
            content = msg.get("content")
            text = block_text(content)
            if has_tool_result(content) or r.get("toolUseResult") is not None:
                if current and isinstance(content, list):
                    current.errors += sum(1 for b in content if isinstance(b, dict) and b.get("is_error"))
                    current.end = r.get("timestamp") or current.end
                continue
            if text.strip().startswith("[Request interrupted") and current:
                current.interrupted = True
                continue
            chosen = effort_command(text)
            if chosen:
                effort, explicit = chosen, chosen
                continue
            if r.get("isMeta") or r.get("isCompactSummary") or r.get("isVisibleInTranscriptOnly"):
                continue
            prompt = clean_prompt(text)
            if not prompt:
                continue
            current = Turn(prompt, r.get("timestamp"), model, effort, explicit, last_reply, len(turns))
            explicit = None
            turns.append(current)
        elif r.get("type") == "assistant" and current:
            m = model_name(msg.get("model"))
            if msg.get("model") == "<synthetic>":
                continue
            if m:
                model = m
                current.models[m] += 1
            if isinstance(r.get("effort"), str):
                effort = EFFORT_ALIASES.get(r["effort"], r["effort"])
                current.efforts[effort] += 1
            mid = msg.get("id") or uid
            current.assistant_messages.add(mid)
            usage = msg.get("usage") or {}
            if isinstance(usage.get("output_tokens"), int):
                current.output_tokens[mid] = max(current.output_tokens.get(mid, 0), usage["output_tokens"])
            for b in msg.get("content") or []:
                if not isinstance(b, dict):
                    continue
                if b.get("type") == "tool_use":
                    current.tool(b.get("name"), b.get("input"))
                elif b.get("type") == "text" and b.get("text", "").strip():
                    current.last_text = b["text"]
                    last_reply = b["text"]
            current.end = r.get("timestamp") or current.end
    finish(turns, out, source, path, session)


def parse_pi(path, source, out):
    turns, current = [], None
    model, effort, explicit, last_reply, session = None, None, None, None, None
    for line in path.open(errors="ignore"):
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if not isinstance(r, dict):
            continue
        kind = r.get("type")
        if kind == "session":
            session = r.get("id")
        elif kind == "model_change":
            model = model_name(r.get("modelId")) or None
        elif kind == "thinking_level_change":
            level = EFFORT_ALIASES.get(r.get("thinkingLevel"))
            # The first change is the session's starting level; later ones are deliberate.
            if effort is not None and level:
                explicit = level
            effort = level or effort
        elif kind == "message":
            msg = r.get("message") if isinstance(r.get("message"), dict) else {}
            role = msg.get("role")
            if role == "user":
                prompt = clean_prompt(block_text(msg.get("content")))
                if not prompt:
                    continue
                current = Turn(prompt, r.get("timestamp"), model, effort, explicit, last_reply, len(turns))
                explicit = None
                turns.append(current)
            elif role == "assistant" and current:
                m = model_name(msg.get("model")) or model
                if m:
                    current.models[m] += 1
                if effort:
                    current.efforts[effort] += 1
                current.assistant_messages.add(r.get("id"))
                usage = msg.get("usage") or {}
                if isinstance(usage.get("output"), int):
                    current.output_tokens[r.get("id")] = usage["output"]
                for b in msg.get("content") or []:
                    if not isinstance(b, dict):
                        continue
                    if b.get("type") == "toolCall":
                        current.tool(b.get("name"), b.get("arguments"))
                    elif b.get("type") == "text" and b.get("text", "").strip():
                        current.last_text = b["text"]
                        last_reply = b["text"]
                current.end = r.get("timestamp") or current.end
            elif role == "toolResult" and current:
                if msg.get("isError"):
                    current.errors += 1
                current.end = r.get("timestamp") or current.end
    finish(turns, out, source, path, session)


def detect(path):
    with path.open(errors="ignore") as f:
        head = f.read(20000)
    if re.search(r'"type"\s*:\s*"session"', head) and '"parentUuid"' not in head:
        return parse_pi
    if '"sessionId"' in head or '"parentUuid"' in head:
        return parse_claude_code
    return None


def main():
    out, counts = [], Counter()
    files = sorted(RAW.rglob("*.jsonl"))
    for path in files:
        parser = detect(path)
        if not parser:
            counts["skipped_files"] += 1
            continue
        source = path.relative_to(RAW).parts[0].replace("__", "/")
        before = len(out)
        try:
            parser(path, source, out)
        except Exception as e:  # One malformed export must not stop the run.
            print(f"{path}: {e}", file=sys.stderr)
        counts[parser.__name__] += 1
        counts["turns"] += len(out) - before
    with OUT.open("w") as f:
        for rec in out:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(dict(counts), "->", OUT)
    print("by model:", Counter(r["model"] for r in out).most_common())


if __name__ == "__main__":
    main()
