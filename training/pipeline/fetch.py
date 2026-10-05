"""Find agent-trace datasets on Hugging Face that contain Claude sessions, and download them.

    uv run python pipeline/fetch.py            # discover + download into data/raw
    uv run python pipeline/fetch.py --probe    # only refresh data/sources.json

Discovery lists every dataset tagged `format:agent-traces`, reads the first 200 kB of up
to three trace files each, and keeps those whose records name a Claude model and look like
Claude Code or Pi sessions. Downloads go file by file with retries, because unauthenticated
bulk snapshots stall; set HF_TOKEN for higher rate limits.
"""
import argparse
import collections
import concurrent.futures as cf
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
API = "https://huggingface.co/api/datasets"
CLAUDE = re.compile(r"claude|opus|sonnet|fable|haiku", re.I)
MODEL_FIELD = re.compile(r'"model"\s*:\s*"([^"]{3,60})"')


def get(url, *, headers=None, timeout=60, retries=8):
    headers = dict(headers or {})
    if os.environ.get("HF_TOKEN"):
        headers["Authorization"] = "Bearer " + os.environ["HF_TOKEN"]
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=timeout) as r:
                return r.read(), r.headers
        except urllib.error.HTTPError as e:
            if attempt == retries - 1 or e.code not in (429, 500, 502, 503, 504):
                raise
            # The Hub rate-limits anonymous clients per 5-minute window.
            wait = int(e.headers.get("Retry-After") or 0) or min(300, 15 * 2 ** attempt)
            time.sleep(wait)
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)


def list_datasets():
    out, url = [], f"{API}?filter=format:agent-traces&limit=1000&full=true"
    while url:
        body, headers = get(url)
        out += json.loads(body)
        m = re.search(r'<([^>]+)>;\s*rel="next"', headers.get("Link") or "")
        url = m.group(1) if m else None
    return out


def tree(repo):
    body, _ = get(f"{API}/{repo}/tree/main?recursive=true")
    return [f for f in json.loads(body) if f["type"] == "file"]


def trace_files(files):
    return [f for f in files if f["path"].endswith(".jsonl")]


def resolve_url(repo, path):
    return f"https://huggingface.co/datasets/{repo}/resolve/main/{urllib.parse.quote(path)}"


def probe(repo):
    try:
        files = trace_files(tree(repo))
    except Exception:
        return None
    models, formats = collections.Counter(), set()
    for f in files[:3]:
        try:
            head, _ = get(resolve_url(repo, f["path"]), headers={"Range": "bytes=0-200000"}, timeout=30, retries=2)
        except Exception:
            continue
        text = head.decode("utf8", "ignore")
        models.update(MODEL_FIELD.findall(text))
        if '"sessionId"' in text and '"parentUuid"' in text:
            formats.add("claude-code")
        if re.search(r'"type"\s*:\s*"session"', text):
            formats.add("pi")
    return {
        "repo": repo,
        "files": len(files),
        "bytes": sum(f.get("size", 0) for f in files),
        "models": dict(models.most_common(6)),
        "formats": sorted(formats),
    }


def discover():
    listed = list_datasets()
    with cf.ThreadPoolExecutor(4) as ex:
        probes = [p for p in ex.map(probe, [d["id"] for d in listed]) if p]
    # Many repos are byte-identical forks of the same release; keep one per signature.
    by_signature = {}
    downloads = {d["id"]: d.get("downloads", 0) for d in listed}
    for p in probes:
        key = (p["files"], p["bytes"])
        if key not in by_signature or downloads[p["repo"]] > downloads[by_signature[key]["repo"]]:
            by_signature[key] = p
    selected = [
        p for p in by_signature.values()
        if any(CLAUDE.search(m) for m in p["models"]) and ({"claude-code", "pi"} & set(p["formats"]))
    ]
    selected.sort(key=lambda p: -p["bytes"])
    DATA.mkdir(exist_ok=True)
    (DATA / "sources.json").write_text(json.dumps(selected, indent=1))
    print(f"{len(listed)} agent-trace datasets, {len(by_signature)} unique, {len(selected)} with Claude sessions "
          f"({sum(p['bytes'] for p in selected) / 1e9:.2f} GB)")
    return selected


def download(repo):
    dest = DATA / "raw" / repo.replace("/", "__")
    done = 0
    for f in trace_files(tree(repo)):
        target = dest / f["path"]
        if target.exists() and target.stat().st_size == f.get("size"):
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        body, _ = get(resolve_url(repo, f["path"]), timeout=300)
        tmp = target.with_suffix(target.suffix + ".part")
        tmp.write_bytes(body)
        tmp.rename(target)
        done += 1
    return repo, done


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--probe", action="store_true", help="Only refresh data/sources.json")
    parser.add_argument("--reuse", action="store_true", help="Download from the existing data/sources.json")
    args = parser.parse_args()
    if args.reuse and (DATA / "sources.json").exists():
        selected = json.loads((DATA / "sources.json").read_text())
    else:
        selected = discover()
    if args.probe:
        return
    with cf.ThreadPoolExecutor(2) as ex:
        futures = {ex.submit(download, p["repo"]): p["repo"] for p in selected}
        for i, fut in enumerate(cf.as_completed(futures), 1):
            try:
                repo, n = fut.result()
                print(f"[{i}/{len(selected)}] {repo}: {n} new files", flush=True)
            except Exception as e:
                print(f"[{i}/{len(selected)}] {futures[fut]} failed: {e}", flush=True)


if __name__ == "__main__":
    main()
