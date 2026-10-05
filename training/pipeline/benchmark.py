"""Benchmark System One decision models on the test turns, each asked the mod's own request.

    uv run python pipeline/benchmark.py                 # every contender
    uv run python pipeline/benchmark.py nisev kev-4b    # some of them
    uv run python pipeline/report.py                    # the tables, benchmark included

Local models are GGUFs from the Hub (downloaded once to models/), served one at a time by
the llama.cpp on your PATH (`llama-server`, or the unified `llama serve`) with the flags
the mod uses, so they all run on the same runtime. Cloud models are called at their
endpoint, with the key in the variable each one names (AUTO_EFFORT_API_KEY for Jev;
CLOUDFLARE_API_TOKEN, a Workers AI token, and CLOUDFLARE_ACCOUNT_ID for Clef-Flash). Nisev gets the category question it was trained on;
the others get the mod's effort question, as the mod sends it to any Jev endpoint.
Results go to results/bench-NAME.json.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parent.parent
PORT = 8790

CONTENDERS = {
    "nisev": {"hub": ("arthur-fontaine/nisev-1.7b-GGUF", "nisev-1.7b-Q8_0.gguf"), "question": "category",
              # The local export, when there is one, so the benchmark runs before publishing.
              "local": ROOT / "runs" / "qwen3-1.7b" / "nisev-1.7b-Q8_0.gguf"},
    "kev-4b": {"hub": ("ggml-org/Kev-4B-GGUF", "Kev-4B-Q8_0.gguf")},
    # Clef reads embeddings, which needs the whole prompt in one physical batch (512 by default).
    "clef-flash": {"hub": ("ggml-org/Clef-Flash-GGUF", "Clef-Flash-Q4_K_M.gguf"), "args": ["-b", "4096", "-ub", "4096"]},
    "jev-1.13": {"endpoint": "https://opencode.ai/zen/v1/systemone", "api_model": "jev-1.13",
                 "provider": "OpenCode Zen", "key": "AUTO_EFFORT_API_KEY"},
    "clef-flash-cloud": {"endpoint": "https://api.cloudflare.com/client/v4/accounts/{CLOUDFLARE_ACCOUNT_ID}"
                                     "/ai/run/@cf/cloudflare/clef-flash",
                         "api_model": "clef-flash", "provider": "Cloudflare Workers AI", "key": "CLOUDFLARE_API_TOKEN"},
}


def llama_command():
    if shutil.which("llama-server"):
        return ["llama-server"]
    if shutil.which("llama"):
        return ["llama", "serve"]
    sys.exit("llama.cpp was not found: put llama-server or llama (build b11361 or later) on your PATH.")


def gguf_path(name, spec):
    if spec.get("local") and spec["local"].exists():
        return spec["local"]
    repo, filename = spec["hub"]
    return Path(hf_hub_download(repo, filename, local_dir=ROOT / "models" / name))


def wait_ready(proc, url, log):
    for _ in range(600):
        if proc.poll() is not None:
            sys.exit(f"llama.cpp exited while loading; see {log}")
        try:
            with urllib.request.urlopen(url, timeout=2):
                return
        except Exception:
            time.sleep(0.5)
    sys.exit(f"llama.cpp did not start in 5 minutes; see {log}")


def evaluate(name, args, env=None):
    cmd = [sys.executable, str(ROOT / "pipeline" / "evaluate.py"), "--name", f"bench-{name}", *args]
    subprocess.run(cmd, check=True, env=env, stdout=subprocess.DEVNULL)


def run_local(name, spec):
    path = gguf_path(name, spec)
    log = ROOT / "logs" / f"bench-{name}.log"
    log.parent.mkdir(exist_ok=True)
    argv = [*llama_command(), "-m", str(path), "--host", "127.0.0.1", "--port", str(PORT),
            "--alias", name, "-c", "8192", "-np", "2", *spec.get("args", [])]
    with log.open("w") as out:
        proc = subprocess.Popen(argv, stdout=out, stderr=subprocess.STDOUT)
    try:
        wait_ready(proc, f"http://127.0.0.1:{PORT}/health", log)
        endpoint = f"http://127.0.0.1:{PORT}/v1/systemone"
        # One request first, so the first measured one doesn't pay for warm-up.
        evaluate(name, ["--baseline", "systemone", "--endpoint", endpoint, "--api-model", name,
                        "--question", spec.get("question", "effort"), "--limit", "1"])
        evaluate(name, ["--baseline", "systemone", "--endpoint", endpoint, "--api-model", name,
                        "--question", spec.get("question", "effort")])
    finally:
        proc.terminate()
        proc.wait()
    version = subprocess.run([*llama_command()[:1], "--version"], capture_output=True, text=True)
    build = next((w.strip("(,") for w in (version.stdout + version.stderr).split() if w.strip("(,").isdigit()), None)
    return {"where": "local", "machine": machine(), "runtime": f"llama.cpp b{build}" if build else "llama.cpp",
            "file": path.name, "size_gb": round(path.stat().st_size / 1e9, 2)}


def machine():
    def sysctl(key):
        return subprocess.run(["sysctl", "-n", key], capture_output=True, text=True).stdout.strip()
    return f"{sysctl('machdep.cpu.brand_string')}, {int(sysctl('hw.memsize')) // 2**30} GB"


def run_cloud(name, spec):
    missing = [v for v in (spec["key"], *(["CLOUDFLARE_ACCOUNT_ID"] if "{" in spec["endpoint"] else [])) if not os.environ.get(v)]
    if missing:
        sys.exit(f"{name}: set {', '.join(missing)}")
    env = dict(os.environ, AUTO_EFFORT_ENDPOINT=spec["endpoint"].format(**os.environ), AUTO_EFFORT_MODEL=spec["api_model"],
               AUTO_EFFORT_API_KEY=os.environ[spec["key"]])
    evaluate(name, ["--baseline", "jev"], env)
    return {"where": "cloud", "provider": spec["provider"], "client": machine()}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("names", nargs="*", help="Default: " + ", ".join(CONTENDERS))
    args = parser.parse_args()
    if unknown := set(args.names) - set(CONTENDERS):
        parser.error("unknown: " + ", ".join(sorted(unknown)))
    for name in args.names or CONTENDERS:
        spec = CONTENDERS[name]
        print(f"{name}…", flush=True)
        info = run_cloud(name, spec) if "endpoint" in spec else run_local(name, spec)
        path = ROOT / "results" / f"bench-{name}.json"
        result = json.loads(path.read_text())
        result["bench"] = info
        path.write_text(json.dumps(result, indent=1))
        lat = result.get("latency_ms", {})
        print(f"  level {result['level_actual_model']['accuracy']:.1%} · p50 {lat.get('p50', 0):.0f} ms")


if __name__ == "__main__":
    main()
