"""Train on a free Colab GPU from this machine, pulling every checkpoint back as it lands.

    uv run python cloud/colab_train.py --base unsloth/Qwen3.5-2B --name qwen3.5-2b

Needs the Colab CLI (`uv tool install google-colab-cli`, run through cloud/colab) and the
bundle from cloud/pack.py. The code and data stay here: the bundle is uploaded, and two
detached processes run on the VM, train_unsloth.py and vm_packer.py, which re-packs
/content/runs/NAME.state.tar.gz whenever a new checkpoint lands. This script only polls: it reads a
small status through the kernel and downloads the archive to runs/cloud/NAME/state.tar.gz
when it changes. Nothing long runs in the kernel, so a hung call can't block the next one.

Free sessions end without warning. Run the same command again: if training is still going
it just resumes watching; if the VM is gone it creates one, uploads the bundle and the last
synced state, and training resumes from that checkpoint.
"""
import argparse
import json
import subprocess
import sys
import tarfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUNDLE = ROOT / "cloud" / "dist" / "auto-effort-bundle.tar.gz"
CLI = ROOT / "cloud" / "colab"


def colab(*args, stdin=None, timeout=3600):
    done = subprocess.run([str(CLI), *args], input=stdin, capture_output=True, text=True, timeout=timeout)
    if done.returncode != 0:
        raise RuntimeError(f"colab {' '.join(args)} failed: {done.stderr[-1500:] or done.stdout[-1500:]}")
    return done.stdout


def remote(session, code, timeout=120):
    # The CLI sometimes never gets the kernel's reply; bound the call on both sides.
    return colab("exec", "-s", session, "--timeout", str(timeout), stdin=code, timeout=timeout + 60)


def shell(session, command, timeout=120):
    return remote(session, f"import subprocess\nsubprocess.run({command!r}, shell=True, check=True)", timeout)


def session_alive(session):
    try:
        return "Status:" in colab("status", "-s", session, timeout=120)
    except Exception:
        return False


STATUS = """
import json, os, subprocess
def running(pattern):
    return subprocess.run(["pgrep", "-f", pattern], capture_output=True).returncode == 0
out = {remote!r}
state = json.load(open(out + ".state.json")) if os.path.exists(out + ".state.json") else {{}}
tail = open(out + "/train.log", errors="ignore").read()[-1200:] if os.path.exists(out + "/train.log") else ""
print(json.dumps({{"training": running("^python3 train_unsloth"), "packer": running("^python3 vm_packer"),
                  "done": os.path.exists(out + "/summary.json"), "state": state, "tail": tail}}))
"""


def status(session, args):
    return json.loads(remote(session, STATUS.format(remote=args.remote)).strip().splitlines()[-1])


def start_packer(session, args):
    shell(session, f"cd /content/auto-effort && setsid nohup nice -n 10 python3 vm_packer.py {args.remote} "
                   f">> {args.remote}/packer.log 2>&1 &")


def start(session, args, out):
    if not session_alive(session):
        print(f"creating session {session} ({args.gpu})", flush=True)
        colab("new", "-s", session, "--gpu", args.gpu, timeout=900)
    print("installing Unsloth (skipped if present)", flush=True)
    print(colab("exec", "-s", session, "--timeout", "3500", "-f", str(ROOT / "cloud" / "setup_vm.py"),
                timeout=3600)[-300:], flush=True)
    colab("upload", "-s", session, str(BUNDLE), "/content/bundle.tar.gz", timeout=900)
    shell(session, f"tar xzf /content/bundle.tar.gz -C /content && mkdir -p {args.remote}")
    state = out / "state.tar.gz"
    if state.exists():
        # Resume: put the last synced checkpoint back where the trainer looks for it.
        print(f"uploading {state} to resume", flush=True)
        colab("upload", "-s", session, str(state), "/content/resume.tar.gz", timeout=1800)
        shell(session, f"tar xzf /content/resume.tar.gz -C {args.remote}", timeout=600)
    shell(session, f"cd /content/auto-effort && setsid nohup python3 train_unsloth.py --base {args.base} "
                   f"--out {args.remote} {' '.join(args.extra)} >> {args.remote}/train.log 2>&1 &")
    start_packer(session, args)
    print(f"training started on {session}: {args.base} -> {args.remote}", flush=True)


def download(session, args, out):
    part = out / "state.tar.gz.part"
    colab("download", "-s", session, f"{args.remote}.state.tar.gz", str(part), timeout=1800)
    with tarfile.open(part) as tar:  # Only keep an archive that reads back whole.
        tar.extractall(out / "synced", filter="data")
    part.rename(out / "state.tar.gz")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--session", default="qwen35")
    parser.add_argument("--gpu", default="T4")
    parser.add_argument("--every", type=int, default=180, help="Seconds between polls")
    parser.add_argument("extra", nargs="*", help="Extra arguments for train_unsloth.py, after --")
    args = parser.parse_args()
    args.remote = f"/content/runs/{args.name}"
    out = ROOT / "runs" / "cloud" / args.name
    out.mkdir(parents=True, exist_ok=True)

    current = None
    if session_alive(args.session):
        try:
            current = status(args.session, args)
        except Exception as e:
            print(f"status failed ({str(e)[:120]}); restarting the kernel, which training does not depend on",
                  flush=True)
            colab("restart-kernel", "-s", args.session, timeout=300)
            current = status(args.session, args)
    if current and (current["training"] or current["done"]):
        print("training already running; watching it", flush=True)
        if not current["packer"] and not (current["done"] and current["state"].get("done")):
            colab("upload", "-s", args.session, str(ROOT / "cloud" / "vm_packer.py"),
                  "/content/auto-effort/vm_packer.py", timeout=300)
            start_packer(args.session, args)
    else:
        start(args.session, args, out)

    synced = json.loads((out / "synced.json").read_text()) if (out / "synced.json").exists() else {}
    failures = 0
    while True:
        try:
            current = status(args.session, args)
            failures = 0
        except Exception as e:
            failures += 1
            print(f"poll failed ({failures}): {str(e)[:200]}", flush=True)
            if failures == 2:
                try:
                    colab("restart-kernel", "-s", args.session, timeout=300)
                except Exception:
                    pass
            if failures >= 4:
                sys.exit("The VM looks gone. Run the same command again to resume from the last synced checkpoint.")
            time.sleep(args.every)
            continue
        lines = [line for line in current["tail"].replace("\r", "\n").splitlines() if line.strip()]
        print(time.strftime("%H:%M"), current["state"].get("checkpoint"),
              "training" if current["training"] else "stopped", "|", lines[-1][:150] if lines else "", flush=True)
        if current["state"] and current["state"] != synced:
            try:
                download(args.session, args, out)
                synced = current["state"]
                (out / "synced.json").write_text(json.dumps(synced))
                print(f"synced {synced['checkpoint']} -> {out / 'state.tar.gz'}", flush=True)
            except Exception as e:  # A dropped transfer is retried at the next poll.
                print(f"download failed, will retry: {str(e)[:200]}", flush=True)
        if synced.get("done"):
            print(f"done: {out / 'synced' / 'summary.json'}", flush=True)
            break
        if not current["training"] and not current["done"]:
            sys.exit(f"Training stopped without finishing; see {out / 'synced' / 'train.log'}")
        time.sleep(args.every)


if __name__ == "__main__":
    main()
