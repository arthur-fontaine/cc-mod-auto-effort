"""Keep RUN.state.tar.gz up to date with a run's latest checkpoint. Runs on the VM.

    setsid nohup nice -n 10 python3 vm_packer.py /content/runs/NAME &

Started by cloud/colab_train.py next to the trainer, so packing never happens inside the
notebook kernel, where a long call blocks every later `colab exec`. Each new checkpoint, the
epoch adapters and the metrics are packed to a temporary file and renamed into place, and
RUN.state.json says which checkpoint the archive holds. Exits once the run has
written summary.json and the final state is packed.
"""
import glob
import json
import os
import sys
import tarfile
import time

run = sys.argv[1]
packed = None
while True:
    done = os.path.exists(f"{run}/summary.json")
    checkpoints = sorted(glob.glob(f"{run}/checkpoints/checkpoint-*"), key=lambda p: int(p.rsplit("-", 1)[1]))
    # The newest directory may still be being written; pack the one before it unless training is over.
    ready = checkpoints if done else checkpoints[:-1]
    latest = os.path.basename(ready[-1]) if ready else None
    key = (latest, done)
    if latest and key != packed:
        time.sleep(5)
        with tarfile.open(f"{run}.state.tar.gz.tmp", "w:gz", compresslevel=1) as tar:
            tar.add(f"{run}/checkpoints/{latest}", arcname=f"checkpoints/{latest}")
            for path in glob.glob(f"{run}/epoch-*") + glob.glob(f"{run}/best"):
                tar.add(path, arcname=os.path.relpath(path, run))
            for name in ("dev_metrics.jsonl", "summary.json", "train.log"):
                if os.path.exists(f"{run}/{name}"):
                    tar.add(f"{run}/{name}", arcname=name)
        os.replace(f"{run}.state.tar.gz.tmp", f"{run}.state.tar.gz")
        with open(f"{run}.state.json.tmp", "w") as f:
            json.dump({"checkpoint": latest, "done": done, "time": time.time()}, f)
        os.replace(f"{run}.state.json.tmp", f"{run}.state.json")
        packed = key
        print(f"packed {latest} done={done}", flush=True)
        if done:
            break
    time.sleep(30)
