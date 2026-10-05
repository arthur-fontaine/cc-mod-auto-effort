"""Install Unsloth for Qwen3.5 on a Colab VM. Runs on the VM: `colab exec -f cloud/setup_vm.py`.

The commands are Unsloth's own install cell from its Qwen3.5 Colab notebooks
(github.com/unslothai/notebooks, nb/Qwen3_5_(2B)_Vision.ipynb). Idempotent: it skips the
install when Unsloth already imports.
"""
import importlib.util
import subprocess
import sys


def sh(cmd):
    print("$", cmd, flush=True)
    done = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if done.returncode != 0:
        print(done.stdout[-2000:], done.stderr[-3000:], flush=True)
        raise SystemExit(f"failed: {cmd}")


def installed():
    probe = subprocess.run([sys.executable, "-c", "import unsloth, transformers; print(transformers.__version__)"],
                           capture_output=True, text=True)
    # Unsloth prints a banner on import, so the version is the last line.
    lines = probe.stdout.strip().splitlines()
    return probe.returncode == 0 and bool(lines) and lines[-1].startswith("5.")


if installed():
    print("unsloth already installed", flush=True)
else:
    import numpy
    import PIL

    sh("pip install --upgrade -qqq uv")
    sh(f'uv pip install --system -qqq "torch==2.8.0" "triton>=3.3.0" numpy=={numpy.__version__} '
       f'pillow=={PIL.__version__} torchvision bitsandbytes xformers==0.0.32.post2 '
       '"unsloth_zoo[base] @ git+https://github.com/unslothai/unsloth-zoo" '
       '"unsloth[base] @ git+https://github.com/unslothai/unsloth"')
    sh('uv pip install --system -qqq --no-deps "torchcodec==0.7.0"')
    sh('uv pip install --system --upgrade --no-deps "tokenizers>=0.22.0,<=0.23.0" trl==0.22.2 unsloth unsloth_zoo')
    sh("uv pip install --system transformers==5.2.0")
    # Unsloth bundles the gated delta net kernels; a leftover flash-linear-attention would shadow them.
    sh("uv pip uninstall --system -qqq flash-linear-attention fla-core || true")
    sh("uv pip install --system --no-build-isolation causal_conv1d==1.6.0")
    sh('uv pip install --system --no-deps --upgrade "torchao>=0.16.0"')
    print("installed" if installed() else "install finished but unsloth does not import", flush=True)
