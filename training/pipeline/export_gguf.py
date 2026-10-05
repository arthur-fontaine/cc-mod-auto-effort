"""Export a fine-tuned run as a llama.cpp decision model (GGUF), served at /v1/systemone.

    uv run --group merge python pipeline/export_gguf.py runs/qwen3-1.7b --llama-cpp ~/src/llama.cpp

llama.cpp (build b11361 and later) serves "decision models" from GGUF metadata. This model
uses the `openjev` decision type: one letter per option, read from the next-token logits
after a prompt that a `systemone` chat template renders from the request. The template
here rebuilds exactly the prompt the model was trained on (pipeline/task.py): the system
prompt, then the JSON task with Python's separators, the request's question, options and
the two state fields the mod sends. The softmax temperature fitted on dev
(pipeline/calibrate.py) is stored as `<arch>.decision.temperature.choice`.

Needs a llama.cpp source tree for convert_hf_to_gguf.py and gguf-py; the run's `fused`
directory is a plain Hugging Face checkpoint.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from task import LETTERS, SYSTEM  # noqa: E402


def systemone_template():
    # Strings from the request go through tojson; everything else is the literal text that
    # json.dumps(..., ensure_ascii=False) produced in training, separators included. Only the
    # two state fields the model saw are rendered: the mod also sends `model`.
    option = ('{"label": "{{ letters[loop.index0] }}", "key": {{ o.key | tojson }}, '
              '"description": {{ o.description | tojson }}}')
    return (
        "{% set letters = '" + LETTERS + "' %}"
        "<|im_start|>system\n" + SYSTEM + "<|im_end|>\n"
        "<|im_start|>user\n"
        '{"question": {{ instructions | tojson }}, "options": ['
        "{% for o in options %}{% if not loop.first %}, {% endif %}" + option + "{% endfor %}"
        '], "state": {"latest_user_message": {{ state.latest_user_message | tojson }}'
        "{% if state.previous_assistant_reply %}"
        ', "previous_assistant_reply": {{ state.previous_assistant_reply | tojson }}'
        "{% endif %}}}<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path)
    parser.add_argument("--llama-cpp", type=Path, required=True, help="llama.cpp source tree (b11361 or later)")
    parser.add_argument("--outtype", default="q8_0")
    parser.add_argument("--temperature", type=float, help="Default: the run's fused-8bit/calibration.json")
    parser.add_argument("--name", default="auto-effort")
    args = parser.parse_args()

    fused = args.run / "fused"
    temperature = args.temperature or json.loads((args.run / "fused-8bit" / "calibration.json").read_text())["temperature"]
    raw = args.run / f"{args.name}.{args.outtype}.raw.gguf"
    out = args.run / f"{args.name}-{args.outtype.upper()}.gguf"
    subprocess.run([sys.executable, str(args.llama_cpp / "convert_hf_to_gguf.py"), str(fused),
                    "--outtype", args.outtype, "--outfile", str(raw)], check=True)

    sys.path.insert(0, str(args.llama_cpp / "gguf-py"))
    import gguf
    from gguf.scripts.gguf_new_metadata import MetadataDetails, copy_with_new_metadata

    reader = gguf.GGUFReader(raw, "r")
    arch = reader.get_field(gguf.Keys.General.ARCHITECTURE).contents()
    string, f32 = gguf.GGUFValueType.STRING, gguf.GGUFValueType.FLOAT32
    new = {
        gguf.Keys.General.NAME: MetadataDetails(string, args.name),
        f"{arch}.decision.type": MetadataDetails(string, "openjev"),
        f"{arch}.decision.temperature.choice": MetadataDetails(f32, temperature),
        "tokenizer.chat_template.systemone": MetadataDetails(string, systemone_template()),
        "tokenizer.chat_templates": MetadataDetails(gguf.GGUFValueType.ARRAY, ["systemone"], sub_type=string),
    }
    writer = gguf.GGUFWriter(out, arch=arch, endianess=reader.endianess)
    copy_with_new_metadata(reader, writer, new, remove_metadata=[])
    raw.unlink()
    print(f"-> {out} ({out.stat().st_size / 1e9:.2f} GB, {arch}, temperature {temperature})")


if __name__ == "__main__":
    main()
