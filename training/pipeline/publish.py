"""Publish Nisev to the Hugging Face Hub: the GGUF and its model card.

    uv run hf auth login     # once, with a token that can write
    uv run python pipeline/publish.py runs/qwen3-1.7b --repo arthur-fontaine/nisev-1.7b-GGUF

The card is MODEL_CARD.md with the benchmark table from results/bench-*.json and the chart
from docs/benchmark.svg (run pipeline/benchmark.py and pipeline/plot.py first). Uploading again replaces both files; the Hub keeps history.
"""
import argparse
import sys
from pathlib import Path

from huggingface_hub import HfApi

sys.path.insert(0, str(Path(__file__).resolve().parent))
from report import benchmark_table  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--file", default="nisev-1.7b-Q8_0.gguf")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--card-only", action="store_true", help="Update the model card alone")
    args = parser.parse_args()

    gguf = args.run / args.file
    if not gguf.exists():
        sys.exit(f"{gguf} not found: run pipeline/export_gguf.py first.")
    card = (ROOT / "MODEL_CARD.md").read_text()
    card = card.replace("{{repo}}", args.repo).replace("{{file}}", args.file).replace("{{benchmark}}", benchmark_table())

    api = HfApi()
    url = api.create_repo(args.repo, private=args.private, exist_ok=True)
    api.upload_file(path_or_fileobj=card.encode(), path_in_repo="README.md", repo_id=args.repo,
                    commit_message="Update the model card")
    api.upload_file(path_or_fileobj=str(ROOT.parent / "docs" / "benchmark.svg"), path_in_repo="benchmark.svg",
                    repo_id=args.repo, commit_message="Update the benchmark chart")
    if not args.card_only:
        api.upload_file(path_or_fileobj=str(gguf), path_in_repo=args.file, repo_id=args.repo,
                        commit_message=f"Upload {args.file}")
    print(f"-> {url}")


if __name__ == "__main__":
    main()
