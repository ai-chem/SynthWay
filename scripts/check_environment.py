"""Check the application dependencies and supplied index without calling an LLM."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = {
    "numpy": "numpy",
    "pandas": "pandas",
    "scipy": "scipy",
    "scikit-learn": "sklearn",
    "torch": "torch",
    "transformers": "transformers",
    "sentence-transformers": "sentence_transformers",
    "faiss-cpu": "faiss",
    "google-genai": "google.genai",
    "openai": "openai",
    "python-dotenv": "dotenv",
    "deepeval": "deepeval",
    "rich": "rich",
    "requests": "requests",
    "pillow": "PIL",
    "matplotlib": "matplotlib",
    "networkx": "networkx",
    "openpyxl": "openpyxl",
    "numbers-parser": "numbers_parser",
    "plotly": "plotly",
    "powerlaw": "powerlaw",
    "python-louvain": "community",
    "pyvis": "pyvis",
    "seaborn": "seaborn",
}


def check_index_files(index, expected_count=4137):
    """Check file presence, LFS pointers and metadata; not binary deserialization."""
    errors = []
    unreadable = set()
    for name in (
        "meta.jsonl",
        "model.json",
        "tfidf.pkl",
        "tfidf.npz",
        "tfidf_row_norms.npy",
        "vocab_terms.json",
        "vocab_index.faiss",
    ):
        try:
            with (index / name).open("rb") as stream:
                header = stream.read(128)
            if not header:
                raise ValueError("empty file")
            if header.startswith(b"version https://git-lfs.github.com/spec"):
                raise ValueError("Git LFS pointer; fetch the repository's LFS files")
        except (OSError, ValueError) as exc:
            errors.append(f"Index {name}: {exc}")
            unreadable.add(name)
    count = None
    if "meta.jsonl" not in unreadable:
        try:
            records = [
                json.loads(line)
                for line in (index / "meta.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            count = len(records)
            if count != expected_count:
                errors.append(
                    f"Expected {expected_count:,} supplied index documents; found {count:,}."
                )
            ids = [
                record.get("synthesis_id") if isinstance(record, dict) else None
                for record in records
            ]
            if any(not isinstance(value, str) or not value.strip() for value in ids):
                errors.append("Index metadata contains a missing or invalid synthesis_id.")
            elif len(set(ids)) != count:
                errors.append("Index metadata contains duplicate synthesis_id values.")
        except (OSError, UnicodeError, ValueError) as exc:
            errors.append(f"Index meta.jsonl cannot be read: {exc}")
    return count, errors


def main(gui):
    os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")
    os.environ.setdefault("DEEPEVAL_UPDATE_WARNING_OPT_OUT", "YES")
    errors = []
    print(
        f"Python {platform.python_version()} | {platform.system()} {platform.machine()}", flush=True
    )
    if sys.version_info[:2] not in {(3, 11), (3, 12)}:
        errors.append("Use Python 3.11 or 3.12 for the documented application environment.")
    packages = {**PACKAGES, **({"FreeSimpleGUI": "FreeSimpleGUI"} if gui else {})}
    for distribution, module in packages.items():
        try:
            importlib.import_module(module)
            print(f"  {distribution} {importlib.metadata.version(distribution)}", flush=True)
        except Exception as exc:
            errors.append(f"{distribution}: {type(exc).__name__}: {exc}")
    index = ROOT / "synthway/index"
    count, index_errors = check_index_files(index)
    errors.extend(index_errors)
    if count is not None:
        print(f"Supplied index: {count:,} documents")
    if gui:
        try:
            import tkinter

            window = tkinter.Tk()
            try:
                window.withdraw()
                window.update()
                print(f"Tk {window.tk.call('info', 'patchlevel')}: display available")
            finally:
                window.destroy()
        except Exception as exc:
            errors.append(f"Tk/display: {exc}. See README.md.")
    if errors:
        for message in errors:
            print(f"ERROR: {message}", file=sys.stderr)
        print(
            "Install requirements.txt in this Python environment. Tkinter is an operating-system/Python component.",
            file=sys.stderr,
        )
        return 1
    print(
        "Environment check passed. Index binaries were not loaded; model access, credentials and BGE cache/download are not tested."
    )
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gui", action="store_true", help="Also check FreeSimpleGUI and the desktop display."
    )
    sys.exit(main(parser.parse_args().gui))
