"""Resolve pipeline modules within the article repository."""
from pathlib import Path


def locate_rag_llm_dir(script_dir: Path) -> Path:
    pipeline=script_dir.resolve().parents[1]
    if not (pipeline/"predictions.py").is_file():
        raise FileNotFoundError(f"Missing SynthWay pipeline: {pipeline / 'predictions.py'}")
    return pipeline
