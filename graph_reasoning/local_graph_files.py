"""Read a local graph collection, resolving unchanged files from the original collection."""
from pathlib import Path
import csv
import hashlib


def graph_paths(directory):
    folder = Path(directory)
    paths = {path.name: path for path in folder.glob("*.graphml")}
    shared = folder / "shared_from_raw.csv"
    if shared.is_file():
        with shared.open(encoding="utf-8", newline="") as stream:
            for entry in csv.DictReader(stream):
                name = entry["filename"]
                if Path(name).name != name or not name.endswith(".graphml"):
                    raise ValueError(f"Invalid shared graph name: {name}")
                if name in paths:
                    continue  # A locally regenerated graph takes precedence.
                source = folder.parent / "local_graphs_without_synthesis" / name
                if hashlib.sha256(source.read_bytes()).hexdigest() != entry["sha256"]:
                    raise ValueError(f"Shared graph differs from its recorded contents: {name}")
                paths[name] = source
    return [str(paths[name]) for name in sorted(paths)]
