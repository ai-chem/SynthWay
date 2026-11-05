#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from random import Random
from typing import Any


def stable_int_from_str(s: str) -> int:
    """Stable (cross-run) int from string, unlike Python's built-in hash()."""
    h = hashlib.md5(s.encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def permute_list(items: list[Any], strategy: str, rnd: Random) -> tuple[list[Any], dict[str, Any]]:
    n = len(items)
    meta: dict[str, Any] = {"strategy": strategy, "n": n}

    if n <= 1:
        meta["note"] = "n<=1, unchanged"
        return items, meta

    if strategy == "shuffle":
        idx = list(range(n))
        rnd.shuffle(idx)
        meta["perm_index"] = idx
        return [items[i] for i in idx], meta

    if strategy == "reverse":
        meta["perm_index"] = list(reversed(range(n)))
        return list(reversed(items)), meta

    if strategy == "rotate1":
        # [0,1,2,3] -> [1,2,3,0]
        idx = list(range(1, n)) + [0]
        meta["perm_index"] = idx
        return [items[i] for i in idx], meta

    if strategy == "swap_ends":
        # swap first and last
        idx = list(range(n))
        idx[0], idx[-1] = idx[-1], idx[0]
        meta["perm_index"] = idx
        return [items[i] for i in idx], meta

    raise ValueError(f"Unknown strategy: {strategy}")


def process_retrieval_context_file(
    src_path: Path, dst_path: Path, seed: int, strategy: str
) -> dict[str, Any]:
    data = json.loads(src_path.read_text(encoding="utf-8"))

    if not isinstance(data, dict):
        raise ValueError(f"{src_path} is not a JSON object")

    ctx = data.get("retrieval_context", None)
    if ctx is None:
        raise ValueError(f"{src_path} has no 'retrieval_context' field")
    if not isinstance(ctx, list):
        raise ValueError(f"{src_path} 'retrieval_context' is not a list")

    # deterministic per-file RNG
    rel = str(src_path)
    file_seed = seed + stable_int_from_str(rel)
    rnd = Random(file_seed)

    new_ctx, meta = permute_list(ctx, strategy=strategy, rnd=rnd)
    data["retrieval_context"] = new_ctx

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    dst_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    meta.update(
        {
            "src": str(src_path),
            "dst": str(dst_path),
            "seed": seed,
            "file_seed": file_seed,
        }
    )
    return meta


def copy_tree_with_context_permutation(
    in_dir: Path, out_dir: Path, seed: int, strategy: str
) -> list[dict[str, Any]]:
    manifests: list[dict[str, Any]] = []

    if out_dir.exists():
        # Require a new output directory.
        raise FileExistsError(
            f"Output directory already exists: {out_dir}\n"
            f"Choose another --out-dir or delete it manually."
        )

    for root, dirs, files in os.walk(in_dir):
        root_p = Path(root)
        rel_root = root_p.relative_to(in_dir)
        dst_root = out_dir / rel_root
        dst_root.mkdir(parents=True, exist_ok=True)

        # copy files
        for fn in files:
            src_f = root_p / fn
            dst_f = dst_root / fn

            if fn == "retrieval_context.json":
                meta = process_retrieval_context_file(src_f, dst_f, seed=seed, strategy=strategy)
                manifests.append(meta)
            else:
                # exact copy for everything else
                shutil.copy2(src_f, dst_f)

    return manifests


def main():
    ap = argparse.ArgumentParser(
        description="Clone routes-dir into out-dir, permuting retrieval_context order in each retrieval_context.json"
    )
    ap.add_argument(
        "--routes-dir", required=True, help="Input dataset root (output_LLM_stratified)"
    )
    ap.add_argument(
        "--out-dir", required=True, help="Output dataset root (e.g., check_metrics/...)"
    )
    ap.add_argument("--seed", type=int, default=42, help="Base seed for deterministic shuffling")
    ap.add_argument(
        "--strategy",
        choices=["shuffle", "reverse", "rotate1", "swap_ends"],
        default="shuffle",
        help="Permutation strategy for retrieval_context",
    )
    ap.add_argument(
        "--manifest-name",
        default="permutation_manifest.jsonl",
        help="Manifest file name in out-dir",
    )
    args = ap.parse_args()

    in_dir = Path(args.routes_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()

    if not in_dir.exists():
        raise FileNotFoundError(f"routes-dir not found: {in_dir}")

    manifests = copy_tree_with_context_permutation(
        in_dir, out_dir, seed=args.seed, strategy=args.strategy
    )

    manifest_path = out_dir / args.manifest_name
    with manifest_path.open("w", encoding="utf-8") as f:
        for rec in manifests:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Done. Written dataset: {out_dir}")
    print(f"Permuted files count: {len(manifests)}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
