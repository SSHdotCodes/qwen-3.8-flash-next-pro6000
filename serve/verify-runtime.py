#!/usr/bin/env python3
"""Verify exported sources locally, or installed sources/packages in the image."""

import argparse
import ast
import hashlib
import importlib.metadata
import json
from pathlib import Path


def build_sources(path):
    """The SRC list that build_qf.py compiles into qwenfast.so."""
    for node in ast.parse(path.read_text(), filename=str(path)).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "SRC" for t in node.targets
        ):
            return ast.literal_eval(node.value)
    raise ValueError(f"No SRC list in {path}")


def verify(installed=False):
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / "runtime.json").read_text())
    for entry in manifest["files"]:
        path = (
            Path(entry["container_path"])
            if installed
            else root / "overlays" / entry["path"]
        )
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != entry["sha256"]:
            raise ValueError(f"Runtime checksum mismatch: {path}")
        if path.suffix == ".py":
            ast.parse(data, filename=str(path))
    kernels = manifest["qwenfast"]
    source_dir = Path(kernels["source_dir"]) if installed else root / "qwenfast"
    for name, expected in kernels["sources"].items():
        path = source_dir / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Kernel source checksum mismatch: {path}")
    # Every file compiled into qwenfast.so must be pinned in runtime.json, and vice versa.
    if set(build_sources(source_dir / "build_qf.py")) != set(kernels["sources"]) - {
        "build_qf.py"
    }:
        raise ValueError("build_qf.py sources do not match runtime.json")
    if installed and not Path(kernels["library"]).is_file():
        raise ValueError(f"Missing compiled kernels: {kernels['library']}")
    token_path = (
        Path("/opt/qwen-mtp-hotmap.json")
        if installed
        else root / "overlays/hot-token-ids.json"
    )
    tokens = json.loads(token_path.read_text())
    if (
        len(tokens) != manifest["draft_hot_ids"]
        or any(type(i) is not int or not 0 <= i < 248320 for i in tokens)
        or tokens != sorted(set(tokens))
        or not set(range(248044, 248320)).issubset(tokens)
    ):
        raise ValueError("Invalid draft token map")
    if installed:
        for name, expected in manifest["packages"].items():
            actual = importlib.metadata.version(name)
            if actual != expected:
                raise ValueError(f"{name}: expected {expected}, found {actual}")
    print(
        f"Verified {len(manifest['files'])} runtime files, {len(kernels['sources'])} kernel sources"
        f" and {manifest['draft_hot_ids']:,} draft IDs"
        + ("; installed package versions match" if installed else "")
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--installed", action="store_true")
    verify(parser.parse_args().installed)
