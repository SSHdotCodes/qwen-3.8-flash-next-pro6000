#!/usr/bin/env python3
"""Verify exported sources locally, or installed sources/packages in the image."""

import argparse
import ast
import hashlib
import importlib.metadata
import json
from pathlib import Path


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
    token_path = (
        Path("/opt/qwen-mtp-hotmap.json")
        if installed
        else root / "overlays/hot-token-ids.json"
    )
    tokens = json.loads(token_path.read_text())
    if (
        len(tokens) != 65800
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
        f"Verified {len(manifest['files'])} runtime files and 65,800 draft IDs"
        + ("; installed package versions match" if installed else "")
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--installed", action="store_true")
    verify(parser.parse_args().installed)
