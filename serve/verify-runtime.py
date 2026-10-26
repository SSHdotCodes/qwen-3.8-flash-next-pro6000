#!/usr/bin/env python3
"""Verify exported sources locally, or installed sources/packages in the image."""

import argparse
import ast
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re


def build_sources(path):
    """The SRC list that build_qf.py compiles into qwenfast.so."""
    for node in ast.parse(path.read_text(), filename=str(path)).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "SRC" for t in node.targets
        ):
            return ast.literal_eval(node.value)
    raise ValueError(f"No SRC list in {path}")


def verify(installed=False, draft_a16=None):
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
    # Pin compilation units separately from headers and optional standalone test bindings.
    units = build_sources(source_dir / "build_qf.py")
    if units != kernels["compilation_units"] or len(units) != len(set(units)):
        raise ValueError("build_qf.py sources do not match runtime.json")
    if not set(units).issubset(kernels["sources"]):
        raise ValueError("Unpinned kernel compilation unit")
    for name in kernels["sources"]:
        for header in re.findall(r'^\s*#include\s+"([^"]+)"', (source_dir / name).read_text(), re.M):
            if header not in kernels["sources"]:
                raise ValueError(f"Unpinned kernel header: {header}")
    recipe = root / manifest["draft_a16"]["preparation"]
    if hashlib.sha256(recipe.read_bytes()).hexdigest() != manifest["draft_a16"]["preparation_sha256"]:
        raise ValueError("Draft preparation checksum mismatch")
    ast.parse(recipe.read_bytes(), filename=str(recipe))
    # These are reference bytes, not a promised hash for a recompiled extension or
    # re-serialized torch artifact. Check a copied measured artifact only on request.
    if draft_a16 is not None:
        with Path(draft_a16).open("rb") as handle:
            actual = hashlib.file_digest(handle, "sha256").hexdigest()
        if actual != manifest["draft_a16"]["measured_sha256"]:
            raise ValueError("Draft A16 differs from the measured artifact")
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
    parser.add_argument("--draft-a16", type=Path, help="Also check a copied measured artifact's exact file hash")
    args = parser.parse_args()
    verify(args.installed, args.draft_a16)
