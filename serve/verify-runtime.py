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


def public_differences(manifest):
    differences = []
    for entry in manifest["files"]:
        has_reference = "installed_sha256" in entry
        if has_reference != bool(entry.get("public_difference")):
            raise ValueError(f"Incomplete public/installed difference: {entry['path']}")
        if has_reference:
            if (not re.fullmatch(r"[0-9a-f]{64}", entry["installed_sha256"])
                    or entry["installed_sha256"] == entry["sha256"]):
                raise ValueError(f"Invalid installed reference checksum: {entry['path']}")
            differences.append(entry)
    return differences


def compare_installed_files(manifest, measured):
    """Compare public pins with a saved source manifest without normalizing away differences."""
    public_differences(manifest)
    measured_hashes = {entry["mount"]: entry["sha256"] for entry in measured["files"].values()}
    for entry in manifest["files"]:
        expected = entry.get("installed_sha256", entry["sha256"])
        if measured_hashes.get(entry["container_path"]) != expected:
            raise ValueError(f"Installed reference checksum mismatch: {entry['path']}")


def verify(installed=False, draft_a16=None, installed_manifest=None):
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / "runtime.json").read_text())
    differences = public_differences(manifest)
    if installed_manifest is not None:
        data = installed_manifest.read_bytes()
        if hashlib.sha256(data).hexdigest() != manifest["installed_manifest_sha256"]:
            raise ValueError("Measured installed manifest checksum mismatch")
        compare_installed_files(manifest, json.loads(data))
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
    for entry in differences:
        print(f"Intentional public/installed difference: {entry['path']}: {entry['public_difference']}")
    if installed_manifest is not None:
        print(f"Compared measured installed manifest: {len(manifest['files']) - len(differences)}"
              f" identical file hashes; {len(differences)} intentional differences with pinned installed hashes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--installed", action="store_true")
    parser.add_argument("--draft-a16", type=Path, help="Also check a copied measured artifact's exact file hash")
    parser.add_argument("--installed-manifest", type=Path,
                        help="Also compare against a saved measured source manifest, reporting intentional differences")
    args = parser.parse_args()
    verify(args.installed, args.draft_a16, args.installed_manifest)
