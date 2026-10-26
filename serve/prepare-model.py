#!/usr/bin/env python3
"""Build the target and draft views of the pinned RedHatAI checkpoint.

Each view is a directory of symlinks to the downloaded snapshot plus its own
config.json and index. No weight file is copied or modified.

- target: SGLang 0.5.21 names the QSA blocks full_attention; the BF16 PLE table is
  kept; the quantization scope is expressed as "routed experts only" (Qwen4's
  runtime flattens several module prefixes that the checkpoint's ignore list names).
- draft: the native MTP tensors only, unquantized config (FP8 is applied online).
"""

import argparse
import json
import os
from pathlib import Path

REPO = "RedHatAI/Qwen3.8-Flash-Next-NVFP4"
REVISION = "c8f2fb1b9869f686b214782036123b10ff96d14a"
DRAFT_TOTAL_SIZE = 5214301696


def snapshot_dir(hf_cache: Path) -> Path:
    name = "models--" + REPO.replace("/", "--")
    return hf_cache / "hub" / name / "snapshots" / REVISION


def write_view(source: Path, dest: Path, kind: str) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for path in source.iterdir():
        # Snapshot entries are symlinks that may only resolve inside the container.
        if path.is_dir() or path.name in ("config.json", "model.safetensors.index.json"):
            continue
        link = dest / path.name
        if not link.is_symlink():
            link.symlink_to(path)
    config = json.loads((source / "config.json").read_text())
    config["text_config"]["layer_types"] = [
        "full_attention" if t == "qwen_sparse_attention" else t
        for t in config["text_config"]["layer_types"]
    ]
    index = json.loads((source / "model.safetensors.index.json").read_text())
    if kind == "target":
        config["vega_redhat_streamed_load"] = True
        config["text_config"]["ple_embedding_dtype"] = "bfloat16"
        config["quantization_config"]["ignore"].append(r"re:^(?!.*\.mlp\.experts(?:\.|$)).*$")
    else:
        config.pop("quantization_config", None)
        config["text_config"].pop("quantization_config", None)
        index["weight_map"] = {n: f for n, f in index["weight_map"].items() if "mtp" in n}
        index["metadata"]["total_size"] = DRAFT_TOTAL_SIZE
    (dest / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (dest / "model.safetensors.index.json").write_text(json.dumps(index, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-cache", default=os.environ.get("HF_CACHE", str(Path.home() / "models/huggingface")))
    parser.add_argument("--views", default=os.environ.get("MODEL_VIEWS", str(Path.home() / "models/qwen38-flash-next-redhat")))
    args = parser.parse_args()
    source = snapshot_dir(Path(args.hf_cache).resolve())
    if not (source / "model.safetensors.index.json").is_file():
        raise SystemExit(f"Pinned snapshot missing at {source}. Run ./serve/download-model.sh first.")
    views = Path(args.views).resolve() / REVISION
    for kind in ("target", "draft"):
        write_view(source, views / kind, kind)
    print(f"Prepared {views}/target and {views}/draft from {source}")


if __name__ == "__main__":
    main()
