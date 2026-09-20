import json, random
from pathlib import Path
import requests
from sustained import request, payload


def run(args):
    R = Path(args.outdir)
    R.mkdir(parents=True, exist_ok=True)
    base = f"http://127.0.0.1:{args.port}"
    tag = args.tag
    out = []
    rng = random.Random(713849)
    words = "network service database table worker task cache request response latency timeout capacity memory owner project billing storage".split()
    for size in [8192, 120000, 240000]:
        doc = rng.choices(words, k=size)
        expected = {"owner": "ZEPHYR-7381", "budget": "46129", "region": "EAST-704"}
        for pos, (key, value) in zip(
            [size // 30, size // 2, size * 29 // 30], expected.items()
        ):
            doc[pos] = f"\nAUTHORITATIVE_LEDGER {key}={value}\n"
        prompt = (
            "Read the following noisy imported operations ledger. Extract the three AUTHORITATIVE_LEDGER values exactly.\n"
            + " ".join(doc)
            + "\nReturn a JSON object with just owner, budget, and region; use string values. Do not infer values from other words."
        )
        if args.flush_cache:
            requests.post(base + "/flush_cache", timeout=30).raise_for_status()
        p = payload("qwen3.8-flash-next", prompt, 8192, "xhigh", 713849 + size)
        row = request(base, p)
        row.update(size_words=size, expected=expected)
        try:
            text = (
                row["content"]
                .strip()
                .removeprefix("```json")
                .removeprefix("```")
                .removesuffix("```")
                .strip()
            )
            row["semantic_pass"] = json.loads(text) == expected
        except Exception:
            row["semantic_pass"] = False
        out.append(row)
        (R / (tag + "-context.json")).write_text(json.dumps(out, indent=2))
        print(
            json.dumps(
                {k: v for k, v in row.items() if k not in ("content", "reasoning")}
            ),
            flush=True,
        )
        assert row["semantic_pass"], "Long-context extraction failed"


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Reproduce the September retrieval checks; output rates are not sustained benchmarks."
    )
    parser.add_argument("tag")
    parser.add_argument("--port", type=int, default=30010)
    parser.add_argument(
        "--outdir", default=str(Path(__file__).resolve().parents[1] / "results/local")
    )
    parser.add_argument("--flush-cache", action="store_true")
    run(parser.parse_args())
