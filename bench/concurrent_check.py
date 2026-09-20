import concurrent.futures, json
from pathlib import Path
import requests
from sustained import payload, request


def run(args):
    R = Path(args.outdir)
    R.mkdir(parents=True, exist_ok=True)
    base = f"http://127.0.0.1:{args.port}"
    if args.flush_cache:
        requests.post(base + "/flush_cache", timeout=30).raise_for_status()

    def check(i):
        expected = f"ACCOUNT-{9173 + i * 104}"
        prompt = (
            "Operations record. " * 8000
        ) + f"\nThe verified account code is {expected}. Return only that code."
        r = request(base, payload("qwen3.8-flash-next", prompt, 2048, "xhigh", 504 + i))
        r.update(
            expected=expected, semantic_pass=r["content"].strip().strip("`") == expected
        )
        return r

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        out = list(pool.map(check, range(2)))
    (R / (args.tag + "-concurrent.json")).write_text(json.dumps(out, indent=2))
    print(
        json.dumps(
            [
                {k: v for k, v in x.items() if k not in ("content", "reasoning")}
                for x in out
            ]
        ),
        flush=True,
    )
    assert all(x["semantic_pass"] for x in out)


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
