#!/usr/bin/env python3
"""Sustained xhigh benchmark; sampled output includes reasoning, excludes TTFT."""

import argparse, json, time, hashlib, random
from pathlib import Path
import requests

A = Path(__file__).resolve().parents[1] / "results/local"
P = {
    "code": "Write a complete Python module implementing a thread-safe LRU cache with TTL expiry, size accounting, metrics, full type hints and unit tests.",
    "agent": "Design and implement a durable Python task queue backed by SQLite. Include task claiming, lease expiration, retry backoff, idempotency and concurrent worker safety. Explain correctness and provide the complete module.",
    "prose": "Explain speculative decoding in detail: draft generation, exact rejection sampling, target verification, KV state rollback and the memory bandwidth argument. Derive the output-distribution guarantee and discuss its limits.",
    "chinese": "请详细解释数据库事务的隔离级别，并用Python和SQL示例说明如何防止丢失更新、幻读和写偏差。比较乐观锁和悲观锁，给出完整的实现和测试。",
}


def request(base, payload):
    t = time.perf_counter()
    first = None
    usage = {}
    content = ""
    reason = ""
    finish = None
    chunks = 0
    with requests.post(
        base + "/v1/chat/completions", json=payload, stream=True, timeout=(20, 1800)
    ) as r:
        r.raise_for_status()
        for raw in r.iter_lines():
            if not raw.startswith(b"data:"):
                continue
            data = raw[5:].strip()
            if data == b"[DONE]":
                break
            d = json.loads(data)
            if d.get("error"):
                raise RuntimeError(d["error"])
            if d.get("usage"):
                usage = d["usage"]
            for ch in d.get("choices", []):
                v = ch.get("delta", {})
                s = v.get("content") or ""
                z = v.get("reasoning_content") or v.get("reasoning") or ""
                if s or z:
                    if first is None:
                        first = time.perf_counter()
                    content += s
                    reason += z
                    chunks += 1
                finish = ch.get("finish_reason") or finish
    end = time.perf_counter()
    assert first is not None and usage, (content, reason, usage)
    n = usage["completion_tokens"]
    full = content + reason
    bad = not full.strip() or (
        full.count("!") > 32 and full.count("!") / len(full) > 0.4
    )
    return dict(
        prompt_tokens=usage["prompt_tokens"],
        completion_tokens=n,
        ttft_s=first - t,
        elapsed_s=end - t,
        decode_tok_s=(n - 1) / (end - first) if n > 1 else None,
        prefill_tok_s=usage["prompt_tokens"] / (first - t),
        content=content,
        reasoning=reason,
        finish_reason=finish,
        collapse=bad,
        chunks=chunks,
    )


def payload(model, prompt, limit, effort, seed):
    return dict(
        model=model,
        messages=[dict(role="user", content=prompt)],
        max_tokens=limit,
        stream=True,
        stream_options=dict(include_usage=True),
        temperature=1.0,
        top_p=0.95,
        top_k=20,
        min_p=0.0,
        presence_penalty=0.0,
        repetition_penalty=1.0,
        reasoning_effort=effort,
        chat_template_kwargs=dict(enable_thinking=True, preserve_thinking=True),
        seed=seed,
    )


def run(args):
    base = f"http://127.0.0.1:{args.port}"
    out = []
    A = Path(args.outdir)
    A.mkdir(parents=True, exist_ok=True)
    dest = A / (args.tag + ".json")
    (A / (args.tag + "-settings.json")).write_text(json.dumps(vars(args), indent=2))
    info = requests.get(base + "/server_info", timeout=30)
    info.raise_for_status()
    (A / (args.tag + "-server-info.json")).write_text(json.dumps(info.json(), indent=2))
    for i in range(2):
        request(base, payload(args.model, "Reply with READY.", 32, "low", 130 + i))
    jobs = []
    if args.prefill:
        rng = random.Random(115)
        words = "the quick brown fox jumps over lazy dog while seven engineers debate whether cache coherence matters more than latency in distributed systems built from commodity hardware and open protocols".split()
        for size in map(int, args.prefill.split(",")):
            doc = " ".join(rng.choice(words) for _ in range(size))

            if args.flush_cache:
                requests.post(base + "/flush_cache", timeout=30).raise_for_status()
            request(base, payload(args.model, doc + "\nReply ACK.", 16, "low", 194))
            for i in range(args.runs):
                jobs.append(
                    (
                        f"prefill{size}",
                        i,
                        "low",
                        f"Document {i}.\n{doc}\nReply with the single word ACK.",
                        16,
                    )
                )
    for effort in filter(None, args.efforts.split(",")):
        for name in filter(None, args.profiles.split(",")):
            for i in range(args.runs):
                jobs.append((name, i, effort, P[name], args.tokens))
    for name, i, effort, prompt, limit in jobs:
        if args.flush_cache:
            requests.post(base + "/flush_cache", timeout=30).raise_for_status()
        row = request(base, payload(args.model, prompt, limit, effort, 20260911 + i))
        row.update(
            profile=name,
            run=i,
            effort=effort,
            prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
        )
        out.append(row)
        dest.write_text(json.dumps(out, indent=2, ensure_ascii=False))
        print(
            json.dumps(
                {
                    k: v
                    for k, v in row.items()
                    if k not in ("content", "reasoning", "prompt_sha256")
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        assert not row["collapse"], "Degenerate output"
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("tag")
    p.add_argument("--port", type=int, default=30010)
    p.add_argument("--model", default="qwen3.8-flash-next")
    p.add_argument("--runs", type=int, default=2)
    p.add_argument("--tokens", type=int, default=8192)
    p.add_argument("--profiles", default="code,agent,prose")
    p.add_argument("--efforts", default="xhigh")
    p.add_argument("--prefill", default="")
    p.add_argument("--outdir", default=str(A))
    p.add_argument(
        "--flush-cache",
        action="store_true",
        help="Clear prefix cache before requests; use only on an idle dedicated server",
    )
    run(p.parse_args())
