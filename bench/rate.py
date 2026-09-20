#!/usr/bin/env python3
"""Legacy greedy degeneration probe. Uses a nonce to defeat prefix reuse.
This heuristic is not a comprehensive correctness test or sampled benchmark.
The earlier /flush_cache corruption hypothesis was not reproduced.
"""
import argparse, json, sys, time, urllib.request

PROMPTS = {
    "think_long": (True, "Design a fault-tolerant distributed job scheduler. Analyze competing consistency "
                         "models, failure modes, leases, idempotency, fairness, and recovery, then recommend "
                         "a design with explicit tradeoffs."),
    "plain_long": (False, "Write a detailed technical explanation of how a modern MoE transformer serves a "
                          "single decode step: routing, expert gather, attention, KV cache reads, and where "
                          "the memory bandwidth goes. Be thorough and specific."),
    "code_long": (False, "Write a complete production-quality TypeScript implementation of a bounded "
                         "asynchronous task queue with backpressure, cancellation, graceful shutdown, typed "
                         "errors, and deterministic tests. Output code only."),
}


def stream(base, payload, timeout=1800):
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    c, r, n = [], [], 0
    t0 = time.perf_counter(); first = None
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data: "):
                continue
            d = line[6:]
            if d == "[DONE]":
                break
            ch = json.loads(d)
            if ch.get("usage"):
                n = ch["usage"].get("completion_tokens", 0)
            cs = ch.get("choices") or []
            if not cs:
                continue
            de = cs[0].get("delta") or {}
            for key, sink in (("content", c), ("reasoning_content", r)):
                if de.get(key):
                    if first is None:
                        first = time.perf_counter()
                    sink.append(de[key])
    t1 = time.perf_counter()
    dec = max(t1 - (first or t0), 1e-9)
    return "".join(r) + "|" + "".join(c), n, round(max(n - 1, 0) / dec, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=30010)
    ap.add_argument("--model", default="qwen3.8-flash-next")
    ap.add_argument("--iters", type=int, default=12)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--cases", default="think_long,plain_long,code_long")
    ap.add_argument("--tag", default="rate")
    args = ap.parse_args()
    base = f"http://127.0.0.1:{args.port}"
    cases = [c for c in args.cases.split(",") if c]

    bad, total, per_case, speeds = 0, 0, {}, []
    for it in range(args.iters):
        for name in cases:
            think, prompt = PROMPTS[name]
            payload = {"model": args.model,
                       "messages": [{"role": "user", "content": "[run %d-%d]\n%s" % (it, hash(name) % 997, prompt)}],
                       "max_tokens": args.max_tokens, "stream": True, "temperature": 0.0,
                       "stream_options": {"include_usage": True},
                       "chat_template_kwargs": {"enable_thinking": think, "preserve_thinking": think}}
            txt, n, tps = stream(base, payload)
            total += 1
            speeds.append(tps)
            if "!!!!" in txt:
                bad += 1
                per_case[name] = per_case.get(name, 0) + 1
                print("  DEGEN it=%d case=%s at_char=%d tokens=%d" % (it, name, txt.find("!!!!"), n), flush=True)
    speeds.sort()
    print(json.dumps({"tag": args.tag, "total": total, "degenerate": bad,
                      "rate": round(bad / max(total, 1), 4), "per_case": per_case,
                      "median_decode_tok_s": speeds[len(speeds) // 2] if speeds else 0}), flush=True)


if __name__ == "__main__":
    main()
