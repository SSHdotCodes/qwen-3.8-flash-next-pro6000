#!/usr/bin/env python3
"""Compare a candidate result file against the pinned baseline, token-for-token.

Speculative decoding is only lossless if greedy output is identical. This reports
the longest common prefix (in characters) and whether the full strings match.
"""
import json, os, sys, difflib

def load(tag, d=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")):
    with open(os.path.join(d, tag + ".json")) as f:
        return json.load(f)

def lcp(a, b):
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i

def main():
    base = load(sys.argv[1])
    cand = load(sys.argv[2])
    rows = []
    for name in base["profiles"]:
        if name not in cand["profiles"]:
            continue
        b, c = base["profiles"][name], cand["profiles"][name]
        for field in ("reasoning", "content"):
            bt, ct = b.get(field, ""), c.get(field, "")
            if not bt and not ct:
                continue
            p = lcp(bt, ct)
            ratio = difflib.SequenceMatcher(None, bt, ct, autojunk=False).ratio()
            rows.append({
                "profile": name, "field": field,
                "identical": bt == ct,
                "base_chars": len(bt), "cand_chars": len(ct),
                "common_prefix_chars": p,
                "prefix_frac": round(p / max(len(bt), 1), 4),
                "similarity": round(ratio, 4),
            })
        btc = json.dumps(b.get("tool_calls") or [], sort_keys=True)
        ctc = json.dumps(c.get("tool_calls") or [], sort_keys=True)
        if btc != "[]" or ctc != "[]":
            rows.append({"profile": name, "field": "tool_calls", "identical": btc == ctc,
                         "base_chars": len(btc), "cand_chars": len(ctc),
                         "common_prefix_chars": lcp(btc, ctc),
                         "prefix_frac": round(lcp(btc, ctc) / max(len(btc), 1), 4),
                         "similarity": round(difflib.SequenceMatcher(None, btc, ctc).ratio(), 4)})
    print(f"{'profile':<12} {'field':<11} {'ident':<6} {'prefix%':>8} {'sim':>7} {'base':>7} {'cand':>7}")
    for r in rows:
        print(f"{r['profile']:<12} {r['field']:<11} {str(r['identical']):<6} "
              f"{r['prefix_frac']*100:7.2f}% {r['similarity']:7.4f} {r['base_chars']:7d} {r['cand_chars']:7d}")
    print()
    print("SPEED  profile      baseline   candidate   speedup")
    for name in base["profiles"]:
        if name not in cand["profiles"]:
            continue
        bs = base["profiles"][name]["median_decode_tok_s"]
        cs = cand["profiles"][name]["median_decode_tok_s"]
        print(f"       {name:<12} {bs:8.2f}   {cs:9.2f}   {cs/max(bs,1e-9):6.2f}x")
    all_ident = all(r["identical"] for r in rows)
    print()
    print("VERDICT: " + ("byte-identical to baseline" if all_ident else "DIVERGES from baseline"))

if __name__ == "__main__":
    main()
