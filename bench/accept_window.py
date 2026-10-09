"""Mean accept length and steps/s from an SGLang server log over a UTC window.

usage: accept_window.py SERVER_LOG HH:MM:SS HH:MM:SS  (for example: docker logs qwen38-flash-next-sglang > server.log)
"""
import re, sys, statistics
log, lo, hi = sys.argv[1:4]
acc, thr = [], []
for line in open(log, errors="ignore"):
    t = re.match(r"\[\d{4}-\d\d-\d\d (\d\d:\d\d:\d\d)\]", line)
    if not t or not (lo <= t.group(1) <= hi):
        continue
    m = re.search(r"#running-req: 1,.*accept len: ([\d.]+).*gen throughput \(token/s\): ([\d.]+)", line)
    if m:
        acc.append(float(m.group(1))); thr.append(float(m.group(2)))
print(f"{log} [{lo}-{hi}]: n={len(acc)} mean accept len {statistics.mean(acc):.3f}  mean logged thr {statistics.mean(thr):.1f}  thr/acc (steps/s) {statistics.mean(thr)/statistics.mean(acc):.1f}")
