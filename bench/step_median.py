"""Median verify steps/s from an SGLang server log: per log line, gen throughput / accept length.

usage: step_median.py SERVER_LOG [HH:MM:SS HH:MM:SS]  (UTC window; for example: docker logs qwen38-flash-next-sglang > server.log)

Used for the October 10 results. Only single-request decode lines are counted, and lines logging 60 tok/s or less
(warm-up, idle tails) are skipped. The per-line median moves less between runs than the ratio of means that
accept_window.py prints, so compare a build only with a baseline measured the same way.
"""
import re, sys, statistics


def step_rates(lines, lo="00:00:00", hi="99:99:99"):
    rates = []
    for line in lines:
        t = re.match(r"\[\d{4}-\d\d-\d\d (\d\d:\d\d:\d\d)\]", line)
        m = re.search(r"#running-req: 1, #full token: \d+,.*accept len: ([\d.]+).*gen throughput \(token/s\): ([\d.]+)", line)
        if t and m and lo <= t.group(1) <= hi and float(m.group(2)) > 60:
            rates.append(float(m.group(2)) / float(m.group(1)))
    return rates


if __name__ == "__main__":
    log = sys.argv[1]
    lo, hi = (sys.argv[2], sys.argv[3]) if len(sys.argv) > 3 else ("00:00:00", "99:99:99")
    s = sorted(step_rates(open(log, errors="ignore"), lo, hi))
    k = len(s) // 10
    print(f"{log} [{lo}-{hi}]: n={len(s)} median steps/s {statistics.median(s):.1f}  trimmed mean {statistics.mean(s[k:len(s) - k]):.1f}")
