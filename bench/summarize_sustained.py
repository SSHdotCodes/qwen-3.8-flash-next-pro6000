#!/usr/bin/env python3
"""Recompute the published September comparison from per-request evidence."""

import argparse
import json
import math
import random
import statistics
from pathlib import Path


def summarize(root):
    def read(name):
        return json.loads((root / f"{name}.json").read_text())

    control = read("baseline") + read("control-final")
    candidate = read("hotmap-results") + read("hotmap-chinese") + read("selected-final")
    for row in control + candidate:
        if row["completion_tokens"] != 8192 or row["collapse"]:
            raise ValueError("Comparison requires non-collapsed 8,192-token records")
    cells = []
    for profile in ("code", "agent", "prose", "chinese"):
        a = [r["decode_tok_s"] for r in control if r["profile"] == profile]
        b = [r["decode_tok_s"] for r in candidate if r["profile"] == profile]
        cells.append(
            dict(
                profile=profile,
                control_mean=statistics.mean(a),
                candidate_mean=statistics.mean(b),
                control_n=len(a),
                candidate_n=len(b),
                control_values=a,
                candidate_values=b,
                gain_percent=100 * (statistics.mean(b) / statistics.mean(a) - 1),
            )
        )
    gain = math.exp(
        statistics.mean(
            math.log(c["candidate_mean"] / c["control_mean"]) for c in cells[:3]
        )
    )
    rng = random.Random(7193)
    bootstrap = []
    for _ in range(5000):
        ratios = []
        for cell in cells[:3]:
            a, b = cell["control_values"], cell["candidate_values"]
            av = statistics.mean(rng.choices(a, k=len(a)))
            bv = statistics.mean(rng.choices(b, k=len(b)))
            ratios.append(math.log(bv / av))
        bootstrap.append(100 * (math.exp(statistics.mean(ratios)) - 1))
    bootstrap.sort()
    return dict(
        cells=cells,
        geomean_gain_percent=100 * (gain - 1),
        bootstrap_interval_percent=[bootstrap[125], bootstrap[4874]],
        note="Exploratory interval for these fixed English workloads only.",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    print(json.dumps(summarize(parser.parse_args().directory), indent=2))
