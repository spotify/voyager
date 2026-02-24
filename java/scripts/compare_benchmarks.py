#!/usr/bin/env python3
"""Compare two JMH JSON benchmark result files and output a markdown summary.

Usage:
    python3 compare_benchmarks.py <base_results.json> <pr_results.json>

If the base results file does not exist (e.g. when benchmarks are first added),
only the PR results are printed.

Uses only Python standard library (json, sys, os).
"""

import json
import os
import sys


def load_results(path):
    """Load JMH JSON results and return a dict keyed by benchmark name + params."""
    with open(path) as f:
        data = json.load(f)

    results = {}
    for entry in data:
        benchmark = entry["benchmark"]
        # Extract short method name from fully qualified name
        short_name = benchmark.rsplit(".", 1)[-1]

        params = entry.get("params", {})
        param_key = ", ".join(f"{k}={v}" for k, v in sorted(params.items()))

        key = f"{short_name}({param_key})" if param_key else short_name

        score = entry["primaryMetric"]["score"]
        error = entry["primaryMetric"]["scoreError"]
        unit = entry["primaryMetric"]["scoreUnit"]

        results[key] = {"score": score, "error": error, "unit": unit}

    return results


def format_score(score, error):
    """Format a score with error margin."""
    return f"{score:.3f} \u00b1 {error:.3f}"


def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <base_results.json> <pr_results.json>", file=sys.stderr)
        sys.exit(1)

    base_path = sys.argv[1]
    pr_path = sys.argv[2]

    if not os.path.exists(pr_path):
        print("Error: PR results file not found.", file=sys.stderr)
        sys.exit(1)

    pr_results = load_results(pr_path)

    if not os.path.exists(base_path):
        # Base results don't exist yet (first PR adding benchmarks)
        print("## Java Benchmark Results\n")
        print("_No base branch results available for comparison._\n")
        print("| Benchmark | Score | Unit |")
        print("|-----------|-------|------|")
        for name in sorted(pr_results.keys()):
            r = pr_results[name]
            print(f"| {name} | {format_score(r['score'], r['error'])} | {r['unit']} |")
        return

    base_results = load_results(base_path)

    print("## Java Benchmark Comparison\n")
    print("| Benchmark | Base | PR | Delta | Status |")
    print("|-----------|------|-----|-------|--------|")

    all_keys = sorted(set(list(base_results.keys()) + list(pr_results.keys())))

    for name in all_keys:
        if name not in base_results:
            r = pr_results[name]
            print(
                f"| {name} | _new_ | {format_score(r['score'], r['error'])} {r['unit']}"
                f" | - | \U0001f195 |"
            )
            continue

        if name not in pr_results:
            r = base_results[name]
            print(
                f"| {name} | {format_score(r['score'], r['error'])} {r['unit']}"
                f" | _removed_ | - | - |"
            )
            continue

        base = base_results[name]
        pr = pr_results[name]

        if base["score"] == 0:
            delta_pct = 0.0
        else:
            delta_pct = ((pr["score"] - base["score"]) / base["score"]) * 100

        # Determine if the change is significant by comparing against combined error margins
        combined_error = base["error"] + pr["error"]
        abs_diff = abs(pr["score"] - base["score"])

        if abs_diff > combined_error:
            # For time-based benchmarks, lower is better
            if pr["score"] < base["score"]:
                status = "\u2705 faster"
            else:
                status = "\u26a0\ufe0f slower"
        else:
            status = "\u2194\ufe0f unchanged"

        print(
            f"| {name}"
            f" | {format_score(base['score'], base['error'])}"
            f" | {format_score(pr['score'], pr['error'])}"
            f" | {delta_pct:+.1f}%"
            f" | {status} |"
        )


if __name__ == "__main__":
    main()
