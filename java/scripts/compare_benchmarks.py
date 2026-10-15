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

TIME_UNITS_TO_MS = {
    "ns/op": 0.000001,
    "us/op": 0.001,
    "ms/op": 1,
    "s/op": 1000,
    "min/op": 60000,
    "hr/op": 3600000,
    "day/op": 86400000,
}


def load_results(path):
    """Load JMH JSON results and return a dict keyed by benchmark name + params."""
    with open(path) as f:
        data = json.load(f)

    results = {}
    for entry in data:
        benchmark = entry["benchmark"]
        params = entry.get("params", {})
        key = (benchmark, tuple(sorted(params.items())))

        score = entry["primaryMetric"]["score"]
        error = entry["primaryMetric"]["scoreError"]
        unit = entry["primaryMetric"]["scoreUnit"]

        # Normalize time scores and their uncertainty before comparing results.
        if unit in TIME_UNITS_TO_MS:
            factor = TIME_UNITS_TO_MS[unit]
            score *= factor
            error *= factor
            unit = "ms/op"

        results[key] = {"score": score, "error": error, "unit": unit}

    return results


def format_name(key):
    """Keep class and method names in labels while retaining full identity in keys."""
    benchmark, params = key
    name = ".".join(benchmark.split(".")[-2:])
    param_label = ", ".join(f"{k}={v}" for k, v in params)
    return f"{name}({param_label})" if params else name


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
        print("| Benchmark | Score | Unit |")
        print("|-----------|-------|------|")
        for key in sorted(pr_results.keys()):
            name = format_name(key)
            r = pr_results[key]
            print(f"| {name} | {format_score(r['score'], r['error'])} | {r['unit']} |")
        print(
            f"\n**Summary:** Baseline unavailable; benchmark results reported: **{len(pr_results)}**. No before/after comparison."
        )
        return

    base_results = load_results(base_path)

    print("## Java Benchmark Comparison\n")
    print("| Benchmark | Base | PR | Delta | Status |")
    print("|-----------|------|-----|-------|--------|")

    all_keys = sorted(set(list(base_results.keys()) + list(pr_results.keys())))
    regressions = []
    improvements = []
    unchanged = new = removed = not_comparable = 0

    for key in all_keys:
        name = format_name(key)
        if key not in base_results:
            new += 1
            r = pr_results[key]
            print(f"| {name} | _new_ | {format_score(r['score'], r['error'])} {r['unit']} | - | \U0001f195 |")
            continue

        if key not in pr_results:
            removed += 1
            r = base_results[key]
            print(f"| {name} | {format_score(r['score'], r['error'])} {r['unit']} | _removed_ | - | - |")
            continue

        base = base_results[key]
        pr = pr_results[key]

        if base["unit"] != "ms/op" or pr["unit"] != "ms/op":
            not_comparable += 1
            print(
                f"| {name}"
                f" | {format_score(base['score'], base['error'])} {base['unit']}"
                f" | {format_score(pr['score'], pr['error'])} {pr['unit']}"
                f" | - | not comparable |"
            )
            continue

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
                improvements.append((delta_pct, name))
            else:
                status = "\u26a0\ufe0f slower"
                regressions.append((delta_pct, name))
        else:
            status = "\u2194\ufe0f unchanged"
            unchanged += 1

        print(
            f"| {name}"
            f" | {format_score(base['score'], base['error'])} {base['unit']}"
            f" | {format_score(pr['score'], pr['error'])} {pr['unit']}"
            f" | {delta_pct:+.1f}%"
            f" | {status} |"
        )

    print()
    if regressions or improvements:
        print(
            f"**Summary:** Regressions: **{len(regressions)}**; "
            f"improvements: **{len(improvements)}**; unchanged: **{unchanged}**."
        )
    elif unchanged:
        print(
            f"**Summary:** All comparable benchmarks are unchanged within measured uncertainty ({unchanged} compared)."
        )
    else:
        print("**Summary:** No comparable benchmarks.")

    if new or removed or not_comparable:
        print(f"\nNew: {new}; removed: {removed}; not comparable: {not_comparable}.")

    changes = [("Regression", delta, name) for delta, name in sorted(regressions, reverse=True)] + [
        ("Improvement", delta, name) for delta, name in sorted(improvements)
    ]
    if changes:
        print()
        for label, delta, name in changes:
            print(f"- **{label}:** `{name}` ({delta:+.1f}%).")


if __name__ == "__main__":
    main()
