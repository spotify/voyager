"""Regression tests: python3 -m unittest discover -s java/scripts -v."""

import json
import subprocess
import sys
import tempfile
import unittest
from itertools import product
from pathlib import Path

SCRIPT = Path(__file__).with_name("compare_benchmarks.py")


def result(benchmark, score, unit="ms/op", error=0.01, params=None):
    return {
        "benchmark": f"com.spotify.voyager.jni.{benchmark}",
        "params": params or {},
        "primaryMetric": {"score": score, "scoreError": error, "scoreUnit": unit},
    }


class CompareBenchmarksTest(unittest.TestCase):
    def report(self, base, pr):
        with tempfile.TemporaryDirectory() as directory:
            base_path = Path(directory) / "base.json"
            pr_path = Path(directory) / "pr.json"
            if base is not None:
                base_path.write_text(json.dumps(base))
            pr_path.write_text(json.dumps(pr))
            return subprocess.run(
                [sys.executable, str(SCRIPT), str(base_path), str(pr_path)],
                check=True,
                capture_output=True,
                text=True,
            ).stdout

    def test_index_regression_is_not_hidden_by_string_index(self):
        base = [
            result("IndexQueryBenchmark.queryK1", 10),
            result("StringIndexQueryBenchmark.queryK1", 20),
        ]
        pr = [
            result("IndexQueryBenchmark.queryK1", 100),
            result("StringIndexQueryBenchmark.queryK1", 20),
        ]
        report = self.report(base, pr)
        index_row = next(row for row in report.splitlines() if row.startswith("| IndexQueryBenchmark."))
        string_row = next(row for row in report.splitlines() if row.startswith("| StringIndexQueryBenchmark."))
        self.assertIn("+900.0%", index_row)
        self.assertIn("slower", index_row)
        self.assertIn("unchanged", string_row)

    def test_equal_timings_in_different_units(self):
        report = self.report(
            [result("IndexQueryBenchmark.queryK1", 1)],
            [result("IndexQueryBenchmark.queryK1", 1000, "us/op", error=10)],
        )
        self.assertIn("+0.0%", report)
        self.assertIn("unchanged", report)
        self.assertEqual(report.count("1.000 ± 0.010 ms/op"), 2)

    def test_error_margins_are_scaled_with_scores(self):
        report = self.report(
            [result("IndexQueryBenchmark.queryK1", 1, error=0.01)],
            [result("IndexQueryBenchmark.queryK1", 1100, "us/op", error=1)],
        )
        self.assertIn("+10.0%", report)
        self.assertIn("slower", report)
        self.assertIn("1.100 ± 0.001 ms/op", report)

    def test_unsupported_units_are_not_compared(self):
        for unit in ("ops/s", "widgets/op"):
            with self.subTest(unit=unit):
                report = self.report(
                    [result("IndexQueryBenchmark.queryK1", 1)],
                    [result("IndexQueryBenchmark.queryK1", 1000, unit)],
                )
                self.assertIn("not comparable", report)
                self.assertIn(unit, report)
                self.assertNotIn("%", report)
                self.assertNotIn("faster", report)
                self.assertNotIn("slower", report)

    def test_parameter_order_does_not_change_identity(self):
        report = self.report(
            [result("IndexQueryBenchmark.queryK1", 1, params={"spaceType": "Cosine", "numElements": "4096"})],
            [result("IndexQueryBenchmark.queryK1", 1, params={"numElements": "4096", "spaceType": "Cosine"})],
        )
        self.assertIn("unchanged", report)
        self.assertNotIn("_new_", report)
        self.assertNotIn("_removed_", report)

    def test_all_54_benchmarks_are_reported(self):
        results = []
        methods = (
            "IndexCreationBenchmark.addItems",
            "StringIndexCreationBenchmark.addItems",
            "IndexQueryBenchmark.queryK1",
            "IndexQueryBenchmark.queryK20",
            "StringIndexQueryBenchmark.queryK1",
            "StringIndexQueryBenchmark.queryK20",
        )
        for method, space, storage in product(
            methods, ("Euclidean", "InnerProduct", "Cosine"), ("Float32", "Float8", "E4M3")
        ):
            params = {
                "numDimensions": "256",
                "numElements": "1024" if "Creation" in method else "4096",
                "spaceType": space,
                "storageDataType": storage,
                "efConstruction": "24",
            }
            results.append(result(method, 1, params=params))
        for base in (None, results):
            with self.subTest(first_release=base is None):
                rows = [row for row in self.report(base, results).splitlines() if "Benchmark." in row]
                self.assertEqual(len(rows), 54)
                self.assertEqual(len(set(rows)), 54)

    def test_new_and_removed_benchmarks_keep_their_names(self):
        report = self.report(
            [result("IndexQueryBenchmark.queryK1", 1)],
            [result("StringIndexQueryBenchmark.queryK1", 1)],
        )
        self.assertIn("| IndexQueryBenchmark.queryK1 | 1.000 ± 0.010 ms/op | _removed_", report)
        self.assertIn("| StringIndexQueryBenchmark.queryK1 | _new_ | 1.000 ± 0.010 ms/op", report)

    def test_summary_lists_regressions_before_improvements_below_table(self):
        benchmarks = (
            "IndexQueryBenchmark.queryK1",
            "IndexQueryBenchmark.queryK20",
            "IndexCreationBenchmark.addItems",
            "StringIndexCreationBenchmark.addItems",
        )
        base = [result(name, 10) for name in benchmarks]
        scores = (12, 20, 5, 10)
        pr = [result(name, scores[i]) for i, name in enumerate(benchmarks)]
        report = self.report(base, pr)
        summary = report.split("\n\n**Summary:**", 1)[1]
        self.assertGreater(report.index("**Summary:**"), report.rindex("|"))
        self.assertIn("Regressions: **2**; improvements: **1**; unchanged: **1**.", summary)
        largest = "- **Regression:** `IndexQueryBenchmark.queryK20` (+100.0%)."
        smaller = "- **Regression:** `IndexQueryBenchmark.queryK1` (+20.0%)."
        improvement = "- **Improvement:** `IndexCreationBenchmark.addItems` (-50.0%)."
        self.assertLess(summary.index(largest), summary.index(smaller))
        self.assertLess(summary.index(smaller), summary.index(improvement))

    def test_summary_uses_measured_uncertainty_for_unchanged_results(self):
        report = self.report(
            [result("IndexQueryBenchmark.queryK1", 1)],
            [result("IndexQueryBenchmark.queryK1", 1.01)],
        )
        summary = report.split("\n\n**Summary:**", 1)[1]
        self.assertIn("All comparable benchmarks are unchanged within measured uncertainty (1 compared).", summary)
        self.assertNotIn("**Regression:**", summary)
        self.assertNotIn("**Improvement:**", summary)

    def test_summary_separates_non_comparable_and_missing_results(self):
        report = self.report(
            [result("IndexQueryBenchmark.queryK1", 1), result("IndexQueryBenchmark.queryK20", 1)],
            [result("IndexQueryBenchmark.queryK1", 1, unit="ops/s"), result("IndexCreationBenchmark.addItems", 1)],
        )
        summary = report.split("\n\n**Summary:**", 1)[1]
        self.assertIn("No comparable benchmarks.", summary)
        self.assertIn("New: 1; removed: 1; not comparable: 1.", summary)
        self.assertNotIn("unchanged", summary)

    def test_missing_baseline_summary_is_below_table(self):
        report = self.report(None, [result("IndexQueryBenchmark.queryK1", 1)])
        self.assertGreater(report.index("**Summary:**"), report.rindex("|"))
        self.assertIn("Baseline unavailable; benchmark results reported: **1**. No before/after comparison.", report)


if __name__ == "__main__":
    unittest.main()
