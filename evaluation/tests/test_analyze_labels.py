from pathlib import Path
import csv
import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("gtr_human_labels", ROOT / "analyze_labels.py")
labels = importlib.util.module_from_spec(spec)
spec.loader.exec_module(labels)


def row(clip="c1", rater="r1", gt=(1, 1, 1), rating=(1, 1, 1)):
    return {"clip_id": clip, "rater_id": rater,
            **{f"gt_{axis}": value for axis, value in zip("gtr", gt)},
            **{f"rating_{axis}": value for axis, value in zip("gtr", rating)}}


class HumanLabelTests(unittest.TestCase):
    def load(self, rows):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "ratings.json"
            path.write_text(json.dumps(rows))
            return labels.load_annotations(path)[0]

    def test_pearson_formula_constant_and_insufficient(self):
        self.assertAlmostEqual(labels.pearson([(1, 3), (2, 2), (3, 1)])["coefficient"], -1)
        self.assertAlmostEqual(labels.pearson([(1, 1), (2, 2), (3, 3)])["coefficient"], 1)
        for pairs in [[], [(1, 1)], [(1, 1), (1, 2)], [(1, 2), (2, 2)],
                      [(1, .4), (2, .4), (3, .4)]]:
            self.assertIsNone(labels.pearson(pairs)["coefficient"])
            self.assertIsNotNone(labels.pearson(pairs)["undefined_reason"])

    def test_rater_mean_pooled_and_clip_mean_are_not_conflated(self):
        annotations = self.load([row(), row("c2", gt=(2, 2, 2), rating=(2, 2, 2)),
                                 row(rater="r2", rating=(4, 5, 2))])
        result, clips = labels.analyze(annotations, expected_clips=50, expected_raters=3)
        self.assertEqual(result["coverage"]["annotation_count"], 3)
        self.assertEqual(result["coverage"]["missing_annotations_within_observed_clip_rater_matrix"], 1)
        self.assertFalse(result["coverage"]["expected_counts"]["clips"]["matches"])
        self.assertEqual(result["per_rater"]["r2"]["missing_clip_ids_from_observed_set"], ["c2"])
        macro = result["unweighted_mean_of_defined_rater_metrics"]
        self.assertAlmostEqual(macro["G"]["mean"], 1)
        self.assertEqual(macro["G"]["undefined_rater_ids"], ["r2"])
        self.assertAlmostEqual(macro["R"]["mean"], .5)
        self.assertAlmostEqual(result["annotation_pooled_metrics"]["R"]["accuracy"], 2 / 3)
        self.assertAlmostEqual(result["pearson_after_unweighted_clip_mean_ratings"]["G"]["coefficient"], -1)
        self.assertEqual(clips[0]["mean_rating_G"], 2.5)
        self.assertNotIn("mean_rating_R", clips[0])
        self.assertFalse(result["paper_results_reproduced"])

    def test_partial_ratings_and_independent_rater_axes_are_preserved(self):
        annotations = self.load([row(rating=(0, None, 3)), row("c2", rating=(None, 2, None))])
        result, _ = labels.analyze(annotations)
        self.assertEqual(result["coverage"]["annotation_count"], 2)
        self.assertEqual(result["coverage"]["rater_cross_axis_inconsistent_annotations_retained"], 1)
        self.assertEqual(result["coverage"]["missing_ratings_by_axis"], {"G": 1, "T": 1, "R": 1})
        self.assertEqual(result["per_rater"]["r1"]["metrics"]["R"]["paired_count"], 1)
        all_missing, _ = labels.analyze(self.load([row(rating=(None, None, None))]))
        self.assertIsNone(all_missing["annotation_pooled_metrics"]["R"]["accuracy"])
        self.assertIsNone(all_missing["unweighted_mean_of_defined_rater_metrics"]["G"]["mean"])

    def test_reject_duplicate_ambiguous_gt_and_mixed_scales(self):
        invalid = [[row(), row()],
                   [row(), row(rater="r2", gt=(2, 2, 2))],
                   [row(gt=(0, 1, 1))], [row(gt=(1, 1, 7))],
                   [row(rating=(1, 1, 7))], [row(rating=(1.0, 1, 1))],
                   [row(rating=(True, 1, 1))], [row(gt=(None, 1, 1))],
                   [{**row(), "email": "private-value-must-not-be-used"}],
                   [row(rater="not-anonymous@example.test")]]
        for rows in invalid:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                self.load(rows)

    def test_csv_cli_source_and_ledger_hashes_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "ratings.csv"
            rows = [row(), row("c2", gt=(2, 2, 2), rating=(2, "", 2))]
            with path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=sorted(labels.FIELDS))
                writer.writeheader(); writer.writerows(rows)
            output = Path(temp) / "analysis"
            command = [sys.executable, str(ROOT / "analyze_labels.py"), "--input", str(path),
                       "--output-dir", str(output), "--expected-clips", "50", "--expected-raters", "3"]
            printed = json.loads(subprocess.check_output(command, text=True))
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(printed["coverage"]["annotation_count"], 2)
            self.assertEqual(summary["input"]["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
            for name, receipt in summary["artifacts"].items():
                self.assertEqual(receipt["sha256"], hashlib.sha256((output / name).read_bytes()).hexdigest())
            self.assertEqual(len((output / "annotations.jsonl").read_text().splitlines()), 2)
            self.assertEqual(summary["coverage"]["missing_ratings_by_axis"]["T"], 1)
            before = (output / "summary.json").read_bytes()
            self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)
            self.assertEqual((output / "summary.json").read_bytes(), before)

    def test_jsonl_and_explicit_schema_version(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "ratings.jsonl"
            path.write_text(json.dumps(row()) + "\n" + json.dumps(row("c2")) + "\n")
            self.assertEqual(len(labels.load_annotations(path)[0]), 2)
        self.assertEqual(len(self.load({"schema_version": 1, "ratings": [row()]})), 1)
        with self.assertRaises(ValueError):
            self.load({"schema_version": 2, "ratings": [row()]})

    def test_duplicate_json_fields_are_not_silently_overwritten(self):
        with tempfile.TemporaryDirectory() as temp:
            duplicate = json.dumps(row())[:-1] + ',"gt_g":2}'
            for suffix, content in [("json", "[" + duplicate + "]"), ("jsonl", duplicate + "\n")]:
                path = Path(temp) / ("ratings." + suffix)
                path.write_text(content)
                with self.subTest(suffix=suffix), self.assertRaisesRegex(ValueError, "Duplicate JSON field"):
                    labels.load_annotations(path)


if __name__ == "__main__":
    unittest.main()
