from pathlib import Path
import csv
import importlib.util
import json
import math
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("gtr_evaluation_analyzer", ROOT / "analyze.py")
analyzer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analyzer)


class ListeningStatisticsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.mapping = self.write("models.json", {
            "schema_version": 1, "name": "synthetic_explicit_ranges", "paper_mapping_confirmed": False,
            "ranges": [{"start": 1, "end": 20, "model": "FastPitch"},
                       {"start": 21, "end": 50, "model": "StyleTTS"}]})

    def tearDown(self):
        self.temp.cleanup()

    def write(self, name, data):
        path = self.root / name
        path.write_text(json.dumps(data))
        return path

    def analyze(self, rows, **kwargs):
        path = self.write("responses.json", rows)
        return analyzer.analyze(path, self.mapping, **kwargs)

    def trialmap(self, entries, counts=None):
        return self.write("trials-map.json", {"schema_version": 1,
            "source": {"description": "Synthetic source fixture, not recovered participants", "sha256": "a" * 64},
            "evaluation_trial_counts": counts or {"evaluation_1": 2}, "entries": entries})

    def test_strict_result_parsing_and_explicit_aliases(self):
        page, trials = analyzer.parse_result("evaluation_50,correct/4/5,variant1/3/4,variant_probe_2/1/2")
        self.assertEqual(page, "evaluation_50")
        self.assertEqual([row["choice"] for row in trials], ["correct", "variant1", "variant2"])
        self.assertEqual([row["correct"] for row in trials], [1, 0, 0])
        for invalid in ["evaluation_0,correct/1/1", "evaluation_1", "evaluation_1,",
                        "evaluation_1,wrong/1/1", "evaluation_1,correct/4.0/5",
                        "evaluation_1,correct/0/5", "evaluation_1,correct/4/6"]:
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                analyzer.parse_result(invalid)

    def test_correct_sample_population_sd_and_sem_units(self):
        values = analyzer.describe([1, 3])
        self.assertEqual(values["mean"], 2)
        self.assertAlmostEqual(values["sample_sd_ddof1"], math.sqrt(2))
        self.assertEqual(values["population_sd_ddof0"], 1)
        self.assertAlmostEqual(values["sem_from_sample_sd"], 1)
        self.assertIsNone(analyzer.describe([1])["sample_sd_ddof1"])

    def test_response_weighting_is_not_trial_weighting(self):
        summary, _ = self.analyze([
            {"response_id": "r1", "result": "evaluation_1,correct/5/5"},
            {"response_id": "r2", "result": "evaluation_1,variant1/1/1,correct/1/1,correct/1/1"}])
        stats = summary["models"]["FastPitch"]
        self.assertEqual(stats["trial_pooled_statistics"]["quality"]["mean"], 2)
        self.assertEqual(stats["per_response_mean_statistics"]["quality"]["mean"], 3)
        self.assertEqual(stats["accuracy"]["correct"], 3)
        self.assertEqual(stats["accuracy"]["total"], 4)
        self.assertIsNone(stats["participant_count"])

    def test_no_implicit_id40_or_trialcount_filter(self):
        rows = [{"response_id": f"r{i}", "result": f"evaluation_{page}," + ",".join(["correct/4/5"] * count)}
                for i, (page, count) in enumerate([(1, 18), (41, 19), (50, 20)])]
        summary, records = self.analyze(rows)
        self.assertEqual(summary["included_response_count"], 3)
        self.assertEqual(summary["excluded_response_count"], 0)
        self.assertEqual(summary["global"]["trial_count"], 57)
        self.assertEqual(summary["models"]["StyleTTS"]["response_count"], 2)
        self.assertEqual(summary["models"]["StyleTTS"]["trial_count"], 39)
        self.assertTrue(all(row["included"] for row in records))
        self.assertFalse(summary["paper_results_reproduced"])

    def test_response_ids_are_not_participant_ids(self):
        summary, _ = self.analyze([
            {"response_id": "r1", "participant_id": "p1", "result": "evaluation_1,correct/4/4"},
            {"response_id": "r2", "participant_id": "p1", "result": "evaluation_1,correct/2/2"},
            {"response_id": "r3", "participant_id": "p2", "result": "evaluation_1,correct/5/5"}])
        stats = summary["models"]["FastPitch"]
        self.assertEqual(stats["response_count"], 3)
        self.assertEqual(stats["participant_count"], 2)
        self.assertEqual(stats["per_declared_participant_mean_statistics"]["quality"]["mean"], 4)
        self.assertNotEqual(stats["per_response_mean_statistics"]["quality"]["mean"], 4)

    def test_included_unknown_model_is_error_not_silent_exclusion(self):
        with self.assertRaisesRegex(ValueError, "no explicit model assignment"):
            self.analyze([{"response_id": "r1", "result": "evaluation_51,correct/4/5"}])

    def test_explicit_exclusion_preserves_input_counts_and_hashed_ledgers(self):
        excluded = self.write("exclusions.json", [{"response_id": "r2", "reason": "Synthetic predeclared exclusion"}])
        summary, records = self.analyze([
            {"response_id": "r1", "result": "evaluation_1,correct/4/5"},
            {"response_id": "r2", "result": "evaluation_50,variant1/1/1,correct/3/3"}], exclusions=excluded)
        self.assertEqual(summary["input_response_count"], 2)
        self.assertEqual(summary["included_response_count"], 1)
        self.assertEqual(summary["excluded_trial_count"], 2)
        output = self.root / "output"
        analyzer.write_outputs(output, summary, records)
        self.assertEqual(summary["ledgers"]["included-responses.jsonl"]["rows"], 1)
        self.assertEqual(summary["ledgers"]["excluded-responses.jsonl"]["rows"], 1)
        self.assertEqual(summary["ledgers"]["trials.jsonl"]["rows"], 3)
        for name, row in summary["ledgers"].items():
            self.assertEqual(row["sha256"], analyzer.file_digest(output / name))

    def test_unmatched_exclusion_is_rejected(self):
        path = self.write("exclusions.json", [{"response_id": "missing", "reason": "typo"}])
        with self.assertRaisesRegex(ValueError, "matches no"):
            self.analyze([{"response_id": "r1", "result": "evaluation_1,correct/4/5"}], exclusions=path)

    def test_no_map_means_no_category_or_chi_square_claim(self):
        summary, _ = self.analyze([{"response_id": "r1", "result": "evaluation_1,correct/4/5"}])
        self.assertEqual(summary["figure1"]["status"], "NOT_COMPUTED_NO_TRIAL_MAP")
        self.assertEqual(summary["figure1"]["categories"], [])
        self.assertEqual(summary["figure1"]["chi_square_tests"], [])

    def test_version_mismatch_blocks_only_category_mapping_not_global_trials(self):
        path = self.trialmap([{"evaluation_id": "evaluation_1", "trial_index": 1, "dimension": "G", "level": 1}])
        summary, records = self.analyze([{"response_id": "r1", "result": "evaluation_1,correct/4/5,correct/4/5,correct/4/5"}], trial_map=path)
        self.assertEqual(summary["global"]["trial_count"], 3)
        self.assertEqual(summary["figure1"]["categories"], [])
        self.assertEqual(summary["figure1"]["coverage"]["source_page_trial_count_mismatch_responses"], 1)
        self.assertEqual(records[0]["trials"][0]["figure_mapping_status"], "source_page_trial_count_mismatch")

    def test_partial_explicit_trial_map_and_chi_square_formula(self):
        path = self.trialmap([
            {"evaluation_id": "evaluation_1", "trial_index": 1, "dimension": "G", "level": 1},
            {"evaluation_id": "evaluation_1", "trial_index": 2, "dimension": "G", "level": 4}])
        rows = [{"response_id": "r1", "result": "evaluation_1,correct/4/5,variant1/3/3"},
                {"response_id": "r2", "result": "evaluation_1,correct/4/5,correct/3/3"},
                {"response_id": "r3", "result": "evaluation_50,correct/5/5"}]
        summary, _ = self.analyze(rows, trial_map=path, pairwise=True)
        self.assertEqual(summary["global"]["trial_count"], 5)
        figure = summary["figure1"]
        self.assertEqual(figure["coverage"]["figure_trials"], 4)
        self.assertEqual(figure["coverage"]["unmapped_trials"], 1)
        self.assertFalse(figure["coverage"]["all_included_trials_have_source_mapping"])
        chi = figure["chi_square_tests"][0]
        self.assertEqual(chi["correct_incorrect_counts"], [[2, 0], [1, 1]])
        self.assertAlmostEqual(chi["pearson"]["statistic"], 4 / 3)
        self.assertFalse(chi["pearson"]["yates_correction"])
        self.assertEqual(chi["pairwise_bonferroni_family_size"], 1)

    def test_trial_map_requires_provenance_and_page_count(self):
        invalid = self.write("bad-map.json", {"entries": [{"evaluation_id": "evaluation_1", "trial_index": 1, "dimension": "G", "level": 1}]})
        with self.assertRaisesRegex(ValueError, "source"):
            self.analyze([{"response_id": "r1", "result": "evaluation_1,correct/4/5"}], trial_map=invalid)

    def test_invalid_resonance_level_is_not_silently_relabelled(self):
        invalid = self.trialmap([{"evaluation_id": "evaluation_1", "trial_index": 1,
                                  "dimension": "R", "level": 7}])
        with self.assertRaisesRegex(ValueError, "valid G/T/R"):
            self.analyze([{"response_id": "r1", "result": "evaluation_1,correct/4/5,correct/3/4"}], trial_map=invalid)

    def test_incomplete_identity_does_not_invent_full_participant_count(self):
        summary, _ = self.analyze([
            {"response_id": "r1", "participant_id": "p1", "result": "evaluation_1,correct/4/5"},
            {"response_id": "r2", "result": "evaluation_1,correct/2/3"}])
        stats = summary["global"]
        self.assertEqual(stats["response_count"], 2)
        self.assertIsNone(stats["participant_count"])
        self.assertEqual(stats["known_unique_participant_ids"], 1)
        self.assertEqual(stats["participant_statistics_scope"], "known_ID_subset_only")

    def test_jsonl_csv_and_participant_only_input(self):
        row = {"participant_id": "p1", "resultstring": "evaluation_1,correct/4/5"}
        jsonl = self.root / "input.jsonl";jsonl.write_text(json.dumps(row) + "\n")
        csvfile = self.root / "input.csv"
        with csvfile.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row));writer.writeheader();writer.writerow(row)
        for path in [jsonl, csvfile]:
            summary, records = analyzer.analyze(path, self.mapping)
            self.assertEqual(summary["global"]["participant_count"], 1)
            self.assertEqual(records[0]["response_id"], "response_000001")

    def test_unrecognized_identity_columns_and_duplicate_response_ids_fail(self):
        with self.assertRaisesRegex(ValueError, "anonymize"):
            self.analyze([{"response_id": "r1", "email": "not-a-real-person@example.invalid", "result": "evaluation_1,correct/4/5"}])
        with self.assertRaisesRegex(ValueError, "Duplicate response_id"):
            self.analyze([{"response_id": "r1", "result": "evaluation_1,correct/4/5"}] * 2)

    def test_cli_new_process_is_functional(self):
        source = self.write("input.json", [{"response_id": "r1", "result": "evaluation_50,correct/4/5"}])
        output = self.root / "cli-output"
        command = [sys.executable, str(ROOT / "analyze.py"), "--input", str(source),
                   "--model-map", str(self.mapping), "--output-dir", str(output)]
        result = json.loads(subprocess.check_output(command, text=True))
        self.assertEqual(result["included_response_count"], 1)
        summary = json.loads((output / "summary.json").read_text())
        self.assertEqual(summary["models"]["StyleTTS"]["accuracy"]["proportion"], 1)
        repeated = subprocess.run(command, capture_output=True, text=True)
        self.assertNotEqual(repeated.returncode, 0)
        self.assertTrue((output / "summary.json").is_file())


if __name__ == "__main__":
    unittest.main()
