from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
import wave

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gtr_data import check_wav_frames, load_records, make_split, parse_label


class DataContractTests(unittest.TestCase):
    def test_leading_zero_and_coupled_whisper(self):
        self.assertEqual(parse_label("010"), (0, 1, 0))
        self.assertEqual(parse_label("456"), (4, 5, 6))
        for invalid in [10, "000", "012", "210", "999", "10"]:
            with self.assertRaises(ValueError):
                parse_label(invalid)

    def test_incomplete_pcm_frame_is_not_silently_decoded(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "original.wav"
            with wave.open(str(path), "wb") as wav:
                wav.setparams((1, 3, 48000, 0, "NONE", "not compressed"))
                wav.writeframes(b"\0" * 301)
            before = path.read_bytes()
            with self.assertRaisesRegex(ValueError, "Incomplete PCM frame"):
                check_wav_frames(path)
            self.assertEqual(before, path.read_bytes())

    def test_duplicate_sources_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "clip.wav"
            path.touch()
            manifest = Path(directory) / "manifest.jsonl"
            rows = [{"id": f"010/{i}", "source_path": str(path), "label": "010"} for i in range(2)]
            manifest.write_text("\n".join(json.dumps(row) for row in rows))
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                load_records(manifest=manifest)

    def test_deterministic_stratified_and_sentence_holdout(self):
        rows = [{"id": f"{label}/{i:02}", "label": label, "utterance_id": str(i)}
                for label in ["010", "111", "222"] for i in range(1, 11)]
        train, test, split = make_split(rows, seed=42)
        train_again, test_again, _ = make_split(rows, seed=42)
        self.assertEqual(train.tolist(), train_again.tolist())
        self.assertEqual(test.tolist(), test_again.tolist())
        self.assertFalse(set(split["train_ids"]) & set(split["test_ids"]))
        _, _, held = make_split(rows, method="sentence", test_sentences=[9, 10])
        self.assertEqual({Path(value).name for value in held["test_ids"]}, {"09", "10"})
        self.assertFalse({Path(value).name for value in held["train_ids"]} & {"09", "10"})
        self.assertEqual(split["source_separation"]["status"], "UNKNOWN_MISSING_SOURCE_SHA256")

    def test_copied_audio_bytes_cannot_cross_any_split(self):
        # Different paths and IDs can still contain the exact same source audio.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first, copied = root / "first.wav", root / "copied.wav"
            with wave.open(str(first), "wb") as wav:
                wav.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                wav.writeframes(b"\0" * 1600)
            copied.write_bytes(first.read_bytes())
            digest = hashlib.sha256(first.read_bytes()).hexdigest()
            rows = [{"id": "clip-a", "source_path": str(first), "label": "111", "utterance_id": "1", "split": "train", "source_sha256": digest},
                    {"id": "clip-b", "source_path": str(copied), "label": "111", "utterance_id": "2", "split": "test", "source_sha256": digest}]
            manifest = root / "manifest.jsonl"
            manifest.write_text("\n".join(json.dumps(row) for row in rows))
            loaded = load_records(manifest=manifest)
            self.assertEqual(len(loaded), 2)  # Paths and IDs alone are distinct.
            for method, kwargs in [("manifest", {}), ("sentence", {"test_sentences": [2]})]:
                with self.subTest(method=method), self.assertRaisesRegex(ValueError, "Same source SHA256"):
                    make_split(loaded, method=method, **kwargs)
            stratified = [{**rows[0], "id": f"copy-{i}"} for i in range(10)]
            with self.assertRaisesRegex(ValueError, "Same source SHA256"):
                make_split(stratified, method="stratified", seed=42)

    def test_partial_source_hashes_are_unknown_and_invalid_hashes_rejected(self):
        rows = [{"id": "a", "label": "111", "utterance_id": "1", "split": "train",
                 "source_sha256": "a" * 64},
                {"id": "b", "label": "222", "utterance_id": "2", "split": "test"}]
        _, _, split = make_split(rows, method="manifest")
        self.assertEqual(split["source_separation"]["status"], "UNKNOWN_MISSING_SOURCE_SHA256")
        self.assertEqual(split["source_separation"]["missing_source_sha256_ids"], {"train": [], "test": ["b"]})
        rows[1]["source_sha256"] = "not-a-hash"
        with self.assertRaisesRegex(ValueError, "Invalid cached source SHA256"):
            make_split(rows, method="manifest")
        rows[1]["source_sha256"] = "b" * 64
        self.assertEqual(make_split(rows, method="manifest")[2]["source_separation"]["status"], "VERIFIED_NO_SHARED_SOURCE_SHA256")


class AcousticPipelineTests(unittest.TestCase):
    def test_all_twelve_families_and_finite_unvoiced_frames(self):
        import numpy as np
        from feature_extractor import FeatureExtractor
        extractor = FeatureExtractor()
        matrix = extractor.extract_features(np.zeros(22050, dtype=np.float32))
        self.assertEqual(matrix.shape[0], 184)
        self.assertGreater(matrix.shape[1], 1)
        self.assertTrue(np.isfinite(matrix).all())
        self.assertTrue((matrix[179] == 0).all())  # Explicit unvoiced F0 policy.

    def test_train_save_new_process_evaluate_and_cache_binding(self):
        import numpy as np
        from run_svm import evaluate, train
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            labels = ["010", "020", "111", "122", "213", "224", "315", "326", "411", "422"]
            rows = [{"id": f"{label}/{i:02}.wav", "label": label, "utterance_id": str(i),
                     "source_sha256": hashlib.sha256(f"{label}/{i}".encode()).hexdigest()}
                    for label in labels for i in range(1, 11)]
            features = np.asarray([parse_label(row["label"]) for row in rows], dtype=float)
            metadata = {"config": {"fixture": "artifact integrity, not paper metrics"}, "records": rows, "rejected": []}
            cache = root / "features.npz"
            np.savez_compressed(cache, features=features, metadata=np.asarray(json.dumps(metadata)))
            args = argparse.Namespace(features=cache, output_dir=root/"model", split="stratified",
                test_size=0.2, seed=42, test_sentences=None, C=1.0, gamma="scale", kernel="rbf", scale=True)
            report = train(args)
            self.assertTrue(report["reload_predictions_equal"])
            self.assertEqual(report["test_samples"], 20)
            self.assertFalse(report["paper_exact_reproduction"])
            self.assertEqual(report["split"]["source_separation"]["status"], "VERIFIED_NO_SHARED_SOURCE_SHA256")
            command = [sys.executable, str(Path(__file__).resolve().parents[1]/"run_svm.py"),
                       "evaluate", "--features", str(cache), "--model", str(root/"model/models.joblib")]
            independent = json.loads(subprocess.check_output(command, text=True))
            self.assertEqual(independent["source_separation"], report["split"]["source_separation"])
            for axis in "GTR":
                self.assertEqual(report["metrics"][axis]["predictions"], independent["metrics"][axis]["predictions"])
            np.savez_compressed(cache, features=features + 0.01, metadata=np.asarray(json.dumps(metadata)))
            with self.assertRaisesRegex(ValueError, "cache bytes differ"):
                evaluate(argparse.Namespace(features=cache, model=root/"model/models.joblib", all_samples=False))

    def test_missing_source_hash_cache_is_rejected_before_svm_fit(self):
        import numpy as np
        from run_svm import train
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [{"id": str(i), "label": "111", "utterance_id": str(i),
                     "split": "train" if i < 2 else "test"} for i in range(4)]
            cache = root / "missing-hashes.npz"
            np.savez_compressed(cache, features=np.zeros((4, 3)),
                                metadata=np.asarray(json.dumps({"config": {}, "records": rows})))
            args = argparse.Namespace(features=cache, output_dir=root / "must-not-exist", split="manifest",
                test_size=.2, seed=42, test_sentences=None, C=.1, gamma=".001", kernel="rbf", scale=True)
            with self.assertRaisesRegex(ValueError, "Source separation is UNKNOWN"):
                train(args)
            self.assertFalse(args.output_dir.exists())


if __name__ == "__main__":
    unittest.main()
