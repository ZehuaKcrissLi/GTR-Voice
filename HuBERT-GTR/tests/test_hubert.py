from pathlib import Path
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


def module(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


class ModelContractTests(unittest.TestCase):
    def test_legacy_split_replay_requires_stable_fields_and_verified_source_guard(self):
        import copy
        main = module("gtr_hubert_legacy_split_test", "main.py")
        current = {"method": "stratified", "seed": 42, "test_size": .2,
                   "test_sentences": [], "train_ids": ["a"], "test_ids": ["b"],
                   "unused_validation_ids": [], "paper_original_split": "not reported",
                   "source_separation": {"status": "VERIFIED_NO_SHARED_SOURCE_SHA256",
                                         "known_shared_source_hashes": 0}}
        old = {key: value for key, value in current.items() if key != "source_separation"}
        result = main.verify_checkpoint_split(current, old)
        self.assertEqual(result["checkpoint_split_metadata"], "legacy_stable_fields_with_recomputed_guard")
        self.assertEqual(result["source_separation"], current["source_separation"])
        self.assertEqual(main.verify_checkpoint_split(current, current)["checkpoint_split_metadata"], "current")
        for changed in [{**old, "test_ids": ["wrong"]}, {**old, "seed": 7},
                        {key: value for key, value in old.items() if key != "train_ids"},
                        {**old, "unknown_field": "cannot-ignore"},
                        {**current, "source_separation": {"status": "UNKNOWN_MISSING_SOURCE_SHA256"}}]:
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                main.verify_checkpoint_split(current, changed)
        unknown = copy.deepcopy(current)
        unknown["source_separation"]["status"] = "UNKNOWN_MISSING_SOURCE_SHA256"
        with self.assertRaisesRegex(ValueError, "source separation is UNKNOWN"):
            main.verify_checkpoint_split(unknown, old)

    def test_all_cli_entries_are_local_by_default_and_download_is_explicit(self):
        import contextlib
        import io

        main = module("gtr_hubert_parser_test", "main.py")
        parser = main.build_parser()
        commands = [
            ["extract", "--manifest", "manifest.jsonl", "--model-path", "fixture/model", "--output-dir", "features"],
            ["train", "--features", "features", "--model-path", "fixture/model", "--output-dir", "trained"],
            ["evaluate", "--features", "features", "--checkpoint", "model.pt"],
        ]
        for command in commands:
            with self.subTest(command=command[0]):
                self.assertTrue(parser.parse_args(command).local_files_only)
                self.assertTrue(parser.parse_args(command + ["--local-files-only"]).local_files_only)
                self.assertFalse(parser.parse_args(command + ["--allow-download"]).local_files_only)
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    parser.parse_args(command + ["--local-files-only", "--allow-download"])
        probe = module("gtr_hubert_probe_test", "hubert_test.py")
        command = ["--model-path", "fixture/model", "--wav", "example.wav"]
        self.assertTrue(probe.parse_args(command).local_files_only)
        self.assertFalse(probe.parse_args(command + ["--allow-download"]).local_files_only)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            probe.parse_args(command + ["--local-files-only", "--allow-download"])

    def test_uncached_model_apis_fail_without_any_http_request_by_default(self):
        from unittest.mock import patch

        main = module("gtr_hubert_offline_main_test", "main.py")
        models = module("gtr_hubert_offline_model_test", "model.py")
        missing = "gtr-local-regression/never-download-this-uncached-model"
        # Guard the real transformers/Hugging Face boundary. A default download
        # attempt fails this test even if transformers wraps the network error.
        with patch("requests.sessions.Session.request", side_effect=AssertionError("Network request forbidden")) as network:
            with self.assertRaises(OSError):
                models.GTRClassifier(missing)
            with self.assertRaises(OSError):
                main.extract_features(missing, None, "unused-no-output", manifest="unused.jsonl")
            network.assert_not_called()

    def test_masks_frozen_backbone_and_singleton_training(self):
        from types import SimpleNamespace
        import torch
        from torch import nn
        models = module("gtr_hubert_model_test", "model.py")

        class FakeHubert(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = SimpleNamespace(hidden_size=4, conv_kernel=[], conv_stride=[])
                self.weight = nn.Parameter(torch.ones(()))

            def forward(self, values, attention_mask=None):
                # Nonzero padding features expose incorrect unmasked pooling.
                hidden = values[:, :, None].repeat(1, 1, 4) * self.weight + 10
                return SimpleNamespace(last_hidden_state=hidden)

        model = models.GTRClassifier(num_classes=5, backbone=FakeHubert())
        model.eval()
        first = model._encode(torch.tensor([[1., 3.]]), torch.tensor([[1, 1]]))
        padded = model._encode(torch.tensor([[1., 3., 99., 99.]]), torch.tensor([[1, 1, 0, 0]]))
        self.assertTrue(torch.equal(first, padded))
        model.train()
        self.assertFalse(model.hubert_model.training)
        self.assertTrue(model.classifier.training)
        output = model(torch.tensor([[1., 3.]]), torch.tensor([[1, 1]]))
        self.assertEqual(tuple(output.shape), (1, 5))
        output.sum().backward()
        self.assertIsNone(model.hubert_model.weight.grad)
        self.assertIsNotNone(model.fc1.weight.grad)

    def test_full_cli_with_local_tiny_encoder_save_new_process_eval(self):
        import numpy as np
        import soundfile as sf
        import torch
        from transformers import HubertConfig, HubertModel, Wav2Vec2FeatureExtractor

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_path = root / "tiny-hubert"
            config = HubertConfig(hidden_size=8, num_hidden_layers=1,
                num_attention_heads=2, intermediate_size=16, conv_dim=(8, 8),
                conv_stride=(5, 2), conv_kernel=(10, 3), num_conv_pos_embeddings=4,
                num_conv_pos_embedding_groups=2, mask_time_prob=0.0,
                feat_extract_norm="layer")
            HubertModel(config).save_pretrained(model_path)
            Wav2Vec2FeatureExtractor(sampling_rate=16000, return_attention_mask=True).save_pretrained(model_path)
            data_root = root / "audio"
            for label in ["010", "020", "111", "122", "213", "224"]:
                folder = data_root / label
                folder.mkdir(parents=True)
                for utterance in range(1, 6):
                    time = np.arange(3200 + utterance * 80) / 16000
                    signal = .1 * np.sin(2 * np.pi * (120 + int(label)) * time)
                    sf.write(folder / f"{utterance:02}.wav", signal, 16000, subtype="PCM_16")
            cache = root / "features"
            command = [sys.executable, str(ROOT / "main.py")]
            extracted = json.loads(subprocess.check_output(command + ["extract", "--data-root", str(data_root),
                "--model-path", str(model_path), "--output-dir", str(cache)], text=True))
            self.assertEqual(extracted["saved"], 30)
            self.assertEqual(len(list(cache.glob("*/*.npy"))), 30)
            output = root / "trained"
            trained = json.loads(subprocess.check_output(command + ["train", "--features", str(cache),
                "--model-path", str(model_path), "--output-dir", str(output),
                "--axis", "T", "--device", "cpu", "--epochs", "1", "--batch-size", "23"], text=True))
            self.assertTrue(trained["reload_predictions_equal"])
            self.assertFalse(trained["paper_reported_method"])
            independent = json.loads(subprocess.check_output(command + ["evaluate", "--features", str(cache),
                "--checkpoint", str(output / "model.pt"), "--device", "cpu"], text=True))
            self.assertEqual(trained["metrics"]["predictions"], independent["metrics"]["predictions"])
            self.assertEqual(trained["metrics"]["label_order"], [1, 2, 3, 4, 5])
            checkpoint = torch.load(output / "model.pt", map_location="cpu", weights_only=True)
            self.assertEqual(checkpoint["feature_index_sha256"], module("gtr_hubert_data_test", "dataset.py").file_sha256(cache / "index.jsonl"))
            # Trusted local copy of a pre-guard checkpoint: identical model,
            # cache and IDs, with only the later-added metadata field absent.
            del checkpoint["split"]["source_separation"]
            legacy_path = root / "trusted-legacy-model.pt"
            torch.save(checkpoint, legacy_path)
            legacy = json.loads(subprocess.check_output(command + ["evaluate", "--features", str(cache),
                "--checkpoint", str(legacy_path), "--device", "cpu"], text=True))
            self.assertEqual(legacy["metrics"], independent["metrics"])
            self.assertEqual(legacy["source_separation"]["status"], "VERIFIED_NO_SHARED_SOURCE_SHA256")
            self.assertEqual(legacy["checkpoint_split_metadata"], "legacy_stable_fields_with_recomputed_guard")
            checkpoint["split"]["test_ids"] = ["a-different-held-out-identity"]
            torch.save(checkpoint, root / "drifted-legacy-model.pt")
            rejected = subprocess.run(command + ["evaluate", "--features", str(cache),
                "--checkpoint", str(root / "drifted-legacy-model.pt"), "--device", "cpu"],
                text=True, capture_output=True)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("Held-out split differs", rejected.stderr)


if __name__ == "__main__":
    unittest.main()
