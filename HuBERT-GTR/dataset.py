"""Deterministic cached-input datasets with global label mappings and masks."""
from pathlib import Path
import json
import sys
import numpy as np
import torch
from torch.utils.data import Dataset

_ACOUSTIC = str(Path(__file__).resolve().parents[1] / "Acoustic-GTR")
if _ACOUSTIC not in sys.path:
    sys.path.append(_ACOUSTIC)
from gtr_data import LABEL_VALUES, file_sha256, make_split, parse_label


class AudioDataset(Dataset):
    def __init__(self, file_path, GTR="G", mode="train", *, seed=42, test_size=0.2,
                 split="stratified", test_sentences=None, verify_features=True):
        if GTR not in LABEL_VALUES or mode not in {"train", "test", "all"}:
            raise ValueError("GTR must be G/T/R; mode must be train/test/all")
        self.root, self.GTR = Path(file_path).resolve(strict=True), GTR
        index = self.root / "index.jsonl"
        if not index.is_file():
            raise ValueError("Feature index missing; run main.py extract first")
        records = [json.loads(line) for line in index.read_text().splitlines() if line.strip()]
        if not records or len({row["id"] for row in records}) != len(records):
            raise ValueError("Empty feature index or duplicate sample IDs")
        records.sort(key=lambda row: row["id"])
        for row in records:
            parse_label(row["label"])
            relative = Path(row["feature_path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Feature path must stay inside the feature root")
            path = self.root / relative
            if path.is_symlink() or not path.is_file() or path.resolve().parent.parent != self.root:
                raise ValueError(f"Invalid feature path: {relative}")
            if verify_features and file_sha256(path) != row["feature_sha256"]:
                raise ValueError(f"Feature hash mismatch: {row['id']}")
        self.index_sha256 = file_sha256(index)
        self.label_order = list(LABEL_VALUES[GTR])
        if mode == "all":
            indices = range(len(records))
            self.split_record = {"method": "all_samples", "paper_original_split": "not reported"}
        else:
            train, test, self.split_record = make_split(records, method=split, test_size=test_size,
                                                       seed=seed, test_sentences=test_sentences)
            indices = train if mode == "train" else test
        self.records = [records[int(i)] for i in indices]
        axis = "GTR".index(GTR)
        self.labels = [self.label_order.index(parse_label(row["label"])[axis]) for row in self.records]
        self.files = [self.root / row["feature_path"] for row in self.records]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        values = np.load(self.files[index], allow_pickle=False)
        if values.ndim == 2 and values.shape[0] == 1:
            values = values[0]
        if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
            raise ValueError(f"Expected nonempty finite input_values: {self.files[index]}")
        return torch.as_tensor(values.copy(), dtype=torch.float32), self.labels[index]

    @staticmethod
    def collation_fn(batch):
        values, labels = zip(*batch)
        lengths = [len(value) for value in values]
        inputs = torch.zeros(len(values), max(lengths), dtype=torch.float32)
        mask = torch.zeros_like(inputs, dtype=torch.long)
        for i, value in enumerate(values):
            inputs[i, :len(value)] = value
            mask[i, :len(value)] = 1
        return inputs, torch.as_tensor(labels, dtype=torch.long), mask
