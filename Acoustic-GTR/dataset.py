"""Portable acoustic-feature datasets without a HuBERT/CUDA dependency."""
from pathlib import Path

from feature_extractor import FeatureExtractor
from gtr_data import parse_label


class AudioDataset:
    """Extract a sample vector and its G/T/R labels from explicit WAV paths.

    SVM batch extraction uses run_svm.py; this small adapter preserves the old
    AudioDataset name without its undefined feature_extractor/device members.
    """
    def __init__(self, file_paths, model_path=None, *, sr=22050, aggregation="mean"):
        if model_path is not None:
            raise ValueError("Acoustic features do not use model_path; see HuBERT-GTR for HuBERT")
        self.file_paths = [Path(path) for path in file_paths]
        self.feature_extractor = FeatureExtractor(sr=sr)
        self.aggregation = aggregation

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        path = self.file_paths[index]
        return self.feature_extractor.extract_file(path, self.aggregation), parse_label(path.parent.name)
