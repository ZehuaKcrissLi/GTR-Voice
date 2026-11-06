"""The twelve acoustic feature families listed in paper section 2.3.

The paper does not specify aggregation, feature counts, or pYIN settings.
Defaults below are explicit released-code-compatible implementation choices.
"""
from __future__ import annotations

import numpy as np


class FeatureExtractor:
    def __init__(self, sr=22050, n_fft=2048, hop_length=512, n_mfcc=40, n_mels=128):
        if sr < 16000 or n_fft < 32 or hop_length < 1:
            raise ValueError("Use sample rate >=16000 and positive frame settings")
        self.sr, self.n_fft, self.hop_length = sr, n_fft, hop_length
        self.n_mfcc, self.n_mels = n_mfcc, n_mels

    @property
    def config(self):
        return {"sample_rate": self.sr, "n_fft": self.n_fft, "hop_length": self.hop_length,
                "n_mfcc": self.n_mfcc, "n_mels": self.n_mels,
                "pitch_range_notes": ["C2", "C7"], "unvoiced_f0_fill": 0.0,
                "rolloff_percent": 0.99, "minimum_rolloff_percent": 0.01,
                "paper_aggregation_and_hyperparameters": "not reported"}

    def extract_features(self, y, sr=None):
        import librosa

        sr = self.sr if sr is None else sr
        y = np.asarray(y, dtype=np.float32)
        if y.ndim != 1 or not len(y) or not np.isfinite(y).all():
            raise ValueError("Expected nonempty finite mono samples")
        if sr != self.sr:
            y = librosa.resample(y, orig_sr=sr, target_sr=self.sr)
        frame = {"n_fft": self.n_fft, "hop_length": self.hop_length}
        common = {"y": y, "sr": self.sr, **frame}
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y, sr=self.sr, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"),
            frame_length=self.n_fft, hop_length=self.hop_length, fill_na=0.0)
        families = [
            librosa.feature.mfcc(**common, n_mfcc=self.n_mfcc),
            librosa.feature.melspectrogram(**common, n_mels=self.n_mels),
            librosa.feature.spectral_contrast(**common),
            librosa.feature.spectral_centroid(**common),
            librosa.feature.spectral_bandwidth(**common),
            librosa.feature.spectral_rolloff(**common, roll_percent=0.99),
            librosa.feature.spectral_rolloff(**common, roll_percent=0.01),
            f0[None, :], voiced_flag[None, :], voiced_probs[None, :],
            librosa.feature.zero_crossing_rate(y, frame_length=self.n_fft, hop_length=self.hop_length),
            librosa.feature.spectral_flatness(y=y, **frame),
        ]
        if len({value.shape[1] for value in families}) != 1:
            raise ValueError("Feature frame alignment mismatch")
        features = np.concatenate(families, axis=0).astype(np.float32)
        if not np.isfinite(features).all():
            raise ValueError("Nonfinite acoustic features")
        return features

    def extract_file(self, path, aggregation="mean"):
        import librosa
        from gtr_data import check_wav_frames

        check_wav_frames(path)
        audio, sr = librosa.load(path, sr=None, mono=True)
        matrix = self.extract_features(audio, sr=sr)
        if aggregation == "mean":
            return matrix.mean(axis=1)
        if aggregation == "mean_std":
            return np.concatenate([matrix.mean(axis=1), matrix.std(axis=1)])
        raise ValueError(f"Unknown aggregation: {aggregation}")
