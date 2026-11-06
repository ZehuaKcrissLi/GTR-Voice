"""Explicit, byte-preserving input and split contracts for GTR classifiers."""
from __future__ import annotations

import hashlib
import json
import re
import struct
from pathlib import Path

LABEL_VALUES = {"G": tuple(range(5)), "T": tuple(range(1, 6)), "R": tuple(range(7))}


def parse_label(label: str) -> tuple[int, int, int]:
    if not isinstance(label, str) or not re.fullmatch(r"[0-4][1-5][0-6]", label):
        raise ValueError(f"Expected a three-digit GTR label, preserving leading zero: {label!r}")
    g, t, r = map(int, label)
    if (g == 0) != (r == 0):
        raise ValueError(f"Whisper G=0 and R=0 must occur together: {label}")
    return g, t, r


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_wav_frames(path: Path) -> dict:
    """Reject truncated PCM frames instead of silently repairing original WAVs.

    libsndfile may decode the complete prefix of a malformed final PCM frame.
    This RIFF check makes that known dataset anomaly an explicit extraction error.
    """
    path = Path(path)
    size = path.stat().st_size
    with path.open("rb") as stream:
        header = stream.read(12)
        if len(header) != 12 or header[:4] != b"RIFF" or header[8:] != b"WAVE":
            raise ValueError(f"Expected a RIFF/WAVE file: {path}")
        declared_end = struct.unpack("<I", header[4:8])[0] + 8
        if declared_end > size:
            raise ValueError(f"Truncated RIFF container: {path}")
        fmt = None
        payload_sizes = []
        while stream.tell() + 8 <= declared_end:
            chunk = stream.read(8)
            kind, length = chunk[:4], struct.unpack("<I", chunk[4:])[0]
            start = stream.tell()
            if start + length > size or start + length > declared_end:
                raise ValueError(f"Truncated WAV chunk: {path}")
            if kind == b"fmt ":
                raw = stream.read(min(length, 40))
                if len(raw) < 16:
                    raise ValueError(f"Incomplete WAV format: {path}")
                audio_format, channels, sample_rate, _, block_align, bits = struct.unpack("<HHIIHH", raw[:16])
                if audio_format not in (1, 3, 0xFFFE) or channels < 1 or block_align < 1:
                    raise ValueError(f"Unsupported WAV format: {path}")
                fmt = {"channels": channels, "sample_rate": sample_rate,
                       "block_align": block_align, "bits_per_sample": bits}
            elif kind == b"data":
                payload_sizes.append(length)
            stream.seek(start + length + (length % 2))
        if fmt is None or not payload_sizes or not any(payload_sizes):
            raise ValueError(f"WAV format or audio payload missing: {path}")
        if any(n % fmt["block_align"] for n in payload_sizes):
            raise ValueError(f"Incomplete PCM frame in original WAV (not repaired): {path}")
    return fmt


def load_records(*, data_root=None, manifest=None) -> list[dict]:
    if bool(data_root) == bool(manifest):
        raise ValueError("Specify exactly one of --data-root and --manifest")
    records = []
    if manifest:
        manifest = Path(manifest).resolve(strict=True)
        for number, line in enumerate(manifest.read_text().splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            raw_path = row.get("source_path", row.get("path"))
            if not raw_path:
                raise ValueError(f"Manifest line {number} has no source_path")
            path = Path(raw_path)
            if not path.is_absolute():
                path = manifest.parent / path
            label = row.get("label", row.get("gtr_label", path.parent.name))
            if not isinstance(label, str):
                raise ValueError("Manifest label must be a string (e.g. '010'), not an integer")
            record = {"id": str(row.get("id", row.get("relative_path", f"{label}/{path.name}"))),
                      "source_path": str(path.absolute()), "label": label,
                      "utterance_id": str(row.get("utterance_id", path.stem))}
            for key in ("split", "source_sha256"):
                if key in row:
                    record[key] = row[key]
            if "source_sha256" not in record and "sha256" in row:
                record["source_sha256"] = row["sha256"]
            records.append(record)
    else:
        root = Path(data_root).resolve(strict=True)
        for path in sorted(root.glob("*/*.wav")):
            records.append({"id": path.relative_to(root).as_posix(), "source_path": str(path),
                            "label": path.parent.name, "utterance_id": path.stem})
    seen_ids, seen_paths = set(), set()
    for row in records:
        parse_label(row["label"])
        path = Path(row["source_path"])
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Source must be an existing regular WAV, not a symlink: {path}")
        resolved = str(path.resolve(strict=True))
        if row["id"] in seen_ids or resolved in seen_paths:
            raise ValueError(f"Duplicate sample ID or source: {row['id']}")
        seen_ids.add(row["id"])
        seen_paths.add(resolved)
        if "source_sha256" in row and not re.fullmatch(r"[a-f0-9]{64}", row["source_sha256"]):
            raise ValueError(f"Invalid source SHA256: {row['id']}")
    if not records:
        raise ValueError("No input WAV records found")
    return sorted(records, key=lambda row: row["id"])


def verify_source_separation(records, train_indices, test_indices):
    """Reject identical original audio bytes crossing train/test partitions.

    Cached extraction records bind these hashes to source bytes. Missing hashes
    remain explicitly unknown: different paths or IDs alone prove no separation.
    """
    hashes, missing = {}, {}
    for name, indices in [("train", train_indices), ("test", test_indices)]:
        hashes[name], missing[name] = set(), []
        for index in indices:
            row = records[int(index)]
            value = row.get("source_sha256")
            if value is None or value == "":
                missing[name].append(row["id"])
            elif not isinstance(value, str) or not re.fullmatch(r"[a-f0-9]{64}", value):
                raise ValueError(f"Invalid cached source SHA256: {row['id']}")
            else:
                hashes[name].add(value)
    shared = hashes["train"] & hashes["test"]
    if shared:
        raise ValueError(f"Same source SHA256 appears in train and test ({len(shared)} distinct audio payloads); "
                         "different paths or IDs do not establish a held-out sample")
    unknown = any(missing.values())
    return {"status": "UNKNOWN_MISSING_SOURCE_SHA256" if unknown else "VERIFIED_NO_SHARED_SOURCE_SHA256",
            "identity_basis": "SHA256 stored by extraction; original sources are not reread during splitting",
            "known_train_source_hashes": len(hashes["train"]),
            "known_test_source_hashes": len(hashes["test"]),
            "known_shared_source_hashes": 0,
            "missing_source_sha256_ids": missing}


def make_split(records, *, method="stratified", test_size=0.2, seed=42, test_sentences=None):
    import numpy as np
    from sklearn.model_selection import train_test_split

    indices = np.arange(len(records))
    unused = []
    if method == "stratified":
        train, test = train_test_split(indices, test_size=test_size, random_state=seed,
                                       stratify=[row["label"] for row in records])
    elif method == "sentence":
        held = {str(int(value)) for value in (test_sentences or [])}
        if not held:
            raise ValueError("Sentence holdout requires --test-sentences")
        test = [i for i, row in enumerate(records) if str(int(row["utterance_id"])) in held]
        train = [i for i in indices if i not in set(test)]
    elif method == "manifest":
        if any(row.get("split") not in {"train", "validation", "test"} for row in records):
            raise ValueError("Every manifest record needs split=train, validation, or test")
        train = [i for i, row in enumerate(records) if row["split"] == "train"]
        test = [i for i, row in enumerate(records) if row["split"] == "test"]
        unused = [i for i, row in enumerate(records) if row["split"] == "validation"]
    else:
        raise ValueError(f"Unknown split method: {method}")
    train, test = np.asarray(sorted(train), dtype=int), np.asarray(sorted(test), dtype=int)
    if not len(train) or not len(test) or set(train).intersection(test):
        raise ValueError("Train and test must be nonempty and disjoint")
    source_separation = verify_source_separation(records, train, test)
    return train, test, {"method": method, "seed": seed, "test_size": test_size,
                         "test_sentences": list(test_sentences or []),
                         "train_ids": [records[i]["id"] for i in train],
                         "test_ids": [records[i]["id"] for i in test],
                         "unused_validation_ids": [records[i]["id"] for i in unused],
                         "source_separation": source_separation,
                         "paper_original_split": "not reported"}
