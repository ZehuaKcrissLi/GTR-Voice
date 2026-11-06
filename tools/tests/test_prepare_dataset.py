import importlib.util
import json
from pathlib import Path
import struct
import subprocess
import sys
import wave
import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'prepare_dataset.py'
spec = importlib.util.spec_from_file_location('prepare_dataset', SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def source(tmp_path):
    root = tmp_path / 'audio'
    (root / '010').mkdir(parents=True)
    valid = root / '010/01.wav'
    with wave.open(str(valid), 'wb') as stream:
        stream.setnchannels(1); stream.setsampwidth(2); stream.setframerate(48000)
        stream.writeframes((np.sin(np.arange(4800)*.03)*10000).astype('<i2').tobytes())
    broken = root / '010/02.wav'
    raw = bytearray(valid.read_bytes())
    raw += b'\x01'
    struct.pack_into('<I', raw, 4, len(raw)-8)
    struct.pack_into('<I', raw, 40, len(raw)-44)
    broken.write_bytes(raw)
    texts = tmp_path / 'transcripts.txt'
    texts.write_text('01.wav|First sentence.\n02.wav|Second sentence.\n')
    return root, texts, valid, broken


def test_partial_frame_excluded_and_original_bytes_unchanged(tmp_path):
    root, texts, valid, broken = source(tmp_path)
    before = {p:p.read_bytes() for p in (valid, broken)}
    output = tmp_path / 'audit'
    result = module.prepare(root, output, texts)
    assert (result['total'], result['clean'], result['excluded']) == (2, 1, 1)
    assert result['experimental_split'] == 'not assigned'
    assert json.loads((output/'clean.jsonl').read_text())['label'] == '010'
    assert 'Incomplete PCM frame' in json.loads((output/'excluded.jsonl').read_text())['exclusion_reason']
    assert all(p.read_bytes() == content for p, content in before.items())
    with pytest.raises(FileExistsError): module.prepare(root, output, texts)


def test_release_counts_fail_before_output_and_cli_new_process(tmp_path):
    root, texts, _, _ = source(tmp_path)
    output = tmp_path/'wrong-release'
    with pytest.raises(ValueError, match='Unexpected release counts'):
        module.prepare(root, output, texts, expect_release=True)
    assert not output.exists()
    result = subprocess.run([sys.executable, str(SCRIPT), '--data-root', str(root), '--output-dir', str(tmp_path/'cli'), '--transcripts', str(texts)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)['excluded'] == 1


def test_output_parent_symlink_cannot_enter_original_audio_tree(tmp_path):
    root, texts, _, _ = source(tmp_path)
    alias = tmp_path/'source-alias'
    alias.symlink_to(root, target_is_directory=True)
    with pytest.raises(ValueError, match='independent'):
        module.prepare(root, alias/'new-audit', texts)
    assert not (root/'new-audit').exists()
