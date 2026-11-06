"""Engineering regression fixtures, not perceptual/paper-reproduction evidence."""
import importlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

MODULE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE))
from gtr_embedder import GTRStyleEncoder, PaperDescribedGTRStyleEncoder, validate_labels


def test_all_125_labels_and_whisper_support():
    labels = torch.tensor([100*g + 10*t + r for g in range(5) for t in range(1, 6)
                           for r in range(7) if (g == 0) == (r == 0)])
    assert len(labels) == 125
    for model in (GTRStyleEncoder(), PaperDescribedGTRStyleEncoder()):
        output = model(labels)
        assert output.shape == (125, 128) and torch.isfinite(output).all()
        output.square().mean().backward()
        assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0
                   for parameter in model.parameters())
    for bad in [0, 11, 230, 557, -1]:
        with pytest.raises(ValueError):
            validate_labels(torch.tensor([bad]))


def test_legacy_default_keys_and_configured_dimensions():
    model = GTRStyleEncoder()
    assert set(model.state_dict()) == {
        'g_embedding.weight', 't_embedding.weight', 'r_embedding.weight',
        'fc_sequnetial.0.weight', 'fc_sequnetial.0.bias',
        'fc_sequnetial.2.weight', 'fc_sequnetial.2.bias',
        'fc_sequnetial.4.weight', 'fc_sequnetial.4.bias'}
    assert GTRStyleEncoder(out_dim=64, style_dim=7)(torch.tensor([231])).shape == (1, 64)


def test_import_entrypoints_has_no_login_or_asset_loading():
    command = "import inference, train_teacher_forcing, sys; assert 'wandb' not in sys.modules; assert 'torch' not in sys.modules; print('IMPORT_OK')"
    result = subprocess.run([sys.executable, '-B', '-c', command], cwd=MODULE,
                            text=True, capture_output=True, check=True)
    assert result.stdout.strip() == 'IMPORT_OK'


def targets(path, ids, labels):
    # Explicitly synthetic vectors test engineering only; no audio/model quality.
    random = np.random.default_rng(4)
    np.savez(path, embeddings=random.normal(size=(len(ids), 128)).astype('float32'),
             labels=np.array(labels, dtype='int64'), ids=np.array(ids),
             source_sha256=np.array([str(index).zfill(64) for index in range(len(ids))]),
             metadata=np.array(json.dumps({'teacher_checkpoint_sha256': 'fixture-only'})))


@pytest.mark.parametrize('architecture', ['legacy', 'paper-described'])
def test_train_save_fresh_process_evaluate_and_resume(tmp_path, architecture):
    data = tmp_path / 'targets.npz'
    targets(data, ['fixture-a', 'fixture-b', 'fixture-c'], [231, 443, 10])
    output = tmp_path / 'training'
    train = [sys.executable, '-B', str(MODULE / 'gtr_pipeline.py'), 'train',
             '--embeddings', str(data), '--output-dir', str(output),
             '--architecture', architecture, '--epochs', '2', '--batch-size', '2']
    subprocess.run(train, check=True, capture_output=True, text=True)
    record = json.loads((output / 'training.json').read_text())
    assert record['epochs_completed'] == 2 and not record['paper_human_metrics_reproduced']
    checkpoint = torch.load(output / 'last.pt', weights_only=True)
    assert checkpoint['epoch'] == 2 and checkpoint['optimizer']['state']
    evaluation = [sys.executable, '-B', str(MODULE / 'gtr_pipeline.py'), 'evaluate',
                  '--embeddings', str(data), '--checkpoint', str(output / 'last.pt'),
                  '--output', str(tmp_path / 'evaluation.json')]
    subprocess.run(evaluation, check=True, capture_output=True, text=True)
    result = json.loads((tmp_path / 'evaluation.json').read_text())
    assert result['samples'] == 3 and np.isfinite(result['mse'])
    assert result['contains_training_sources'] is True
    assert result['evaluation_scope'] == 'CONTAINS_TRAINING_SOURCES'
    train[train.index('--epochs') + 1] = '3'
    subprocess.run(train + ['--resume', str(output / 'last.pt')], check=True,
                   capture_output=True, text=True)
    assert torch.load(output / 'last.pt', weights_only=True)['epoch'] == 3


def test_overlapping_validation_is_rejected(tmp_path):
    data = tmp_path / 'targets.npz'
    targets(data, ['fixture-a'], [231])
    result = subprocess.run([sys.executable, '-B', str(MODULE / 'gtr_pipeline.py'), 'train',
                             '--embeddings', str(data), '--validation-embeddings', str(data),
                             '--output-dir', str(tmp_path / 'out'), '--epochs', '1'],
                            text=True, capture_output=True)
    assert result.returncode and 'overlap' in result.stderr


def test_malformed_pcm_frame_is_rejected_without_editing_original(tmp_path):
    import struct
    import wave
    from gtr_pipeline import check_source_wav
    source = tmp_path / 'broken.wav'
    with wave.open(str(source), 'wb') as stream:
        stream.setnchannels(1); stream.setsampwidth(2); stream.setframerate(24000)
        stream.writeframes(b'\x00\x01' * 2400)
    raw = bytearray(source.read_bytes()) + b'\x01'
    struct.pack_into('<I', raw, 4, len(raw) - 8)
    struct.pack_into('<I', raw, 40, len(raw) - 44)
    source.write_bytes(raw)
    before = source.read_bytes()
    with pytest.raises(ValueError, match='Incomplete PCM frame'):
        check_source_wav(source)
    assert source.read_bytes() == before
