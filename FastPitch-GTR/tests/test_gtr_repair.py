"""CPU engineering regressions; random model fixtures are not paper results."""
from pathlib import Path
import sys
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fastpitch.gtr import from_path, parse_label
from fastpitch.model import FastPitch


def tiny_config():
    return dict(n_mel_channels=4, n_symbols=10, padding_idx=0,
                symbols_embedding_dim=8, in_fft_n_layers=1, in_fft_n_heads=1,
                in_fft_d_head=8, in_fft_conv1d_kernel_size=3,
                in_fft_conv1d_filter_size=16, in_fft_output_size=8,
                p_in_fft_dropout=0., p_in_fft_dropatt=0., p_in_fft_dropemb=0.,
                out_fft_n_layers=1, out_fft_n_heads=1, out_fft_d_head=8,
                out_fft_conv1d_kernel_size=3, out_fft_conv1d_filter_size=16,
                out_fft_output_size=8, p_out_fft_dropout=0.,
                p_out_fft_dropatt=0., p_out_fft_dropemb=0.,
                dur_predictor_kernel_size=3, dur_predictor_filter_size=8,
                p_dur_predictor_dropout=0., dur_predictor_n_layers=1,
                pitch_predictor_kernel_size=3, pitch_predictor_filter_size=8,
                p_pitch_predictor_dropout=0., pitch_predictor_n_layers=1,
                pitch_embedding_kernel_size=3, energy_conditioning=False,
                energy_predictor_kernel_size=3, energy_predictor_filter_size=8,
                p_energy_predictor_dropout=0., energy_predictor_n_layers=1,
                energy_embedding_kernel_size=3, n_speakers=1,
                speaker_emb_weight=1., gtr_emb=True)


def test_labels_from_original_tree_and_cached_prefix():
    assert from_path('/data/gtr_125/010/04.wav') == (0, 1, 0)
    assert from_path('/data/mels/231_04.pt') == (2, 3, 1)
    for value in ['000', '230', '011', '557', '23', 'abc']:
        with pytest.raises(ValueError):
            parse_label(value)
    with pytest.raises(ValueError):
        from_path('/data/unlabelled.wav')


def test_cpu_real_forward_backward_save_reload_and_conditioned_inference(tmp_path):
    torch.manual_seed(7)
    model = FastPitch(**tiny_config())
    text = torch.tensor([[1, 2, 3], [4, 5, 0]])
    lengths = torch.tensor([3, 2])
    mel = torch.randn(2, 4, 12)
    mel_lengths = torch.tensor([12, 10])
    batch = [text, lengths, mel, mel_lengths, torch.ones(2, 1, 12),
             torch.ones(2, 12), None, torch.ones(2, 12, 3) / 3,
             ['fixture-a', 'fixture-b'], torch.tensor([2, 4]),
             torch.tensor([3, 4]), torch.tensor([1, 3])]
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    result = model(batch)
    loss = result[0].square().mean()
    loss.backward()
    assert torch.isfinite(loss) and model.g_emb.weight.grad.abs().sum() > 0
    optimizer.step()
    model.eval()
    durations = torch.tensor([[4., 4., 4.], [5., 5., 0.]])
    with torch.inference_mode():
        expected = model.infer(text, dur_tgt=durations, g=2, t=3, r=1)[0]
        changed = model.infer(text, dur_tgt=durations, g=4, t=3, r=1)[0]
    assert expected.shape == (2, 4, 12) and not torch.equal(expected, changed)
    path = tmp_path / 'engineering-fixture.pt'
    torch.save(model.state_dict(), path)
    fresh = FastPitch(**tiny_config()).eval()
    fresh.load_state_dict(torch.load(path, weights_only=True), strict=True)
    with torch.inference_mode():
        assert torch.equal(expected, fresh.infer(text, dur_tgt=durations, g=2, t=3, r=1)[0])


def test_illegal_control_rejected_before_embedding_lookup():
    model = FastPitch(**tiny_config())
    with pytest.raises(ValueError):
        model.infer(torch.tensor([[1, 2]]), g=2, t=3, r=0)
    with pytest.raises(ValueError, match='integers'):
        model.infer(torch.tensor([[1, 2]]), g=2.5, t=3, r=1)


@pytest.mark.parametrize('module_name', ['fastpitch.data_function', 'hifigan.data_function'])
def test_multiple_mel_configurations_do_not_share_incompatible_cache(module_name):
    import importlib
    mel_spectrogram = importlib.import_module(module_name).mel_spectrogram
    wave = torch.zeros(1, 4096)
    first = mel_spectrogram(wave, 512, 40, 16000, 128, 512, 0, 8000)
    second = mel_spectrogram(wave, 1024, 100, 24000, 256, 1024, 0, 12000)
    again = mel_spectrogram(wave, 512, 40, 16000, 128, 512, 0, 8000)
    assert first.shape[1] == 40 and second.shape[1] == 100
    assert torch.equal(first, again)


def test_derived_preparation_preserves_source_and_matches_pitch_grid(tmp_path):
    import hashlib
    import json
    import subprocess
    import numpy as np
    import soundfile as sf
    module = Path(__file__).resolve().parents[1]
    source = tmp_path / 'original' / '010' / '04.wav'
    source.parent.mkdir(parents=True)
    wave = .1 * np.sin(2 * np.pi * 220 * np.arange(16000) / 16000)
    sf.write(source, wave, 16000, subtype='PCM_16')
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    manifest = tmp_path / 'input.jsonl'
    manifest.write_text(json.dumps({'id': '010-04', 'source_path': str(source),
                                   'source_sha256': source_sha, 'label': '010',
                                   'utterance_id': '04', 'text': 'A test.',
                                   'split': 'train'}) + '\n')
    output = tmp_path / 'derived'
    subprocess.run([sys.executable, '-B', str(module / 'prepare_gtr_dataset.py'),
                    '--manifest', str(manifest), '--output', str(output), '--resample',
                    '--with-pitch'], cwd=module, text=True, capture_output=True, check=True)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == source_sha
    row = json.loads((output / 'manifest.jsonl').read_text())
    mel = torch.load(row['mel_path'], weights_only=True)
    pitch = torch.load(row['pitch_path'], weights_only=True)
    assert mel.shape[0] == 100 and mel.shape[-1] == pitch.shape[-1]
    assert torch.isfinite(mel).all() and torch.isfinite(pitch).all()
    assert float(pitch[pitch > 0].median()) == pytest.approx(220, abs=5)
    receipt = json.loads((output / 'preparation.json').read_text())
    assert receipt['source_bytes_modified'] is False
    assert receipt['samples'] == 1 and receipt['features']['sampling_rate'] == 24000
    assert receipt['pitch_training_voiced_mean'] == pytest.approx(220, abs=5)


def test_dataset_split_must_be_explicit(tmp_path):
    import json
    from prepare_gtr_dataset import records
    manifest = tmp_path / 'manifest.jsonl'
    manifest.write_text(json.dumps({'id': 'x', 'source_path': '/unused',
                                   'source_sha256': 'fixture-only', 'label': '231',
                                   'utterance_id': '04', 'text': 'Test.'}) + '\n')
    with pytest.raises(ValueError, match='explicit split'):
        records(manifest)
    assert records(manifest, {'04'})[0]['split'] == 'validation'
    row = json.loads(manifest.read_text()); row['split'] = 'test'
    manifest.write_text(json.dumps(row) + '\n')
    assert records(manifest, {'04'})[0]['split'] == 'test'


def test_cli_training_save_resume_and_fresh_process_inference(tmp_path):
    """Exercise the real CLI/optimizer/checkpointer, using labelled fixtures only."""
    import json
    import subprocess
    import numpy as np
    import soundfile as sf
    module = Path(__file__).resolve().parents[1]
    for name in ('wavs', 'mels', 'pitch'):
        (tmp_path / name).mkdir()
    for index, label in enumerate(('231', '443')):
        stem = label + '_engineering-fixture'
        sf.write(tmp_path / 'wavs' / (stem + '.wav'), np.zeros(4096), 24000)
        torch.save(torch.randn(4, 12), tmp_path / 'mels' / (stem + '.pt'))
        torch.save(torch.full((1, 12), 220.), tmp_path / 'pitch' / (stem + '.pt'))
        (tmp_path / ('train.txt' if index == 0 else 'validation.txt')).write_text(
            str(tmp_path / 'wavs' / (stem + '.wav')) + '|' + str(tmp_path / 'pitch' / (stem + '.pt')) + '|A.\n')
    output = tmp_path / 'train-output'
    command = [sys.executable, '-B', str(module / 'train.py'), '-o', str(output),
               '--dataset-path', str(tmp_path), '--training-files', str(tmp_path / 'train.txt'),
               '--validation-files', str(tmp_path / 'validation.txt'), '--epochs', '1',
               '--epochs-per-checkpoint', '1', '--batch-size', '1', '--num-workers', '0',
               '--trainloader-repeats', '1', '--optimizer', 'adam', '--learning-rate', '.001',
               '--warmup-steps', '1', '--load-mel-from-disk', '--load-pitch-from-disk',
               '--pitch-mean', '220', '--pitch-std', '10', '--p-arpabet', '0',
               '--symbols-embedding-dim', '8', '--in-fft-n-layers', '1', '--in-fft-d-head', '8',
               '--in-fft-conv1d-filter-size', '16', '--in-fft-output-size', '8',
               '--out-fft-n-layers', '1', '--out-fft-d-head', '8',
               '--out-fft-conv1d-filter-size', '16', '--out-fft-output-size', '8',
               '--dur-predictor-filter-size', '8', '--dur-predictor-n-layers', '1',
               '--pitch-predictor-filter-size', '8', '--pitch-predictor-n-layers', '1',
               '--n-mel-channels', '4', '--sampling-rate', '24000', '--mel-fmax', '12000']
    subprocess.run(command, cwd=module, check=True, text=True, capture_output=True)
    first = torch.load(output / 'FastPitch_checkpoint_1.pt', weights_only=True)
    assert first['epoch'] == 1 and first['optimizer']['state'] and first['config']['gtr_emb']
    assert first['iteration'] == 2
    command[command.index('--epochs') + 1] = '2'
    subprocess.run(command + ['--resume'], cwd=module, check=True, text=True, capture_output=True)
    checkpoint = output / 'FastPitch_checkpoint_2.pt'
    assert torch.load(checkpoint, weights_only=True)['epoch'] == 2
    assert torch.load(checkpoint, weights_only=True)['iteration'] == 3
    phrases = tmp_path / 'phrases.txt'
    phrases.write_text('A.\n')
    destination = tmp_path / 'inference'
    result = subprocess.run([sys.executable, '-B', str(module / 'inference.py'),
                             '--fastpitch', str(checkpoint), '--input', str(phrases),
                             '--output', str(destination), '--save-mels', '--affinity', 'disabled',
                             '--batch-size', '1', '--g', '2', '--t', '3', '--r', '1', '--pace', '.05'],
                            cwd=module, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr + result.stdout
    mel = np.load(destination / 'mel_0.npy', allow_pickle=False)
    assert mel.ndim == 2 and mel.shape[1] == 4 and mel.shape[0] > 0 and np.isfinite(mel).all()


def test_missing_cmudict_fails_without_network_or_writes(tmp_path, monkeypatch):
    from common.text.cmudict import CMUDict
    import urllib.request
    monkeypatch.setattr(urllib.request, 'urlretrieve', lambda *args, **kwargs: pytest.fail('implicit network attempt'))
    with pytest.raises(FileNotFoundError, match='No download was attempted'):
        CMUDict().initialize(tmp_path / 'missing-dict', None)
    assert list(tmp_path.iterdir()) == []
    dictionary = tmp_path / 'existing-dict'
    dictionary.write_text('TEST  T EH1 S T\n')
    model = CMUDict(dictionary, None)
    assert model.lookup('test') == ['T EH1 S T']


@pytest.mark.parametrize('module_name', ['prepare_gtr_dataset', 'vocode_bigvgan'])
def test_malformed_pcm_frame_is_rejected_without_editing_original(tmp_path, module_name):
    import importlib
    import struct
    import wave
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
        importlib.import_module(module_name).check_source_wav(source)
    assert source.read_bytes() == before
