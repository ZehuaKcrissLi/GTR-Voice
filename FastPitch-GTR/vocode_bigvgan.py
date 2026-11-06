"""Explicit local-source BigVGAN waveform export; no model/code downloads.

Accepts FastPitch's saved (1, channels, frames) mel or a hash-verified source
waveform for a ground-truth vocoder probe. Ground-truth reconstruction is not
GTR-conditioned TTS and is recorded separately in the output receipt.
"""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()



def check_source_wav(path):
    # Reuse the repository's audited RIFF/frame validator without loading model
    # packages or making another implementation of anomaly handling.
    import importlib.util
    validator = Path(__file__).resolve().parents[1] / "Acoustic-GTR" / "gtr_data.py"
    spec = importlib.util.spec_from_file_location("_gtr_audio_integrity", validator)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.check_wav_frames(Path(path))


def export(args):
    import numpy as np
    import soundfile as sf
    import torch
    import librosa
    source = Path(args.source_dir).resolve()
    config_path = Path(args.config).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    output = Path(args.output)
    if output.exists() or Path(str(output) + '.json').exists():
        raise FileExistsError('use a fresh waveform/receipt output path')
    if not (source / 'bigvgan.py').is_file():
        raise FileNotFoundError(source / 'bigvgan.py')
    # BigVGAN's original code uses absolute local imports. Refuse to borrow an
    # unrelated already-imported module with the same generic name.
    for name in ('bigvgan', 'activations', 'utils', 'env', 'meldataset', 'alias_free_activation'):
        module = sys.modules.get(name)
        if module is not None:
            file = getattr(module, '__file__', None)
            if file is None or source not in Path(file).resolve().parents:
                raise RuntimeError(f'conflicting loaded module {name}; run this entry point in a fresh process')
    config = json.loads(config_path.read_text())
    required = ('num_mels', 'sampling_rate', 'n_fft', 'hop_size', 'win_size', 'fmin', 'fmax')
    for key in required:
        if key not in config:
            raise ValueError(f'vocoder config lacks {key}')
    if int(np.prod(config['upsample_rates'])) != config['hop_size']:
        raise ValueError('vocoder upsampling ratio and mel hop differ')
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA is unavailable')
    torch.manual_seed(args.seed)
    input_metadata = {}
    if args.audio:
        path = Path(args.audio)
        check_source_wav(path)
        if not args.source_sha256 or digest(path) != args.source_sha256:
            raise ValueError('--audio requires its matching --source-sha256')
        wave, sample_rate = sf.read(path, dtype='float32', always_2d=True)
        if wave.shape[1] != 1 or not np.isfinite(wave).all():
            raise ValueError('expected finite mono source')
        wave = wave[:, 0]
        original_samples = len(wave)
        if sample_rate != config['sampling_rate']:
            if not args.resample:
                raise ValueError('source/vocoder sample rates differ; explicitly permit a derivative with --resample')
            wave = librosa.resample(wave, orig_sr=sample_rate, target_sr=config['sampling_rate'])
        if args.max_seconds is not None:
            if args.max_seconds <= 0:
                raise ValueError('--max-seconds must be positive')
            wave = wave[:int(args.max_seconds * config['sampling_rate'])]
        wave = librosa.util.normalize(wave) * .95
        from fastpitch.data_function import mel_spectrogram
        mel = mel_spectrogram(torch.from_numpy(wave)[None], config['n_fft'], config['num_mels'],
                              config['sampling_rate'], config['hop_size'], config['win_size'],
                              config['fmin'], config['fmax'], center=False)
        input_metadata = {'mode': 'GROUND_TRUTH_VOCODER_RECONSTRUCTION', 'source_path': str(path),
                          'source_sha256': args.source_sha256, 'source_sample_rate': sample_rate,
                          'source_samples': original_samples, 'processed_samples': len(wave),
                          'max_seconds': args.max_seconds, 'original_bytes_modified': False}
    else:
        path = Path(args.mel)
        array = np.load(path, allow_pickle=False)
        if array.ndim != 3 or array.shape[0] != 1:
            raise ValueError('mel array must have shape (1, channels, frames), as exported by inference_bigv.py')
        mel = torch.from_numpy(array.astype('float32'))
        input_metadata = {'mode': 'FASTPITCH_MEL_TO_WAVEFORM', 'mel_path': str(path), 'mel_sha256': digest(path)}
    if mel.shape[1] != config['num_mels'] or not torch.isfinite(mel).all() or mel.shape[-1] < 1:
        raise ValueError('mel channels/values differ from the vocoder configuration')
    # Inputs have already been validated; load only the caller-supplied local
    # implementation and tensor-only weights. Never use from_pretrained().
    sys.path.insert(0, str(source))
    implementation = importlib.import_module('bigvgan')
    configuration = importlib.import_module('env').AttrDict(config)
    vocoder = implementation.BigVGAN(configuration, use_cuda_kernel=False)
    state = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    vocoder.load_state_dict(state['generator'], strict=True)
    vocoder.eval().requires_grad_(False).to(device)
    vocoder.remove_weight_norm()
    with torch.inference_mode():
        waveform = vocoder(mel.to(device)).squeeze().cpu().numpy()
    if waveform.ndim != 1 or not len(waveform) or not np.isfinite(waveform).all() or np.max(np.abs(waveform)) <= 0:
        raise RuntimeError('invalid/silent vocoder output')
    output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output, waveform, config['sampling_rate'], subtype='FLOAT')
    sources = {str(file.relative_to(source)): digest(file) for file in sorted(source.rglob('*.py'))}
    receipt = dict(input_metadata, state='STRICT_LOAD_REAL_BIGVGAN_WAVEFORM_EXPORTED',
                   source_dir=str(source), source_code_sha256=sources,
                   checkpoint_sha256=digest(checkpoint_path), config_sha256=digest(config_path),
                   mel_shape=list(mel.shape), mel_float32_sha256=hashlib.sha256(mel.numpy().tobytes()).hexdigest(),
                   output_sha256=digest(output), pcm_float32_sha256=hashlib.sha256(waveform.astype('float32').tobytes()).hexdigest(),
                   samples=len(waveform), sample_rate=config['sampling_rate'],
                   waveform_peak=float(np.max(np.abs(waveform))), paper_metrics_reproduced=False)
    Path(str(output) + '.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({'output': str(output), 'samples': len(waveform), 'sample_rate': config['sampling_rate'], 'mode': receipt['mode']}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('--source-dir', '--config', '--checkpoint', '--output'):
        parser.add_argument(flag, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--mel')
    group.add_argument('--audio')
    parser.add_argument('--source-sha256')
    parser.add_argument('--resample', action='store_true')
    parser.add_argument('--max-seconds', type=float)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--seed', type=int, default=1234)
    export(parser.parse_args())


if __name__ == '__main__':
    main()
