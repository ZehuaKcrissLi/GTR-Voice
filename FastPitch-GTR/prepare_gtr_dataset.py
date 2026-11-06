"""Build an independent, provenance-recorded FastPitch cache from a JSONL manifest.

Source audio is read and hashed, never edited. No implicit downloads or splits.
Default features follow this repository's released 100-mel/24kHz configuration;
they are not interchangeable with the StyleTTS 80-mel feature space.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def records(path, validation_utterances=None):
    from fastpitch.gtr import parse_label
    result, ids, source_hashes = [], set(), set()
    for index, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        item = json.loads(line)
        for field in ('id', 'source_path', 'source_sha256', 'label', 'utterance_id', 'text'):
            if field not in item:
                raise ValueError(f'manifest line {index} lacks {field}')
        parse_label(item['label'])
        if not isinstance(item['label'], str):
            raise ValueError('label must be a string preserving three digits')
        if item.get('eligible') is False or item.get('decode_ok') is False:
            raise ValueError(f'manifest includes ineligible source: {item["id"]}')
        if item['id'] in ids or item['source_sha256'] in source_hashes:
            raise ValueError(f'duplicate sample id or source bytes: {item["id"]}')
        if not isinstance(item['text'], str) or not item['text'].strip() or any(c in item['text'] for c in '|\r\n'):
            raise ValueError(f'text is empty or cannot be represented by a filelist: {item["id"]}')
        if validation_utterances is not None and item.get("split") != "test":
            item['split'] = 'validation' if str(item['utterance_id']).zfill(2) in validation_utterances else 'train'
        if item.get('split') == 'val':
            item['split'] = 'validation'
        if item.get('split') not in ('train', 'validation', 'test'):
            raise ValueError('supply explicit split on every row or --validation-utterances; no paper split is inferred')
        ids.add(item['id'])
        source_hashes.add(item['source_sha256'])
        result.append(item)
    if not result:
        raise ValueError('manifest is empty')
    return result



def check_source_wav(path):
    # Reuse the repository's audited RIFF/frame validator without loading model
    # packages or making another implementation of anomaly handling.
    import importlib.util
    validator = Path(__file__).resolve().parents[1] / "Acoustic-GTR" / "gtr_data.py"
    spec = importlib.util.spec_from_file_location("_gtr_audio_integrity", validator)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.check_wav_frames(Path(path))


def prepare(args):
    import librosa
    import numpy as np
    import soundfile as sf
    import torch
    from fastpitch.data_function import load_wav, mel_spectrogram, estimate_pitch
    validation = None if args.validation_utterances is None else {x.strip().zfill(2) for x in args.validation_utterances.split(',') if x.strip()}
    data = records(args.manifest, validation)
    output = Path(args.output).resolve()
    for item in data:
        source = Path(item['source_path']).resolve()
        if source == output or output in source.parents or source in output.parents:
            raise ValueError('output and source audio paths must be independent')
    if output.exists():
        raise FileExistsError('use a fresh output directory; interrupted preparations are preserved for diagnosis')
    if args.mel_fmax > args.sampling_rate / 2 or args.hop_length > args.filter_length:
        raise ValueError('invalid spectrogram configuration')
    output.mkdir(parents=True)
    for name in ('wavs', 'mels', 'pitch', 'filelists'):
        (output / name).mkdir()
    rows, used_names = [], set()
    filelists = {split: [] for split in ('train', 'validation', 'test')}
    for item in data:
        source = Path(item['source_path'])
        check_source_wav(source)
        if digest(source) != item['source_sha256']:
            raise ValueError(f'source SHA256 mismatch: {item["id"]}')
        wave, rate = sf.read(source, dtype='float32', always_2d=True)
        if wave.shape[1] != 1 or not np.isfinite(wave).all():
            raise ValueError(f'expected finite mono audio: {item["id"]}')
        wave = wave[:, 0]
        if rate != args.sampling_rate:
            if not args.resample:
                raise ValueError(f'{item["id"]}: source {rate}Hz differs from target; pass --resample for an independent derivative')
            wave = librosa.resample(wave, orig_sr=rate, target_sr=args.sampling_rate)
        if len(wave) <= args.filter_length or np.abs(wave).max() <= 0 or np.abs(wave).max() > 1:
            raise ValueError(f'source is short, silent or exceeds PCM range: {item["id"]}')
        safe_id = re.sub(r'[^A-Za-z0-9_.-]+', '_', str(Path(str(item['id'])).with_suffix('')))
        stem = f'{item["label"]}_{safe_id}'
        if stem in used_names:
            raise ValueError('sample ids collide after filename sanitization')
        used_names.add(stem)
        wav_path, mel_path, pitch_path = (output / name / (stem + suffix) for name, suffix in [('wavs', '.wav'), ('mels', '.pt'), ('pitch', '.pt')])
        sf.write(wav_path, wave, args.sampling_rate, subtype='PCM_16')
        pcm, _ = load_wav(wav_path, args.sampling_rate)
        normalized = librosa.util.normalize(pcm.astype('float32') / 32768.) * .95
        mel = mel_spectrogram(torch.from_numpy(normalized)[None], args.filter_length,
                              args.n_mel_channels, args.sampling_rate, args.hop_length,
                              args.win_length, args.mel_fmin, args.mel_fmax, center=False)[0]
        if not torch.isfinite(mel).all():
            raise ValueError(f'non-finite mel: {item["id"]}')
        torch.save(mel, mel_path)
        fields = [str(wav_path)]
        if args.with_pitch:
            pitch = estimate_pitch(str(wav_path), mel.shape[-1], sampling_rate=args.sampling_rate,
                                   hop_length=args.hop_length, frame_length=args.filter_length)
            if not torch.isfinite(pitch).all():
                raise ValueError(f'non-finite pitch: {item["id"]}')
            torch.save(pitch, pitch_path)
            fields.append(str(pitch_path))
        fields.append(item['text'])
        if args.speaker_id is not None:
            fields.append(str(args.speaker_id))
        filelists[item['split']].append('|'.join(fields))
        rows.append(dict(item, derived_wav=str(wav_path), derived_wav_sha256=digest(wav_path),
                         mel_path=str(mel_path), mel_sha256=digest(mel_path),
                         pitch_path=str(pitch_path) if args.with_pitch else None,
                         pitch_sha256=digest(pitch_path) if args.with_pitch else None,
                         source_sample_rate=rate, target_sample_rate=args.sampling_rate,
                         mel_frames=mel.shape[-1]))
        print(json.dumps({'prepared': len(rows), 'total': len(data), 'id': item['id']}), flush=True)
    (output / 'manifest.jsonl').write_text(''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in rows))
    for split, lines in filelists.items():
        if lines:
            (output / 'filelists' / (split + '.txt')).write_text('\n'.join(lines) + '\n')
    # Only training sources contribute to pitch normalization statistics.
    train_pitch = []
    if args.with_pitch:
        for row in rows:
            if row['split'] == 'train':
                pitch = torch.load(row['pitch_path'], weights_only=True)
                train_pitch.extend(pitch[pitch > 0].tolist())
    mean, std = (float(np.mean(train_pitch)), float(np.std(train_pitch))) if train_pitch else (None, None)
    receipt = {'state': 'DERIVED_DATASET_PREPARED', 'samples': len(rows),
               'source_manifest_sha256': digest(args.manifest), 'source_bytes_modified': False,
               'splits': {key: len(value) for key, value in filelists.items()},
               'split_note': 'explicit user manifest or utterance list; original paper split is not inferred',
               'features': {key: getattr(args, key) for key in ('sampling_rate', 'filter_length', 'win_length', 'hop_length', 'n_mel_channels', 'mel_fmin', 'mel_fmax')},
               'pitch_implementation': 'librosa.pyin explicit sample_rate/frame_length/hop_length; corrected original implicit 22050Hz bug',
               'pitch_training_voiced_mean': mean, 'pitch_training_voiced_std': std,
               'speaker_id': args.speaker_id, 'paper_metrics_reproduced': False,
               'manifest_sha256': digest(output / 'manifest.jsonl')}
    (output / 'preparation.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--validation-utterances', help='explicit comma-separated held-out utterance IDs; preserves existing test rows; overrides other manifest splits')
    parser.add_argument('--resample', action='store_true')
    parser.add_argument('--with-pitch', action='store_true')
    parser.add_argument('--speaker-id', type=int)
    parser.add_argument('--sampling-rate', type=int, default=24000)
    parser.add_argument('--filter-length', type=int, default=1024)
    parser.add_argument('--win-length', type=int, default=1024)
    parser.add_argument('--hop-length', type=int, default=256)
    parser.add_argument('--n-mel-channels', type=int, default=100)
    parser.add_argument('--mel-fmin', type=float, default=0.)
    parser.add_argument('--mel-fmax', type=float, default=12000.)
    args = parser.parse_args()
    prepare(args)


if __name__ == '__main__':
    main()
