"""Audit original WAV bytes and make explicit clean/excluded manifests.

No downloads, repair, transcoding or inferred experimental split. A malformed
PCM frame can decode partially in libsndfile; it is still excluded explicitly.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'Acoustic-GTR'))
from gtr_data import check_wav_frames, file_sha256, parse_label


def transcripts(path):
    result = {}
    for number, line in enumerate(Path(path).read_text(encoding='utf-8').splitlines(), 1):
        if not line.strip():
            continue
        name, text = line.split('|', 1)
        utterance = Path(name).stem.zfill(2)
        if utterance in result or not text.strip() or any(x in text for x in '\r\n|'):
            raise ValueError(f'invalid or duplicate transcript at line {number}')
        result[utterance] = text.strip()
    return result


def prepare(data_root, output_dir, transcript_path, expect_release=False):
    import numpy as np
    import soundfile as sf
    root = Path(data_root).resolve(strict=True)
    output = Path(output_dir).resolve(strict=False)
    if output.exists():
        raise FileExistsError('Use a fresh output directory; existing audit results are never overwritten')
    if output == root or root in output.parents or output in root.parents:
        raise ValueError('output and original audio paths must be independent')
    text = transcripts(transcript_path)
    paths = sorted(root.glob('*/*.wav'))
    if not paths:
        raise ValueError('No WAV files found; --data-root must contain label directories such as 010/01.wav')
    all_rows, clean, excluded = [], [], []
    for path in paths:
        if path.is_symlink() or path.parent.is_symlink() or not path.is_file():
            raise ValueError(f'Expected regular source audio, not a symlink: {path}')
        label = path.parent.name
        g, t, r = parse_label(label)
        utterance = path.stem.zfill(2)
        if utterance not in text:
            raise ValueError(f'Missing transcript for {path.name}')
        before = file_sha256(path)
        row = dict(id=path.relative_to(root).as_posix(), source_path=str(path),
                   source_sha256=before, source_bytes=path.stat().st_size,
                   label=label, utterance_id=utterance, text=text[utterance], G=g, T=t, R=r)
        try:
            header = check_wav_frames(path)
            with sf.SoundFile(path) as stream:
                if stream.channels != 1 or stream.samplerate <= 0 or stream.frames == 0:
                    raise ValueError('Expected nonempty mono source audio')
                frames = 0
                for block in stream.blocks(blocksize=65536, dtype='float32', always_2d=True):
                    if not np.isfinite(block).all():
                        raise ValueError('Decoded samples contain nonfinite values')
                    frames += len(block)
                if frames != stream.frames:
                    raise ValueError('Decoded frame count differs from WAV metadata')
            row.update(eligible=True, decode_ok=True, sample_rate=header['sample_rate'])
            clean.append(row)
        except (ValueError, RuntimeError) as error:
            row.update(eligible=False, decode_ok=False, exclusion_reason=str(error).replace(str(path), row['id']))
            excluded.append(row)
        if file_sha256(path) != before:
            raise ValueError(f'Source changed during audit: {row["id"]}')
        all_rows.append(row)
    if expect_release and (len(all_rows), len(clean), len(excluded)) != (2500, 2486, 14):
        raise ValueError(f'Unexpected release counts: {len(all_rows)}/{len(clean)}/{len(excluded)}; no audit output written')
    output.mkdir(parents=True)
    for name, rows in [('all.jsonl', all_rows), ('clean.jsonl', clean), ('excluded.jsonl', excluded)]:
        with (output / name).open('x', encoding='utf-8') as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    receipt = dict(total=len(all_rows), clean=len(clean), excluded=len(excluded),
                   source_bytes=sum(row['source_bytes'] for row in all_rows),
                   source_bytes_modified=False, experimental_split='not assigned',
                   transcript_sha256=file_sha256(transcript_path),
                   manifest_sha256={name:file_sha256(output/name) for name in ['all.jsonl','clean.jsonl','excluded.jsonl']})
    (output / 'audit.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--transcripts', default=str(Path(__file__).resolve().parents[1] / 'StyleTTS-GTR/Data/gtr_trans.txt'))
    parser.add_argument('--expect-release', action='store_true', help='Require the original 2500/2486/14 counts')
    args = parser.parse_args()
    print(json.dumps(prepare(args.data_root, args.output_dir, args.transcripts, args.expect_release), indent=2))


if __name__ == '__main__':
    main()
