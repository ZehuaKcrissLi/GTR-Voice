# *****************************************************************************
#  Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#      * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#      * Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#      * Neither the name of the NVIDIA CORPORATION nor the
#        names of its contributors may be used to endorse or promote products
#        derived from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# *****************************************************************************

import functools
import json
import pdb
import re
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
from scipy.stats import betabinom
import soundfile as sf
import torchaudio

import torch.utils.data
from librosa.util import normalize
from scipy.io.wavfile import read
from librosa.filters import mel as librosa_mel_fn
import pathlib
from tqdm import tqdm

import common.layers as layers
from fastpitch.gtr import from_path as gtr_from_path
from common.utils import load_wav_to_torch, load_filepaths_and_text, to_gpu


class BetaBinomialInterpolator:
    """Interpolates alignment prior matrices to save computation.

    Calculating beta-binomial priors is costly. Instead cache popular sizes
    and use img interpolation to get priors faster.
    """
    def __init__(self, round_mel_len_to=100, round_text_len_to=20):
        self.round_mel_len_to = round_mel_len_to
        self.round_text_len_to = round_text_len_to
        self.bank = functools.lru_cache(beta_binomial_prior_distribution)

    def round(self, val, to):
        return max(1, int(np.round((val + 1) / to))) * to

    def __call__(self, w, h):
        bw = self.round(w, to=self.round_mel_len_to)
        bh = self.round(h, to=self.round_text_len_to)
        ret = ndimage.zoom(self.bank(bw, bh).T, zoom=(w / bw, h / bh), order=1)
        assert ret.shape[0] == w, ret.shape
        assert ret.shape[1] == h, ret.shape
        return ret


def beta_binomial_prior_distribution(phoneme_count, mel_count, scaling=1.0):
    P = phoneme_count
    M = mel_count
    x = np.arange(0, P)
    mel_text_probs = []
    for i in range(1, M+1):
        a, b = scaling * i, scaling * (M + 1 - i)
        rv = betabinom(P, a, b)
        mel_i_prob = rv.pmf(x)
        mel_text_probs.append(mel_i_prob)
    return torch.tensor(np.array(mel_text_probs))


def estimate_pitch(wav, mel_len, method='pyin', normalize_mean=None,
                   normalize_std=None, n_formants=1, sampling_rate=24000,
                   hop_length=256, frame_length=1024):

    if type(normalize_mean) is float or type(normalize_mean) is list:
        normalize_mean = torch.tensor(normalize_mean)

    if type(normalize_std) is float or type(normalize_std) is list:
        normalize_std = torch.tensor(normalize_std)

    if method == 'pyin':
        snd, sr = librosa.load(wav, sr=sampling_rate)

        pitch_mel, voiced_flag, voiced_probs = librosa.pyin(
            snd, fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'), sr=sr,
            frame_length=frame_length, hop_length=hop_length)
        if abs(mel_len - pitch_mel.shape[0]) > 1:
            raise ValueError('pitch/mel frame counts differ by more than one')

        pitch_mel = np.where(np.isnan(pitch_mel), 0.0, pitch_mel)
        pitch_mel = torch.from_numpy(pitch_mel).unsqueeze(0)
        pitch_mel = pitch_mel[..., :mel_len]
        pitch_mel = F.pad(pitch_mel, (0, mel_len - pitch_mel.size(1)))

        if n_formants > 1:
            raise NotImplementedError

    else:
        raise ValueError

    pitch_mel = pitch_mel.float()

    if normalize_mean is not None:
        assert normalize_std is not None
        pitch_mel = normalize_pitch(pitch_mel, normalize_mean, normalize_std)

    return pitch_mel


def normalize_pitch(pitch, mean, std):
    zeros = (pitch == 0.0)
    pitch -= mean[:, None]
    pitch /= std[:, None]
    pitch[zeros] = 0.0
    return pitch


def load_wav(full_path, sr_target):
    sampling_rate, data = read(full_path)
    if sampling_rate != sr_target:
        raise RuntimeError("Sampling rate of the file {} is {} Hz, but the model requires {} Hz".
              format(full_path, sampling_rate, sr_target))
    return data, sampling_rate


def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)


def dynamic_range_decompression(x, C=1):
    return np.exp(x) / C


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output


mel_basis = {}
hann_window = {}
def mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
    if torch.min(y) < -1.:
        print('min value is ', torch.min(y))
    if torch.max(y) > 1.:
        print('max value is ', torch.max(y))

    global mel_basis, hann_window
    key = (sampling_rate, n_fft, num_mels, fmin, fmax, str(y.device), str(y.dtype))
    if key not in mel_basis:
        mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        mel_basis[key] = torch.from_numpy(mel).float().to(y.device)
    window_key = (win_size, str(y.device), str(y.dtype))
    if window_key not in hann_window:
        hann_window[window_key] = torch.hann_window(win_size, device=y.device, dtype=y.dtype)

    y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
    y = y.squeeze(1)

    # complex tensor as default, then use view_as_real for future pytorch compatibility
    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[window_key],
                      center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex=True)
    spec = torch.view_as_real(spec)
    spec = torch.sqrt(spec.pow(2).sum(-1)+(1e-9))

    spec = torch.matmul(mel_basis[key], spec)
    spec = spectral_normalize_torch(spec)

    return spec


class TTSDataset(torch.utils.data.Dataset):
    """
        1) loads audio,text pairs
        2) normalizes text and converts them to sequences of one-hot vectors
        3) computes mel-spectrograms from audio files.
    """
    def __init__(self,
                 dataset_path,
                 audiopaths_and_text,
                 text_cleaners,
                 n_mel_channels,
                 symbol_set='english_basic',
                 p_arpabet=1.0,
                 n_speakers=1,
                 load_mel_from_disk=True,
                 load_pitch_from_disk=True,
                 pitch_mean=214.72203,  # LJSpeech defaults
                 pitch_std=65.72038,
                 max_wav_value=None,
                 sampling_rate=None,
                 filter_length=None,
                 hop_length=None,
                 win_length=None,
                 mel_fmin=None,
                 mel_fmax=None,
                 prepend_space_to_text=False,
                 append_space_to_text=False,
                 pitch_online_dir=None,
                 betabinomial_online_dir=None,
                 use_betabinomial_interpolator=True,
                 pitch_online_method='pyin',
                 gtr_emb=True,
                 **ignored):

        # Expect a list of filenames
        if type(audiopaths_and_text) is str:
            audiopaths_and_text = [audiopaths_and_text]

        self.dataset_path = dataset_path
        self.audiopaths_and_text = load_filepaths_and_text(
            dataset_path, audiopaths_and_text,
            has_speakers=(n_speakers > 1))
        self.load_mel_from_disk = load_mel_from_disk
        self.gtr_emb = gtr_emb
        self.mel_config = (filter_length, n_mel_channels, sampling_rate, hop_length, win_length, mel_fmin, mel_fmax)
        if not load_mel_from_disk:
            self.max_wav_value = max_wav_value
            self.sampling_rate = sampling_rate
            # self.stft = layers.TacotronSTFT(
            #     filter_length, hop_length, win_length,
            #     n_mel_channels, sampling_rate, mel_fmin, mel_fmax)
            self.to_mel = torchaudio.transforms.MelSpectrogram(
                n_mels=n_mel_channels, n_fft=filter_length, win_length=win_length,
                hop_length=hop_length, f_min=mel_fmin, f_max=mel_fmax)

        self.load_pitch_from_disk = load_pitch_from_disk

        self.prepend_space_to_text = prepend_space_to_text
        self.append_space_to_text = append_space_to_text

        assert p_arpabet == 0.0 or p_arpabet == 1.0, (
            'Only 0.0 and 1.0 p_arpabet is currently supported. '
            'Variable probability breaks caching of betabinomial matrices.')

        from common.text.text_processing import get_text_processing
        self.tp = get_text_processing(symbol_set, text_cleaners, p_arpabet)
        self.n_speakers = n_speakers
        self.pitch_tmp_dir = pitch_online_dir
        self.f0_method = pitch_online_method
        self.betabinomial_tmp_dir = betabinomial_online_dir
        self.use_betabinomial_interpolator = use_betabinomial_interpolator

        if use_betabinomial_interpolator:
            self.betabinomial_interpolator = BetaBinomialInterpolator()

        expected_columns = (2 + int(load_pitch_from_disk) + (n_speakers > 1))

        assert not (load_pitch_from_disk and self.pitch_tmp_dir is not None)

        if len(self.audiopaths_and_text[0]) < expected_columns:
            raise ValueError(f'Expected {expected_columns} columns in audiopaths file. '
                             'The format is <mel_or_wav>|[<pitch>|]<text>[|<speaker_id>]')

        if len(self.audiopaths_and_text[0]) > expected_columns:
            print('WARNING: Audiopaths file has more columns than expected')

        to_tensor = lambda x: torch.Tensor([x]) if type(x) is float else x
        self.pitch_mean = to_tensor(pitch_mean)
        self.pitch_std = to_tensor(pitch_std)

    def __getitem__(self, index):
        # Separate filename and text
        if self.n_speakers > 1:
            audiopath, *extra, text, speaker = self.audiopaths_and_text[index]
            speaker = int(speaker)
        else:
            audiopath, *extra, text = self.audiopaths_and_text[index]
            speaker = None

        if self.gtr_emb:
            g, t, r = gtr_from_path(audiopath)
        else:
            g = t = r = None

        mel = self.get_mel(audiopath)
        text = self.get_text(text)
        pitch = self.get_pitch(index, mel.size(-1))
        energy = torch.norm(mel.float(), dim=0, p=2)
        attn_prior = self.get_prior(index, mel.shape[1], text.shape[0])

        assert pitch.size(-1) == mel.size(-1)

        # No higher formants?
        if len(pitch.size()) == 1:
            pitch = pitch[None, :]

        return (text, mel, len(text), pitch, energy, speaker, attn_prior,
                audiopath, g, t, r)

    def __len__(self):
        return len(self.audiopaths_and_text)

    def get_mel(self, filename):
        if not self.load_mel_from_disk:
            # audio, sampling_rate = load_wav_to_torch(filename)
            # if sampling_rate != self.stft.sampling_rate:
            #     raise ValueError("{} SR doesn't match target {} SR".format(
            #         sampling_rate, self.stft.sampling_rate))
            # audio_norm = audio / self.max_wav_value
            # audio_norm = audio_norm.unsqueeze(0)
            # audio_norm = torch.autograd.Variable(audio_norm,
            #                                      requires_grad=False)
            # melspec = self.stft.mel_spectrogram(audio_norm)
            # melspec = torch.squeeze(melspec, 0)

            # audio, sampling_rate = sf.read(filename)
            # audio = torch.from_numpy(audio).float()
            # mel = self.to_mel(audio)
            # mean = -4
            # std = 4
            # melspec = (torch.log(1e-5 + mel.unsqueeze(0)) - mean) / std
            # melspec = torch.squeeze(melspec, 0)
            # print(melspec.shape)

            #bigvgan
            audio, sampling_rate = load_wav(filename, self.sampling_rate)
            audio = audio / self.max_wav_value
            audio = normalize(audio) * 0.95

            audio = torch.FloatTensor(audio)
            audio = audio.unsqueeze(0)

            # if (audio.size(1) % 256) != 0:
            #     audio = audio[:, :-(audio.size(1) % 256)]

            melspec = mel_spectrogram(audio, *self.mel_config, center=False)

            melspec = torch.squeeze(melspec, 0)

        else:
            mel_path = Path(filename).with_suffix('.pt')
            mel_path = Path(*('mels' if part == 'wavs' else part for part in mel_path.parts))
            melspec = torch.load(mel_path, weights_only=True)
            # melspec = torch.load(filename)

            # assert melspec.size(0) == self.stft.n_mel_channels, (
            #     'Mel dimension mismatch: given {}, expected {}'.format(
            #         melspec.size(0), self.stft.n_mel_channels))
        return melspec

    def get_text(self, text):
        text = self.tp.encode_text(text)
        space = [self.tp.encode_text("A A")[1]]

        if self.prepend_space_to_text:
            text = space + text

        if self.append_space_to_text:
            text = text + space

        return torch.LongTensor(text)

    def get_prior(self, index, mel_len, text_len):

        if self.use_betabinomial_interpolator:
            return torch.from_numpy(self.betabinomial_interpolator(mel_len,
                                                                   text_len))

        if self.betabinomial_tmp_dir is not None:
            audiopath, *_ = self.audiopaths_and_text[index]
            fname = Path(audiopath).relative_to(self.dataset_path)
            fname = fname.with_suffix('.pt')
            cached_fpath = Path(self.betabinomial_tmp_dir, fname)

            if cached_fpath.is_file():
                return torch.load(cached_fpath, weights_only=True)

        attn_prior = beta_binomial_prior_distribution(text_len, mel_len)

        if self.betabinomial_tmp_dir is not None:
            cached_fpath.parent.mkdir(parents=True, exist_ok=True)
            torch.save(attn_prior, cached_fpath)

        return attn_prior

    def get_pitch(self, index, mel_len=None):
        audiopath, *fields = self.audiopaths_and_text[index]

        if self.n_speakers > 1:
            spk = int(fields[-1])
        else:
            spk = 0

        if self.load_pitch_from_disk:
            pitchpath = fields[0]
            pitch = torch.load(pitchpath, weights_only=True)
            if self.pitch_mean is not None:
                assert self.pitch_std is not None
                pitch = normalize_pitch(pitch, self.pitch_mean, self.pitch_std)
            return pitch

        if self.pitch_tmp_dir is not None:
            fname = Path(audiopath).relative_to(self.dataset_path)
            fname_method = fname.with_suffix('.pt')
            cached_fpath = Path(self.pitch_tmp_dir, fname_method)
            if cached_fpath.is_file():
                return torch.load(cached_fpath, weights_only=True)

        # No luck so far - calculate
        wav = audiopath
        if not wav.endswith('.wav'):
            wav = re.sub('/mels/', '/wavs/', wav)
            wav = re.sub('.pt$', '.wav', wav)

        pitch_mel = estimate_pitch(wav, mel_len, self.f0_method,
                                   self.pitch_mean, self.pitch_std,
                                   sampling_rate=self.mel_config[2],
                                   hop_length=self.mel_config[3],
                                   frame_length=self.mel_config[0])

        if self.pitch_tmp_dir is not None and not cached_fpath.is_file():
            cached_fpath.parent.mkdir(parents=True, exist_ok=True)
            torch.save(pitch_mel, cached_fpath)

        return pitch_mel


def ensure_disjoint(*tts_datasets):
    paths = [set(list(zip(*d.audiopaths_and_text))[0]) for d in tts_datasets]
    assert sum(len(p) for p in paths) == len(set().union(*paths)), (
        "Your datasets (train, val) are not disjoint. "
        "Review filelists and restart training."
    )


class TTSCollate:
    """Zero-pads model inputs and targets based on number of frames per step"""

    def __call__(self, batch):
        """Collate training batch from normalized text and mel-spec"""
        # Right zero-pad all one-hot text sequences to max input length
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[0]) for x in batch]),
            dim=0, descending=True)
        max_input_len = input_lengths[0]

        text_padded = torch.LongTensor(len(batch), max_input_len)
        text_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            text = batch[ids_sorted_decreasing[i]][0]
            text_padded[i, :text.size(0)] = text

        # Right zero-pad mel-spec
        num_mels = batch[0][1].size(0)
        max_target_len = max([x[1].size(1) for x in batch])

        # Include mel padded and gate padded
        mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
        mel_padded.zero_()
        output_lengths = torch.LongTensor(len(batch))
        for i in range(len(ids_sorted_decreasing)):
            mel = batch[ids_sorted_decreasing[i]][1]
            mel_padded[i, :, :mel.size(1)] = mel
            output_lengths[i] = mel.size(1)

        n_formants = batch[0][3].shape[0]
        pitch_padded = torch.zeros(mel_padded.size(0), n_formants,
                                   mel_padded.size(2), dtype=batch[0][3].dtype)
        energy_padded = torch.zeros_like(pitch_padded[:, 0, :])

        for i in range(len(ids_sorted_decreasing)):
            pitch = batch[ids_sorted_decreasing[i]][3]
            energy = batch[ids_sorted_decreasing[i]][4]
            pitch_padded[i, :, :pitch.shape[1]] = pitch
            energy_padded[i, :energy.shape[0]] = energy

        if batch[0][5] is not None:
            speaker = torch.zeros_like(input_lengths)
            for i in range(len(ids_sorted_decreasing)):
                speaker[i] = batch[ids_sorted_decreasing[i]][5]
        else:
            speaker = None

        attn_prior_padded = torch.zeros(len(batch), max_target_len,
                                        max_input_len)
        attn_prior_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            prior = batch[ids_sorted_decreasing[i]][6]
            attn_prior_padded[i, :prior.size(0), :prior.size(1)] = prior

        # Count number of items - characters in text
        len_x = [x[2] for x in batch]
        len_x = torch.Tensor(len_x)

        audiopaths = [batch[i][7] for i in ids_sorted_decreasing]

        if batch[0][8] is not None:
            g = torch.zeros_like(input_lengths)
            for i in range(len(ids_sorted_decreasing)):
                g[i] = batch[ids_sorted_decreasing[i]][8]
        else:
            g = None

        if batch[0][9] is not None:
            t = torch.zeros_like(input_lengths)
            for i in range(len(ids_sorted_decreasing)):
                t[i] = batch[ids_sorted_decreasing[i]][9]
        else:
            t = None

        if batch[0][10] is not None:
            r = torch.zeros_like(input_lengths)
            for i in range(len(ids_sorted_decreasing)):
                r[i] = batch[ids_sorted_decreasing[i]][10]
        else:
            r = None

        return (text_padded, input_lengths, mel_padded, output_lengths, len_x,
                pitch_padded, energy_padded, speaker, attn_prior_padded,
                audiopaths, g, t, r)


def batch_to_gpu(batch, device=None):
    move = to_gpu if device is None else lambda tensor: tensor.contiguous().to(device)
    (text_padded, input_lengths, mel_padded, output_lengths, len_x,
     pitch_padded, energy_padded, speaker, attn_prior, audiopaths, g, t, r) = batch

    text_padded = move(text_padded).long()
    input_lengths = move(input_lengths).long()
    mel_padded = move(mel_padded).float()
    output_lengths = move(output_lengths).long()
    pitch_padded = move(pitch_padded).float()
    energy_padded = move(energy_padded).float()
    attn_prior = move(attn_prior).float()
    if speaker is not None:
        speaker = move(speaker).long()
    if g is not None:
        g = move(g).long()
    if t is not None:
        t = move(t).long()
    if r is not None:
        r = move(r).long()

    # Alignments act as both inputs and targets - pass shallow copies
    x = [text_padded, input_lengths, mel_padded, output_lengths,
         pitch_padded, energy_padded, speaker, attn_prior, audiopaths, g, t, r]
    y = [mel_padded, input_lengths, output_lengths]
    len_x = torch.sum(output_lengths)
    return (x, y, len_x)
