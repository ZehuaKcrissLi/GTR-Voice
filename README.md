

<p align="center">

  <h2 align="center">[Interspeech'24] GTR-Voice: Articulatory Phonetics Informed Controllable Expressive Speech Synthesis </h2>
  <p align="center">
    <strong>Zehua Kcriss Li</strong><sup>1</sup>
    · 
    <strong>Meiying Melissa Chen</strong><sup>1</sup>
    · 
    <strong>Yi Zhong</strong><sup>1</sup>
    ·
    <strong>Pinxin Liu</strong><sup>2</sup>
    ·
    <strong>Zhiyao Duan</strong><sup>1</sup>
    ·
    <br>
    <sup>1</sup>Department of Electrical and Computer Engineering, University of Rochester; 
    <br>
    <sup>2</sup>Department of Computer Science, University of Rochester
    <br>
    </br>
        <a href="https://arxiv.org/pdf/2406.10514v1">
        <img src='https://camo.githubusercontent.com/d8617043c5be969759376b28adaf0620c3360f34567782a825b13e9945717a7a/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f61725869762d50617065722d253343434f4c4f522533452e737667' alt='Paper Arxiv'></a> &nbsp; &nbsp; 
        <a href='https://demo.gtr-voice.com/'>
        <img src='https://img.shields.io/badge/Webpage-Demo-blue' alt='Demo'></a> &nbsp;&nbsp;

  </p>
    </p>
<div align="center">
  <img src="./assets/gtr_poster.png" alt="GTR-Voice"></a>
</div>

## Dataset and demo

[Interactive demo](https://gtr-voice.com/) · [Paper](https://arxiv.org/abs/2406.10514) · [Original dataset ZIP](https://github.com/ZehuaKcrissLi/GTR-Voice/releases/download/gtr-voice-dataset-v1/GTR-Voice.zip)

The original ZIP is 1,044,673,451 bytes. SHA256:
`d2b1301acbdeac7ca8e3808eae0d0f6396b9bb6c5afe4c84585c22a307c826e0`.
It contains 2,500 recordings across 125 GTR labels and 20 utterances. Fourteen
original files contain incomplete PCM frames; the download preserves their
original bytes. Training preparation records these exclusions explicitly.

## Runnable research paths

This repository contains the original Interspeech 2024 implementations and
repairs to their data preparation, training, inference and analysis entry points.
Use the following guides instead of the historical notebook paths:

| Path | Guide |
|---|---|
| Original audio integrity and clean manifests | [Dataset preparation](docs/reproduction.md#dataset-preparation) |
| Twelve acoustic feature families and three SVM classifiers | [Acoustic-GTR](Acoustic-GTR/README.md) |
| Mandarin FastPitch with GTR conditioning | [FastPitch-GTR](FastPitch-GTR/README.md) |
| Frozen StyleTTS base and trainable GTR embedder | [StyleTTS-GTR](StyleTTS-GTR/README.md) |
| Anonymous listening responses, MOS and control accuracy | [Evaluation statistics](evaluation/README.md) |
| Optional HuBERT classifier extension | [HuBERT-GTR](HuBERT-GTR/README.md) |

HuBERT is an additional repository experiment, not the paper's acoustic-SVM
method. Original final paper checkpoints and the full original listening-study
protocol are not distributed here. See [reproduction notes](docs/reproduction.md)
for tested commands, measured results and the remaining scientific gaps.

## Install and engineering checks

Use Python 3.10 in an isolated environment. The root requirements describe the
versions used for these repaired paths; retained upstream dependency files and
old absolute-path data lists are historical references.

```sh
python3.10 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
```

The pinned PyTorch 2.1.2 CPU path is supported. For NVIDIA GPU training, install
matching PyTorch/torchaudio CUDA wheels as described by
[PyTorch](https://pytorch.org/get-started/previous-versions/) before the remaining
requirements. No import automatically downloads weights or starts training.
StyleTTS text inference needs eSpeak-NG; an explicit loader option is documented
in its guide.

Run each component separately because the original projects have overlapping
module names:

```sh
python -m pytest -q tools/tests Acoustic-GTR/tests
python -m pytest -q HuBERT-GTR/tests
python -m pytest -q FastPitch-GTR/tests
python -m pytest -q StyleTTS-GTR/tests
python -m unittest discover -s evaluation/tests -p 'test_*.py' -v
```

These tests use small synthetic fixtures to check optimizer steps, strict
checkpoint replay, conditioning, malformed inputs and statistical accounting.
They do not measure human speech quality or reproduce the paper's benchmark
numbers. Full-data commands and separately measured outcomes are documented in
the [reproduction guide](docs/reproduction.md).

## 🙏 Acknowledgments

Our code follows several excellent repositories. We appreciate them for making their codes available to the public.
* [StyleTTS](https://github.com/yl4579/StyleTTS)
* [FastPitch](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/SpeechSynthesis/FastPitch)


## ✏️ Citation
If you find our work useful, please consider citing:
```BibTeX
@inproceedings{li2024gtr,
  title={GTR-Voice: Articulatory Phonetics Informed Controllable Expressive Speech Synthesis},
  author={Li, Zehua Kcriss and Chen, Meiying Melissa and Zhong, Yi and Liu, Pinxin and Duan, Zhiyao},
  booktitle={Proc. Interspeech 2024},
  pages={1775--1779},
  year={2024}
}
```
