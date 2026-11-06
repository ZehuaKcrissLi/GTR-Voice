# StyleTTS-GTR: explicit restoration pipeline

The restored GTR path is `gtr_pipeline.py`: verified source audio → frozen original StyleTTS style targets → GTR embedder training → strict fresh-process evaluation → text/GTR waveform synthesis. It does not download assets, log into W&B, choose a GPU, or load a private path at import. `train_teacher_forcing.py` and `inference.py` are compatibility entry points for its `train` and `infer` commands.

## Models and paper scope

The paper uses the original **StyleTTS**, pretrained on LibriTTS train-clean-460 for 200 epochs across two stages (stage allocation is unspecified), then freezes the base and trains the GTR embedder for **500 epochs**. Use an explicit compatible original StyleTTS checkpoint, its YAML config, and its **80-mel / 24 kHz** HiFi-GAN vocoder/config. A JDC pitch extractor by itself is not the GTR model. Pretrained model archives are linked by [the official original StyleTTS repository](https://github.com/yl4579/StyleTTS); obtain and verify them separately. No original final GTR checkpoint is included here. The separately verified public `Models.zip` contains `epoch_2nd_00050.pth` with config `epochs_1st: 80` and `epochs_2nd: 50`; this public base differs from the paper's 200-epoch description and is identified as a public-base-derived run, not the recovered original paper model.

Two embedder configurations are intentionally distinct:

- `legacy`: released repository architecture, three separate G/T/R embeddings, concatenation and three Linear layers (two intervening ReLUs). Defaults preserve the released state keys, including `fc_sequnetial` spelling, with widths 12→48→96→128.
- `paper-described`: one joint 125-class embedding and two Linear layers with ReLU. The paper does not report the joint encoding or these dimensions; the compact class coding and default widths here are a **derived configuration**, not an exact architecture reproduction. Never load a legacy checkpoint into it by renaming keys.

## Commands

Install the tested environment with the repository root `requirements.txt`; run commands from this directory. The root dependency list is authoritative for the restored path. All paths below are explicit inputs. Set `OMP_NUM_THREADS=2` and place caches, temporary data and outputs on the task volume when working on the Mac mini.

```bash
python gtr_pipeline.py extract --manifest /data/eligible-train.jsonl \
  --checkpoint /models/epoch_2nd_00199.pth --output /work/train-targets.npz --device cpu
python gtr_pipeline.py extract --manifest /data/eligible-validation.jsonl \
  --checkpoint /models/epoch_2nd_00199.pth --output /work/validation-targets.npz --device cpu
python gtr_pipeline.py train --embeddings /work/train-targets.npz \
  --validation-embeddings /work/validation-targets.npz --output-dir /work/gtr-training \
  --architecture legacy --epochs 500 --batch-size 64 --learning-rate 0.001 --device cpu
python gtr_pipeline.py evaluate --embeddings /work/validation-targets.npz \
  --checkpoint /work/gtr-training/last.pt --output /work/evaluation.json --device cpu
python gtr_pipeline.py infer --checkpoint /models/epoch_2nd_00199.pth \
  --model-config /models/config.yml --gtr-checkpoint /work/gtr-training/last.pt \
  --vocoder-checkpoint /models/vocoder/generator --vocoder-config /models/vocoder/config.json \
  --text 'An example sentence.' --gtr 231 --output /work/generated-231.wav --device cpu --espeakng-loader
```

The learning rate/batch defaults for embedder training are explicit implementation settings, not fully specified paper hyperparameters. `--resume /work/gtr-training/last.pt` resumes the same architecture and training-target hash. Inference uses the official English predictor architecture (`ProsodyPredictorOLD` in this repository); checkpoints with the experimental Mandarin tone embedding require a matching separately documented architecture and fail strict loading here. `--phonemes` accepts already phonemized text if eSpeak is not installed; otherwise the local eSpeak library is required. Unknown symbols fail instead of being silently removed.

Each manifest JSONL row must contain `id`, `source_path`, `source_sha256` and `label` as a three-character string (`"010"`, `"231"`). Labels enforce the 125 legal combinations: G=0 iff R=0; T=1…5, G=0…4, R=0…6. Split manifests must be independently disjoint by original source hash. Ineligible or undecodable rows must be audited and excluded explicitly before extraction; the restored original dataset contains 14 known decoding anomalies, and this pipeline never repairs their original bytes.

Source WAVs are preflighted with the repository's shared RIFF/PCM frame validator before libsndfile decoding, so a malformed final PCM frame cannot be silently used as a complete prefix. Extraction records source hashes, teacher hash, original 24 kHz waveform processing and the **16 kHz mel-filter sample rate used implicitly by official StyleTTS code**. Changing that filter rate silently would change the pretrained feature space. Extraction accepts `--device mps` on supported Macs while preserving the exact CPU resampling/trim/mel arithmetic and moving only the frozen style encoder; device and float32 precision are recorded. No MPS fallback or time padding is applied. Training checkpoints record the paired teacher hash and target device metadata; inference rejects a different base checkpoint. Vocoder code is vendored from the pinned original StyleTTS revision recorded in `vocoder/SOURCE.json`, with its MIT provenance preserved; no vocoder weights are vendored.

`training.json`, evaluation JSON and waveform-sidecar JSON record actual completed work and model/output hashes. Evaluation separately reports overlap with recorded training source hashes; training replay is never labelled held-out evaluation. An engineering test with random fixtures or a completed 500-epoch fit does not reproduce the paper's MOS, reported standard deviations, human G/T/R ratings or reported speech quality. Those require the stated original models/data protocol and new human ratings, or a clearly identified recovery of the original study results.

## Validation

```bash
python -m pytest -q tests
```

Tests cover legal labels, gradients, strict checkpoint replay, fresh-process evaluation/resume, split leakage rejection and safe imports. They are engineering regression tests, not paper quality measurements. The original upstream base-model training files and archived experiments remain below for provenance; use the explicit pipeline above for the restored GTR path.

---

# StyleTTS: A Style-Based Generative Model for Natural and Diverse Text-to-Speech Synthesis

### Yinghao Aaron Li, Cong Han, Nima Mesgarani

> Text-to-Speech (TTS) has recently seen great progress in synthesizing high-quality speech owing to the rapid development of parallel TTS systems, but producing speech with naturalistic prosodic variations, speaking styles and emotional tones remains challenging. Moreover, since duration and speech are generated separately, parallel TTS models still have problems finding the best monotonic alignments that are crucial for naturalistic speech synthesis. Here, we propose StyleTTS, a style-based generative model for parallel TTS that can synthesize diverse speech with natural prosody from a reference speech utterance. With novel Transferable Monotonic Aligner (TMA) and duration-invariant data augmentation schemes, our method significantly outperforms state-of-the-art models on both single and multi-speaker datasets in subjective tests of speech naturalness and speaker similarity. Through self-supervised learning of the speaking styles, our model can synthesize speech with the same prosodic and emotional tone as any given reference speech without the need for explicitly labeling these categories.

Paper: [https://arxiv.org/abs/2107.10394](https://arxiv.org/abs/2205.15439)

Audio samples: [https://styletts.github.io/](https://styletts.github.io/)

## Pre-requisites
1. Python >= 3.7
2. Clone this repository:
```bash
git clone https://github.com/yl4579/StyleTTS.git
cd StyleTTS
```
3. Install python requirements: 
```bash
pip install SoundFile torchaudio munch torch pydub pyyaml librosa git+https://github.com/resemble-ai/monotonic_align.git
```
4. Download and extract the [LJSpeech dataset](https://keithito.com/LJ-Speech-Dataset/), unzip to the data folder and upsample the data to 24 kHz. The vocoder, text aligner and pitch extractor are pre-trained on 24 kHz data, but you can easily change the preprocessing and re-train them using your own preprocessing. I will provide more receipes and pre-trained models later if I have time. If you are willing to help, feel free to work on other preprocessing methods. 
For LibriTTS, you will need to combine train-clean-360 with train-clean-100 and rename the folder train-clean-460 (see [val_list_libritts.txt](https://github.com/yl4579/StyleTTS/blob/main/Data/val_list_libritts.txt) as an example).

## Training
First stage training:
```bash
python train_first.py --config_path ./Configs/config.yml
```
Second stage training:
```bash
python train_second.py --config_path ./Configs/config.yml
```
You can run both consecutively and it will train both the first and second stage. The model will be saved in the format "epoch_1st_%05d.pth" and "epoch_2nd_%05d.pth". Checkpoints and Tensorboard logs will be saved at `log_dir`. 

The data list format needs to be `filename.wav|transcription`, see [val_list_libritts.txt](https://github.com/yl4579/StyleTTS/blob/main/Data/val_list_libritts.txt) as an example. 

## Inference

Please refer to [inference.ipynb](https://github.com/yl4579/StyleTTS/blob/main/Demo/Inference_LJSpeech.ipynb) for details. 

The pretrained StyleTTS and Hifi-GAN on LJSpeech corpus in 24 kHz can be downloaded at [StyleTTS Link](https://drive.google.com/file/d/1aqOExU7NroGHdIVjgkzqRYrK5q_694cj/view?usp=sharing) and [Hifi-GAN Link](https://drive.google.com/file/d/1h_h0GFdC6VOiZ-oFDClqy2bVonA1xDiw/view?usp=sharing). 

The pretrained StyleTTS and Hifi-GAN on LibriTTS corpus can be downloaded at [StyleTTS Link](https://drive.google.com/file/d/1nm0yB6Y5QWF3FYGfJCwQ6zYNlOAYVSet/view?usp=sharing) and [Hifi-GAN Link](https://drive.google.com/file/d/1RDxYknrzncGzusYeVeDo38ErNdczzbik/view?usp=sharing). You also need to download test-clean from LibriTTS if you want to run the zero-shot demo. 

Please unzip to `Models` and `Vocoder` respectivey and run each cell in the notebook. You will also need to install [phonemizer](https://github.com/bootphon/phonemizer) to run this inference demo. 

## Preprocessing

The pretrained text aligner and pitch extractor models are provided under the `Utils` folder. Both the text aligner and pitch extractor models are trained with melspectrograms preprocessed using [meldataset.py](https://github.com/yl4579/StyleTTS/blob/main/meldataset.py). 

You can edit the [meldataset.py](https://github.com/yl4579/StyleTTS/blob/main/meldataset.py) with your own melspectrogram preprocessing, but the provided pretrained models will no longer work. You will need to train your own text aligner and pitch extractor with the new preprocessing. 

The code for training new text aligner model is available [here](https://github.com/yl4579/AuxiliaryASR) and that for training new pitch extractor models is available [here](https://github.com/yl4579/PitchExtractor).

I will provide more recepies with existing preprocessing like those in official [HifiGAN](https://github.com/jik876/hifi-gan) and [ESPNet](https://github.com/espnet/espnet) in the future if I have extra time. If you are willing to help, feel free to make receipes with [ESPNet](https://github.com/espnet/espnet). 
