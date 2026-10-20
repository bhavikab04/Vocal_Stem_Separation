# Vocal_Stem_Separation
In this project, instrument classification and vocal source separation are done on music mixtures using several architectures on the MoisesDB dataset. It compares feature-based and neural classifiers, trains a Transformer-based spectrogram model and a Hybrid Spectrogram-TasNet from scratch to isolate vocals, then fine-tunes a pretrained HTDemucs model and serves it through a web app.

Trained on the [MoisesDB](https://arxiv.org/abs/2307.15913) dataset — used 80 songs, each with isolated stems (vocals, drums, guitar, bass, etc.). The goal is to take a full mixture track and pull out just the vocals.

---

## What's in the notebook

The project went through a few stages, each one motivated by what the previous stage got wrong.

### Stage 1 — Instrument classification (baseline)

Before trying to separate anything, I wanted to understand the dataset. I extracted audio features (MFCCs, chroma, spectral centroid, RMS, zero-crossing rate) from each isolated stem and trained three classifiers: an SVM, a Random Forest, and a small MLP, all on the same train/test split. After that, CNN nad R-CNN were also trained for classification and CNN gave the best results.

This was done to see which instruments are spectrally distinct. Drums came out as the easiest to identify. That's actually why we chose **vocals** for separation rather than drums: drums are easy to classify because their energy spreads across all frequencies, which makes them very difficult to mask cleanly. Vocals are harmonically structured and sit in a narrower band, so a mask-based approach had more potential.

### Stage 2 — Vocal separation with a Vision Transformer

The first separation model treats the spectrogram like an image. The mixture's log-magnitude spectrogram (513 freq bins × 1024 time frames, roughly 16 seconds) gets carved into 2D patches of size 27×32, giving 608 tokens. Those go through a 6-layer Transformer encoder with 8-head attention, get reshaped back to a 2D feature map, upsampled with bilinear interpolation, then passed through a small conv decoder that outputs a soft mask in [0, 1].

**Why patch size (27, 32)?** At the original patch size from ViT papers (9×8), a 513×1024 spectrogram produces 7,296 tokens — way too many for a T4 GPU. (27, 32) brings that down to 608 tokens while still giving the decoder enough spatial resolution to reconstruct a clean mask.

**Why log-magnitude loss instead of raw L1?** Raw L1 on spectrograms is dominated by low-frequency, high-energy bins. The model's easiest solution is to output a near-zero mask everywhere — which minimises the loss but produces silence. Working in log1p space equalises all frequency bins and forces the model to actually learn structure.

**Results:** Log-Spec L1 improved from 0.41 to 0.03 (a 93% reduction). SI-SDR and SDR were negative, which sounds bad but isn't a model failure — it's a known limitation of magnitude masking. When you reconstruct audio using a predicted magnitude mask but the original mixture's phase, the phase mismatch destroys waveform-level metrics even if the spectrogram looks clean. This is documented in the literature (Im et al., 2022 showed that even ground-truth vocoder reconstructions score poorly on SDR for the same reason).

### Stage 3 — Hybrid Spectrogram-TasNet (HS-TasNet-Small)

Motivated by Venkatesh et al. (2024). The core insight from that paper: spectrogram-based models are good at harmonic structure (vocals, keys) but bad at phase. Waveform-based models handle phase naturally but struggle with fine spectral detail. A hybrid model gets both.

HS-TasNet-Small runs two branches in parallel:

- **Spectrogram branch** — BiLSTM over time frames of the log-magnitude spectrogram, predicts a soft mask, reconstructed via iSTFT
- **Waveform branch** — learned Conv1D encoder (replaces the STFT entirely), BiLSTM masker, transposed Conv1D decoder back to raw audio. A Hanning window is applied to the decoder filters to prevent discontinuities at chunk boundaries — a specific trick from the paper

The branches are combined by summation rather than concatenation (that's the "Small" variant — keeps the parameter count around 16M instead of 42M, matching the dataset scale).

**Why BiLSTM instead of Transformer here?** With 80 songs, a Transformer's data hunger becomes a real problem. Rouard et al. (2022) showed that Hybrid Transformer Demucs only beats the simpler Hybrid Demucs baseline when trained on 800+ extra songs. BiLSTM is more data-efficient and has a proven track record on similar dataset sizes (MusDB has 86 training songs, almost identical to ours).

**Multi-domain loss:** spectral L1 in log-space + waveform L1. The waveform term is ramped up over the first 10 epochs so the spectrogram branch stabilises before the waveform branch starts contributing gradients — otherwise the harder waveform optimization drags the whole model down early on.

---

### Stage 4 — Fine-tuning pretrained HTDemucs

Both from-scratch models struggled at this data scale (waveform-level SDR stayed negative), so the next step was to start from a model trained on far more data. I fine-tuned pretrained HTDemucs ([Rouard et al., 2022](https://arxiv.org/abs/2211.08553)) on MoisesDB (`finetune_vocals.py`).

- **Data pipeline:** MoisesDB stems are grouped into HTDemucs's four sources (vocals, drums + percussion, bass, other), and every wav in a stem folder is summed. Training samples are random 7.8 s stereo crops at 44.1 kHz with gain, polarity and channel-swap augmentation. Half of the samples are remixes built from stems of different songs.
- **Training:** AdamW, learning rate 3e-5 with cosine decay, mixed precision, 12 epochs × 60 steps with an effective batch of 8 (micro-batch 1, gradient accumulation 8, to fit a T4). Loss is waveform L1 with vocals weighted 3×.
- **Evaluation:** 10 songs were held out. Vocal SDR was computed on 30 s vocal-heavy excerpts, comparing the pretrained baseline against the fine-tuned model. Training took about 43 minutes on one T4.

| Model | Held-out vocal SDR |
|---|---|
| Pretrained HTDemucs | 11.51 dB |
| Fine-tuned on MoisesDB | 11.54 dB |

The gain of about 0.03 dB is within noise for a 10-song validation set (and the best epoch was selected on the same songs), so the practical conclusion is that fine-tuning on ~70 songs does not meaningfully improve an already strong pretrained model. The pretrained weights are the app's default; the fine-tuned checkpoint can be dropped in as `htdemucs_vocals_ft.pt`.

### Stage 5 — Web app

A FastAPI backend (`server.py`) runs HTDemucs separation and a single-page front end (`static/index.html`) lets you upload a song and get isolated vocals and an instrumental. The page shows upload progress, a synced three-track player (original, vocals, instrumental) with waveforms, solo/mute/volume per track, looping, and downloads.

**Run it locally**

1. Install Python 3.10+ and ffmpeg (`winget install ffmpeg` / `sudo apt install ffmpeg` / `brew install ffmpeg`).
2. Windows: double-click `run.bat`. Linux/Mac: `bash run.sh`.
3. Open http://localhost:8000.

The first launch downloads the pretrained model (internet needed). On a CPU a full song takes several minutes, so try a 30-second clip first.

## Architecture summary

| | Transformer | HS-TasNet-Small |
|---|---|---|
| Input | Log-magnitude spectrogram | Log-mag spec + raw waveform |
| Core | 6-layer Transformer encoder | 2× BiLSTM (one per branch) |
| Output | Soft mask → iSTFT | Spec mask + waveform estimate |
| Parameters | ~8M | ~16M |
| Phase handling | Mixture phase (approximation) | Waveform branch (implicit) |
| Log-Spec L1 | 0.034 | 0.034 |

---

## Key references

- Venkatesh et al. (2024) — *Real-Time Low-Latency Music Source Separation Using Hybrid Spectrogram-TasNet*
- Rouard et al. (2022) — *Hybrid Transformers for Music Source Separation*
- Im et al. (2022) — *Neural Vocoder Feature Estimation for Dry Singing Voice Separation*
- Stoter et al. (2019) — *Open-Unmix: A Reference Implementation for Music Source Separation*
- Pereira et al. (2023) — *MoisesDB: A Dataset for Source Separation Beyond 4-Stems*

---

## Dataset

[MoisesDB](https://zenodo.org/record/10520051) — 80 songs with isolated stems. Uploaded to Kaggle as a private dataset. Each song folder contains per-instrument subdirectories with `.wav` files. No external library required — stems are loaded directly from the file structure.

All audio resampled to 16kHz mono. STFT computed with n_fft=1024, hop=256 (≈16ms per frame). Each training example covers ~16 seconds of audio (MAX_LEN=1024 frames).

### Citation

```bibtex
@misc{pereira2023moisesdb,
      title={Moisesdb: A dataset for source separation beyond 4-stems}, 
      author={Igor Pereira and Felipe Araújo and Filip Korzeniowski and Richard Vogl},
      year={2023},
      eprint={2307.15913},
      archivePrefix={arXiv},
      primaryClass={cs.SD}
}
```