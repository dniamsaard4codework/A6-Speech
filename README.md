# A6 — Speech Processing

**Student:** Dechathon Niamsa-ard **[st126235]** · **Course:** DL-AIT Assignment 6

Speech tokenization, CTC alignment, wav2vec 2.0 self-supervised probing, and voice cloning
(OpenVoice V2 + MeloTTS) — all four exercises run end-to-end on the **GPU** in a single
reproducible `uv` environment. Everything is also consolidated in one runnable notebook:
[A6_Speech_Processing.ipynb](A6_Speech_Processing.ipynb).

---

## Repository structure

```
A6-Speech/
├── A6_Speech_Processing.ipynb   # all 4 exercises in one self-contained, executed notebook
├── run.py                       # training / inference CLI (exact assignment command spec)
├── pyproject.toml               # core deps (cu128 GPU torch, numpy<2, transformers 4.46)
├── requirements-voice.txt       # OpenVoice/MeloTTS extras (installed into the same venv)
├── scripts/setup.ps1            # one-shot reproducible environment setup
├── src/a6_speech/               # canonical implementation imported by run.py
│   ├── tokenizer.py             #   Ex1  SpeechTokenizer + comparison figure
│   ├── ctc.py                   #   Ex2  collapse, forward algo, TinyCTC, CER training
│   ├── probe.py                 #   Ex3  wav2vec2 + mel probes, t-SNE
│   ├── voice_clone.py           #   Ex4  OpenVoice V2 + MeloTTS wrappers
│   └── utils.py                 #   seeding, device, logging, edit distance
├── figures/                     # all generated visualizations (embedded below)
├── logs/                        # saved training logs (ctc_train.log, probe.log, voice_clone.log)
├── outputs/                     # generated audio (.wav), tone-color embedding, metrics json
└── third_party/                 # OpenVoice + MeloTTS (cloned by setup.ps1; git-ignored)
```

## Environment & setup

The RTX 5060 Ti is a **Blackwell (`sm_120`)** GPU, which requires **PyTorch cu128 (≥ 2.7)**.
The whole assignment — including OpenVoice/MeloTTS — runs in one Python 3.10 `uv` venv:

```powershell
# one-shot setup (creates .venv, installs the GPU stack + OpenVoice/MeloTTS, verifies imports)
./scripts/setup.ps1
```

<details><summary>What that runs, step by step</summary>

```powershell
uv sync                                                      # torch 2.8.0+cu128 (GPU) + Ex1-3 stack
git clone --depth 1 https://github.com/myshell-ai/OpenVoice.git third_party/OpenVoice
git clone --depth 1 https://github.com/myshell-ai/MeloTTS.git  third_party/MeloTTS
uv pip install --python .venv/Scripts/python.exe -r requirements-voice.txt
uv pip install --python .venv/Scripts/python.exe "setuptools==75.8.2"
uv pip install --python .venv/Scripts/python.exe --no-deps -e third_party/OpenVoice -e third_party/MeloTTS
```
</details>

> **Why one env works:** OpenVoice/MeloTTS normally pin old deps, but their runtime only needs
> `numpy<2` + a stable `transformers` (they use just `AutoTokenizer`/`AutoModelForMaskedLM`).
> Pinning `numpy==1.26.4` + `transformers==4.46.3` satisfies both wav2vec2 (Ex3) and MeloTTS (Ex4).
> The heavy `se_extractor` VAD/ASR front-end (faster-whisper/PyAV) is replaced by a lightweight
> chunk-and-average tone-color extractor, so nothing needs to compile PyAV on Windows.

## Commands used

```bash
# Exercise 2 — train the toy CTC model (also runs the Ex2d (1,2)-duration study)
python run.py --model ctc --epochs 300 --train

# Exercise 3 — linear-probe a frozen wav2vec2 vs a raw mel-spectrogram baseline
python run.py --model wav2vec2-probe --dataset speechcommands --classes yes,no,stop,go --train
python run.py --model wav2vec2-probe --dataset speechcommands --classes yes,no,stop,go,up,down --train  # Ex3c (6-way)

# Exercise 4 — voice cloning (place your ~10-30s clip at data/voice_clone/my_voice.wav first)
python run.py --model voice-clone --extract-se --reference data/voice_clone/my_voice.wav
python run.py --model voice-clone --accent us  --text "I got the job!" --generate
python run.py --model voice-clone --accent all --text "I got the job!" --generate
python run.py --model voice-clone --language es --text "Hola, como estas?" --generate

# Run the full notebook end-to-end
jupyter nbconvert --to notebook --execute A6_Speech_Processing.ipynb
```

## Results

| Task | Model / Method | Result | Notes |
|---|---|---|---|
| Tokenization (Ex 1) | `SpeechTokenizer` | char tokens ≫ word tokens; accent = 1 token | see char-vs-word table below; accent IDs 36/37/38 |
| CTC character error rate (Ex 2) | Toy BiLSTM + CTC | **CER < 10% by step 66** (→ ~3%) | error-rate-vs-step curve below |
| wav2vec2 vs raw-feature probe (Ex 3) | Linear probe (4-way) | **89.6%** vs **68.8%** (random 25%) | wav2vec2 **+20.8 pts**; 6-way: 79.2% vs 51.4% |
| Voice cloning: accent + cross-lingual (Ex 4) | OpenVoice V2 + MeloTTS | cosine sim **high & ~equal** across 4 accents | identity preserved while accent/language change |

**Exercise 1(a) — character vs token counts**

| Sentence | # Char tokens | # Tokens (BOS/EOS) | Accent tag ID |
|---|---|---|---|
| Hello, how are you? | 19 | 21 | — |
| Dr. Smith prescribed 10 tablets. | 36 | 38 | — |
| [EN-US] I got the job! | 14 | 17 | 36 |
| [EN-BR] I lost my wallet. | 17 | 20 | 37 |
| [EN-INDIA] This is completely unacceptable! | 32 | 35 | 38 |

**Exercise 2 — CER & the doubled-letter study (Ex2d)**

| frames_per_char | Overall word accuracy | `hello` accuracy | CER < 10% at step |
|---|---|---|---|
| (3, 8) | 83.0% | 12.5% | 66 |
| (1, 2) | 98.0% | 88.5% | 21 |

Shorter durations are **better** here: the only hard word is the doubled-letter `hello`, which
greedy decoding collapses to `helo` when the model emits a long unbroken run of `l` without the
separating blank. Shorter character runs let that blank survive.

**Exercise 3 — wav2vec2 vs mel-spectrogram linear probe**

| Feature | 4-way acc | 6-way acc |
|---|---|---|
| Raw mel-spectrogram (mean-pooled) | 68.8% | 51.4% |
| wav2vec2 (frozen, mean-pooled) | **89.6%** | **79.2%** |
| (random baseline) | 25.0% | 16.7% |

Both probes are trained for 150 epochs with **train/validation loss + accuracy logged per epoch** (curve below).

**Exercise 4 — same cloned voice across 4 accents** (reference: `data/voice_clone/my_voice.wav`)

| Accent | Duration (s) | RMS Energy | Mel Spectral Centroid (Hz) | cos(reference, clip) |
|---|---|---|---|---|
| us | 1.846 | 0.0598 | 629.8 | 0.511 |
| br | 1.324 | 0.0800 | 369.3 | 0.482 |
| india | 1.800 | 0.0347 | 250.1 | 0.596 |
| au | 1.730 | 0.0756 | 242.5 | 0.558 |

The cosine similarities are clustered (~0.48–0.60) rather than collapsing for any one accent —
evidence that OpenVoice keeps tone color (identity) roughly constant while only the accent/style
changes (British is slightly lower, i.e. a little more style leakage into identity). Cross-lingual
cloning (EN/ES/FR) reuses the *same* embedding.

> Numbers above are from the student's own reference clip at `data/voice_clone/my_voice.wav`
> (~47 s). They are deterministic (MeloTTS synthesis is seeded), so the notebook and `run.py` agree.

## Visualizations

**1. Tokenization comparison — NLP tokens vs speech chars vs accent token (Part 1)**

![tokenization comparison](figures/ex1_tokenization_comparison.png)

**2. CTC greedy decoding grid + train/val loss & character-error-rate curves (Part 3 / Ex 2)**

![CTC train/val loss and CER](figures/ex2_loss_cer.png)
![CTC greedy grid (3,8)](figures/ex2_greedy_grid_3_8.png)
![CTC greedy grid (1,2)](figures/ex2_greedy_grid_1_2.png)

**3. wav2vec2 vs mel-spectrogram probe + t-SNE (Part 4 / Ex 3)**

Linear-probe training curves (train vs. validation loss, validation accuracy per epoch):

![probe training curves](figures/ex3_probe_training_4way.png)

![probe comparison 4-way](figures/ex3_probe_comparison_4way.png)
![t-SNE of wav2vec2 embeddings](figures/ex3_tsne_4way.png)

**4. Mel spectrogram grid: same cloned voice across 4 accents (Part 5.4)**

![mel grid across accents](figures/ex4_mel_grid_accents.png)

## Discussion

Understanding speech tokenization and CTC alignment reframes ASR/TTS training as fundamentally
an **alignment** problem rather than a plain sequence-to-sequence mapping: the model never sees a
frame→character labelling, so the architecture (CTC's blank + collapse, or attention) has to
*discover* which of the hundreds of audio frames belong to each of a handful of text tokens. That
makes design choices like the blank token, character-vs-phoneme units, and text normalization
load-bearing — they decide whether a word like `hello` can even be represented over a long noisy
frame sequence, as Exercise 2(d)'s doubled-letter failure made concrete. A **tone-color
embedding** is a fundamentally different kind of object from a text token or a CTC blank, even
though all three "condition" the output: a text token and a blank are **discrete symbols from a
fixed vocabulary carrying linguistic/structural content** (what to say, and where one unit ends),
whereas a tone-color embedding is a **continuous vector extracted from audio with no linguistic
content at all** — a point in a learned speaker-identity space describing *how a voice sounds*. The
token and blank are consumed as part of the symbol stream the model must align and pronounce; the
tone-color vector is injected as a global conditioning signal that re-paints timbre while leaving
the linguistic alignment untouched — which is exactly why the same embedding can clone a voice
across both accents and languages.

---

### Training logs

Saved under [logs/](logs/): `ctc_train.log` (Ex2, incl. per-word breakdown), `probe_4way.log` +
`probe_6way.log` (Ex3), `voice_clone.log` (Ex4). Reproducible: seeds fixed to 42 across
NumPy / PyTorch / scikit-learn, and SpeechCommands clip selection is sorted for determinism.
