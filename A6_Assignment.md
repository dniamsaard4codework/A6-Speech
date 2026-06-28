# Exercises

## Exercise 1: Speech vs NLP Tokenization — Building Intuition

a) Using the `SpeechTokenizer` defined in Part 1, tokenize the following sentences and fill in the table:

```python
sentences = [
    "Hello, how are you?",
    "Dr. Smith prescribed 10 tablets.",
    "[EN-US] I got the job!",
    "[EN-BR] I lost my wallet.",
    "[EN-INDIA] This is completely unacceptable!",
]
```

| Sentence | # Char tokens | # Tokens (with BOS/EOS) | Accent tag token ID |
|---|---|---|---|
| Hello, how are you? | ? | ? | — |
| Dr. Smith prescribed 10 tablets. | ? | ? | — |
| [EN-US] I got the job! | ? | ? | ? |
| [EN-BR] I lost my wallet. | ? | ? | ? |
| [EN-INDIA] This is completely unacceptable! | ? | ? | ? |

b) In the tokenizer's `normalize()` method, `"Dr. Smith"` becomes `"doctor smith"`. Why is this normalization critical for TTS? What would happen if the model received `"Dr."` as input?

c) In NLP, `[CLS]` is a special token that summarizes the sequence. In speech, the accent tag `[EN-US]` is also a single special token at the beginning. Explain the **architectural similarity** — how does a single token influence the model's entire output?

---

## Exercise 2: CTC — Verifying the Forward Algorithm

a) Pick 3 short hand-built alignments (lists of characters, like the `examples` list in Part 3.2) that should all collapse to the same word. Run them through `ctc_collapse` and confirm they agree.

b) Using `ctc_forward_log_prob` from Part 3.4, compute $P_{CTC}(Y \mid X)$ for the same random `log_probs` matrix but for two *different* target words of the same length (e.g. `"HEL"` vs `"LEH"`). They should generally get different probabilities — explain in 1-2 sentences why this is expected given the model's output at each frame.

c) Modify the toy training loop in Part 3.5 to track **character error rate** (edit distance between decoded and true word, divided by word length) instead of just loss. Plot character error rate vs. training step. At what step does it drop below 10%?

d) Re-run the greedy decoding visualization (Part 3.5's last cell) with `frames_per_char=(1, 2)` instead of `(3, 8)` in `synthesize_frames`. Does accuracy get better or worse with much shorter character durations? Explain why, referencing how many frames the model has to "agree" on a character before collapsing.

---

## Exercise 3: wav2vec2 — How Much Does Self-Supervision Actually Buy You?

a) Repeat the Part 4.3 linear probe, but using **raw mel-spectrogram features** (mean-pooled over time, like in Part 2) instead of wav2vec2 embeddings, as your baseline. Fill in:

| Feature | Test Accuracy |
|---|---|
| Raw mel-spectrogram (mean-pooled) | ? |
| wav2vec2 (frozen, mean-pooled) | ? |

b) By how much does wav2vec2 improve over the raw-feature baseline? Compare this gap to the gap you measured between an MLP-on-raw-pixels baseline and a pretrained SSL encoder (SimCLR/DINO/MAE) in the SSL lab — is the speech gap bigger, smaller, or about the same order of magnitude?

c) Increase `PROBE_WORDS` to include at least 6 SpeechCommands classes instead of 4. Does linear-probe accuracy drop, and if so, is the drop proportional to the increase in classes (i.e., does it stay well above the new random baseline)?

d) wav2vec2 was pretrained with a **contrastive** loss against quantized targets, while MAE (SSL lab) used **reconstruction**. Based on what you've now measured in both labs, which inductive bias seems to transfer better to a completely different downstream task — and is the comparison even fair given the very different data modalities?

---

## Exercise 4: Voice Cloning — Identity, Style, and Language

a) Using your own ~10-30s recording, extract a tone color embedding and synthesize the test sentence from Part 5 in all four accents (`us`, `br`, `india`, `au`). For each, record:

| Accent | Duration (s) | RMS Energy | Mel Spectral Centroid |
|---|---|---|---|
| us | ? | ? | ? |
| br | ? | ? | ? |
| india | ? | ? | ? |
| au | ? | ? | ? |

b) Listen to all four clips. Does it still sound like *you* in every style, or does the cloned voice drift toward a different-sounding speaker in some accents? Compute the cosine similarity between the tone color embedding extracted from your **reference clip** and a tone color embedding extracted from each of the **four generated clips** (re-run `se_extractor.get_se` on the outputs). If OpenVoice's disentanglement is working well, what should these similarities look like — high and roughly equal across all four accents, or otherwise?

---

## Submission

Submit your work to GitHub. Your repository should contain:

### 1. Training Script (`run.py`)

```bash
# Train the toy CTC model (Part 3 / Exercise 2)
python3 run.py --model ctc --epochs 300 --train

# Linear-probe a pretrained wav2vec2 checkpoint (Part 4 / Exercise 3)
python3 run.py --model wav2vec2-probe --dataset speechcommands --classes yes,no,stop,go --train

# Extract tone color from your reference clip
python3 run.py --model voice-clone --extract-se --reference my_voice.wav

# Synthesize in a given style with your cloned voice
python3 run.py --model voice-clone --accent us --text "I got the job!" --generate

# Synthesize all styles for comparison
python3 run.py --model voice-clone --accent all --text "Hello world" --generate

# Cross-lingual cloning
python3 run.py --model voice-clone --language es --text "Hola, como estas?" --generate
```

### 2. `README.md`

Your `README.md` must include:

**Commands used** (exact commands you ran)

**Results table:**

| Task | Model / Method | Result | Notes |
|---|---|---|---|
| Tokenization (Ex 1) | SpeechTokenizer | — | char vs word count table |
| CTC character error rate (Ex 2) | Toy BiLSTM + CTC | ? % CER | error rate vs training step |
| wav2vec2 vs raw-feature probe (Ex 3) | Linear probe | ? % vs ? % | wav2vec2 vs mel-spectrogram baseline |
| Voice cloning: accent + cross-lingual (Ex 4) | OpenVoice | ? cosine sim / ? quality | identity-vs-accent and language transfer |

**Visualizations** (include in README or as separate image files):
- CTC greedy decoding grid + character error rate curve (Part 3 / Exercise 2)
- wav2vec2 vs mel-spectrogram linear probe comparison + t-SNE plot (Part 4 / Exercise 3)
- Mel spectrogram grid: same cloned voice across 4 accents (Part 5.4)
- Tokenization comparison: NLP tokens vs speech chars vs accent tokens (Part 1)

**Discussion** (3–5 sentences): How does understanding speech tokenization and CTC alignment change how you think about training a TTS or ASR model? Why is a tone color embedding fundamentally a different kind of object than a text token or a CTC blank, even though all three "condition" or shape a model's output?