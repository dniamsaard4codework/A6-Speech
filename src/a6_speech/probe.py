"""Exercise 3 — wav2vec 2.0 linear probing vs. a raw mel-spectrogram baseline.

Part 4.3 reproduced, plus the Exercise-3 additions:
  (a) a raw mel-spectrogram (mean-pooled) baseline probe alongside the wav2vec2 probe,
  (c) support for >= 6 SpeechCommands classes,
  + the wav2vec2-vs-mel comparison bar chart and the t-SNE embedding plot.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from .utils import set_seed, get_device, DATA_DIR

W2V_NAME = "facebook/wav2vec2-base"
DEFAULT_PROBE_WORDS = ["yes", "no", "stop", "go"]
EXTENDED_PROBE_WORDS = ["yes", "no", "stop", "go", "up", "down"]  # Ex 3c (>=6 classes)
SAMPLE_RATE = 16000


# ---------------------------------------------------------------------------
# Data: a small balanced SpeechCommands subset
# ---------------------------------------------------------------------------
def _resolve_path(ds, meta):
    """Absolute wav path from a SPEECHCOMMANDS metadata tuple (relpath, sr, label, ...)."""
    import os

    relpath = meta[0]
    if os.path.isabs(relpath) and os.path.exists(relpath):
        return relpath
    base = getattr(ds, "_archive", None) or getattr(ds, "_path", None)
    cand = os.path.join(str(base), relpath)
    if os.path.exists(cand):
        return cand
    # last resort: search under the dataset path
    return os.path.join(str(getattr(ds, "_path", base)), relpath)


def build_speechcommands_subset(words, n_per_class=40, root: str | Path = DATA_DIR):
    """Return {label: [waveform, ...]} with `n_per_class` clips per word.

    Uses `get_metadata` to scan labels WITHOUT decoding audio, then loads only the
    chosen clips with `soundfile` (avoids torchaudio.load -> TorchCodec, which is
    not available on this Windows + torchaudio 2.11 setup).
    """
    import torch
    import torchaudio
    import soundfile as sf

    root = str(root)
    ds = torchaudio.datasets.SPEECHCOMMANDS(root=root, download=True)
    n_total = len(ds)
    # Collect ALL metadata for the target labels (no audio decode), then sort by path and
    # take the first n_per_class. Sorting makes clip selection deterministic across processes
    # (the dataset's internal walker order is not stable), so the notebook and run.py agree.
    buckets = {w: [] for w in words}
    for i in range(n_total):
        meta = ds.get_metadata(i)  # (relpath, sample_rate, label, speaker_id, utt_no)
        if meta[2] in buckets:
            buckets[meta[2]].append(meta)
    chosen = {w: sorted(buckets[w], key=lambda m: m[0])[:n_per_class] for w in words}

    missing = {w: n_per_class - len(v) for w, v in chosen.items() if len(v) < n_per_class}
    if missing:
        raise RuntimeError(f"Not enough clips for {missing} in SpeechCommands.")

    by_label = {}
    for w, metas in chosen.items():
        clips = []
        for meta in metas:
            wav, sr = sf.read(_resolve_path(ds, meta), dtype="float32")
            clips.append(torch.from_numpy(wav).unsqueeze(0))  # (1, samples)
        by_label[w] = clips
    total = sum(len(v) for v in by_label.values())
    print(f"SpeechCommands subset: {total} clips, {n_per_class}/class, classes={words}")
    return by_label


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------
def load_wav2vec2(device):
    import transformers
    from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor

    transformers.logging.set_verbosity_error()  # silence the pretraining-head LOAD REPORT
    extractor = Wav2Vec2FeatureExtractor.from_pretrained(W2V_NAME)
    model = Wav2Vec2Model.from_pretrained(W2V_NAME).to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded {W2V_NAME} ({n_params:,} frozen parameters)")
    return model, extractor


def extract_w2v_features(model, extractor, by_label, words, device):
    """Frozen wav2vec2 last_hidden_state, mean-pooled over time -> (N, 768)."""
    import torch

    feats, labels = [], []
    with torch.no_grad():
        for label, clips in by_label.items():
            for wvf in clips:
                inputs = extractor(
                    wvf.squeeze(0).numpy(), sampling_rate=SAMPLE_RATE, return_tensors="pt"
                ).to(device)
                out = model(**inputs).last_hidden_state  # (1, T, 768)
                pooled = out.mean(dim=1).squeeze(0).cpu()
                feats.append(pooled)
                labels.append(words.index(label))
    X = torch.stack(feats).numpy()
    y = np.array(labels)
    print(f"wav2vec2 features: {X.shape[0]} clips x {X.shape[1]} dims")
    return X, y


def extract_mel_features(by_label, words, n_mels=80):
    """Raw log-mel spectrogram, mean-pooled over time -> (N, n_mels). Baseline (Part 2)."""
    import torch
    import torchaudio.transforms as T

    mel_tf = T.MelSpectrogram(sample_rate=SAMPLE_RATE, n_fft=1024, hop_length=256, n_mels=n_mels)
    feats, labels = [], []
    with torch.no_grad():
        for label, clips in by_label.items():
            for wvf in clips:
                mel = mel_tf(wvf)  # (1, n_mels, T)
                log_mel = torch.log(mel + 1e-9)
                pooled = log_mel.mean(dim=-1).squeeze(0)  # (n_mels,)
                feats.append(pooled)
                labels.append(words.index(label))
    X = torch.stack(feats).numpy()
    y = np.array(labels)
    print(f"mel-spectrogram features: {X.shape[0]} clips x {X.shape[1]} dims")
    return X, y


# ---------------------------------------------------------------------------
# Linear probe
# ---------------------------------------------------------------------------
def train_linear_probe(X, y, n_classes, epochs=150, lr=5e-2, seed=42, standardize=True,
                       weight_decay=1e-2, name="probe", log_every=30, verbose=True):
    """Train a single linear layer on frozen features, recording train/val loss + accuracy
    every epoch.

    The probe is trained on CPU and its weights are initialized from a NumPy RandomState
    (not the torch global generator), so the whole thing is deterministic and independent of
    torch/GPU state — the notebook, run.py and the README all produce identical numbers.
    """
    import torch
    import torch.nn.functional as F
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=seed, stratify=y
    )
    if standardize:
        scaler = StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

    Xtr = torch.tensor(X_train, dtype=torch.float32)
    ytr = torch.tensor(y_train, dtype=torch.long)
    Xte = torch.tensor(X_test, dtype=torch.float32)
    yte = torch.tensor(y_test, dtype=torch.long)

    d = X.shape[1]
    rng = np.random.RandomState(seed)
    W = (rng.randn(n_classes, d) / np.sqrt(d)).astype("float32")
    b = np.zeros(n_classes, dtype="float32")
    probe = torch.nn.Linear(d, n_classes)
    with torch.no_grad():
        probe.weight.copy_(torch.from_numpy(W))
        probe.bias.copy_(torch.from_numpy(b))
    opt = torch.optim.Adam(probe.parameters(), lr=lr, weight_decay=weight_decay)

    hist = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    if verbose:
        print(f"  training {name} probe ({d}-dim -> {n_classes} classes, "
              f"{len(y_train)} train / {len(y_test)} val):")
    for ep in range(epochs):
        probe.train()
        logits = probe(Xtr)
        loss = F.cross_entropy(logits, ytr)
        opt.zero_grad()
        loss.backward()
        opt.step()
        probe.eval()
        with torch.no_grad():
            val_logits = probe(Xte)
            vl = F.cross_entropy(val_logits, yte).item()
            ta = (logits.argmax(1) == ytr).float().mean().item()
            va = (val_logits.argmax(1) == yte).float().mean().item()
        hist["train_loss"].append(loss.item())
        hist["val_loss"].append(vl)
        hist["train_acc"].append(ta)
        hist["val_acc"].append(va)
        if verbose and ((ep + 1) % log_every == 0 or ep == 0):
            print(f"    epoch {ep + 1:3d}/{epochs} | train_loss {loss.item():.4f} "
                  f"| val_loss {vl:.4f} | train_acc {ta*100:5.1f}% | val_acc {va*100:5.1f}%")

    return {"test_acc": hist["val_acc"][-1], "n_test": len(y_test), "history": hist}


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------
def plot_probe_comparison(mel_acc, w2v_acc, n_classes, save_path: str | Path | None = None):
    random_baseline = 1.0 / n_classes
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    names = ["Random\nbaseline", "Raw mel-spec\n(mean-pooled)", "wav2vec2\n(frozen)"]
    vals = [random_baseline * 100, mel_acc * 100, w2v_acc * 100]
    colors = ["lightgray", "#fdae6b", "#6baed6"]
    bars = ax.bar(names, vals, color=colors, edgecolor="black")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.1f}%", ha="center", fontweight="bold")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_ylim(0, 105)
    ax.axhline(random_baseline * 100, color="gray", ls="--", lw=1)
    ax.set_title(f"Linear probe: wav2vec2 vs. mel-spectrogram ({n_classes}-way)")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=130, bbox_inches="tight")
        print(f"saved {save_path}")
    return fig


def plot_probe_training(hist_mel, hist_w2v, n_classes, save_path: str | Path | None = None):
    """Train vs. validation loss (left) and validation accuracy (right) per epoch, for the
    raw-mel and wav2vec2 linear probes."""
    rnd = 100.0 / n_classes
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    series = [("mel-spec", hist_mel, "#e6802a"), ("wav2vec2", hist_w2v, "#2c7fb8")]
    for nm, h, c in series:
        ep = np.arange(1, len(h["train_loss"]) + 1)
        axes[0].plot(ep, h["train_loss"], color=c, lw=1.8, label=f"{nm} train")
        axes[0].plot(ep, h["val_loss"], color=c, lw=1.8, ls="--", label=f"{nm} val")
        axes[1].plot(ep, np.array(h["val_acc"]) * 100, color=c, lw=1.8, label=f"{nm} val acc")
    axes[0].set_title("Linear-probe training: train vs. validation loss")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("cross-entropy loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[1].axhline(rnd, color="gray", ls=":", lw=1, label=f"random ({rnd:.0f}%)")
    axes[1].set_title("Linear-probe validation accuracy")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("validation accuracy (%)")
    axes[1].set_ylim(0, 100)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=130, bbox_inches="tight")
        print(f"saved {save_path}")
    return fig


def plot_tsne(X, y, words, save_path: str | Path | None = None, perplexity=15, seed=42):
    from sklearn.manifold import TSNE

    proj = TSNE(n_components=2, random_state=seed, perplexity=perplexity,
                init="pca").fit_transform(X)
    fig, ax = plt.subplots(figsize=(7, 6))
    colors_map = plt.cm.tab10(np.linspace(0, 1, len(words)))
    for i, word in enumerate(words):
        mask = y == i
        ax.scatter(proj[mask, 0], proj[mask, 1], c=[colors_map[i]], label=word, alpha=0.75, s=32)
    ax.legend()
    ax.set_title("Frozen wav2vec2 Embeddings (t-SNE)\nNo transcripts used during pretraining")
    ax.axis("off")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=130, bbox_inches="tight")
        print(f"saved {save_path}")
    return fig


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def run_probe(words=DEFAULT_PROBE_WORDS, n_per_class=40, epochs=150, seed=42, device=None,
              figure_dir: str | Path | None = None, tag=""):
    """Full Exercise-3 pipeline: extract both feature types, probe both, plot."""
    set_seed(seed)
    device = device if device is not None else get_device()
    n_classes = len(words)

    by_label = build_speechcommands_subset(words, n_per_class=n_per_class)

    w2v_model, w2v_extractor = load_wav2vec2(device)
    X_w2v, y = extract_w2v_features(w2v_model, w2v_extractor, by_label, words, device)
    X_mel, y_mel = extract_mel_features(by_label, words)
    assert np.array_equal(y, y_mel)

    mel_res = train_linear_probe(X_mel, y, n_classes, epochs=epochs, seed=seed, name="mel-spec")
    w2v_res = train_linear_probe(X_w2v, y, n_classes, epochs=epochs, seed=seed, name="wav2vec2")
    rand = 100.0 / n_classes
    print(f"\n=== {n_classes}-way linear probe ({words}) ===")
    print(f"Raw mel-spectrogram (mean-pooled): {mel_res['test_acc'] * 100:5.1f}%")
    print(f"wav2vec2 (frozen, mean-pooled):    {w2v_res['test_acc'] * 100:5.1f}%")
    print(f"Random baseline:                   {rand:5.1f}%")
    print(f"wav2vec2 improvement over mel:     +{(w2v_res['test_acc'] - mel_res['test_acc']) * 100:.1f} pts")

    figs = {}
    if figure_dir is not None:
        figure_dir = Path(figure_dir)
        suffix = f"_{tag}" if tag else ""
        figs["training"] = plot_probe_training(
            mel_res["history"], w2v_res["history"], n_classes,
            save_path=figure_dir / f"ex3_probe_training{suffix}.png")
        figs["bar"] = plot_probe_comparison(
            mel_res["test_acc"], w2v_res["test_acc"], n_classes,
            save_path=figure_dir / f"ex3_probe_comparison{suffix}.png")
        figs["tsne"] = plot_tsne(
            X_w2v, y, words, save_path=figure_dir / f"ex3_tsne{suffix}.png")

    return {
        "words": words,
        "n_classes": n_classes,
        "mel_acc": mel_res["test_acc"],
        "w2v_acc": w2v_res["test_acc"],
        "random_baseline": rand / 100,
        "mel_history": mel_res["history"],
        "w2v_history": w2v_res["history"],
        "X_w2v": X_w2v,
        "y": y,
        "figures": figs,
    }


if __name__ == "__main__":
    from .utils import FIGURE_DIR, LOG_DIR, Tee

    with Tee(LOG_DIR / "probe.log"):
        run_probe(DEFAULT_PROBE_WORDS, n_per_class=40, figure_dir=FIGURE_DIR, tag="4way")
