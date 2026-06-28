"""Exercise 2 — CTC (Connectionist Temporal Classification).

Faithful reproduction of the lab's CTC code (Part 3): collapsing, the log-domain forward
algorithm, the synthetic frame task, the TinyCTC BiLSTM model, plus the Exercise-2 additions:
character-error-rate tracking (2c) and a configurable `frames_per_char` for the duration
study (2d).
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from .utils import set_seed, get_device, char_error_rate

# ---------------------------------------------------------------------------
# Part 3.2 — CTC collapsing
# ---------------------------------------------------------------------------
BLANK = "_"


def ctc_collapse(alignment):
    """Merge consecutive duplicates, then remove blanks."""
    merged = []
    for ch in alignment:
        if not merged or ch != merged[-1]:
            merged.append(ch)
    return "".join(ch for ch in merged if ch != BLANK)


# Exercise 2(a): three hand-built alignments that should ALL collapse to "hello".
# Note the blank `_` between the two l's — without it the repeated l's would merge into
# one l ("helo"), which is exactly why CTC needs the blank token.
EX2A_EXAMPLES = [
    list("hel_lo"),
    list("hheell_lloo"),
    list("_hhh_eee_lll_lll_ooo_"),
]


# ---------------------------------------------------------------------------
# Part 3.4 — Forward algorithm (log domain)
# ---------------------------------------------------------------------------
NEG_INF = -1e9


def log_add(a, b):
    """log(exp(a) + exp(b)), numerically stable."""
    if a == NEG_INF:
        return b
    if b == NEG_INF:
        return a
    m = max(a, b)
    return m + np.log(np.exp(a - m) + np.exp(b - m))


def ctc_forward_log_prob(log_probs, labels, blank=0):
    """Compute log P_CTC(labels | log_probs).

    Args:
        log_probs: (T, V) log-softmax per frame
        labels: list of label indices (length L, no blanks)
        blank: blank token index
    Returns: log probability (float)
    """
    T, V = log_probs.shape
    L = len(labels)

    # Extended sequence: [blank, label1, blank, label2, blank, ...]
    ext = [blank]
    for lab in labels:
        ext += [lab, blank]
    S = len(ext)  # = 2L + 1

    alpha = np.full((T, S), NEG_INF)
    alpha[0, 0] = log_probs[0, ext[0]]
    if S > 1:
        alpha[0, 1] = log_probs[0, ext[1]]

    for t in range(1, T):
        for s in range(S):
            stay = alpha[t - 1, s]
            prev = alpha[t - 1, s - 1] if s - 1 >= 0 else NEG_INF
            skip = NEG_INF
            if s - 2 >= 0 and ext[s] != blank and ext[s] != ext[s - 2]:
                skip = alpha[t - 1, s - 2]
            best_prev = log_add(log_add(stay, prev), skip)
            alpha[t, s] = best_prev + log_probs[t, ext[s]]

    if S == 1:
        return alpha[T - 1, S - 1]
    return log_add(alpha[T - 1, S - 1], alpha[T - 1, S - 2])


# ---------------------------------------------------------------------------
# Part 3.5 — Synthetic frame task + TinyCTC model
# ---------------------------------------------------------------------------
ALPHABET = list("helo wrd")
CHAR2IDX = {c: i + 1 for i, c in enumerate(ALPHABET)}  # 0 = blank
IDX2CHAR = {i + 1: c for i, c in enumerate(ALPHABET)}
VOCAB_SIZE = len(ALPHABET) + 1  # 8 chars + 1 blank = 9
N_MELS = 20
WORDS = ["hello", "world", "hero", "red", "led", "doer"]


def synthesize_frames(word, frames_per_char=(3, 8)):
    """Synthesize a frame sequence where each character spans a variable number of
    frames (sampled in `frames_per_char`) plus Gaussian noise."""
    frames, char_at_frame = [], []
    for ch in word:
        n = random.randint(*frames_per_char)
        base = np.zeros(N_MELS)
        base[CHAR2IDX[ch] % N_MELS] = 3.0  # signature peak
        for _ in range(n):
            frames.append(base + np.random.randn(N_MELS) * 0.5)
            char_at_frame.append(ch)
    return np.stack(frames), char_at_frame


def _build_model():
    import torch.nn as nn
    import torch.nn.functional as F

    class TinyCTCModel(nn.Module):
        """BiLSTM encoder -> linear -> log-softmax."""

        def __init__(self, in_dim=N_MELS, hidden=64, vocab=VOCAB_SIZE):
            super().__init__()
            self.lstm = nn.LSTM(in_dim, hidden, batch_first=True, bidirectional=True)
            self.fc = nn.Linear(hidden * 2, vocab)

        def forward(self, x):
            h, _ = self.lstm(x)
            return F.log_softmax(self.fc(h), dim=-1)  # (B, T, V)

    return TinyCTCModel()


def greedy_decode(model, x):
    """Greedy CTC decode: argmax per frame -> ctc_collapse."""
    import torch

    with torch.no_grad():
        log_probs = model(x).squeeze(0)  # (T, V)
    pred_ids = log_probs.argmax(dim=-1).tolist()
    pred_chars = [IDX2CHAR.get(i, BLANK) if i != 0 else BLANK for i in pred_ids]
    return ctc_collapse(pred_chars), pred_chars


def _make_eval_set(words=WORDS, frames_per_char=(3, 8), seed=123):
    """Fixed evaluation clips (isolated RNG) so the CER curve is smooth/comparable."""
    np_state, py_state = np.random.get_state(), random.getstate()
    random.seed(seed)
    np.random.seed(seed)
    eval_set = [(w, synthesize_frames(w, frames_per_char)[0]) for w in words]
    np.random.set_state(np_state)
    random.setstate(py_state)
    return eval_set


def eval_mean_cer(model, eval_set, device):
    import torch

    cers = []
    for word, frames in eval_set:
        x = torch.tensor(frames, dtype=torch.float32).unsqueeze(0).to(device)
        decoded, _ = greedy_decode(model, x)
        cers.append(char_error_rate(decoded, word))
    return float(np.mean(cers))


def eval_mean_loss(model, eval_set, ctc_loss_fn, device):
    """Mean CTC loss over the held-out eval set (validation loss)."""
    import torch

    losses = []
    with torch.no_grad():
        for word, frames in eval_set:
            x = torch.tensor(frames, dtype=torch.float32).unsqueeze(0).to(device)
            targets = torch.tensor([CHAR2IDX[c] for c in word], dtype=torch.long).to(device)
            log_probs = model(x).transpose(0, 1)
            il = torch.tensor([log_probs.size(0)])
            tl = torch.tensor([len(targets)])
            losses.append(ctc_loss_fn(log_probs, targets, il, tl).item())
    return float(np.mean(losses))


def train_ctc(epochs=300, frames_per_char=(3, 8), lr=1e-2, seed=42, device=None, log_every=50):
    """Train the TinyCTC model, tracking CTC loss AND character error rate per step (Ex 2c)."""
    import torch
    import torch.nn as nn

    set_seed(seed)
    device = device if device is not None else get_device()
    model = _build_model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    ctc_loss_fn = nn.CTCLoss(blank=0, zero_infinity=True)

    eval_set = _make_eval_set(frames_per_char=frames_per_char)

    losses, val_losses, cers = [], [], []
    step_below_10 = None
    print(f"Training TinyCTC | epochs={epochs} frames_per_char={frames_per_char} lr={lr} device={device}")
    for step in range(epochs):
        word = random.choice(WORDS)
        frames, _ = synthesize_frames(word, frames_per_char)
        x = torch.tensor(frames, dtype=torch.float32).unsqueeze(0).to(device)
        targets = torch.tensor([CHAR2IDX[c] for c in word], dtype=torch.long).to(device)

        log_probs = model(x).transpose(0, 1)  # CTCLoss wants (T, B, V)
        input_lengths = torch.tensor([log_probs.size(0)])
        target_lengths = torch.tensor([len(targets)])

        loss = ctc_loss_fn(log_probs, targets, input_lengths, target_lengths)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        # Validation: mean CTC loss and Exercise-2c character error rate over the eval set
        val_losses.append(eval_mean_loss(model, eval_set, ctc_loss_fn, device))
        cer = eval_mean_cer(model, eval_set, device)
        cers.append(cer)
        if step_below_10 is None and cer < 0.10:
            step_below_10 = step + 1

        if (step + 1) % log_every == 0:
            print(
                f"Step {step + 1:3d} | train loss {np.mean(losses[-log_every:]):.4f} "
                f"| val loss {val_losses[-1]:.4f} | mean CER {cer * 100:5.1f}%"
            )

    if step_below_10 is not None:
        print(f"==> CER first dropped below 10% at step {step_below_10}.")
    else:
        print("==> CER never dropped below 10% within the training budget.")
    return {
        "model": model,
        "losses": losses,
        "val_losses": val_losses,
        "cers": cers,
        "step_below_10": step_below_10,
        "frames_per_char": frames_per_char,
        "device": str(device),
        "eval_set": eval_set,
    }


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------
def plot_loss_cer(result, save_path: str | Path | None = None):
    """Two-panel: train vs. validation CTC loss, and character error rate vs. step (Ex 2c)."""
    losses, cers = result["losses"], result["cers"]
    val_losses = result.get("val_losses")
    step_below_10 = result["step_below_10"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))

    axes[0].plot(losses, color="steelblue", lw=1, alpha=0.7, label="train loss (per step)")
    if val_losses is not None:
        axes[0].plot(val_losses, color="crimson", lw=1.8, label="val loss (eval set)")
    axes[0].set_title("CTC train vs. validation loss (toy frame-to-character task)")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("CTC loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    cer_pct = np.array(cers) * 100
    axes[1].plot(cer_pct, color="crimson", label="mean CER")
    axes[1].axhline(10, color="gray", ls="--", lw=1, label="10% threshold")
    if step_below_10 is not None:
        axes[1].axvline(step_below_10 - 1, color="green", ls=":", lw=1.5,
                        label=f"CER<10% @ step {step_below_10}")
    axes[1].set_title("Character Error Rate vs. Training Step (Ex 2c)")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("CER (%)")
    axes[1].set_ylim(-2, max(100, cer_pct[:5].max() + 5))
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=130, bbox_inches="tight")
        print(f"saved {save_path}")
    return fig


def plot_greedy_grid(model, frames_per_char=(3, 8), words=WORDS, seed=7,
                     save_path: str | Path | None = None, device=None):
    """Reproduce the lab's greedy-decoding grid: per-frame predicted char + collapsed text.
    Returns (fig, n_correct)."""
    import torch

    device = device if device is not None else next(model.parameters()).device
    model.eval()

    # Fixed clips for a reproducible figure
    np_state, py_state = np.random.get_state(), random.getstate()
    random.seed(seed)
    np.random.seed(seed)
    clips = [(w, synthesize_frames(w, frames_per_char)) for w in words]
    np.random.set_state(np_state)
    random.setstate(py_state)

    fig, axes = plt.subplots(len(words), 1, figsize=(12, 2 * len(words)))
    if len(words) == 1:
        axes = [axes]
    n_correct = 0
    colors = plt.cm.tab10(np.linspace(0, 1, len(ALPHABET) + 1))

    for ax, (word, (frames, _true)) in zip(axes, clips):
        x = torch.tensor(frames, dtype=torch.float32).unsqueeze(0).to(device)
        decoded, pred_chars_raw = greedy_decode(model, x)
        T = len(pred_chars_raw)
        for t, ch in enumerate(pred_chars_raw):
            idx = 0 if ch == BLANK else ALPHABET.index(ch) + 1
            ax.bar(t, 1, color=colors[idx], edgecolor="white", linewidth=0.3)
            if ch != BLANK:
                ax.text(t, 0.5, ch, ha="center", va="center", fontsize=8, color="white")
        ax.set_xlim(0, T)
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_xticks([])
        ok = decoded == word
        n_correct += int(ok)
        ax.set_ylabel(f'"{word}"', rotation=0, labelpad=35, fontsize=10, va="center")
        ax.set_title(
            f'Raw greedy output ({T} frames) -> collapsed: "{decoded}"  [{"correct" if ok else "wrong"}]',
            fontsize=9, loc="left",
        )

    plt.suptitle(
        f"CTC Greedy Decoding: frames_per_char={frames_per_char}  "
        f"({n_correct}/{len(words)} correct)",
        fontsize=13,
    )
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=130, bbox_inches="tight")
        print(f"saved {save_path}")
    model.train()
    return fig, n_correct


def per_word_accuracy(model, frames_per_char=(3, 8), n=300, seed=123, device=None):
    """Per-word greedy-decode accuracy (reveals the doubled-letter failure in Ex 2d)."""
    import torch

    device = device if device is not None else next(model.parameters()).device
    model.eval()
    np_state, py_state = np.random.get_state(), random.getstate()
    random.seed(seed)
    np.random.seed(seed)
    counts = {w: [0, 0] for w in WORDS}
    for _ in range(n):
        word = random.choice(WORDS)
        frames, _ = synthesize_frames(word, frames_per_char)
        x = torch.tensor(frames, dtype=torch.float32).unsqueeze(0).to(device)
        decoded, _ = greedy_decode(model, x)
        counts[word][1] += 1
        counts[word][0] += int(decoded == word)
    np.random.set_state(np_state)
    random.setstate(py_state)
    model.train()
    return {w: (c[0] / c[1] if c[1] else float("nan")) for w, c in counts.items()}


def evaluate_accuracy(model, frames_per_char=(3, 8), n=300, seed=999, device=None):
    """Word-level accuracy over `n` fresh random clips (robust number for Ex 2d)."""
    import torch

    device = device if device is not None else next(model.parameters()).device
    model.eval()
    np_state, py_state = np.random.get_state(), random.getstate()
    random.seed(seed)
    np.random.seed(seed)
    correct = 0
    for _ in range(n):
        word = random.choice(WORDS)
        frames, _ = synthesize_frames(word, frames_per_char)
        x = torch.tensor(frames, dtype=torch.float32).unsqueeze(0).to(device)
        decoded, _ = greedy_decode(model, x)
        correct += int(decoded == word)
    np.random.set_state(np_state)
    random.setstate(py_state)
    model.train()
    return correct / n


# ---------------------------------------------------------------------------
# Exercise 2(a) and 2(b) demonstrations
# ---------------------------------------------------------------------------
def demo_collapse(examples=EX2A_EXAMPLES):
    print("Exercise 2(a) — alignments that collapse to the same word:")
    results = []
    for ex in examples:
        out = ctc_collapse(ex)
        results.append(out)
        print(f"  {''.join(ex):20s} -> {out}")
    agree = len(set(results)) == 1
    print(f"  All agree: {agree}  (-> '{results[0]}')")
    return results, agree


def demo_forward_hel_leh(seed=42, T=6):
    """Exercise 2(b): P_CTC for 'hel' vs 'leh' on the SAME random log_probs matrix."""
    import torch
    import torch.nn.functional as F

    set_seed(seed)
    logits = torch.randn(T, VOCAB_SIZE)
    log_probs = F.log_softmax(logits, dim=-1).numpy()

    labels_hel = [CHAR2IDX["h"], CHAR2IDX["e"], CHAR2IDX["l"]]
    labels_leh = [CHAR2IDX["l"], CHAR2IDX["e"], CHAR2IDX["h"]]
    lp_hel = ctc_forward_log_prob(log_probs, labels_hel, blank=0)
    lp_leh = ctc_forward_log_prob(log_probs, labels_leh, blank=0)
    print("Exercise 2(b) — same random log_probs, two targets of equal length:")
    print(f"  log P_CTC('hel') = {lp_hel:.4f}   P = {np.exp(lp_hel):.3e}")
    print(f"  log P_CTC('leh') = {lp_leh:.4f}   P = {np.exp(lp_leh):.3e}")
    print(f"  Different: {abs(lp_hel - lp_leh) > 1e-6}")
    return {"log_probs": log_probs, "hel": lp_hel, "leh": lp_leh}


if __name__ == "__main__":
    from .utils import FIGURE_DIR, LOG_DIR, Tee

    with Tee(LOG_DIR / "ctc_train.log"):
        demo_collapse()
        print()
        demo_forward_hel_leh()
        print()
        res = train_ctc(epochs=300)
        plot_loss_cer(res, save_path=FIGURE_DIR / "ex2_loss_cer.png")
        plot_greedy_grid(res["model"], frames_per_char=(3, 8),
                         save_path=FIGURE_DIR / "ex2_greedy_grid_3_8.png")
        acc = evaluate_accuracy(res["model"], frames_per_char=(3, 8))
        print(f"Word accuracy @ frames_per_char=(3,8): {acc * 100:.1f}%")
