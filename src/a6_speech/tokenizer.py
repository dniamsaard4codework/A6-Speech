"""Exercise 1 — Speech tokenization.

Faithful reproduction of the lab's `SpeechTokenizer` (Part 1) plus helpers to build the
Exercise-1(a) table and the "NLP tokens vs speech chars vs accent tokens" comparison figure.
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ---------------------------------------------------------------------------
# Part 1 — SpeechTokenizer (verbatim from the lab)
# ---------------------------------------------------------------------------
class SpeechTokenizer:
    """Character-level tokenizer for TTS."""

    ACCENTS = ["[EN-US]", "[EN-BR]", "[EN-INDIA]", "[EN-AU]", "[EN-DEFAULT]"]

    def __init__(self):
        chars = " !',-.?abcdefghijklmnopqrstuvwxyz"
        self.vocab = {c: i + 3 for i, c in enumerate(chars)}
        self.vocab["<PAD>"] = 0
        self.vocab["<BOS>"] = 1
        self.vocab["<EOS>"] = 2
        for i, a in enumerate(self.ACCENTS):
            self.vocab[a] = len(self.vocab)
        self.inv_vocab = {v: k for k, v in self.vocab.items()}

    def normalize(self, text):
        text = text.lower()
        text = re.sub(r"dr\.", "doctor", text)
        text = re.sub(r"mr\.", "mister", text)
        text = re.sub(r"(\d+)", lambda m: self._num_to_words(int(m.group())), text)
        text = re.sub(r"[^a-z !',\-.?\[\]]", "", text)
        return text.strip()

    def _num_to_words(self, n):
        words = {
            0: "zero", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
            6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten",
        }
        return words.get(n, str(n))

    def encode(self, text, add_special=True):
        """Tokenize text. Accent tags are single tokens, not character-by-character."""
        tag_pattern = "|".join(re.escape(a) for a in self.ACCENTS)
        parts = re.split(f"({tag_pattern})", text)
        tokens = []
        if add_special:
            tokens.append(self.vocab["<BOS>"])
        for part in parts:
            if part in self.ACCENTS:
                tokens.append(self.vocab[part])  # accent = single token
            else:
                normalized = self.normalize(part)
                for ch in normalized:
                    if ch in self.vocab:
                        tokens.append(self.vocab[ch])
        if add_special:
            tokens.append(self.vocab["<EOS>"])
        return tokens

    def decode(self, ids):
        return "".join(
            self.inv_vocab.get(i, "?")
            for i in ids
            if i not in (self.vocab["<PAD>"], self.vocab["<BOS>"], self.vocab["<EOS>"])
        )

    def __len__(self):
        return len(self.vocab)

    # -- convenience -------------------------------------------------------
    @property
    def accent_ids(self) -> set[int]:
        return {self.vocab[a] for a in self.ACCENTS}


# Sentences from Exercise 1(a)
EX1_SENTENCES = [
    "Hello, how are you?",
    "Dr. Smith prescribed 10 tablets.",
    "[EN-US] I got the job!",
    "[EN-BR] I lost my wallet.",
    "[EN-INDIA] This is completely unacceptable!",
]


def analyze_sentence(tok: SpeechTokenizer, text: str) -> dict:
    """Return char-token count, total tokens (with BOS/EOS), and accent tag id (if any)."""
    ids_no_special = tok.encode(text, add_special=False)
    ids_with_special = tok.encode(text, add_special=True)
    accent_ids = tok.accent_ids
    n_char = sum(1 for i in ids_no_special if i not in accent_ids)
    present_accents = [i for i in ids_no_special if i in accent_ids]
    accent_id = present_accents[0] if present_accents else None
    accent_name = tok.inv_vocab[accent_id] if accent_id is not None else None
    return {
        "sentence": text,
        "n_char_tokens": n_char,
        "n_tokens_with_special": len(ids_with_special),
        "accent_id": accent_id,
        "accent_name": accent_name,
        "normalized": tok.decode(ids_with_special),
        "ids": ids_with_special,
    }


def build_ex1_table(tok: SpeechTokenizer | None = None) -> list[dict]:
    """Compute the Exercise-1(a) table for all five sentences."""
    tok = tok or SpeechTokenizer()
    return [analyze_sentence(tok, s) for s in EX1_SENTENCES]


def print_ex1_table(rows: list[dict]) -> None:
    print(f"{'Sentence':45s} | {'#Char':>5s} | {'#Tok(BOS/EOS)':>13s} | {'Accent id':>9s}")
    print("-" * 82)
    for r in rows:
        acc = "—" if r["accent_id"] is None else f"{r['accent_id']} {r['accent_name']}"
        s = r["sentence"] if len(r["sentence"]) <= 44 else r["sentence"][:41] + "..."
        print(f"{s:45s} | {r['n_char_tokens']:5d} | {r['n_tokens_with_special']:13d} | {acc:>9s}")


# ---------------------------------------------------------------------------
# Required Part-1 visualization: NLP tokens vs speech chars vs accent tokens
# ---------------------------------------------------------------------------
def plot_tokenization_comparison(
    sentence: str = "[EN-US] I got the job!",
    save_path: str | Path | None = None,
):
    """Three-row figure contrasting how NLP, speech-char, and accent tokenization
    segment the same sentence: word-level (few tokens) vs character-level (many
    tokens) vs a single accent control token."""
    tok = SpeechTokenizer()

    # Strip the accent tag for the "text body"
    accent_match = re.match(r"^(\[[A-Z\-]+\])\s*(.*)$", sentence)
    if accent_match:
        accent_tag, body = accent_match.group(1), accent_match.group(2)
    else:
        accent_tag, body = None, sentence

    # NLP (word-level) tokens
    nlp_tokens = body.split()
    # Speech character tokens (normalized)
    char_tokens = list(tok.normalize(body))
    char_tokens_disp = ["␣" if c == " " else c for c in char_tokens]

    fig, axes = plt.subplots(3, 1, figsize=(13, 5.2))
    fig.suptitle(
        f'Tokenization of "{sentence}" — three different granularities',
        fontsize=13, fontweight="bold",
    )

    def draw_row(ax, tokens, color, title, max_show=None):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.set_title(title, loc="left", fontsize=11, fontweight="bold")
        show = tokens if max_show is None else tokens[:max_show]
        n = len(show)
        if n == 0:
            return
        pad = 0.004
        w = (1.0 - pad * (n + 1)) / max(n, 1)
        w = min(w, 0.14)
        x = pad
        for t in show:
            ax.add_patch(
                mpatches.FancyBboxPatch(
                    (x, 0.25), w, 0.5,
                    boxstyle="round,pad=0.006,rounding_size=0.02",
                    linewidth=1.2, edgecolor="black", facecolor=color, alpha=0.85,
                )
            )
            ax.text(x + w / 2, 0.5, str(t), ha="center", va="center",
                    fontsize=10, fontweight="bold")
            x += w + pad

    # Row 1: NLP word tokens
    draw_row(axes[0], nlp_tokens, "#9ecae1",
             f"NLP word tokens  —  {len(nlp_tokens)} tokens (each token = a whole word / sub-word)")
    # Row 2: speech character tokens
    draw_row(axes[1], char_tokens_disp, "#fdae6b",
             f"Speech character tokens  —  {len(char_tokens)} tokens (model must stretch each over many audio frames)")
    # Row 3: accent control token
    if accent_tag is not None:
        acc_id = tok.vocab[accent_tag]
        draw_row(axes[2], [f"{accent_tag}  (id {acc_id})"], "#a1d99b",
                 "Accent control token  —  1 single token conditions the ENTIRE utterance (like [CLS] in NLP)")
    else:
        axes[2].axis("off")
        axes[2].set_title("Accent control token — none in this sentence", loc="left",
                          fontsize=11, fontweight="bold")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if save_path:
        fig.savefig(save_path, dpi=130, bbox_inches="tight")
        print(f"saved {save_path}")
    return fig


if __name__ == "__main__":
    from .utils import FIGURE_DIR

    tokenizer = SpeechTokenizer()
    print(f"Vocab size: {len(tokenizer)}")
    print("Accent token IDs:", {a: tokenizer.vocab[a] for a in tokenizer.ACCENTS})
    print()
    table = build_ex1_table(tokenizer)
    print_ex1_table(table)
    plot_tokenization_comparison(save_path=FIGURE_DIR / "ex1_tokenization_comparison.png")
