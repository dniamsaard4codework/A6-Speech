"""Exercise 4 — Voice cloning + accent/cross-lingual TTS with OpenVoice V2 + MeloTTS.

Reproduces Part 5 and fills in the pieces the teaching notebook leaves to the student
(proper MeloTTS init, real speaker-id keys, consistent accent list). It also adds the
Exercise-4 measurements: per-accent audio metrics (duration / RMS / mel spectral centroid)
and the cosine similarity between the reference tone-color embedding and the embedding
re-extracted from each generated clip.

Runs in the single project env (.venv, py3.10, torch 2.8 cu128) — on the **GPU**.

Design note: instead of OpenVoice's `se_extractor.get_se` (which pulls faster-whisper /
whisper-timestamped / PyAV — none of which build cleanly on this Windows machine and are
unnecessary for a clean reference clip), we extract the tone color directly with the
converter's own `ToneColorConverter.extract_se`, splitting the reference into ~10 s chunks
and averaging — exactly what get_se does internally, minus the VAD/ASR front-end.
"""
from __future__ import annotations

import os

# Windows without Developer Mode can't create the symlinks huggingface_hub uses in its
# cache (WinError 1314) — tell HF to copy files instead. Must be set before any HF import.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import json
import tempfile
from pathlib import Path

import numpy as np

from .utils import OUTPUT_DIR, FIGURE_DIR, DATA_DIR, set_seed

DEFAULT_TEXT = "I got the job!"
ACCENTS = ["us", "br", "india", "au"]

# accent -> (base-speaker SE file in OpenVoiceV2, MeloTTS English speaker-id key candidates)
ACCENT_MAP = {
    "us": ("en-us.pth", ["EN-US"]),
    "br": ("en-br.pth", ["EN-BR"]),
    "india": ("en-india.pth", ["EN_INDIA", "EN-INDIA"]),
    "au": ("en-au.pth", ["EN-AU"]),
}

# Cross-lingual base speakers (OpenVoiceV2 ships es/fr/zh/jp/kr SE files + MeloTTS langs)
LANG_MAP = {
    "en": ("en-default.pth", "EN"),
    "es": ("es.pth", "ES"),
    "fr": ("fr.pth", "FR"),
    "zh": ("zh.pth", "ZH"),
    "jp": ("jp.pth", "JP"),
    "kr": ("kr.pth", "KR"),
}

CROSS_LINGUAL_TEXTS = {
    "EN": "Hello, this is a test of cross lingual voice cloning.",
    "ES": "Hola, esta es una prueba de clonacion de voz entre idiomas.",
    "FR": "Bonjour, ceci est un test de clonage vocal interlingue.",
}

PROCESSED_DIR = DATA_DIR / "voice_clone" / "processed"


def _patch_torch_load():
    """OpenVoice/MeloTTS checkpoints are trusted local files saved the pre-2.6 way;
    torch>=2.6 defaults to weights_only=True, which rejects them. Default it to False."""
    import torch

    if getattr(torch.load, "_a6_patched", False):
        return
    _orig = torch.load

    def _load(*a, **k):
        k.setdefault("weights_only", False)
        return _orig(*a, **k)

    _load._a6_patched = True
    torch.load = _load


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_converter(device="cuda"):
    """Download OpenVoiceV2 checkpoints and load the ToneColorConverter."""
    from huggingface_hub import snapshot_download
    from openvoice.api import ToneColorConverter

    _patch_torch_load()
    ckpt_dir = snapshot_download(repo_id="myshell-ai/OpenVoiceV2")
    converter = ToneColorConverter(f"{ckpt_dir}/converter/config.json", device=device)
    converter.load_ckpt(f"{ckpt_dir}/converter/checkpoint.pth")
    print(f"OpenVoiceV2 loaded (device={device}).")
    return converter, ckpt_dir


_MELO_CACHE: dict = {}


def _ensure_nltk():
    """g2p_en (MeloTTS English front-end) needs these nltk resources. Newer nltk renamed
    the tagger with an `_eng` suffix, so download both spellings to be safe."""
    import nltk

    for res in ["averaged_perceptron_tagger", "averaged_perceptron_tagger_eng",
                "cmudict", "punkt", "punkt_tab"]:
        try:
            nltk.download(res, quiet=True)
        except Exception:
            pass


def load_melo(language="EN", device="cuda"):
    """Load (and cache) a MeloTTS base-speaker model for the given language."""
    if language in _MELO_CACHE:
        return _MELO_CACHE[language]
    _ensure_nltk()
    from melo.api import TTS

    _patch_torch_load()
    tts = TTS(language=language, device=device)
    spk2id = tts.hps.data.spk2id
    _MELO_CACHE[language] = (tts, spk2id)
    print(f"MeloTTS[{language}] loaded. Speaker ids: {list(spk2id.keys())}")
    return tts, spk2id


def _resolve_spk_id(spk2id, candidates):
    for key in candidates:
        if key in spk2id:
            return key, spk2id[key]
    for cand in candidates:
        suffix = cand.split("-")[-1].split("_")[-1].upper()
        for k in spk2id:
            if suffix in k.upper():
                return k, spk2id[k]
    k = list(spk2id.keys())[0]
    return k, spk2id[k]


# ---------------------------------------------------------------------------
# Tone color extraction (chunk-and-average; replaces se_extractor.get_se)
# ---------------------------------------------------------------------------
def _extract_se_from_path(path, converter, chunk_s=10.0, save_path=None):
    import soundfile as sf
    import librosa

    sr = converter.hps.data.sampling_rate
    y, _ = librosa.load(str(path), sr=sr, mono=True)
    peak = float(np.max(np.abs(y))) or 1.0
    y = (y / peak) * 0.95  # peak-normalize for stable embedding

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    proc = Path(tempfile.mkdtemp(prefix="se_", dir=str(PROCESSED_DIR)))
    chunk = int(chunk_s * sr)
    seg_paths = []
    if len(y) <= chunk:
        p = proc / "seg0.wav"
        sf.write(p, y, sr)
        seg_paths.append(str(p))
    else:
        for i, start in enumerate(range(0, len(y), chunk)):
            seg = y[start:start + chunk]
            if len(seg) < sr * 1.5:  # skip a short tail
                continue
            p = proc / f"seg{i}.wav"
            sf.write(p, seg, sr)
            seg_paths.append(str(p))
    se = converter.extract_se(seg_paths, se_save_path=str(save_path) if save_path else None)
    return se


def extract_se(reference_path, converter=None, device="cuda", out_path=None):
    out_path = Path(out_path or (OUTPUT_DIR / "target_se.pth"))
    if converter is None:
        converter, _ = load_converter(device)
    se = _extract_se_from_path(reference_path, converter, save_path=out_path)
    print(f"Extracted tone color embedding: shape {tuple(se.shape)} -> saved {out_path}")
    print('This single vector now encodes "what you sound like" — independent of what you say.')
    return se


# ---------------------------------------------------------------------------
# Synthesis (Exercise 4a)
# ---------------------------------------------------------------------------
def generate_accent(text, accent, target_se, converter, ckpt_dir, device="cuda",
                    out_dir=None, tau=0.3):
    import torch

    out_dir = Path(out_dir or OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    se_file, spk_candidates = ACCENT_MAP[accent]

    tts, spk2id = load_melo("EN", device=device)
    spk_key, spk_id = _resolve_spk_id(spk2id, spk_candidates)

    base_path = str(out_dir / f"base_{accent}.wav")
    out_path = str(out_dir / f"cloned_{accent}.wav")
    set_seed(42)  # MeloTTS (VITS) samples inference noise — seed for reproducible clips
    tts.tts_to_file(text, spk_id, base_path, speed=1.0)

    source_se = torch.load(f"{ckpt_dir}/base_speakers/ses/{se_file}",
                           map_location=device, weights_only=False)
    converter.convert(audio_src_path=base_path, src_se=source_se, tgt_se=target_se,
                      output_path=out_path, tau=tau)
    print(f"[{accent:6}] se={se_file} spk={spk_key} -> {out_path}")
    return out_path


def cross_lingual(texts=None, target_se=None, device="cuda", out_dir=None):
    import torch

    texts = texts or CROSS_LINGUAL_TEXTS
    converter, ckpt_dir = load_converter(device)
    out_dir = Path(out_dir or OUTPUT_DIR)
    if target_se is None:
        target_se = torch.load(OUTPUT_DIR / "target_se.pth", map_location=device, weights_only=False)
    paths = {}
    for lang, text_lang in texts.items():
        se_file, melo_lang = LANG_MAP[lang.lower()]
        tts, spk2id = load_melo(melo_lang, device=device)
        spk_key = list(spk2id.keys())[0]
        base_path = str(out_dir / f"base_{lang}.wav")
        out_path = str(out_dir / f"cloned_{lang}.wav")
        set_seed(42)
        tts.tts_to_file(text_lang, spk2id[spk_key], base_path, speed=1.0)
        source_se = torch.load(f"{ckpt_dir}/base_speakers/ses/{se_file}",
                               map_location=device, weights_only=False)
        converter.convert(audio_src_path=base_path, src_se=source_se, tgt_se=target_se,
                          output_path=out_path, tau=0.3)
        print(f"[{lang}] '{text_lang}' -> {out_path}")
        paths[lang] = out_path
    return paths


# ---------------------------------------------------------------------------
# Exercise 4a metrics + 4b cosine similarity
# ---------------------------------------------------------------------------
def audio_metrics(path, sr_target=22050):
    """Duration (s), RMS energy, and mel spectral centroid (Hz) for one clip."""
    import librosa

    y, sr = librosa.load(str(path), sr=sr_target, mono=True)
    duration = len(y) / sr
    rms = float(np.sqrt(np.mean(y ** 2)))
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=256, n_mels=80)
    mel_freqs = librosa.mel_frequencies(n_mels=80, fmin=0.0, fmax=sr / 2)
    denom = S.sum(axis=0) + 1e-9
    centroid = float(((mel_freqs[:, None] * S).sum(axis=0) / denom).mean())
    return {"duration_s": round(duration, 3), "rms_energy": round(rms, 5),
            "mel_centroid_hz": round(centroid, 1)}


def cosine_to_reference(reference_se, clip_paths, converter, device="cuda"):
    """Re-extract a tone-color embedding from each generated clip and compute cosine
    similarity to the reference embedding (Exercise 4b)."""
    import torch
    import torch.nn.functional as F

    ref = reference_se.flatten().float().cpu()
    sims = {}
    for name, path in clip_paths.items():
        gen = _extract_se_from_path(path, converter).flatten().float().cpu()
        sims[name] = round(F.cosine_similarity(ref.unsqueeze(0), gen.unsqueeze(0)).item(), 4)
        print(f"cos(reference, {name}) = {sims[name]}")
    return sims


# ---------------------------------------------------------------------------
# Part 5.4 — mel spectrogram grid (required visualization)
# ---------------------------------------------------------------------------
def plot_mel_grid(clip_paths, save_path=None):
    import matplotlib.pyplot as plt
    import torch
    import torchaudio
    import torchaudio.transforms as T

    style_colors = {"us": "purple", "br": "teal", "india": "royalblue", "au": "darkorange"}
    mel_tf = T.MelSpectrogram(sample_rate=22050, n_fft=1024, hop_length=256, n_mels=80)
    items = list(clip_paths.items())
    fig, axes = plt.subplots(1, len(items), figsize=(4.5 * len(items), 4))
    if len(items) == 1:
        axes = [axes]
    for ax, (style, path) in zip(axes, items):
        wvf, sr = torchaudio.load(str(path))
        if sr != 22050:
            wvf = T.Resample(sr, 22050)(wvf)
        mel = mel_tf(wvf[0].unsqueeze(0)).squeeze()
        log_mel = torch.log(mel + 1e-9)
        ax.imshow(log_mel.numpy(), aspect="auto", origin="lower", cmap="magma")
        ax.set_title(f"[{style.upper()}]", color=style_colors.get(style, "black"), fontweight="bold")
        ax.set_xlabel("Time frames")
        ax.set_ylabel("Mel bins")
    plt.suptitle("Same Cloned Voice, Four Accents — Tone Color Held Constant",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=130, bbox_inches="tight")
        print(f"saved {save_path}")
    return fig


# ---------------------------------------------------------------------------
# Full Exercise-4 pipeline
# ---------------------------------------------------------------------------
def run_full(reference_path, text=DEFAULT_TEXT, device="cuda", out_dir=None,
             do_cross_lingual=True):
    set_seed(42)
    out_dir = Path(out_dir or OUTPUT_DIR)
    converter, ckpt_dir = load_converter(device)

    target_se = extract_se(reference_path, converter=converter, device=device)

    paths = {acc: generate_accent(text, acc, target_se, converter, ckpt_dir,
                                  device=device, out_dir=out_dir) for acc in ACCENTS}

    metrics = {acc: audio_metrics(p) for acc, p in paths.items()}
    print("\nExercise 4a — per-accent audio metrics:")
    print(f"{'Accent':8s} {'Dur(s)':>8s} {'RMS':>10s} {'MelCentroid(Hz)':>16s}")
    for acc, m in metrics.items():
        print(f"{acc:8s} {m['duration_s']:8.3f} {m['rms_energy']:10.5f} {m['mel_centroid_hz']:16.1f}")

    print("\nExercise 4b — cosine similarity (reference vs. generated clip):")
    sims = cosine_to_reference(target_se, paths, converter, device=device)

    plot_mel_grid(paths, save_path=FIGURE_DIR / "ex4_mel_grid_accents.png")

    cl_paths = {}
    if do_cross_lingual:
        print("\nCross-lingual cloning:")
        cl_paths = cross_lingual(target_se=target_se, device=device, out_dir=out_dir)

    summary = {"text": text, "reference": str(reference_path), "accents": paths,
               "metrics": metrics, "cosine_sim": sims, "cross_lingual": cl_paths}
    with open(out_dir / "ex4_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nsaved {out_dir / 'ex4_summary.json'}")
    return summary
