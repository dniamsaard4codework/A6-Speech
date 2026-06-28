#!/usr/bin/env python3
"""A6 Speech Processing — training / inference CLI.

Student: Dechathon Niamsa-ard [st126235]

Examples (exactly the commands from A6_Assignment.md)
-----------------------------------------------------
    # Train the toy CTC model (Part 3 / Exercise 2)
    python run.py --model ctc --epochs 300 --train

    # Linear-probe a pretrained wav2vec2 checkpoint (Part 4 / Exercise 3)
    python run.py --model wav2vec2-probe --dataset speechcommands --classes yes,no,stop,go --train

    # Extract tone color from your reference clip   [run with the voice env]
    python run.py --model voice-clone --extract-se --reference data/voice_clone/my_voice.wav

    # Synthesize in a given style with your cloned voice
    python run.py --model voice-clone --accent us --text "I got the job!" --generate

    # Synthesize all styles for comparison (+ metrics, cosine sims, mel grid)
    python run.py --model voice-clone --accent all --text "Hello world" --generate

    # Cross-lingual cloning
    python run.py --model voice-clone --language es --text "Hola, como estas?" --generate

Note: `ctc` and `wav2vec2-probe` use the main cu128 (GPU) env (.venv).
      `voice-clone` requires the dedicated voice env (.venv-voice) where OpenVoice + MeloTTS
      are installed (CPU torch). See README.md.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the src/ package importable whether or not it's pip-installed.
REPO_ROOT = Path(__file__).resolve().parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import matplotlib

matplotlib.use("Agg")  # headless: save figures, never try to open a window


def cmd_ctc(args):
    from a6_speech import ctc
    from a6_speech.utils import FIGURE_DIR, LOG_DIR, Tee, get_device

    with Tee(LOG_DIR / "ctc_train.log"):
        get_device()
        ctc.demo_collapse()
        print()
        ctc.demo_forward_hel_leh()
        print()
        res = ctc.train_ctc(epochs=args.epochs)
        ctc.plot_loss_cer(res, save_path=FIGURE_DIR / "ex2_loss_cer.png")
        ctc.plot_greedy_grid(res["model"], frames_per_char=(3, 8),
                             save_path=FIGURE_DIR / "ex2_greedy_grid_3_8.png")
        acc_38 = ctc.evaluate_accuracy(res["model"], frames_per_char=(3, 8))
        print(f"Word accuracy @ frames_per_char=(3,8): {acc_38 * 100:.1f}%")

        # Exercise 2d: shorter character durations
        print("\n=== Exercise 2d: frames_per_char=(1,2) ===")
        res_short = ctc.train_ctc(epochs=args.epochs, frames_per_char=(1, 2))
        ctc.plot_greedy_grid(res_short["model"], frames_per_char=(1, 2),
                             save_path=FIGURE_DIR / "ex2_greedy_grid_1_2.png")
        acc_12 = ctc.evaluate_accuracy(res_short["model"], frames_per_char=(1, 2))
        print(f"Word accuracy @ frames_per_char=(1,2): {acc_12 * 100:.1f}%")
        print(f"\nSummary: (3,8) accuracy={acc_38*100:.1f}%  vs  (1,2) accuracy={acc_12*100:.1f}%")

        # Per-word breakdown reveals the doubled-letter ("hello") failure mode (Ex 2d)
        pw_38 = ctc.per_word_accuracy(res["model"], frames_per_char=(3, 8))
        pw_12 = ctc.per_word_accuracy(res_short["model"], frames_per_char=(1, 2))
        print(f"\n{'word':6s} {'(3,8)':>8s} {'(1,2)':>8s}")
        for w in ctc.WORDS:
            print(f"{w:6s} {pw_38[w]*100:7.1f}% {pw_12[w]*100:7.1f}%")


def cmd_probe(args):
    from a6_speech import probe
    from a6_speech.utils import FIGURE_DIR, LOG_DIR, Tee, get_device

    words = [w.strip() for w in args.classes.split(",") if w.strip()]
    tag = f"{len(words)}way"
    with Tee(LOG_DIR / f"probe_{tag}.log"):
        get_device()
        # Note: probe uses its own fixed epoch budget (run_probe default), not --epochs,
        # so notebook and CLI always agree.
        probe.run_probe(words, n_per_class=args.n_per_class, figure_dir=FIGURE_DIR, tag=tag)


def cmd_voice(args):
    from a6_speech import voice_clone as vc
    from a6_speech.utils import LOG_DIR, OUTPUT_DIR, Tee
    import torch

    device = "cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
    with Tee(LOG_DIR / "voice_clone.log", mode="a"):
        print(f"voice-clone | device={device}")
        if args.extract_se:
            ref = args.reference or "data/voice_clone/my_voice.wav"
            vc.extract_se(ref, device=device)
            return

        if args.generate:
            text = args.text or vc.DEFAULT_TEXT

            # --accent all runs the complete Ex4 pipeline and extracts the SE itself.
            if args.accent == "all":
                vc.run_full(args.reference or "data/voice_clone/my_voice.wav",
                            text=text, device=device)
                return

            # Single accent / language need a previously-extracted tone color.
            target_se_path = OUTPUT_DIR / "target_se.pth"
            if not target_se_path.exists():
                sys.exit("No target_se.pth found. Run with --extract-se --reference <clip> first.")
            target_se = torch.load(target_se_path, map_location=device, weights_only=False)

            if args.language:
                texts = {args.language.upper(): text}
                vc.cross_lingual(texts=texts, target_se=target_se, device=device)
            elif args.accent:
                converter, ckpt_dir = vc.load_converter(device)
                vc.generate_accent(text, args.accent, target_se, converter, ckpt_dir, device=device)
            else:
                sys.exit("Specify --accent {us,br,india,au,all} or --language <code> with --generate.")
            return

        sys.exit("voice-clone: nothing to do. Use --extract-se or --generate.")


def build_parser():
    p = argparse.ArgumentParser(description="A6 Speech Processing CLI [st126235]")
    p.add_argument("--model", required=True,
                   choices=["ctc", "wav2vec2-probe", "voice-clone"])
    # ctc
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--train", action="store_true", help="run training")
    # probe
    p.add_argument("--dataset", default="speechcommands")
    p.add_argument("--classes", default="yes,no,stop,go")
    p.add_argument("--n-per-class", type=int, default=40)
    # voice-clone
    p.add_argument("--extract-se", action="store_true")
    p.add_argument("--reference", default=None)
    p.add_argument("--accent", default=None, choices=[None, "us", "br", "india", "au", "all"])
    p.add_argument("--language", default=None)
    p.add_argument("--text", default=None)
    p.add_argument("--generate", action="store_true")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.model == "ctc":
        cmd_ctc(args)
    elif args.model == "wav2vec2-probe":
        cmd_probe(args)
    elif args.model == "voice-clone":
        cmd_voice(args)


if __name__ == "__main__":
    main()
