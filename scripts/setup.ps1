# =====================================================================================
# A6 Speech Processing — reproducible single-environment setup (Windows / PowerShell)
# Student: Dechathon Niamsa-ard [st126235]
#
# One uv venv (Python 3.10) with GPU PyTorch (cu128, for the RTX 5060 Ti / sm_120) that
# runs ALL FOUR exercises, including OpenVoice V2 + MeloTTS voice cloning (Exercise 4).
#
# Run from the repository root:   ./scripts/setup.ps1
# =====================================================================================
$ErrorActionPreference = "Stop"

Write-Host "[1/4] Creating the venv + core stack (torch cu128, transformers, sklearn, librosa)..."
uv sync

Write-Host "[2/4] Cloning OpenVoice V2 + MeloTTS into third_party/ (if missing)..."
New-Item -ItemType Directory -Force third_party | Out-Null
if (-not (Test-Path third_party/OpenVoice)) {
    git clone --depth 1 https://github.com/myshell-ai/OpenVoice.git third_party/OpenVoice
}
if (-not (Test-Path third_party/MeloTTS)) {
    git clone --depth 1 https://github.com/myshell-ai/MeloTTS.git third_party/MeloTTS
}

Write-Host "[3/4] Installing the voice-cloning extras into the SAME venv..."
# Includes OpenVoice's real se_extractor.get_se front-end (faster-whisper / whisper-timestamped
# / PyAV / imageio-ffmpeg). whisper-timestamped drags in numpy>=2, so re-pin numpy afterwards.
uv pip install --python .venv/Scripts/python.exe -r requirements-voice.txt
uv pip install --python .venv/Scripts/python.exe "setuptools==75.8.2"   # pkg_resources for pykakasi
uv pip install --python .venv/Scripts/python.exe --no-deps -e third_party/OpenVoice -e third_party/MeloTTS
uv pip install --python .venv/Scripts/python.exe "numpy==1.26.4"        # wav2vec2 (Ex3) + MeloTTS need numpy<2

Write-Host "[4/4] Verifying GPU + key imports..."
.\.venv\Scripts\python.exe -c "import torch, transformers, librosa, numpy; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'); from openvoice.api import ToneColorConverter; from melo.api import TTS; print('OpenVoice + MeloTTS import OK')"

Write-Host ""
Write-Host "Setup complete. Next:"
Write-Host "  python run.py --model ctc --epochs 300 --train"
Write-Host "  python run.py --model wav2vec2-probe --dataset speechcommands --classes yes,no,stop,go --train"
Write-Host "  # place your ~10-30s clip at data/voice_clone/my_voice.wav, then:"
Write-Host "  python run.py --model voice-clone --extract-se --reference data/voice_clone/my_voice.wav"
Write-Host "  python run.py --model voice-clone --accent all --text 'I got the job!' --generate"
