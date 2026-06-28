"""A6 Speech Processing — DL-AIT Assignment 6.

Student: Dechathon Niamsa-ard [st126235]

Modules
-------
tokenizer     : SpeechTokenizer (Exercise 1)
ctc           : ctc_collapse / ctc_forward_log_prob / TinyCTCModel / training (Exercise 2)
probe         : wav2vec2 vs mel-spectrogram linear probes (Exercise 3)
voice_clone   : OpenVoice V2 + MeloTTS voice cloning (Exercise 4)
utils         : seeding, device, logging tee, edit-distance / CER helpers
"""

__all__ = ["tokenizer", "ctc", "probe", "voice_clone", "utils"]
__student__ = "Dechathon Niamsa-ard [st126235]"
