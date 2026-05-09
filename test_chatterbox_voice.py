from pathlib import Path
from math import gcd

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from chatterbox.tts_turbo import ChatterboxTurboTTS


ROOT = Path("G:/LocalAgent")
REFERENCE = ROOT / "voices" / "rafie_one.wav"
CLEAN_REFERENCE = ROOT / "voices" / "rafie_one_clean.wav"
OUTPUT = ROOT / "tts" / "chatterbox_test.wav"

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
CLEAN_REFERENCE.parent.mkdir(parents=True, exist_ok=True)


def make_clean_reference(input_path: Path, output_path: Path) -> Path | None:
    if not input_path.exists():
        print("No reference voice found. Using Chatterbox default voice.")
        return None

    print(f"Cleaning reference voice: {input_path}")

    audio, sr = sf.read(str(input_path), dtype="float32", always_2d=True)

    audio = audio.mean(axis=1).astype(np.float32)

    target_sr = 16000
    if sr != target_sr:
        divisor = gcd(int(sr), target_sr)
        up = target_sr // divisor
        down = int(sr) // divisor
        audio = resample_poly(audio, up, down).astype(np.float32)

    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 0:
        audio = audio / peak * 0.85

    audio = audio.astype(np.float32)

    sf.write(str(output_path), audio, target_sr, subtype="PCM_16")

    print(f"Saved clean reference voice: {output_path}")
    return output_path


def save_wav(output_path: Path, wav, sample_rate: int) -> None:
    wav = wav.detach().cpu().float()

    if wav.ndim == 2:
        wav_np = wav.squeeze(0).numpy()
    else:
        wav_np = wav.numpy()

    wav_np = wav_np.astype(np.float32)

    peak = float(np.max(np.abs(wav_np))) if wav_np.size else 0.0
    if peak > 1.0:
        wav_np = wav_np / peak * 0.95

    sf.write(str(output_path), wav_np, sample_rate, subtype="PCM_16")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Torch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    clean_reference = make_clean_reference(REFERENCE, CLEAN_REFERENCE)

    print(f"Loading Chatterbox Turbo on {device}.")
    model = ChatterboxTurboTTS.from_pretrained(device=device)

    text = (
        "Hey, I am Rafie. This is voice one. "
        "I should sound smoother, warmer, and more natural now."
    )

    kwargs = {}

    if clean_reference:
        kwargs["audio_prompt_path"] = str(clean_reference)
        print(f"Using clean reference voice: {clean_reference}")

    print("Generating voice test...")

    with torch.inference_mode():
        wav = model.generate(
            text,
            norm_loudness=False,
            exaggeration=0.45,
            cfg_weight=0.35,
            **kwargs,
        )

    save_wav(OUTPUT, wav, model.sr)

    print(f"Saved test voice to: {OUTPUT}")


if __name__ == "__main__":
    main()