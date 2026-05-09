from __future__ import annotations

import json
import time
import wave
from datetime import datetime, timezone
from pathlib import Path

import sounddevice as sd


TRAINING_PHRASES = [
    "A E I O U.",
    "A, B, C, D, E, F, G.",
    "H, I, J, K, L, M, N, O, P.",
    "Q, R, S, T, U, V, W, X, Y, Z.",
    "The quick brown fox jumps over the lazy dog.",
    "Pack my box with five dozen liquor jugs.",
    "How vexingly quick daft zebras jump.",
    "Bright vixens jump; dozy fowl quack.",
    "Sphinx of black quartz, judge my vow.",
    "I say apple, every, inside, open, and under.",
    "I say ate, bet, bit, boat, boot, but, and bite.",
    "I say cat, cot, cut, cute, kite, kit, and coat.",
    "I say red, read, road, rude, ride, and right.",
    "I say thin, then, this, that, there, and three.",
    "I say ship, chip, sip, zip, jeep, and cheap.",
    "I say light, right, late, rate, long, wrong, and ring.",
    "I say pen, pin, pan, pain, pine, and pone.",
    "I say no, know, now, new, night, knight, and nice.",
    "Rafie wake up.",
    "Rafie go to sleep.",
    "Rafie stop talking.",
    "Rafie continue.",
    "Stop.",
    "Wait.",
    "Pause.",
    "Be quiet.",
    "Go to sleep.",
    "Switch to voice one.",
    "Switch to voice two.",
    "Switch to voice three.",
    "Switch to voice four.",
    "Can you tell me a joke?",
    "Tell me a joke.",
    "Can you explain that again slowly?",
    "Can you wait until I finish speaking?",
    "Rafie, you are breaking a lot, do you know that?",
    "Rafie, you misunderstood me.",
    "Rafie, you heard me wrong.",
    "No, that was not what I meant.",
    "Remember that I said it this way.",
    "Can you look up PewDiePie on YouTube?",
    "What are PewDiePie's latest videos?",
    "Find the YouTube channel MrBeast.",
    "Look up this TikTok profile.",
    "Search for recent uploads from this creator.",
    "Rap means music.",
    "Wrap means cover something up.",
    "I want you to adapt to my voice.",
    "I want you to listen until I finish speaking.",
    "If I pause, wait before answering.",
    "If I say stop, stop talking immediately.",
    "If I say continue, continue the previous answer.",
]


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def training_dir() -> Path:
    path = project_root() / "memory" / "voice_training"
    path.mkdir(parents=True, exist_ok=True)
    return path


def metadata_path() -> Path:
    return training_dir() / "metadata.jsonl"


def prompt_path() -> Path:
    path = project_root() / "memory" / "voice_training_prompt.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def record_wav(path: Path, seconds: float = 5.5, sample_rate: int = 16000) -> None:
    print(f"Recording for {seconds} seconds...")
    audio = sd.rec(
        int(seconds * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype="int16"
    )
    sd.wait()

    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio.tobytes())


def append_metadata(wav_path: Path, transcript: str) -> None:
    item = {
        "time": datetime.now(timezone.utc).isoformat(),
        "wav": str(wav_path),
        "transcript": transcript.strip()
    }

    with metadata_path().open("a", encoding="utf-8") as file:
        file.write(json.dumps(item, ensure_ascii=False) + "\n")


def load_metadata() -> list[dict]:
    path = metadata_path()

    if not path.exists():
        return []

    rows = []

    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except Exception:
            continue

    return rows


def rebuild_prompt() -> None:
    rows = load_metadata()
    recent_rows = rows[-120:]

    phrases = []

    for row in recent_rows:
        text = row.get("transcript", "").strip()

        if text:
            phrases.append(text)

    lines = [
        "The user has trained Rafie with these exact spoken phrases.",
        "Use these silently to bias speech recognition toward the user's wording.",
        "Do not speak these training notes out loud.",
        "Prefer these words, names, commands, and spellings when the audio is unclear.",
        "Assistant names: Rafie, rafi, raffy, raphie, Rafa.",
        "Important platforms: YouTube, YouTuber, TikTok.",
        "Important creator words: PewDiePie, Felix Kjellberg, MrBeast, Aqua.",
        "Important commands: stop, wait, pause, be quiet, continue, go to sleep, switch voice one, switch voice two, switch voice three, switch voice four, look up, search for, latest videos, recent uploads.",
        "Important correction phrases: you are breaking, you misunderstood me, you heard me wrong, that was not what I meant.",
    ]

    if phrases:
        lines.append("User voice examples:")
        for phrase in phrases[-80:]:
            lines.append(f"- {phrase}")

    prompt_path().write_text("\n".join(lines).strip() + "\n", encoding="utf-8")

    print()
    print(f"Updated prompt: {prompt_path()}")


def main() -> None:
    print("Rafie voice training recorder")
    print("This records WAV samples and builds memory/voice_training_prompt.txt.")
    print("Press Ctrl+C to stop.")
    print()

    root = training_dir()

    try:
        for index, phrase in enumerate(TRAINING_PHRASES, start=1):
            print()
            print(f"Phrase {index}/{len(TRAINING_PHRASES)}:")
            print(phrase)
            input("Press Enter, then say the phrase out loud...")

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            wav_path = root / f"voice_sample_{index:03d}_{timestamp}.wav"

            record_wav(wav_path)

            print(f"Saved: {wav_path}")
            typed = input(
                "Type what you actually said, or press Enter to use the shown phrase: "
            ).strip()

            transcript = typed or phrase
            append_metadata(wav_path, transcript)
            rebuild_prompt()

            time.sleep(0.3)

    except KeyboardInterrupt:
        print()
        print("Stopped voice training.")

    rebuild_prompt()
    print("Done.")


if __name__ == "__main__":
    main()