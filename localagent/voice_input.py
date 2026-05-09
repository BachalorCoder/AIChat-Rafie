from __future__ import annotations

import json
import queue
import time
from pathlib import Path


def listen_once(config: dict, max_seconds: float | None = None) -> str:
    import sounddevice as sd
    from vosk import KaldiRecognizer, Model

    voice_config = config.get("voice", {})
    sample_rate = int(voice_config.get("sample_rate", 16000))
    timeout_seconds = float(max_seconds or voice_config.get("max_seconds", 12))
    vosk_path = Path(config["paths"]["vosk_model"])

    if not vosk_path.exists():
        raise FileNotFoundError(f"Vosk model folder not found: {vosk_path}")

    audio_queue: queue.Queue[bytes] = queue.Queue()
    recognizer = KaldiRecognizer(Model(str(vosk_path)), sample_rate)

    def callback(indata, frames, time_info, status):
        if status:
            print(status)
        audio_queue.put(bytes(indata))

    print(f"Listening for up to {timeout_seconds:.0f}s...")
    deadline = time.monotonic() + timeout_seconds

    with sd.RawInputStream(
        samplerate=sample_rate,
        blocksize=8000,
        dtype="int16",
        channels=1,
        callback=callback,
    ):
        while time.monotonic() < deadline:
            try:
                data = audio_queue.get(timeout=0.25)
            except queue.Empty:
                continue

            if recognizer.AcceptWaveform(data):
                result = json.loads(recognizer.Result())
                text = result.get("text", "").strip()
                if text:
                    return text

    final = json.loads(recognizer.FinalResult())
    return final.get("text", "").strip()

