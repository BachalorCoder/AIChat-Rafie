from __future__ import annotations

import json
import queue
import re
import time
from pathlib import Path

from localagent.transcript import normalize_for_command


def normalize_heard(text: str) -> str:
    return normalize_for_command(text)


def contains_phrase(text: str, phrases: list[str]) -> str | None:
    normalized = normalize_heard(text)
    for phrase in phrases:
        normalized_phrase = normalize_heard(phrase)
        if normalized_phrase and normalized_phrase in normalized:
            return normalized_phrase
    return None


def text_after_phrase(text: str, phrase: str) -> str:
    normalized = normalize_heard(text)
    index = normalized.find(phrase)
    if index < 0:
        return ""
    return normalized[index + len(phrase) :].strip()


class VoskWakeListener:
    def __init__(self, config: dict):
        import sounddevice as sd
        from vosk import KaldiRecognizer, Model

        self.sd = sd
        self.KaldiRecognizer = KaldiRecognizer
        self.config = config
        self.voice_config = config.get("voice", {})
        self.sample_rate = int(self.voice_config.get("sample_rate", 16000))
        self.blocksize = int(self.voice_config.get("blocksize", 4000))
        self.max_seconds = float(self.voice_config.get("max_seconds", 12))
        self.vosk_path = Path(config["paths"]["vosk_model"])
        if not self.vosk_path.exists():
            raise FileNotFoundError(f"Vosk model folder not found: {self.vosk_path}")
        self.model = Model(str(self.vosk_path))

    def wait_for_wake(self) -> str:
        phrases = self.voice_config.get("wake_phrases", ["rafie wake up"])
        audio_queue: queue.Queue[bytes] = queue.Queue()
        recognizer = self.KaldiRecognizer(self.model, self.sample_rate)

        def callback(indata, frames, time_info, status):
            if status:
                print(status)
            audio_queue.put(bytes(indata))

        with self.sd.RawInputStream(
            samplerate=self.sample_rate,
            blocksize=self.blocksize,
            dtype="int16",
            channels=1,
            callback=callback,
        ):
            while True:
                data = audio_queue.get()
                text = ""
                if recognizer.AcceptWaveform(data):
                    text = json.loads(recognizer.Result()).get("text", "")
                else:
                    text = json.loads(recognizer.PartialResult()).get("partial", "")

                matched = contains_phrase(text, phrases)
                if matched:
                    return text_after_phrase(text, matched)

    def listen_for_command(self) -> str:
        audio_queue: queue.Queue[bytes] = queue.Queue()
        recognizer = self.KaldiRecognizer(self.model, self.sample_rate)
        deadline = time.monotonic() + self.max_seconds
        last_text = ""

        def callback(indata, frames, time_info, status):
            if status:
                print(status)
            audio_queue.put(bytes(indata))

        with self.sd.RawInputStream(
            samplerate=self.sample_rate,
            blocksize=self.blocksize,
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
                    text = json.loads(recognizer.Result()).get("text", "").strip()
                    if text:
                        return text
                else:
                    partial = json.loads(recognizer.PartialResult()).get("partial", "").strip()
                    if partial:
                        last_text = partial

        final = json.loads(recognizer.FinalResult()).get("text", "").strip()
        return final or last_text
