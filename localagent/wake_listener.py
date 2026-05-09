from __future__ import annotations

import json
import queue
import tempfile
import time
import wave
from pathlib import Path

import numpy as np

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

    return normalized[index + len(phrase):].strip()


class VoskWakeListener:
    def __init__(self, config: dict):
        import sounddevice as sd
        from vosk import KaldiRecognizer, Model

        self.sd = sd
        self.KaldiRecognizer = KaldiRecognizer

        self.config = config
        self.voice_config = config.get("voice", {})
        self.stt_config = config.get("stt", {})

        self.sample_rate = int(self.voice_config.get("sample_rate", 16000))
        self.blocksize = int(self.voice_config.get("blocksize", 2000))
        self.max_seconds = float(self.voice_config.get("max_seconds", 7))

        self.vosk_path = Path(config["paths"]["vosk_model"])
        if not self.vosk_path.exists():
            raise FileNotFoundError(f"Vosk model folder not found: {self.vosk_path}")

        self.model = Model(str(self.vosk_path))

        self._whisper_model = None
        self._whisper_model_name = None
        self._whisper_device = None
        self._whisper_compute_type = None

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

                if recognizer.AcceptWaveform(data):
                    text = json.loads(recognizer.Result()).get("text", "")
                else:
                    text = json.loads(recognizer.PartialResult()).get("partial", "")

                matched = contains_phrase(text, phrases)
                if matched:
                    return text_after_phrase(text, matched)

    def listen_for_command(
        self,
        max_seconds: float | None = None,
        priority_phrases: list[str] | None = None,
    ) -> str:
        priority_phrases = priority_phrases or []

        # When Rafie is currently speaking, keep Vosk because it can catch
        # "rafi stop" from partial speech faster than Whisper can.
        if priority_phrases:
            return self._listen_for_command_vosk(
                max_seconds=max_seconds,
                priority_phrases=priority_phrases,
            )

        command_engine = self.stt_config.get("command_engine", "vosk").lower().strip()

        if command_engine == "whisper":
            return self._listen_for_command_whisper(max_seconds=max_seconds)

        return self._listen_for_command_vosk(
            max_seconds=max_seconds,
            priority_phrases=priority_phrases,
        )

    def _listen_for_command_vosk(
        self,
        max_seconds: float | None = None,
        priority_phrases: list[str] | None = None,
    ) -> str:
        priority_phrases = priority_phrases or []

        audio_queue: queue.Queue[bytes] = queue.Queue()
        recognizer = self.KaldiRecognizer(self.model, self.sample_rate)

        deadline = time.monotonic() + float(max_seconds or self.max_seconds)
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
                    data = audio_queue.get(timeout=0.15)
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

                        if priority_phrases and contains_phrase(partial, priority_phrases):
                            return partial

            final = json.loads(recognizer.FinalResult()).get("text", "").strip()
            return final or last_text

    def _listen_for_command_whisper(self, max_seconds: float | None = None) -> str:
        audio_path = self._record_command_to_wav(max_seconds=max_seconds)

        if not audio_path:
            return ""

        model = self._get_whisper_model()

        beam_size = int(self.stt_config.get("beam_size", 3))
        initial_prompt = self.stt_config.get("initial_prompt", "")

        try:
            segments, info = model.transcribe(
                str(audio_path),
                language="en",
                beam_size=beam_size,
                vad_filter=False,
                condition_on_previous_text=False,
                initial_prompt=initial_prompt,
            )

            text = " ".join(segment.text.strip() for segment in segments).strip()
            return text

        except Exception as exc:
            print(f"Whisper transcription failed, falling back to Vosk next time: {exc}")
            return ""

        finally:
            try:
                audio_path.unlink(missing_ok=True)
            except Exception:
                pass

    def _record_command_to_wav(self, max_seconds: float | None = None) -> Path | None:
        audio_queue: queue.Queue[bytes] = queue.Queue()

        record_max_seconds = float(
            max_seconds
            or self.stt_config.get("record_max_seconds", self.max_seconds)
        )
        min_record_seconds = float(self.stt_config.get("min_record_seconds", 0.8))
        silence_seconds = float(self.stt_config.get("silence_seconds", 0.9))
        silence_rms = float(self.stt_config.get("silence_rms", 250))

        chunks: list[bytes] = []
        pre_roll: list[bytes] = []
        started = False
        start_time = time.monotonic()
        speech_start_time = None
        last_loud_time = None

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
            while time.monotonic() - start_time < record_max_seconds:
                try:
                    data = audio_queue.get(timeout=0.2)
                except queue.Empty:
                    continue

                samples = np.frombuffer(data, dtype=np.int16)
                rms = float(np.sqrt(np.mean(samples.astype(np.float32) ** 2))) if samples.size else 0.0

                if not started:
                    pre_roll.append(data)
                    pre_roll = pre_roll[-4:]

                    if rms >= silence_rms:
                        started = True
                        speech_start_time = time.monotonic()
                        last_loud_time = speech_start_time
                        chunks.extend(pre_roll)
                    continue

                chunks.append(data)

                now = time.monotonic()

                if rms >= silence_rms:
                    last_loud_time = now

                if speech_start_time and now - speech_start_time < min_record_seconds:
                    continue

                if last_loud_time and now - last_loud_time >= silence_seconds:
                    break

        if not chunks:
            return None

        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        temp_path = Path(temp_file.name)
        temp_file.close()

        with wave.open(str(temp_path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(b"".join(chunks))

        return temp_path

    def _get_whisper_model(self):
        from faster_whisper import WhisperModel

        model_name = self.stt_config.get("whisper_model", "medium.en")
        device = self.stt_config.get("device", "cuda")
        compute_type = self.stt_config.get("compute_type", "float16")

        if (
            self._whisper_model is not None
            and self._whisper_model_name == model_name
            and self._whisper_device == device
            and self._whisper_compute_type == compute_type
        ):
            return self._whisper_model

        print(f"Loading Whisper command model: {model_name} on {device} using {compute_type}.")

        self._whisper_model = WhisperModel(
            model_name,
            device=device,
            compute_type=compute_type,
        )

        self._whisper_model_name = model_name
        self._whisper_device = device
        self._whisper_compute_type = compute_type

        print("Whisper command model is ready.")

        return self._whisper_model