from __future__ import annotations

import json
import queue
import tempfile
import time
import wave
from collections import deque
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
        self.max_seconds = float(self.voice_config.get("max_seconds", 10))

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

        command_engine = self.stt_config.get("command_engine", "vosk").lower().strip()

        if command_engine == "whisper":
            return self._listen_for_command_whisper(
                max_seconds=max_seconds,
                priority_phrases=priority_phrases,
            )

        return self._listen_for_command_vosk(
            max_seconds=max_seconds,
            priority_phrases=priority_phrases,
            priority_only=bool(priority_phrases),
        )

    def _listen_for_command_vosk(
        self,
        max_seconds: float | None = None,
        priority_phrases: list[str] | None = None,
        priority_only: bool = False,
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

                    if not text:
                        continue

                    if priority_phrases:
                        matched = contains_phrase(text, priority_phrases)

                        if matched:
                            return text

                        if priority_only:
                            last_text = text
                            continue

                    return text

                partial = json.loads(recognizer.PartialResult()).get("partial", "").strip()

                if partial:
                    last_text = partial

                    if priority_phrases and contains_phrase(partial, priority_phrases):
                        return partial

            final = json.loads(recognizer.FinalResult()).get("text", "").strip()

            if priority_only:
                if final and contains_phrase(final, priority_phrases):
                    return final

                if last_text and contains_phrase(last_text, priority_phrases):
                    return last_text

                return ""

            return final or last_text

    def _listen_for_command_whisper(
        self,
        max_seconds: float | None = None,
        priority_phrases: list[str] | None = None,
    ) -> str:
        result = self._record_command_to_wav(
            max_seconds=max_seconds,
            priority_phrases=priority_phrases or [],
        )

        if not result:
            print("No command audio was recorded.")
            return ""

        if isinstance(result, str):
            return result

        audio_path = result
        model = self._get_whisper_model()

        beam_size = int(self.stt_config.get("beam_size", 5))
        initial_prompt = self._build_whisper_prompt()

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
            print(f"Whisper transcription failed: {exc}")
            return ""

        finally:
            try:
                audio_path.unlink(missing_ok=True)
            except Exception:
                pass

    def _record_command_to_wav(
        self,
        max_seconds: float | None = None,
        priority_phrases: list[str] | None = None,
    ) -> Path | str | None:
        priority_phrases = priority_phrases or []
        audio_queue: queue.Queue[bytes] = queue.Queue()

        record_max_seconds = float(
            max_seconds or self.stt_config.get("record_max_seconds", self.max_seconds)
        )

        min_record_seconds = float(self.stt_config.get("min_record_seconds", 1.2))
        silence_seconds = float(self.stt_config.get("silence_seconds", 2.8))

        configured_silence_rms = float(self.stt_config.get("silence_rms", 60))
        minimum_silence_rms = float(self.stt_config.get("minimum_silence_rms", 25))

        start_timeout_seconds = float(self.stt_config.get("start_timeout_seconds", 7))
        auto_start_after_seconds = float(self.stt_config.get("auto_start_after_seconds", 0.6))

        pre_roll_chunks_count = int(self.stt_config.get("pre_roll_chunks", 8))
        debug_audio_levels = bool(self.stt_config.get("debug_audio_levels", False))

        chunks: list[bytes] = []
        pre_roll: deque[bytes] = deque(maxlen=pre_roll_chunks_count)

        started = False
        had_loud_audio = False

        start_time = time.monotonic()
        speech_start_time = None
        last_loud_time = None
        last_debug_print_time = start_time

        noise_rms_values: list[float] = []
        peak_rms = 0.0

        priority_recognizer = None

        if priority_phrases:
            priority_recognizer = self.KaldiRecognizer(self.model, self.sample_rate)

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

                priority_text = self._check_priority_phrase(
                    priority_recognizer,
                    data,
                    priority_phrases,
                )

                if priority_text:
                    return priority_text

                now = time.monotonic()
                elapsed = now - start_time

                rms = self._audio_rms(data)
                peak_rms = max(peak_rms, rms)

                if not started:
                    noise_rms_values.append(rms)

                    if len(noise_rms_values) > 20:
                        noise_rms_values = noise_rms_values[-20:]

                effective_silence_rms = self._effective_silence_rms(
                    configured_silence_rms=configured_silence_rms,
                    minimum_silence_rms=minimum_silence_rms,
                    noise_rms_values=noise_rms_values,
                )

                is_loud = rms >= effective_silence_rms

                if debug_audio_levels and now - last_debug_print_time >= 1.0:
                    print(
                        "Mic level:",
                        f"rms={rms:.1f}",
                        f"peak={peak_rms:.1f}",
                        f"threshold={effective_silence_rms:.1f}",
                        f"started={started}",
                    )
                    last_debug_print_time = now

                if not started:
                    pre_roll.append(data)

                    should_auto_start = elapsed >= auto_start_after_seconds

                    if is_loud or should_auto_start:
                        started = True
                        speech_start_time = now
                        chunks.extend(list(pre_roll))
                        last_loud_time = now

                        if is_loud:
                            had_loud_audio = True

                        continue

                    if elapsed >= start_timeout_seconds:
                        return None

                    continue

                chunks.append(data)

                if is_loud:
                    had_loud_audio = True
                    last_loud_time = now

                if speech_start_time and now - speech_start_time < min_record_seconds:
                    continue

                if last_loud_time and now - last_loud_time >= silence_seconds:
                    break

        if not chunks:
            return None

        duration_seconds = self._audio_duration_seconds(chunks)

        if duration_seconds < 0.4:
            return None

        if debug_audio_levels:
            print(
                "Recorded command audio:",
                f"duration={duration_seconds:.2f}s",
                f"peak_rms={peak_rms:.1f}",
                f"had_loud_audio={had_loud_audio}",
            )

        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        temp_path = Path(temp_file.name)
        temp_file.close()

        with wave.open(str(temp_path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(b"".join(chunks))

        return temp_path

    def _check_priority_phrase(
        self,
        recognizer,
        data: bytes,
        priority_phrases: list[str],
    ) -> str:
        if not recognizer or not priority_phrases:
            return ""

        try:
            if recognizer.AcceptWaveform(data):
                text = json.loads(recognizer.Result()).get("text", "").strip()
            else:
                text = json.loads(recognizer.PartialResult()).get("partial", "").strip()
        except Exception:
            return ""

        if not text:
            return ""

        if contains_phrase(text, priority_phrases):
            return text

        return ""

    def _audio_rms(self, data: bytes) -> float:
        samples = np.frombuffer(data, dtype=np.int16)

        if samples.size == 0:
            return 0.0

        samples_float = samples.astype(np.float32)
        return float(np.sqrt(np.mean(samples_float ** 2)))

    def _audio_duration_seconds(self, chunks: list[bytes]) -> float:
        total_bytes = sum(len(chunk) for chunk in chunks)
        total_samples = total_bytes / 2
        return float(total_samples / self.sample_rate)

    def _effective_silence_rms(
        self,
        configured_silence_rms: float,
        minimum_silence_rms: float,
        noise_rms_values: list[float],
    ) -> float:
        if not noise_rms_values:
            return max(minimum_silence_rms, configured_silence_rms)

        noise_floor = float(np.median(np.array(noise_rms_values, dtype=np.float32)))
        adaptive_threshold = noise_floor + 45.0

        return max(
            minimum_silence_rms,
            min(configured_silence_rms, adaptive_threshold),
        )

    def _build_whisper_prompt(self) -> str:
        prompt_parts = []

        base_prompt = self.stt_config.get("initial_prompt", "").strip()

        if base_prompt:
            prompt_parts.append(base_prompt)

        try:
            root = Path(self.config["paths"]["project_root"])
            training_prompt_path = root / "memory" / "voice_training_prompt.txt"

            if training_prompt_path.exists():
                training_prompt = training_prompt_path.read_text(
                    encoding="utf-8"
                ).strip()

                if training_prompt:
                    prompt_parts.append(training_prompt)

        except Exception as exc:
            print(f"Could not load voice training prompt: {exc}")

        return "\n".join(prompt_parts).strip()

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