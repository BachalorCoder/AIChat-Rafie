from __future__ import annotations

import contextlib
import io
import re
import subprocess
import threading
import time
import uuid
import wave
import winsound
from math import gcd
from pathlib import Path


def _normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def clean_spoken_text(text: str) -> str:
    if not text:
        return ""

    text = text.strip()

    while re.match(r"^\s*(rafie|rafa|raffy)\s*:\s*", text, flags=re.IGNORECASE):
        text = re.sub(r"^\s*(rafie|rafa|raffy)\s*:\s*", "", text, flags=re.IGNORECASE).strip()

    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"\*[^*\n]{1,80}\*", "", text)
    text = re.sub(r"(?m)^\s*[-*]\s+", "", text)
    text = re.sub(r"(?m)^\s*\d+\.\s+", "", text)
    text = re.sub(r"^(certainly|absolutely|of course)[,!.\s]+", "", text, flags=re.IGNORECASE)

    text = text.replace("—", ", ")
    text = text.replace(";", ",")
    text = text.replace(":", ",")

    text = re.sub(r"\s+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)

    while re.match(r"^\s*(rafie|rafa|raffy)\s*:\s*", text, flags=re.IGNORECASE):
        text = re.sub(r"^\s*(rafie|rafa|raffy)\s*:\s*", "", text, flags=re.IGNORECASE).strip()

    return text.strip()


class InterruptibleTTS:
    def __init__(self, config: dict):
        self.config = config
        self.root = Path(config["paths"]["project_root"])

        self._lock = threading.RLock()
        self._stop_event = threading.Event()

        self._thread: threading.Thread | None = None
        self._run_id = 0

        self._chunks: list[str] = []
        self._paused_chunks: list[str] = []
        self._current_index = 0

        self._speaking = False
        self._last_text = ""

        self._chatterbox_model = None
        self._chatterbox_device = None
        self._chatterbox_model_class_name = None

    def preload(self, wait: bool = False) -> None:
        engine = self.config.get("tts", {}).get("engine", "piper").lower().strip()

        if engine not in {"chatterbox_turbo", "chatterbox"}:
            return

        if wait:
            self._preload_worker()
            return

        thread = threading.Thread(target=self._preload_worker, daemon=True)
        thread.start()

    def _preload_worker(self) -> None:
        try:
            engine = self.config.get("tts", {}).get("engine", "piper").lower().strip()

            if engine == "chatterbox_turbo":
                from chatterbox.tts_turbo import ChatterboxTurboTTS

                self._get_chatterbox_model(ChatterboxTurboTTS)

            elif engine == "chatterbox":
                from chatterbox.tts import ChatterboxTTS

                self._get_chatterbox_model(ChatterboxTTS)

            print("Chatterbox voice model is ready.")

        except Exception as exc:
            print(f"Could not preload Chatterbox voice model: {exc}")

    def set_voice_profile(self, profile_name: str) -> tuple[bool, str]:
        profile_key = profile_name.lower().strip()
        profiles = self.config.get("tts", {}).get("voice_profiles", {})

        if profile_key not in profiles:
            available = ", ".join(sorted(profiles.keys())) or "none"
            return False, f"I do not have a voice called {profile_name}. Available voices are: {available}."

        profile = profiles[profile_key]
        tts_config = self.config.setdefault("tts", {})

        tts_config["voice_profile"] = profile_key

        for key in ["voice_reference", "exaggeration", "cfg_weight", "chunk_chars"]:
            if key in profile:
                tts_config[key] = profile[key]

        reference = Path(tts_config.get("voice_reference", ""))

        if not reference.exists():
            return (
                False,
                f"I switched to voice {profile_key}, but its reference file is missing. I may fall back to voice one.",
            )

        return True, f"I switched to voice {profile_key}."

    def is_speaking(self) -> bool:
        with self._lock:
            return self._speaking and self._thread is not None and self._thread.is_alive()

    def stop(self) -> None:
        self.interrupt(save=False)

    def interrupt(self, save: bool = True) -> None:
        with self._lock:
            if save and self._speaking and self._chunks:
                index = max(0, min(self._current_index, len(self._chunks)))
                self._paused_chunks = self._chunks[index:]
            elif not save:
                self._paused_chunks = []

            self._run_id += 1
            self._stop_event.set()
            self._speaking = False

        winsound.PlaySound(None, winsound.SND_PURGE)

    def continue_speaking(self, wait: bool = False) -> bool:
        with self._lock:
            chunks = list(self._paused_chunks)
            self._paused_chunks = []

        if not chunks:
            return False

        self._start_chunks(chunks, wait=wait)
        return True

    def speak(self, text: str, wait: bool = False) -> None:
        text = clean_spoken_text(text)
        if not text:
            return

        self.interrupt(save=False)

        chunks = self._split_into_speech_chunks(text)
        if not chunks:
            return

        with self._lock:
            self._last_text = text

        self._start_chunks(chunks, wait=wait)

    def looks_like_echo(self, heard_text: str) -> bool:
        heard = _normalize_text(heard_text)
        if not heard:
            return False

        with self._lock:
            last = _normalize_text(self._last_text)

        if not last:
            return False

        words = heard.split()
        return len(words) <= 8 and heard in last

    def _start_chunks(self, chunks: list[str], wait: bool) -> None:
        with self._lock:
            self._run_id += 1
            run_id = self._run_id

            self._stop_event.clear()
            self._chunks = chunks
            self._current_index = 0
            self._speaking = True

        if wait:
            self._play_worker(run_id)
            return

        self._thread = threading.Thread(target=self._play_worker, args=(run_id,), daemon=True)
        self._thread.start()

    def _should_stop(self, run_id: int) -> bool:
        with self._lock:
            return self._stop_event.is_set() or run_id != self._run_id

    def _play_worker(self, run_id: int) -> None:
        try:
            for index, chunk in enumerate(list(self._chunks)):
                with self._lock:
                    self._current_index = index

                if self._should_stop(run_id):
                    break

                self._speak_chunk(chunk, run_id)

                if self._should_stop(run_id):
                    break

        finally:
            with self._lock:
                if run_id == self._run_id:
                    self._speaking = False

    def _speak_chunk(self, text: str, run_id: int) -> None:
        output_dir = self.root / "tts"
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / f"rafie_voice_{uuid.uuid4().hex}.wav"

        engine = self.config.get("tts", {}).get("engine", "piper").lower().strip()

        try:
            if engine == "chatterbox_turbo":
                self._generate_chatterbox_turbo(text, output_file)
            elif engine == "chatterbox":
                self._generate_chatterbox_standard(text, output_file)
            else:
                self._generate_piper(text, output_file)
        except Exception as exc:
            print(f"TTS engine '{engine}' failed: {exc}")

            fallback = self.config.get("tts", {}).get("fallback_engine", "piper").lower().strip()

            if fallback == "piper" and engine != "piper":
                try:
                    self._generate_piper(text, output_file)
                except Exception as fallback_exc:
                    print(f"Piper fallback failed: {fallback_exc}")
                    return
            else:
                return

        if self._should_stop(run_id):
            return

        self._play_wav_interruptible(output_file, run_id)

        try:
            output_file.unlink(missing_ok=True)
        except Exception:
            pass

    def _generate_chatterbox_turbo(self, text: str, output_file: Path) -> None:
        import torch
        from chatterbox.tts_turbo import ChatterboxTurboTTS

        model = self._get_chatterbox_model(ChatterboxTurboTTS)
        reference_audio = self._reference_audio_path()
        kwargs = self._chatterbox_generate_kwargs(reference_audio)

        quiet = bool(self.config.get("tts", {}).get("quiet", True))

        with torch.inference_mode():
            if quiet:
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    wav = model.generate(
                        text,
                        norm_loudness=False,
                        **kwargs,
                    )
            else:
                wav = model.generate(
                    text,
                    norm_loudness=False,
                    **kwargs,
                )

        self._save_tensor_wav(output_file, wav, model.sr)

    def _generate_chatterbox_standard(self, text: str, output_file: Path) -> None:
        import torch
        from chatterbox.tts import ChatterboxTTS

        model = self._get_chatterbox_model(ChatterboxTTS)
        reference_audio = self._reference_audio_path()
        kwargs = self._chatterbox_generate_kwargs(reference_audio)

        quiet = bool(self.config.get("tts", {}).get("quiet", True))

        with torch.inference_mode():
            try:
                if quiet:
                    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                        wav = model.generate(
                            text,
                            norm_loudness=False,
                            **kwargs,
                        )
                else:
                    wav = model.generate(
                        text,
                        norm_loudness=False,
                        **kwargs,
                    )
            except TypeError:
                if quiet:
                    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                        wav = model.generate(text, **kwargs)
                else:
                    wav = model.generate(text, **kwargs)

        self._save_tensor_wav(output_file, wav, model.sr)

    def _chatterbox_generate_kwargs(self, reference_audio: Path | None) -> dict:
        tts_config = self.config.get("tts", {})
        kwargs = {}

        if reference_audio:
            kwargs["audio_prompt_path"] = str(reference_audio)

        exaggeration = tts_config.get("exaggeration")
        cfg_weight = tts_config.get("cfg_weight")

        if exaggeration is not None:
            kwargs["exaggeration"] = float(exaggeration)

        if cfg_weight is not None:
            kwargs["cfg_weight"] = float(cfg_weight)

        return kwargs

    def _get_chatterbox_model(self, model_class):
        with self._lock:
            wanted_device = self._choose_torch_device()
            class_name = model_class.__name__

            if (
                self._chatterbox_model is not None
                and self._chatterbox_device == wanted_device
                and self._chatterbox_model_class_name == class_name
            ):
                return self._chatterbox_model

            print(f"Loading Chatterbox voice model on {wanted_device}. First load may take a while.")

            quiet = bool(self.config.get("tts", {}).get("quiet", True))

            if quiet:
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    self._chatterbox_model = model_class.from_pretrained(device=wanted_device)
            else:
                self._chatterbox_model = model_class.from_pretrained(device=wanted_device)

            self._chatterbox_device = wanted_device
            self._chatterbox_model_class_name = class_name

            return self._chatterbox_model

    def _choose_torch_device(self) -> str:
        import torch

        device = self.config.get("tts", {}).get("device", "auto").lower().strip()

        if device in {"cuda", "cpu"}:
            return device

        return "cuda" if torch.cuda.is_available() else "cpu"

    def _reference_audio_path(self) -> Path | None:
        tts_config = self.config.get("tts", {})

        candidate = (
            tts_config.get("voice_reference")
            or self.config.get("paths", {}).get("voice_reference")
        )

        if not candidate:
            return None

        path = Path(candidate)

        if not path.exists():
            fallback = Path(self.config.get("paths", {}).get("voice_reference", ""))
            if fallback.exists():
                path = fallback
            else:
                return None

        clean_path = path.with_name(path.stem + "_clean.wav")

        try:
            self._make_clean_reference(path, clean_path)
            return clean_path
        except Exception as exc:
            print(f"Could not clean reference voice, using original: {exc}")
            return path

    def _make_clean_reference(self, input_path: Path, output_path: Path) -> None:
        import numpy as np
        import soundfile as sf
        from scipy.signal import resample_poly

        audio, sr = sf.read(str(input_path), dtype="float32", always_2d=True)

        audio = audio.mean(axis=1).astype(np.float32)

        target_sr = 16000
        if int(sr) != target_sr:
            divisor = gcd(int(sr), target_sr)
            up = target_sr // divisor
            down = int(sr) // divisor
            audio = resample_poly(audio, up, down).astype(np.float32)

        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        if peak > 0:
            audio = audio / peak * 0.85

        audio = audio.astype(np.float32)

        sf.write(str(output_path), audio, target_sr, subtype="PCM_16")

    def _save_tensor_wav(self, output_file: Path, wav, sample_rate: int) -> None:
        import numpy as np
        import soundfile as sf

        wav = wav.detach().cpu().float()

        if wav.ndim == 2:
            wav_np = wav.squeeze(0).numpy()
        else:
            wav_np = wav.numpy()

        wav_np = wav_np.astype(np.float32)

        peak = float(np.max(np.abs(wav_np))) if wav_np.size else 0.0
        if peak > 1.0:
            wav_np = wav_np / peak * 0.95

        sf.write(str(output_file), wav_np, sample_rate, subtype="PCM_16")

    def _generate_piper(self, text: str, output_file: Path) -> None:
        piper = Path(self.config["paths"]["piper"])
        voice = Path(self.config["paths"]["piper_voice"])

        if not piper.exists() or not voice.exists():
            raise FileNotFoundError("Piper or Piper voice file is missing.")

        process = subprocess.run(
            [
                str(piper),
                "--model",
                str(voice),
                "--output_file",
                str(output_file),
            ],
            input=text,
            text=True,
            capture_output=True,
        )

        if process.returncode != 0:
            raise RuntimeError(process.stderr)

    def _play_wav_interruptible(self, output_file: Path, run_id: int) -> None:
        duration = self._wav_duration(output_file)

        winsound.PlaySound(str(output_file), winsound.SND_FILENAME | winsound.SND_ASYNC)

        end_time = time.monotonic() + duration + 0.08

        while time.monotonic() < end_time:
            if self._should_stop(run_id):
                winsound.PlaySound(None, winsound.SND_PURGE)
                return

            time.sleep(0.03)

    def _wav_duration(self, output_file: Path) -> float:
        try:
            with wave.open(str(output_file), "rb") as wav_file:
                frames = wav_file.getnframes()
                rate = wav_file.getframerate()

                if rate > 0:
                    return frames / float(rate)

        except Exception:
            pass

        return 2.0

    def _split_into_speech_chunks(self, text: str) -> list[str]:
        text = clean_spoken_text(text)
        if not text:
            return []

        tts_config = self.config.get("tts", {})
        voice_config = self.config.get("voice", {})

        max_chars = int(
            tts_config.get(
                "chunk_chars",
                voice_config.get("tts_chunk_chars", 280),
            )
        )

        paragraphs = [p.strip() for p in re.split(r"\n+", text) if p.strip()]

        chunks: list[str] = []
        current = ""

        for paragraph in paragraphs:
            sentences = re.split(r"(?<=[.!?])\s+", paragraph)

            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue

                if len(sentence) > max_chars:
                    if current:
                        chunks.append(current)
                        current = ""

                    chunks.extend(self._split_long_piece(sentence, max_chars))
                    continue

                if not current:
                    current = sentence
                elif len(current) + len(sentence) + 1 <= max_chars:
                    current = f"{current} {sentence}".strip()
                else:
                    chunks.append(current)
                    current = sentence

        if current:
            chunks.append(current)

        return chunks

    def _split_long_piece(self, text: str, max_chars: int) -> list[str]:
        words = text.split()
        chunks: list[str] = []
        current = ""

        for word in words:
            if not current:
                current = word
            elif len(current) + len(word) + 1 <= max_chars:
                current = f"{current} {word}"
            else:
                chunks.append(current)
                current = word

        if current:
            chunks.append(current)

        return chunks