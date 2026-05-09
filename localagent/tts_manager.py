from __future__ import annotations

import re
import subprocess
import threading
import winsound
from pathlib import Path


def _normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def clean_spoken_text(text: str) -> str:
    """Make model output sound normal when spoken out loud."""
    if not text:
        return ""

    text = text.strip()

    # Remove repeated assistant prefixes.
    while re.match(r"^\s*(rafie|rafa|raffy)\s*:\s*", text, flags=re.IGNORECASE):
        text = re.sub(r"^\s*(rafie|rafa|raffy)\s*:\s*", "", text, flags=re.IGNORECASE).strip()

    # Remove markdown code fences.
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)

    # Keep the words inside bold markdown, but remove the symbols.
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)

    # Remove stage directions like *concerned*, *smiles warmly*, *chuckles softly*.
    text = re.sub(r"\*[^*\n]{1,80}\*", "", text)

    # Remove common markdown bullet symbols.
    text = re.sub(r"(?m)^\s*[-*]\s+", "", text)

    # Clean repeated spaces/newlines.
    text = re.sub(r"\s+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)

    # Remove any prefix that may appear again after cleanup.
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

        self._chunks: list[str] = []
        self._paused_chunks: list[str] = []
        self._current_index = 0
        self._speaking = False
        self._last_text = ""

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
        if len(words) <= 8 and heard in last:
            return True

        return False

    def _start_chunks(self, chunks: list[str], wait: bool) -> None:
        with self._lock:
            self._stop_event.clear()
            self._chunks = chunks
            self._current_index = 0
            self._speaking = True

        if wait:
            self._play_worker()
            return

        self._thread = threading.Thread(target=self._play_worker, daemon=True)
        self._thread.start()

    def _play_worker(self) -> None:
        try:
            for index, chunk in enumerate(list(self._chunks)):
                with self._lock:
                    self._current_index = index

                if self._stop_event.is_set():
                    break

                self._speak_chunk(chunk)

                if self._stop_event.is_set():
                    break
        finally:
            with self._lock:
                self._speaking = False

    def _speak_chunk(self, text: str) -> None:
        piper = Path(self.config["paths"]["piper"])
        voice = Path(self.config["paths"]["piper_voice"])
        output_file = self.root / "tts" / "last_response.wav"

        if not piper.exists() or not voice.exists():
            print("Piper or voice file missing, skipping speech.")
            return

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
            print("Piper error:")
            print(process.stderr)
            return

        if self._stop_event.is_set():
            return

        winsound.PlaySound(str(output_file), winsound.SND_FILENAME)

    def _split_into_speech_chunks(self, text: str) -> list[str]:
        text = clean_spoken_text(text)
        if not text:
            return []

        # Split by sentences so Rafie can pause/continue naturally.
        pieces = re.split(r"(?<=[.!?])\s+", text)
        chunks: list[str] = []
        current = ""

        for piece in pieces:
            piece = piece.strip()
            if not piece:
                continue

            if len(current) + len(piece) < 350:
                current = f"{current} {piece}".strip()
            else:
                if current:
                    chunks.append(current)
                current = piece

        if current:
            chunks.append(current)

        return chunks