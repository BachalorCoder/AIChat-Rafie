from __future__ import annotations

import subprocess
import threading
import winsound
from pathlib import Path


class InterruptibleTTS:
    def __init__(self, config: dict):
        self.config = config
        self.root = Path(config["paths"]["project_root"])
        self._lock = threading.Lock()
        self._generation: subprocess.Popen | None = None

    def stop(self) -> None:
        with self._lock:
            if self._generation and self._generation.poll() is None:
                self._generation.terminate()
            self._generation = None
        winsound.PlaySound(None, winsound.SND_PURGE)

    def speak(self, text: str, wait: bool = False) -> None:
        if not text.strip():
            return

        self.stop()

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

        flags = winsound.SND_FILENAME
        if not wait:
            flags |= winsound.SND_ASYNC
        winsound.PlaySound(str(output_file), flags)
