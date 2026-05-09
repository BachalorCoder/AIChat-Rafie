from pathlib import Path
import mss
from PIL import Image
import ollama

ROOT = Path(r"G:\LocalAgent")
SCREENSHOTS = ROOT / "screenshots"
SCREENSHOTS.mkdir(parents=True, exist_ok=True)

shot_path = SCREENSHOTS / "current_screen.png"

with mss.MSS() as sct:
    monitor = sct.monitors[1]
    shot = sct.grab(monitor)
    img = Image.frombytes("RGB", shot.size, shot.rgb)
    img.save(shot_path)

print(f"Saved screenshot: {shot_path}")

response = ollama.chat(
    model="qwen3-vl:32b",
    messages=[
        {
            "role": "user",
            "content": (
                "Describe what is visible on this screen. "
                "Do not click anything. Do not suggest risky actions. "
                "Just summarize the visible apps, UI elements, and possible user intent."
            ),
            "images": [str(shot_path)]
        }
    ]
)

print("\nAI screen description:\n")
print(response["message"]["content"])
