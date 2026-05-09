import mss
from PIL import Image
from pathlib import Path

out_dir = Path(r"G:\LocalAgent\screenshots")
out_dir.mkdir(parents=True, exist_ok=True)

with mss.mss() as sct:
    monitor = sct.monitors[1]
    shot = sct.grab(monitor)
    img = Image.frombytes("RGB", shot.size, shot.rgb)
    img.save(out_dir / "test_screen.png")

print(r"Saved G:\LocalAgent\screenshots\test_screen.png")