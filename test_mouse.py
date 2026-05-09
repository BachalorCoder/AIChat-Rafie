import pyautogui
import time

pyautogui.FAILSAFE = True

print("Move your mouse to the top-left corner to emergency stop.")
time.sleep(2)

print("Current mouse position:", pyautogui.position())