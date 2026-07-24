import time
from pathlib import Path

import cv2
from src.camera.camera import Camera

DURATION = 20          # seconds
INTERVAL = 0.1         # save one frame every 0.1 s (10 FPS)

output_dir = Path("frames")
output_dir.mkdir(exist_ok=True)

camera = Camera()
camera.start()

time.sleep(1.0)

start_time = time.time()
frame_count = 0

try:
    while time.time() - start_time < DURATION:
        frame, timestamp = camera.read()

        # Convert RGB -> BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        filename = output_dir / f"frame_{frame_count:04d}.png"
        cv2.imwrite(str(filename), frame_bgr)

        frame_count += 1
        time.sleep(INTERVAL)

finally:
    camera.stop()

print(f"Saved {frame_count} frames to '{output_dir}'.")