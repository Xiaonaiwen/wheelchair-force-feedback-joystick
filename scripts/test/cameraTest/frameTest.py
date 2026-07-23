import cv2
from src.camera.camera import Camera

camera = Camera()
camera.start()

try:
    frame, timestamp = camera.read()

    # Convert to BGR because OpenCV expects BGR when saving
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    cv2.imwrite("frame.png", frame_bgr)
    print("Saved frame.png")

finally:
    camera.stop()