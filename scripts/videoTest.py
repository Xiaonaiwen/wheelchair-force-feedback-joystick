import time

import cv2
from picamera2 import Picamera2

WIDTH = 640
HEIGHT = 480

def main():
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"size": (WIDTH, HEIGHT), "format": "RGB888"}
    )
    picam2.configure(config)
    picam2.start()
    time.sleep(1.0)

    cv2.namedWindow("Camera Test", cv2.WINDOW_NORMAL)

    try:
        while True:
            frame = picam2.capture_array()

            # Picamera2 gives RGB, OpenCV expects BGR for display
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            cv2.imshow("Camera Test", frame_bgr)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        picam2.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()