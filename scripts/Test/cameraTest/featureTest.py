import time
from pathlib import Path

import cv2
import numpy as np
from picamera2 import Picamera2


# -----------------------------
# Settings
# -----------------------------
WIDTH = 640
HEIGHT = 480
FPS = 20
SAVE_EVERY_N_FRAMES = 30
MAX_CORNERS = 100

feature_params = dict(
    maxCorners=MAX_CORNERS,
    qualityLevel=0.3,
    minDistance=7,
    blockSize=7
)

lk_params = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
)


def make_output_dirs():
    base_dir = Path.home() / "captures"
    session_dir = base_dir / time.strftime("%Y%m%d_%H%M%S")
    frames_dir = session_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    return session_dir, frames_dir


def main():
    session_dir, frames_dir = make_output_dirs()
    video_path = session_dir / "tracked.mp4"

    print(f"Saving to: {session_dir}")

    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"size": (WIDTH, HEIGHT), "format": "RGB888"}
    )
    picam2.configure(config)
    picam2.start()
    time.sleep(1.0)

    frame = picam2.capture_array()
    old_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, FPS, (WIDTH, HEIGHT))

    mask = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    frame_idx = 0
    num = 0

    cv2.namedWindow("Tracked frame", cv2.WINDOW_NORMAL)

    while num < 100:
        frame = picam2.capture_array()
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        if p0 is None or len(p0) < 10:
            p0 = cv2.goodFeaturesToTrack(frame_gray, mask=None, **feature_params)
            old_gray = frame_gray.copy()

            writer.write(frame_bgr)

            if frame_idx % SAVE_EVERY_N_FRAMES == 0:
                cv2.imwrite(str(frames_dir / f"frame_{frame_idx:06d}.jpg"), frame_bgr)

            cv2.imshow("Tracked frame", frame_bgr)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_idx += 1
            num += 1
            continue

        p1, st, err = cv2.calcOpticalFlowPyrLK(
            old_gray, frame_gray, p0, None, **lk_params
        )

        if p1 is None or st is None:
            p0 = None
            old_gray = frame_gray.copy()
            continue

        good_new = p1[st == 1]
        good_old = p0[st == 1]

        annotated = frame_bgr.copy()

        for new, old in zip(good_new, good_old):
            a, b = new.ravel()
            c, d = old.ravel()
            a, b, c, d = map(int, [a, b, c, d])

            mask = cv2.line(mask, (a, b), (c, d), (0, 255, 0), 2)
            annotated = cv2.circle(annotated, (a, b), 5, (0, 0, 255), -1)

        output = cv2.add(annotated, mask)

        writer.write(output)

        if frame_idx % SAVE_EVERY_N_FRAMES == 0:
            cv2.imwrite(str(frames_dir / f"frame_{frame_idx:06d}.jpg"), output)

        cv2.imshow("Tracked frame", output)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)
        frame_idx += 1
        num += 1

    writer.release()
    picam2.stop()
    cv2.destroyAllWindows()

    print(f"Saved video: {video_path}")
    print(f"Saved frames in: {frames_dir}")


if __name__ == "__main__":
    main()