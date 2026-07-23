import time
from pathlib import Path

import cv2
import numpy as np

from src.camera.camera import Camera
from src.vision.speedCalculate_and_terrainFrameOutput import Optical_Flow
from src.friction_model.friction_predict import predict_frame


# -----------------------------
# Settings
# -----------------------------
WIDTH = 640
HEIGHT = 480
FPS = 20
DURATION = 20
FEATURE_REFRESH = 5

# Change these to your real wheel/ground regions
WHEEL_MASK = (0, 220, 185, 328)      # x_low, x_high, y_low, y_high
GROUND_MASK = (0, 500, 410, 480)   # x_low, x_high, y_low, y_high


def draw_points(frame, points, color, radius=4):
    if points is None:
        return frame
    for p in points:
        x, y = p.ravel()
        cv2.circle(frame, (int(x), int(y)), radius, color, -1)
    return frame


def main():
    video_path = Path("tracked.mp4")

    camera = Camera()
    optic = Optical_Flow(
        width=WIDTH,
        height=HEIGHT,
        feature_frame_refresh=FEATURE_REFRESH
    )
    optic.default_set()
    optic.set_wheel_mask(*WHEEL_MASK)
    optic.set_ground_mask(*GROUND_MASK)

    camera.start()
    time.sleep(1.0)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, FPS, (WIDTH, HEIGHT))


    previousGroundVelocity = None
    previousTime = None
    standardFrameTime = None

    start_time = time.time()

    try:
        while time.time() - start_time < DURATION:
            frame, currentTime = camera.read()
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            success, wheel_distance_move, ground_distance_move = optic.lk(frame_bgr)

            output = frame_bgr.copy()

            # draw current feature points
            output = draw_points(output, getattr(optic, "wheel_feature", None), (0, 0, 255))
            output = draw_points(output, getattr(optic, "ground_feature", None), (0, 255, 0))

            # draw masks
            cv2.rectangle(
                output,
                (WHEEL_MASK[0], WHEEL_MASK[2]),
                (WHEEL_MASK[1], WHEEL_MASK[3]),
                (0, 0, 255),
                2
            )
            cv2.rectangle(
                output,
                (GROUND_MASK[0], GROUND_MASK[2]),
                (GROUND_MASK[1], GROUND_MASK[3]),
                (0, 255, 0),
                2
            )

            terrainFrame = optic.get_terrain_frame(frame_bgr)
            coefficient = predict_frame(terrainFrame)

            y = 25
            cv2.putText(
                output,
                f"terrain coefficient: {coefficient:.3f}",
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 0),
                2
            )
            y += 28

            if success:
                period = currentTime - standardFrameTime
                if period <= 0:
                    period = 1e-6

                if previousTime is None or previousGroundVelocity is None:
                    slip, s, noMove, v, w = optic.slipRatuib_and_noAcceleration(wheel_distance_move, ground_distance_move, period)
                    a = None
                else:
                    slip, s, a, noMove, v, w = optic.slipRatio_and_currentAcceleration(
                        wheel_distance_move,
                        ground_distance_move,
                        period,
                        currentTime,
                        previousGroundVelocity,
                        previousTime
                    )

                previousTime = currentTime
                previousGroundVelocity = v

                cv2.putText(
                    output,
                    f"period: {period:.3f} s",
                    (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2
                )
                y += 28

                cv2.putText(
                    output,
                    f"slip: {slip}",
                    (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 0, 255),
                    2
                )
                y += 28

                cv2.putText(
                    output,
                    f"noMove: {noMove}",
                    (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),   # choose any colour you like
                    2
                )
                y += 28

                if a is not None:
                    cv2.putText(
                        output,
                        f"acceleration: {a:.3f}",
                        (20, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (0, 255, 255),
                        2
                    )
                    y += 28

                cv2.putText(
                    output,
                    f"velocity: {v:.3f}",
                    (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 0),
                    2
                )
                y += 28

                cv2.putText(
                    output,
                    f"angular velocity: {w:.3f}",
                    (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 0, 255),
                    2
                )
            else:
                standardFrameTime = currentTime
                cv2.putText(
                    output,
                    "REFERENCE FRAME UPDATED",
                    (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (0, 255, 255),
                    2
                )

            writer.write(output)

    finally:
        writer.release()
        camera.stop()
        cv2.destroyAllWindows()

    print(f"Saved video: {video_path.resolve()}")


if __name__ == "__main__":
    main()