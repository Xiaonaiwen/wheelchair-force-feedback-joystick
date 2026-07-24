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
MAX_FRAMES = 30
FEATURE_REFRESH = 5

# Change these to your real wheel/ground regions
WHEEL_MASK = (0, 220, 185, 328)      # x_low, x_high, y_low, y_high
GROUND_MASK = (0, 500, 420, 480)     # x_low, x_high, y_low, y_high


def draw_points(frame, points, color, radius=4):
    if points is None:
        return frame
    for p in points:
        x, y = p.ravel()
        cv2.circle(frame, (int(x), int(y)), radius, color, -1)
    return frame


def draw_two_column_text(
    frame,
    left_items,
    right_items,
    start_y=25,
    left_x=20,
    right_x=330,
    line_gap=28,
    font_scale=0.6,
    thickness=2,
):
    """Draw two aligned text columns on the frame."""
    max_rows = max(len(left_items), len(right_items))
    for row in range(max_rows):
        y = start_y + row * line_gap
        if row < len(left_items):
            text, color = left_items[row]
            cv2.putText(
                frame,
                text,
                (left_x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                color,
                thickness,
            )
        if row < len(right_items):
            text, color = right_items[row]
            cv2.putText(
                frame,
                text,
                (right_x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                color,
                thickness,
            )
    return frame


def main():
    video_path = Path("tracked.mp4")

    camera = Camera()
    optic = Optical_Flow(
        width=WIDTH,
        height=HEIGHT,
        feature_frame_refresh=FEATURE_REFRESH,
    )
    optic.default_set()
    optic.set_wheel_mask(*WHEEL_MASK)
    optic.set_ground_mask(*GROUND_MASK)

    camera.start()
    time.sleep(1.0)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, FPS, (WIDTH, HEIGHT))

    previousGroundVelocity = 0
    previousTime = 0
    previousWheelMove = 0
    previousGroundMove = 0

    count = 0

    try:
        while True:
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
                2,
            )
            cv2.rectangle(
                output,
                (GROUND_MASK[0], GROUND_MASK[2]),
                (GROUND_MASK[1], GROUND_MASK[3]),
                (0, 255, 0),
                2,
            )

            terrainFrame = optic.get_terrain_frame(frame_bgr)
            coefficient = predict_frame(terrainFrame)

            left_items = [(f"terrain coefficient: {coefficient:.3f}", (255, 255, 0))]
            right_items = []

            if success:
                period = currentTime - previousTime
                slip, s, a, noMove, v, w = optic.slipRatio_and_currentAcceleration(
                    wheel_distance_move - previousWheelMove,
                    ground_distance_move - previousGroundMove,
                    period,
                    currentTime,
                    previousGroundVelocity,
                    previousTime,
                )

                previousTime = currentTime
                previousGroundVelocity = v
                previousWheelMove = wheel_distance_move
                previousGroundMove = ground_distance_move

                left_items.extend(
                    [
                        (f"period: {period:.3f} s", (255, 255, 255)),
                        (f"slip: {slip}", (0, 0, 255)),
                        (f"s: {s:.3f}", (0, 255, 255)),
                    ]
                )
                right_items.extend(
                    [
                        (f"noMove: {noMove}", (255, 255, 255)),
                        (f"acceleration: {a:.3f}", (0, 255, 255)),
                        (f"velocity: {v:.3f}", (0, 255, 0)),
                        (f"angular velocity: {w:.3f}", (255, 0, 255)),
                    ]
                )
            else:
                previousWheelMove = 0
                previousGroundMove = 0
                left_items.append(("updating frame", (0, 255, 255)))

            output = draw_two_column_text(output, left_items, right_items)

            writer.write(output)

            count += 1
            if count == MAX_FRAMES:
                break

    finally:
        writer.release()
        camera.stop()
        cv2.destroyAllWindows()

    print(f"Saved video: {video_path.resolve()}")


if __name__ == "__main__":
    main()