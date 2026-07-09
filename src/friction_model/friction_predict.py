from pathlib import Path

import cv2
import numpy as np
import torch

from friction_model.friction_train_model import FrictionCNN, IMAGE_SIZE


def get_project_root():
    return Path(__file__).resolve().parents[2]


def frame_to_tensor(frame):
    frame = torch.from_numpy(frame).float() / 255.0
    frame = frame.permute(2, 0, 1)  # HWC -> CHW
    return frame.unsqueeze(0)  # add batch dimension


def predict_frame(frame):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = get_project_root()

    checkpoint = torch.load(
        root / "models" / "friction_regression_cnn.pth",
        map_location=device,
    )

    model = FrictionCNN().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    image = frame_to_tensor(frame).to(device)

    with torch.no_grad():
        prediction = model(image).item()

    print(f"\nEstimated Static Friction Coefficient (μs): {prediction:.3f}")
    return prediction