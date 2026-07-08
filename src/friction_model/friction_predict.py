from pathlib import Path

import numpy as np
import torch
from PIL import Image

from friction_model.friction_train_model import FrictionCNN, IMAGE_SIZE


def get_project_root():
    return Path(__file__).resolve().parents[2]


def load_image(image_path):

    image = Image.open(image_path).convert("RGB")
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))

    image = torch.from_numpy(np.array(image)).float() / 255.0
    image = image.permute(2, 0, 1)

    return image.unsqueeze(0)


def predict(image_path):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    root = get_project_root()

    checkpoint = torch.load(
        root / "models" / "friction_regression_cnn.pth",
        map_location=device,
    )

    model = FrictionCNN().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    image_path = Path(image_path)

    if not image_path.is_absolute():
        image_path = root / image_path

    image = load_image(image_path).to(device)

    with torch.no_grad():
        prediction = model(image).item()

    print(f"\nEstimated Static Friction Coefficient (μs): {prediction:.3f}")

    return prediction


if __name__ == "__main__":

    image_path = input("Image path: ")

    predict(image_path)