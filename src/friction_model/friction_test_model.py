from pathlib import Path
import csv

import numpy as np
import torch
from PIL import Image

from friction_train_model import FrictionCNN, IMAGE_SIZE


def get_project_root():
    return Path(__file__).resolve().parents[2]


def load_image(path):
    image = Image.open(path).convert("RGB")
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))

    image = torch.from_numpy(np.array(image)).float() / 255.0
    image = image.permute(2, 0, 1)

    return image.unsqueeze(0)


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    root = get_project_root()

    csv_file = (
        root
        / "datasets"
        / "processed"
        / "friction regression"
        / "test"
        / "labels.csv"
    )

    checkpoint = torch.load(
        root / "models" / "friction_cnn_best.pth", 
        map_location=device,
    )

    model = FrictionCNN().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    errors = []
    squared_errors = []

    with open(csv_file, newline="", encoding="utf-8") as f:

        reader = csv.DictReader(f)

        for i, row in enumerate(reader):

            image_path = root / row["image_path"]
            target = float(row["mu_s"])

            image = load_image(image_path).to(device)

            with torch.no_grad():
                prediction = model(image).item()

            error = abs(prediction - target)

            errors.append(error)
            squared_errors.append(error ** 2)

    mae = np.mean(errors)
    rmse = np.sqrt(np.mean(squared_errors))

    print("\n==============================")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Root Mean Square Error (RMSE): {rmse:.4f}")
    print("==============================")


if __name__ == "__main__":
    main()