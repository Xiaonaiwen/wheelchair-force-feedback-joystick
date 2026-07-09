from pathlib import Path
import csv
import random

from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS = 20
LEARNING_RATE = 1e-3
SEED = 42


def get_project_root():
    return Path(__file__).resolve().parents[2]


class FrictionDataset(Dataset):
    def __init__(self, csv_path: Path):
        self.project_root = get_project_root()
        self.samples = []

        with csv_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                image_path = self.project_root / row["image_path"]
                mu_s = float(row["mu_s"])
                self.samples.append((image_path, mu_s))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, mu_s = self.samples[idx]

        image = Image.open(image_path).convert("RGB")
        image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
        image = torch.from_numpy(np.array(image)).float() / 255.0
        image = image.permute(2, 0, 1)  # HWC -> CHW

        target = torch.tensor([mu_s], dtype=torch.float32)
        return image, target


class FrictionCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=5)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(32 * 24 * 24, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 1)   # regression output

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    total_n = 0

    criterion = nn.SmoothL1Loss(reduction="sum")

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)
            loss = criterion(outputs, targets)

            preds = outputs.squeeze(1)
            truth = targets.squeeze(1)

            mae = torch.abs(preds - truth).sum()

            batch_size = targets.size(0)
            total_loss += loss.item()
            total_mae += mae.item()
            total_n += batch_size

    avg_loss = total_loss / total_n
    avg_mae = total_mae / total_n
    return avg_loss, avg_mae


def train_model():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    project_root = get_project_root()
    train_csv = project_root / "datasets" / "processed" / "friction regression" / "train" / "labels.csv"
    test_csv = project_root / "datasets" / "processed" / "friction regression" / "test" / "labels.csv"

    train_dataset = FrictionDataset(train_csv)
    test_dataset = FrictionDataset(test_csv)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    model = FrictionCNN().to(device)

    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    model_dir = project_root / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    normal_save_path = model_dir / "friction_cnn.pth"
    best_save_path = model_dir / "friction_cnn_best.pth"

    best_test_mae = float("inf")
    start_epoch = 0

    # Resume from the normal/latest checkpoint if it exists
    if normal_save_path.exists():
        checkpoint = torch.load(normal_save_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint.get("epoch", -1) + 1
        best_test_mae = checkpoint.get("best_test_mae", float("inf"))
        print(f"Resumed from normal checkpoint: {normal_save_path}")

    def save_checkpoint(path, epoch, test_mae):
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "image_size": IMAGE_SIZE,
                "batch_size": BATCH_SIZE,
                "learning_rate": LEARNING_RATE,
                "best_test_mae": test_mae,
            },
            path,
        )

    for epoch in range(start_epoch, NUM_EPOCHS):
        model.train()

        running_loss = 0.0
        running_mae = 0.0
        running_n = 0

        for images, targets in train_loader:
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                preds = outputs.squeeze(1)
                truth = targets.squeeze(1)
                mae = torch.abs(preds - truth).sum()

            batch_size = targets.size(0)
            running_loss += loss.item() * batch_size
            running_mae += mae.item()
            running_n += batch_size

        train_loss = running_loss / running_n
        train_mae = running_mae / running_n

        test_loss, test_mae = evaluate(model, test_loader, device)

        print(
            f"Epoch {epoch + 1}/{NUM_EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train MAE: {train_mae:.4f} | "
            f"Test Loss: {test_loss:.4f} | "
            f"Test MAE: {test_mae:.4f}"
        )

        # Always save the latest checkpoint
        save_checkpoint(normal_save_path, epoch, best_test_mae)

        # Save the best checkpoint when test MAE improves
        if test_mae < best_test_mae:
            best_test_mae = test_mae
            save_checkpoint(best_save_path, epoch, best_test_mae)
            print(f"Saved best model to: {best_save_path}")

            # Update the normal checkpoint too, so it stores the new best_test_mae
            save_checkpoint(normal_save_path, epoch, best_test_mae)

    # Reload the best model before returning
    if best_save_path.exists():
        checkpoint = torch.load(best_save_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

    return model


def main():
    train_model()


if __name__ == "__main__":
    main()