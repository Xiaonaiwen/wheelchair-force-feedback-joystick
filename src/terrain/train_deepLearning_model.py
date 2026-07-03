from pathlib import Path
import random
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter

CLASS_TO_IDX = {}
IDX_TO_CLASS = {}

IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 0.001


def get_project_root():
    return Path(__file__).resolve().parents[2]


def create_class_to_idx():
    global CLASS_TO_IDX, IDX_TO_CLASS

    train_dir = get_project_root() / "datasets" / "wheelchair_combined" / "train"

    classes = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])

    CLASS_TO_IDX = {name: idx for idx, name in enumerate(classes)}
    IDX_TO_CLASS = {idx: name for name, idx in CLASS_TO_IDX.items()}


def create_train_dataset():
    TRAIN_DIR = get_project_root() / "datasets" / "wheelchair_combined" / "train"
    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

    dataset = []

    for class_name in sorted(TRAIN_DIR.iterdir()):
        if not class_name.is_dir():
            continue

        label = CLASS_TO_IDX[class_name.name]

        for image_path in class_name.iterdir():
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            dataset.append((image_path, label))

    return dataset


def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    image = torch.from_numpy(np.array(image)).float() / 255.0
    image = image.permute(2, 0, 1)  # HWC -> CHW
    return image


class TerrainCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=5)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(32 * 24 * 24, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train_model(dataset):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_classes = len(CLASS_TO_IDX)
    model = TerrainCNN(num_classes).to(device)
    model.train()

    # --------------------------------------------------
    # Class weights for imbalanced data
    # --------------------------------------------------
    labels = [label for _, label in dataset]
    counts = Counter(labels)

    print("Class counts:")
    for i in range(num_classes):
        print(f"{IDX_TO_CLASS[i]:20s}: {counts[i]}")

    class_weights = torch.tensor(
        [1.0 / counts[i] for i in range(num_classes)],
        dtype=torch.float32
    ).to(device)

    # Normalize weights so the average weight is ~1
    class_weights = class_weights / class_weights.sum() * num_classes

    print("\nClass weights:")
    for i in range(num_classes):
        print(f"{IDX_TO_CLASS[i]:20s}: {class_weights[i].item():.4f}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --------------------------------------------------
    # Training loop
    # --------------------------------------------------
    for epoch in range(NUM_EPOCHS):
        random.shuffle(dataset)

        running_loss = 0.0
        running_correct = 0
        running_total = 0

        for start in range(0, len(dataset), BATCH_SIZE):
            batch = dataset[start:start + BATCH_SIZE]

            batch_images = []
            batch_labels = []

            for image_path, label in batch:
                image = load_image(image_path)
                batch_images.append(image)
                batch_labels.append(label)

            batch_images = torch.stack(batch_images).to(device)
            batch_labels = torch.tensor(batch_labels, dtype=torch.long).to(device)

            optimizer.zero_grad()

            outputs = model(batch_images)
            loss = criterion(outputs, batch_labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch_labels.size(0)

            predictions = outputs.argmax(dim=1)
            running_correct += (predictions == batch_labels).sum().item()
            running_total += batch_labels.size(0)

        epoch_loss = running_loss / running_total
        epoch_acc = running_correct / running_total

        print(
            f"Epoch {epoch + 1}/{NUM_EPOCHS} | "
            f"Loss: {epoch_loss:.4f} | "
            f"Acc: {epoch_acc:.4f}"
        )

    return model


def main():
    create_class_to_idx()
    dataset = create_train_dataset()
    model = train_model(dataset)

    project_root = get_project_root()
    model_dir = project_root / "models"
    model_dir.mkdir(exist_ok=True)

    save_path = model_dir / "terrain_cnn.pth"

    torch.save({
        "model_state_dict": model.state_dict(),
        "class_to_idx": CLASS_TO_IDX,
        "idx_to_class": IDX_TO_CLASS,
    }, save_path)

    print(f"Model saved to: {save_path}")


if __name__ == "__main__":
    main()