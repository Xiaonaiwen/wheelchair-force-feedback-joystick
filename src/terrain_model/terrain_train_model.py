from pathlib import Path
from PIL import Image
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter

CLASS_TO_IDX = {}
IDX_TO_CLASS = {}

IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS = 20
LEARNING_RATE = 1e-3


def get_project_root():
    return Path(__file__).resolve().parents[2]


def create_class_to_idx():
    global CLASS_TO_IDX, IDX_TO_CLASS

    train_dir = get_project_root() / "datasets" / "processed" / "terrain classifier" / "train"

    classes = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])

    CLASS_TO_IDX = {name: idx for idx, name in enumerate(classes)}
    IDX_TO_CLASS = {idx: name for name, idx in CLASS_TO_IDX.items()}


def create_train_dataset():
    TRAIN_DIR = get_project_root() / "datasets" / "processed" / "terrain classifier" / "train"
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

def create_test_dataset():
    TEST_DIR = get_project_root() / "datasets" / "processed" / "terrain classifier" / "test"
    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

    dataset = []

    for class_name in sorted(TEST_DIR.iterdir()):
        if not class_name.is_dir():
            continue

        label = CLASS_TO_IDX[class_name.name]

        for image_path in class_name.iterdir():
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            dataset.append((image_path, label))

    return dataset
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



def train_model(train_dataset, test_dataset):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_classes = len(CLASS_TO_IDX)
    model = TerrainCNN(num_classes).to(device)

    labels = [label for _, label in train_dataset]
    counts = Counter(labels)

    print("Class counts:")
    for i in range(num_classes):
        print(f"{IDX_TO_CLASS[i]:20s}: {counts[i]}")

    class_weights = torch.tensor(
        [1.0 / counts[i] for i in range(num_classes)],
        dtype=torch.float32,
        device=device,
    )
    class_weights = class_weights / class_weights.sum() * num_classes

    print("\nClass weights:")
    for i in range(num_classes):
        print(f"{IDX_TO_CLASS[i]:20s}: {class_weights[i].item():.4f}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    project_root = get_project_root()
    model_dir = project_root / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    latest_path = model_dir / "terrain_cnn.pth"
    best_path = model_dir / "terrain_cnn_best.pth"

    start_epoch = 0
    best_test_loss = float("inf")
    best_test_acc = 0.0

    # Resume from latest checkpoint if it exists
    if latest_path.exists():
        checkpoint = torch.load(latest_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint.get("epoch", -1) + 1
        best_test_loss = checkpoint.get("best_test_loss", float("inf"))
        best_test_acc = checkpoint.get("best_test_acc", 0.0)
        print(f"Resumed from latest checkpoint: {latest_path}")

    def save_checkpoint(path, epoch, test_loss, test_acc):
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_test_loss": test_loss,
                "best_test_acc": test_acc,
                "class_to_idx": CLASS_TO_IDX,
                "idx_to_class": IDX_TO_CLASS,
            },
            path,
        )

    def evaluate(test_dataset):
        model.eval()

        total_loss = 0.0
        total_correct = 0
        total = 0

        with torch.no_grad():
            for start in range(0, len(test_dataset), BATCH_SIZE):
                batch = test_dataset[start:start + BATCH_SIZE]

                batch_images = []
                batch_labels = []

                for image_path, label in batch:
                    image = load_image(image_path)
                    batch_images.append(image)
                    batch_labels.append(label)

                batch_images = torch.stack(batch_images).to(device)
                batch_labels = torch.tensor(batch_labels, dtype=torch.long).to(device)

                outputs = model(batch_images)
                loss = criterion(outputs, batch_labels)
                preds = outputs.argmax(dim=1)

                total_correct += (preds == batch_labels).sum().item()
                total += batch_labels.size(0)
                total_loss += loss.item() * batch_labels.size(0)

        avg_loss = total_loss / total
        avg_acc = total_correct / total
        return avg_loss, avg_acc

    for epoch in range(start_epoch, NUM_EPOCHS):
        model.train()

        random.shuffle(train_dataset)  # shuffle once per epoch

        running_loss = 0.0
        running_correct = 0
        running_total = 0

        for start in range(0, len(train_dataset), BATCH_SIZE):
            batch = train_dataset[start:start + BATCH_SIZE]

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

            preds = outputs.argmax(dim=1)
            running_correct += (preds == batch_labels).sum().item()
            running_total += batch_labels.size(0)

        train_loss = running_loss / running_total
        train_acc = running_correct / running_total

        test_loss, test_acc = evaluate(test_dataset)

        print(
            f"Epoch {epoch + 1}/{NUM_EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}"
        )

        # Always save latest checkpoint
        save_checkpoint(latest_path, epoch, best_test_loss, best_test_acc)

        # Save best checkpoint when test loss improves
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_test_acc = test_acc
            save_checkpoint(best_path, epoch, best_test_loss, best_test_acc)
            print(f"Saved best model to: {best_path}")

            # Keep latest checkpoint in sync with the new best metrics
            save_checkpoint(latest_path, epoch, best_test_loss, best_test_acc)

    print(f"\nBest Test Loss: {best_test_loss:.4f}")
    print(f"Best Test Acc : {best_test_acc:.4f}")

    if best_path.exists():
        checkpoint = torch.load(best_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

    return model


def main():
    create_class_to_idx()

    train_dataset = create_train_dataset()
    test_dataset = create_test_dataset()

    train_model(train_dataset, test_dataset)

    print("Training complete.")


if __name__ == "__main__":
    main()