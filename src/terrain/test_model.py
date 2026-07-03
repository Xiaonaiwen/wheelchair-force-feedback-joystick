from pathlib import Path
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

IMAGE_SIZE = 224


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


def get_project_root():
    return Path(__file__).resolve().parents[2]


def get_test_dir():
    return get_project_root() / "datasets" / "wheelchair_combined" / "test"


def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    image = torch.from_numpy(np.array(image)).float() / 255.0
    image = image.permute(2, 0, 1)  # HWC -> CHW
    return image


def load_checkpoint():
    checkpoint_path = get_project_root() / "models" / "terrain_cnn.pth"
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    return checkpoint


def create_test_dataset(class_to_idx):
    test_dir = get_test_dir()
    image_extensions = {".jpg", ".jpeg", ".png"}

    dataset = []

    for class_dir in sorted(test_dir.iterdir()):
        if not class_dir.is_dir():
            continue

        class_name = class_dir.name
        if class_name not in class_to_idx:
            continue

        label = class_to_idx[class_name]

        for image_path in class_dir.iterdir():
            if image_path.suffix.lower() not in image_extensions:
                continue

            dataset.append((image_path, label))

    return dataset


def plot_confusion_matrix(cm, class_names, save_path):
    plt.figure(figsize=(12, 10))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)

    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def evaluate_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = load_checkpoint()
    class_to_idx = checkpoint["class_to_idx"]
    idx_to_class = checkpoint["idx_to_class"]

    # If idx_to_class was saved as a dict with string keys, convert safely.
    if isinstance(idx_to_class, dict):
        class_names = [idx_to_class[str(i)] if str(i) in idx_to_class else idx_to_class[i]
                       for i in range(len(class_to_idx))]
    else:
        class_names = list(idx_to_class)

    test_dataset = create_test_dataset(class_to_idx)

    model = TerrainCNN(num_classes=len(class_to_idx)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    num_classes = len(class_to_idx)
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

    correct = 0
    total = 0

    with torch.no_grad():
        for image_path, label in test_dataset:
            image = load_image(image_path).unsqueeze(0).to(device)
            true_label = int(label)

            outputs = model(image)
            pred_label = int(outputs.argmax(dim=1).item())

            confusion[true_label, pred_label] += 1

            if pred_label == true_label:
                correct += 1
            total += 1

    accuracy = correct / total if total > 0 else 0.0
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test Samples: {total}")
    print(f"Classes: {len(class_names)}")

    print("\nPer-class accuracy:")
    for i, class_name in enumerate(class_names):
        class_total = confusion[i].sum()
        class_correct = confusion[i, i]
        class_acc = class_correct / class_total if class_total > 0 else 0.0
        print(f"{class_name:20s} {class_acc:.4f}  ({class_correct}/{class_total})")

    save_dir = get_project_root() / "results"
    save_dir.mkdir(exist_ok=True)
    plot_path = save_dir / "confusion_matrix.png"
    plot_confusion_matrix(confusion, class_names, plot_path)
    print(f"\nSaved confusion matrix to: {plot_path}")
     
    print("\nMost common misclassification:")
    for i, true_name in enumerate(class_names):
        row = confusion[i].copy()
        row[i] = 0  # Ignore correct predictions

        if row.sum() == 0:
            print(f"{true_name:20s} -> No mistakes")
            continue

        pred_idx = np.argmax(row)
        pred_name = class_names[pred_idx]
        count = row[pred_idx]

        print(f"{true_name:20s} -> {pred_name:20s} ({count})")
    print("\nTop 3 misclassifications:")
    for i, true_name in enumerate(class_names):
        row = confusion[i].copy()
        row[i] = 0

        top3 = np.argsort(row)[::-1][:3]

        print(f"\n{true_name}:")
        for idx in top3:
            if row[idx] > 0:
                print(f"    {class_names[idx]:20s} {row[idx]}")



def main():
    evaluate_model()


if __name__ == "__main__":
    main()