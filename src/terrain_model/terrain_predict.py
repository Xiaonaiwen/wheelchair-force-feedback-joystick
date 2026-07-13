from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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


def load_checkpoint():
    checkpoint_path = get_project_root() / "models" / "terrain_cnn_best.pth"
    return torch.load(checkpoint_path, map_location="cpu")


def frame_to_tensor(frame):
    """
    frame: RGB NumPy array with shape (224, 224, 3)
    """
    frame = np.asarray(frame)

    if frame.shape != (IMAGE_SIZE, IMAGE_SIZE, 3):
        raise ValueError(f"Expected frame shape {(IMAGE_SIZE, IMAGE_SIZE, 3)}, got {frame.shape}")

    frame = torch.from_numpy(frame).float() / 255.0
    frame = frame.permute(2, 0, 1)   # HWC -> CHW
    return frame.unsqueeze(0)        # add batch dimension


def predict_frame(frame):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = load_checkpoint()
    class_to_idx = checkpoint["class_to_idx"]
    idx_to_class = {int(k): v for k, v in checkpoint["idx_to_class"].items()}

    model = TerrainCNN(num_classes=len(class_to_idx)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    image = frame_to_tensor(frame).to(device)

    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.softmax(outputs, dim=1)
        pred_idx = probabilities.argmax(dim=1).item()
        pred_class = idx_to_class[pred_idx]
        confidence = probabilities[0, pred_idx].item()

    print(f"Predicted class: {pred_class}")
    print(f"Confidence: {confidence:.4f}")
    return pred_class, confidence