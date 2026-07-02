from pathlib import Path
import torch

CLASS_TO_IDX = {}
IDX_TO_CLASS = {}

def create_class_to_idx():
    global CLASS_TO_IDX, IDX_TO_CLASS

    PROJECT_ROOT = Path(__file__).resolve().parents[2]

    train_dir = PROJECT_ROOT / "datasets" / "wheelchair_combined" / "train"

    classes = sorted(
        [d.name for d in train_dir.iterdir() if d.is_dir()]
    )

    CLASS_TO_IDX = {
        name: idx
        for idx, name in enumerate(classes)
    }

    IDX_TO_CLASS = {
        idx: name
        for name, idx in CLASS_TO_IDX.items()
    }

def train():

def main():
    create_class_to_idx()


if __name__ == "__main__":
    main()