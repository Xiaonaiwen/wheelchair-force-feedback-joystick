from pathlib import Path
import random
import shutil
import zipfile

from datasets import load_dataset
from PIL import Image

# =========================
# SETTINGS
# =========================
ROOT = Path(__file__).resolve().parents[2]
TARGET_SIZE = (224, 224)
TRAIN_RATIO = 0.8
SEED = 42
FORCE_REBUILD = False  # set False after everything works

GTOS_RAW_NAME = "iSolver-AI/GTOS-Mobile"
EXTREME_RAW = ROOT / "datasets" / "raw" / "Extreme Road Image Dataset"
COMBINED_OUT = ROOT / "datasets" / "processed" / "terrain classifier"

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

FINAL_CLASSES = [
    "high",
    "medium_high",
    "medium_low",
    "natural_ground",
    "low",
    "very_low",
    "pebble",
    "wet_asphalt",
]

# GTOS-Mobile source labels -> final labels
GTOS_TO_FINAL = {
    # High
    "asphalt": "high",
    "stone asphalt": "high",
    "cement": "high",
    "stone cement": "high",

    # Medium High
    "brick": "medium_high",
    "stone brick": "medium_high",
    "large limestone": "medium_high",
    "small limestone": "medium_high",

    # Medium Low
    "sand": "medium_low",
    "wood chips": "medium_low",

    # Natural Ground
    "grass": "natural_ground",
    "turf": "natural_ground",
    "painting turf": "natural_ground",
    "soil": "natural_ground",

    # Special
    "pebble": "pebble",
}

# Extreme Road source labels -> final labels
EXTREME_TO_FINAL = {
    "1 ice surface": "very_low",
    "2 rough ice surface": "very_low",
    "3 loose snow surface": "very_low",
    "4 muddy road after snow": "very_low",
    "5 waterlogged pavement": "low",
    "6 semi impregnated asphalt pavement": "wet_asphalt",
}


def normalize(name: str) -> str:
    return (
        name.lower()
        .strip()
        .replace("_", " ")
        .replace("-", " ")
        .replace("  ", " ")
    )


def clear_dir(path: Path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def maybe_skip(path: Path) -> bool:
    return path.exists() and not FORCE_REBUILD


def unzip_all_zips(root: Path):
    for zip_path in root.rglob("*.zip"):
        extract_dir = zip_path.with_suffix("")
        if extract_dir.exists():
            continue
        print(f"Unzipping: {zip_path.name}")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(zip_path.parent)


def find_folder_recursive(root: Path, target_name: str) -> Path | None:
    target = normalize(target_name)
    for p in root.rglob("*"):
        if p.is_dir() and normalize(p.name) == target:
            return p
    return None


def collect_images(folder: Path) -> list[Path]:
    images = []
    for p in folder.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in VALID_EXTS:
            continue
        if "__MACOSX" in p.parts:
            continue
        if p.name.startswith("._"):
            continue
        images.append(p)
    return images


def save_resized(img_path: Path, out_path: Path):
    with Image.open(img_path) as img:
        img = img.convert("RGB")
        img = img.resize(TARGET_SIZE, Image.Resampling.LANCZOS)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(out_path, quality=95)


def ensure_final_dirs():
    for split in ["train", "test"]:
        for cls in FINAL_CLASSES:
            (COMBINED_OUT / split / cls).mkdir(parents=True, exist_ok=True)


def export_gtos_into_combined():
    print("Loading GTOS-Mobile...")
    dataset = load_dataset(GTOS_RAW_NAME)

    class_names = dataset["train"].features["label"].names
    reverse_map = {normalize(k): v for k, v in GTOS_TO_FINAL.items()}

    for split in ["train", "test"]:
        copied = 0
        skipped = 0

        for i, sample in enumerate(dataset[split]):
            label_name = normalize(class_names[sample["label"]])

            if label_name not in reverse_map:
                skipped += 1
                continue

            final_class = reverse_map[label_name]
            img = sample["image"].convert("RGB")
            img = img.resize(TARGET_SIZE, Image.Resampling.LANCZOS)

            out_name = f"GTOS_{split}_{i:06d}.jpg"
            out_path = COMBINED_OUT / split / final_class / out_name
            img.save(out_path, quality=95)
            copied += 1

        print(f"GTOS {split}: copied={copied}, skipped={skipped}")


def export_extreme_into_combined():
    if not EXTREME_RAW.exists():
        raise FileNotFoundError(f"Extreme Road dataset folder not found: {EXTREME_RAW}")

    unzip_all_zips(EXTREME_RAW)

    rng = random.Random(SEED)
    reverse_map = {normalize(k): v for k, v in EXTREME_TO_FINAL.items()}

    for final_class in ["very_low", "low", "wet_asphalt"]:
        source_names = [k for k, v in EXTREME_TO_FINAL.items() if v == final_class]
        all_images = []

        for source_name in source_names:
            src_folder = find_folder_recursive(EXTREME_RAW, source_name)
            if src_folder is None:
                print(f"Missing folder: {source_name}")
                continue

            print(f"Found folder: {src_folder}")
            all_images.extend(collect_images(src_folder))

        if not all_images:
            print(f"No images found for {final_class}")
            continue

        rng.shuffle(all_images)
        split_idx = int(len(all_images) * TRAIN_RATIO)
        train_images = all_images[:split_idx]
        test_images = all_images[split_idx:]

        for idx, img_path in enumerate(train_images):
            out_path = COMBINED_OUT / "train" / final_class / f"EXT_{final_class}_{idx:06d}.jpg"
            save_resized(img_path, out_path)

        for idx, img_path in enumerate(test_images):
            out_path = COMBINED_OUT / "test" / final_class / f"EXT_{final_class}_{idx:06d}.jpg"
            save_resized(img_path, out_path)

        print(f"Extreme Road {final_class}: train={len(train_images)}, test={len(test_images)}")


def main():
    if maybe_skip(COMBINED_OUT):
        print("Combined dataset already exists. Skipping.")
        return

    clear_dir(COMBINED_OUT)
    ensure_final_dirs()

    export_gtos_into_combined()
    export_extreme_into_combined()

    print(f"Combined dataset saved to: {COMBINED_OUT}")


if __name__ == "__main__":
    main()