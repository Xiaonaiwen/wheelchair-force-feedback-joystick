from pathlib import Path
import csv
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
FORCE_REBUILD = True  # set False once everything works

GTOS_RAW_NAME = "iSolver-AI/GTOS-Mobile"
EXTREME_RAW = ROOT / "datasets" / "Extreme-Road-Image-Dataset"
OUT_ROOT = ROOT / "datasets" / "processed" / "friction_regression"

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Normalized label -> target friction coefficient
LABEL_TO_MU = {
    # High-ish hard surfaces
    "asphalt": 0.81,
    "stone asphalt": 0.81,
    "cement": 0.725,
    "stone cement": 0.725,

    # Medium-high hard surfaces
    "brick": 0.65,
    "stone brick": 0.65,
    "large limestone": 0.625,
    "small limestone": 0.625,

    # Medium-low surfaces
    "grass": 0.295,
    "turf": 0.295,
    "painting turf": 0.295,
    "sand": 0.375,
    "wood chips": 0.375,

    # Natural ground / variable terrain
    "soil": 0.54,

    # Special surfaces
    "pebble": 0.60,
    "metal cover": 0.60,
    "aluminum": 0.60,
    "steel": 0.60,

    # Low / very low
    "waterlogged pavement": 0.20,
    "ice": 0.085,
    "rough ice surface": 0.085,
    "loose snow surface": 0.08,
    "muddy road after snow": 0.10,

    # Wet asphalt
    "semi impregnated asphalt pavement": 0.50,
}

# Extreme Road folder name -> normalized label
EXTREME_FOLDER_TO_LABEL = {
    "1 ice surface": "ice",
    "2 rough ice surface": "rough ice surface",
    "3 loose snow surface": "loose snow surface",
    "4 muddy road after snow": "muddy road after snow",
    "5 waterlogged pavement": "waterlogged pavement",
    "6 semi impregnated asphalt pavement": "semi impregnated asphalt pavement",
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


def save_resized_image_from_path(img_path: Path, out_path: Path):
    with Image.open(img_path) as img:
        img = img.convert("RGB")
        img = img.resize(TARGET_SIZE, Image.Resampling.LANCZOS)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(out_path, quality=95)


def save_resized_image_from_pil(img: Image.Image, out_path: Path):
    img = img.convert("RGB")
    img = img.resize(TARGET_SIZE, Image.Resampling.LANCZOS)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, quality=95)


def write_csv(rows: list[dict], csv_path: Path):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["image_path", "mu_s", "source", "original_label"]
        )
        writer.writeheader()
        writer.writerows(rows)


def process_gtos():
    print("Loading GTOS-Mobile...")
    dataset = load_dataset(GTOS_RAW_NAME)

    class_names = dataset["train"].features["label"].names

    out_rows = {"train": [], "test": []}

    for split in ["train", "test"]:
        copied = 0
        skipped = 0

        for i, sample in enumerate(dataset[split]):
            original_label = normalize(class_names[sample["label"]])

            if original_label not in LABEL_TO_MU:
                skipped += 1
                continue

            mu_s = LABEL_TO_MU[original_label]
            out_path = OUT_ROOT / split / "images" / f"g_{split}_{i:06d}.jpg"
            save_resized_image_from_pil(sample["image"], out_path)

            out_rows[split].append({
                "image_path": str(out_path.relative_to(ROOT)),
                "mu_s": mu_s,
                "source": "GTOS",
                "original_label": original_label,
            })
            copied += 1

        print(f"GTOS {split}: copied={copied}, skipped={skipped}")

    write_csv(out_rows["train"], OUT_ROOT / "train" / "labels.csv")
    write_csv(out_rows["test"], OUT_ROOT / "test" / "labels.csv")


def process_extreme_road():
    if not EXTREME_RAW.exists():
        raise FileNotFoundError(f"Extreme Road dataset folder not found: {EXTREME_RAW}")

    unzip_all_zips(EXTREME_RAW)
    rng = random.Random(SEED)

    out_rows = {"train": [], "test": []}

    for folder_name, label_name in EXTREME_FOLDER_TO_LABEL.items():
        src_folder = find_folder_recursive(EXTREME_RAW, folder_name)
        if src_folder is None:
            print(f"Missing folder: {folder_name}")
            continue

        images = collect_images(src_folder)
        if not images:
            print(f"No images found for: {folder_name}")
            continue

        rng.shuffle(images)
        split_idx = int(len(images) * TRAIN_RATIO)
        train_images = images[:split_idx]
        test_images = images[split_idx:]

        mu_s = LABEL_TO_MU[label_name]

        for idx, img_path in enumerate(train_images):
            out_path = OUT_ROOT / "train" / "images" / f"e_{normalize(label_name).replace(' ', '_')}_{idx:06d}.jpg"
            save_resized_image_from_path(img_path, out_path)

            out_rows["train"].append({
                "image_path": str(out_path.relative_to(ROOT)),
                "mu_s": mu_s,
                "source": "ExtremeRoad",
                "original_label": label_name,
            })

        for idx, img_path in enumerate(test_images):
            out_path = OUT_ROOT / "test" / "images" / f"e_{normalize(label_name).replace(' ', '_')}_{idx:06d}.jpg"
            save_resized_image_from_path(img_path, out_path)

            out_rows["test"].append({
                "image_path": str(out_path.relative_to(ROOT)),
                "mu_s": mu_s,
                "source": "ExtremeRoad",
                "original_label": label_name,
            })

        print(f"Extreme Road {label_name}: train={len(train_images)}, test={len(test_images)}")

    # Append Extreme Road rows to the existing GTOS CSVs
    train_csv = OUT_ROOT / "train" / "labels.csv"
    test_csv = OUT_ROOT / "test" / "labels.csv"

    existing_train = []
    existing_test = []

    if train_csv.exists():
        with train_csv.open("r", newline="", encoding="utf-8") as f:
            existing_train = list(csv.DictReader(f))
    if test_csv.exists():
        with test_csv.open("r", newline="", encoding="utf-8") as f:
            existing_test = list(csv.DictReader(f))

    existing_train.extend(out_rows["train"])
    existing_test.extend(out_rows["test"])

    write_csv(existing_train, train_csv)
    write_csv(existing_test, test_csv)


def main():
    if maybe_skip(OUT_ROOT):
        print("Dataset already prepared. Skipping.")
        return

    clear_dir(OUT_ROOT)
    (OUT_ROOT / "train" / "images").mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "test" / "images").mkdir(parents=True, exist_ok=True)

    process_gtos()
    process_extreme_road()

    print(f"Prepared regression dataset saved to: {OUT_ROOT}")


if __name__ == "__main__":
    main()