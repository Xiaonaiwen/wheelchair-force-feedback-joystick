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
FORCE_REBUILD = True   # set False after everything works

GTOS_OUT = ROOT / "datasets" / "processed" / "GTOS-Mobile-224"
EXTREME_RAW = ROOT / "datasets" / "Extreme-Road-Image-Dataset"
EXTREME_OUT = ROOT / "datasets" / "processed" / "Extreme-Road-224"
COMBINED_OUT = ROOT / "datasets" / "wheelchair_combined"

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Final labels in the combined dataset
CLASS_MAP = {
    "asphalt": ["asphalt", "stone_asphalt"],
    "brick": ["brick", "stone_brick"],
    "cement": ["cement", "stone_cement"],
    "grass": ["grass", "turf", "painting_turf"],
    "pebble": ["pebble"],
    "sand": ["sand"],
    "soil": ["soil"],
    "metal_cover": ["metal_cover"],
    "limestone": ["large_limestone", "small_limestone"],
    "wood_chips": ["wood_chips"],

    "ice": ["1-ice surface", "2-rough ice surface"],
    "snow": ["3-loose snow surface"],
    "muddy_snow": ["4-muddy road after snow"],
    "waterlogged_pavement": ["5-waterlogged pavement"],
    "wet_asphalt": ["6-semi-impregnated asphalt pavement"],
}
# =========================


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


def export_gtos():
    if maybe_skip(GTOS_OUT):
        print("GTOS already processed. Skipping.")
        return

    print("Loading GTOS-Mobile...")
    dataset = load_dataset("iSolver-AI/GTOS-Mobile")

    reverse_map = {}
    for out_class, in_classes in CLASS_MAP.items():
        for in_class in in_classes:
            reverse_map[normalize(in_class)] = out_class

    clear_dir(GTOS_OUT)
    for split in ["train", "test"]:
        for out_class in CLASS_MAP.keys():
            (GTOS_OUT / split / out_class).mkdir(parents=True, exist_ok=True)

    for split in ["train", "test"]:
        copied = 0
        skipped = 0
        class_names = dataset[split].features["label"].names

        for i, sample in enumerate(dataset[split]):
            img = sample["image"].convert("RGB")
            label_name = normalize(class_names[sample["label"]])

            if label_name not in reverse_map:
                skipped += 1
                continue

            out_class = reverse_map[label_name]
            img = img.resize(TARGET_SIZE, Image.Resampling.LANCZOS)

            save_name = f"{split}_{i:06d}.jpg"
            save_path = GTOS_OUT / split / out_class / save_name
            img.save(save_path, quality=95)
            copied += 1

        print(f"GTOS {split}: copied {copied}, skipped {skipped}")


def export_extreme_road():
    if maybe_skip(EXTREME_OUT):
        print("Extreme Road already processed. Skipping.")
        return

    if not EXTREME_RAW.exists():
        raise FileNotFoundError(f"Extreme Road dataset folder not found: {EXTREME_RAW}")

    unzip_all_zips(EXTREME_RAW)

    clear_dir(EXTREME_OUT)
    for split in ["train", "test"]:
        for out_class in ["ice", "snow", "muddy_snow", "waterlogged_pavement", "wet_asphalt"]:
            (EXTREME_OUT / split / out_class).mkdir(parents=True, exist_ok=True)

    rng = random.Random(SEED)

    extreme_map = {
        "ice": ["1-Ice Surface", "2-Rough Ice Surface"],
        "snow": ["3-Loose snow surface"],
        "muddy_snow": ["4-Muddy Road After Snow"],
        "waterlogged_pavement": ["5-Waterlogged Pavement"],
        "wet_asphalt": ["6-Semi-impregnated Asphalt Pavement"],
    }

    for out_class, source_names in extreme_map.items():
        all_images = []

        for source_name in source_names:
            src_folder = find_folder_recursive(EXTREME_RAW, source_name)
            if src_folder is None:
                print(f"Missing folder: {source_name}")
                continue

            print(f"Found folder: {src_folder}")
            all_images.extend(collect_images(src_folder))

        if not all_images:
            print(f"No images found for {out_class}")
            continue

        rng.shuffle(all_images)
        split_idx = int(len(all_images) * TRAIN_RATIO)
        train_images = all_images[:split_idx]
        test_images = all_images[split_idx:]

        for idx, img_path in enumerate(train_images):
            out_path = EXTREME_OUT / "train" / out_class / f"{out_class}_{idx:06d}.jpg"
            save_resized(img_path, out_path)

        for idx, img_path in enumerate(test_images):
            out_path = EXTREME_OUT / "test" / out_class / f"{out_class}_{idx:06d}.jpg"
            save_resized(img_path, out_path)

        print(f"Extreme Road {out_class}: train={len(train_images)}, test={len(test_images)}")


def combine_datasets():
    if maybe_skip(COMBINED_OUT):
        print("Combined dataset already exists. Skipping.")
        return

    clear_dir(COMBINED_OUT)
    for split in ["train", "test"]:
        for out_class in CLASS_MAP.keys():
            (COMBINED_OUT / split / out_class).mkdir(parents=True, exist_ok=True)

    # GTOS
    for split in ["train", "test"]:
        src_split = GTOS_OUT / split
        if src_split.exists():
            for class_dir in src_split.iterdir():
                if class_dir.is_dir() and class_dir.name in CLASS_MAP:
                    for img_path in class_dir.iterdir():
                        if img_path.is_file():
                            dst = COMBINED_OUT / split / class_dir.name / f"GTOS_{img_path.name}"
                            shutil.copy2(img_path, dst)

    # Extreme Road
    for split in ["train", "test"]:
        src_split = EXTREME_OUT / split
        if src_split.exists():
            for class_dir in src_split.iterdir():
                if class_dir.is_dir() and class_dir.name in CLASS_MAP:
                    for img_path in class_dir.iterdir():
                        if img_path.is_file():
                            dst = COMBINED_OUT / split / class_dir.name / f"EXT_{img_path.name}"
                            shutil.copy2(img_path, dst)

    print(f"Combined dataset saved to: {COMBINED_OUT}")


def main():
    export_gtos()
    export_extreme_road()
    combine_datasets()


if __name__ == "__main__":
    main()