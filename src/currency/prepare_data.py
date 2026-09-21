"""
Split the Indian currency dataset into train and validation sets, grouped
by ORIGINAL photo rather than by individual file.

The source dataset contains several augmented copies of each original
photo (rotated, zoomed, flipped). Splitting file-by-file placed copies of
the same photo on both sides, producing 84% leakage and an inflated 99.4%
validation accuracy. Grouping by original guarantees every copy of a
photo lands in the same split.

Filenames look like: 10_original_103.jpg_<uuid>.jpg
Everything before ".jpg_" identifies the original photo.
"""

import random
import shutil
from collections import defaultdict
from pathlib import Path

SOURCE = Path(r"D:\currenc data\Indian Currency Dataset")
DESTINATION = Path(r"D:\GUIDE\data\currency")
VALIDATION_SPLIT = 0.2

# Fixed seed so the split is reproducible
random.seed(42)


def original_id(image_path):
    """Return the identifier of the original photo this file came from."""
    return image_path.name.split(".jpg_")[0]


def main():
    # Refuse to run over an old split, which would mix the two
    if DESTINATION.exists():
        print(f"{DESTINATION} already exists. Delete it first, then re-run.")
        return

    class_folders = [f for f in SOURCE.iterdir() if f.is_dir()]
    print(f"Found {len(class_folders)} classes")

    for class_folder in class_folders:
        images = list(class_folder.glob("*.jpg")) + list(class_folder.glob("*.png"))

        # Group every file under the original photo it was derived from
        groups = defaultdict(list)
        for image in images:
            groups[original_id(image)].append(image)

        # Shuffle and split the ORIGINALS, not the individual files
        originals = list(groups.keys())
        random.shuffle(originals)

        split_point = int(len(originals) * (1 - VALIDATION_SPLIT))
        train_originals = originals[:split_point]
        val_originals = originals[split_point:]

        train_dir = DESTINATION / "train" / class_folder.name
        val_dir = DESTINATION / "val" / class_folder.name
        train_dir.mkdir(parents=True, exist_ok=True)
        val_dir.mkdir(parents=True, exist_ok=True)

        train_count = 0
        for original in train_originals:
            for image in groups[original]:
                shutil.copy2(image, train_dir / image.name)
                train_count += 1

        val_count = 0
        for original in val_originals:
            for image in groups[original]:
                shutil.copy2(image, val_dir / image.name)
                val_count += 1

        print(f"  {class_folder.name}: {len(originals)} originals -> "
              f"{train_count} train files, {val_count} val files")

    print(f"\nDone. Dataset written to {DESTINATION}")


if __name__ == "__main__":
    main()