"""
Select a small, diverse set of currency images for hand-labelling
bounding boxes.

- Picks ONE file per original photo. The dataset contains several
  augmented copies of each photo, and labelling near-duplicates would
  waste effort without adding variety.
- Draws from the existing train and val splits separately, preserving the
  leak-free split built on Day 7.
- Renames each copy to "Rs.500__<original name>", so the denomination can
  be recovered from the filename later. This means the labelling tool only
  needs one generic label, "note", which makes labelling faster.
"""

import random
import shutil
from collections import defaultdict
from pathlib import Path

DATA = Path(r"D:\GUIDE\data\currency")
OUTPUT = Path(r"D:\GUIDE\data\labelling")

# How many images to pick per denomination, from each split
PER_CLASS = {"train": 25, "val": 6}

random.seed(42)


def original_id(image_path):
    """Identifier of the original photo a file was derived from."""
    return image_path.name.split(".jpg_")[0]


def main():
    if OUTPUT.exists():
        print(f"{OUTPUT} already exists. Delete it first to reselect.")
        return

    for split, count in PER_CLASS.items():
        print(f"{split}:")
        destination = OUTPUT / split
        destination.mkdir(parents=True, exist_ok=True)

        for class_folder in sorted((DATA / split).iterdir()):
            if not class_folder.is_dir():
                continue

            # Group files by the original photo they came from
            groups = defaultdict(list)
            for image in class_folder.glob("*.jpg"):
                groups[original_id(image)].append(image)

            # Choose distinct originals, then one file from each
            chosen = random.sample(list(groups), min(count, len(groups)))
            for original in chosen:
                image = random.choice(groups[original])
                shutil.copy2(image, destination / f"{class_folder.name}__{image.name}")

            print(f"  {class_folder.name}: {len(chosen)} images")

    print(f"\nImages ready for labelling in {OUTPUT}")


if __name__ == "__main__":
    main()