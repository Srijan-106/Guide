"""
Build a YOLO detection dataset from the hand-labelled seed set.

Boxes were drawn in makesense.ai with one generic label, "note" (class 0).
Each image's filename begins with its denomination ("Rs.500__..."), so
this script replaces class 0 with the correct denomination class.

Images keep the train/val split chosen by select_for_labelling.py, which
was itself drawn from the leak-free Day 7 split.

Output, in the folder layout Ultralytics expects for detection:
  data/currency_det/images/train    data/currency_det/labels/train
  data/currency_det/images/val      data/currency_det/labels/val
  data/currency_det/data.yaml
"""

import shutil
from pathlib import Path

IMAGES = Path(r"D:\GUIDE\data\labelling")
LABELS = Path(r"D:\currenc test\detection lable data")
OUTPUT = Path(r"D:\GUIDE\data\currency_det")

# Class order for the detector. Its position in this list becomes the
# class number written into each label file.
CLASSES = ["Rs.10", "Rs.20", "Rs.50", "Rs.100", "Rs.200", "Rs.500"]


def denomination(image_path):
    """'Rs.500__500_original_12.jpg_...jpg' -> 'Rs.500'"""
    return image_path.name.split("__")[0]


def valid_box(parts):
    """A YOLO box line has 5 numbers; the last 4 must lie between 0 and 1."""
    if len(parts) != 5:
        return False
    try:
        return all(0.0 <= float(v) <= 1.0 for v in parts[1:])
    except ValueError:
        return False


def main():
    if OUTPUT.exists():
        print(f"{OUTPUT} already exists. Delete it first to rebuild.")
        return

    label_stems = {p.stem for p in LABELS.glob("*.txt")}
    matched = set()
    warnings = []

    for split in ["train", "val"]:
        (OUTPUT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT / "labels" / split).mkdir(parents=True, exist_ok=True)

        labelled = 0
        unlabelled = 0

        for image in sorted((IMAGES / split).glob("*.jpg")):
            label_file = LABELS / f"{image.stem}.txt"

            if not label_file.exists():
                unlabelled += 1
                continue

            matched.add(image.stem)
            lines = [line.split() for line in label_file.read_text().splitlines()
                     if line.strip()]

            if not lines:
                warnings.append(f"{image.name}: empty label file - skipped")
                unlabelled += 1
                continue

            if len(lines) > 1:
                warnings.append(f"{image.name}: {len(lines)} boxes (expected 1)")

            if not all(valid_box(parts) for parts in lines):
                warnings.append(f"{image.name}: malformed box - skipped")
                continue

            # Replace the generic class 0 with this image's denomination
            class_index = CLASSES.index(denomination(image))
            converted = [" ".join([str(class_index)] + parts[1:]) for parts in lines]

            (OUTPUT / "labels" / split / f"{image.stem}.txt").write_text(
                "\n".join(converted) + "\n"
            )
            shutil.copy2(image, OUTPUT / "images" / split / image.name)
            labelled += 1

        print(f"{split}: {labelled} labelled, {unlabelled} without a label")

    # Label files that matched no image would indicate a naming problem
    orphans = label_stems - matched
    if orphans:
        print(f"\n{len(orphans)} label files matched no image, e.g.:")
        for name in sorted(orphans)[:5]:
            print(f"  {name}")

    if warnings:
        print("\nWarnings:")
        for w in warnings:
            print(f"  {w}")

    # The dataset description file Ultralytics reads during training
    names = "\n".join(f"  {i}: {name}" for i, name in enumerate(CLASSES))
    yaml_text = (
        f"path: {OUTPUT.as_posix()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"names:\n{names}\n"
    )
    (OUTPUT / "data.yaml").write_text(yaml_text)
    print(f"\nDataset written to {OUTPUT}")


if __name__ == "__main__":
    main()