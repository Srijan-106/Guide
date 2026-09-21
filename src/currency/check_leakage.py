"""
Check whether augmented copies of the same original photo ended up in
both the training and validation sets.

Filenames look like: 10_original_1_jpg_<uuid>.jpg
Everything before "_jpg_" identifies the original photo.
"""

from pathlib import Path

DATA = Path(r"D:\GUIDE\data\currency")


def original_ids(split):
    """Collect the original-photo identifier for every image in a split."""
    ids = set()
    for image in (DATA / split).rglob("*.jpg"):
        ids.add(image.name.split(".jpg_")[0])
    return ids


train_ids = original_ids("train")
val_ids = original_ids("val")
shared = train_ids & val_ids

print(f"Original photos in train: {len(train_ids)}")
print(f"Original photos in val:   {len(val_ids)}")
print(f"Appearing in BOTH:        {len(shared)}")
print(f"Leakage: {len(shared) / len(val_ids) * 100:.1f}% of val originals also in train")