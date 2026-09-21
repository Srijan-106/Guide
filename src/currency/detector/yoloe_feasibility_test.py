"""
Feasibility test: can YOLOE-26 draw boxes around banknotes automatically?

Detection training needs a box around every note. The currency dataset
has none - only folder labels. If an open-vocabulary detector, prompted
with words, can reliably find the note in each image, those boxes can be
combined with the folder labels to build a detection dataset without
drawing thousands of boxes by hand. This is called pseudo-labelling.

Version 2: the first run found zero boxes with the single prompt
"banknote". This version tries several wordings, lowers the confidence
cut-off, and adds "hand" as a CONTROL - hands appear in most of these
photos and are a common category, so if YOLOE finds hands but not notes,
the model works and the problem is the wording.
"""

import random
from pathlib import Path

import cv2
from ultralytics import YOLOE

# ---------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------

MODEL_PATH = "models/yoloe-26s-seg.pt"

# Several wordings, since zero-shot models respond strongly to phrasing
NOTE_PROMPTS = ["banknote", "money", "paper money", "currency", "cash"]

# Control: a common object that appears in most of these photos
CONTROL_PROMPTS = ["hand"]

# Low cut-off, to reveal weak detections the default 0.25 would hide
CONFIDENCE = 0.05

DATA = Path(r"D:\GUIDE\data\currency\train")
OUTPUT = Path(r"D:\GUIDE\experiments\autolabel_test")

IMAGES_PER_CLASS = 4

random.seed(42)


def fmt(detections):
    """Format a list of (name, confidence) pairs for printing."""
    if not detections:
        return "-"
    return ", ".join(f"{name} {conf:.2f}" for name, conf in detections)


def main():
    model = YOLOE(MODEL_PATH)
    model.set_classes(NOTE_PROMPTS + CONTROL_PROMPTS)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    images_with_note = 0
    images_with_hand = 0
    total = 0

    for class_folder in sorted(DATA.iterdir()):
        if not class_folder.is_dir():
            continue

        images = list(class_folder.glob("*.jpg"))
        sample = random.sample(images, min(IMAGES_PER_CLASS, len(images)))

        for image in sample:
            # agnostic_nms=False stops an overlapping hand box from
            # suppressing the note box, or vice versa
            result = model.predict(str(image), conf=CONFIDENCE,
                                   agnostic_nms=False, verbose=False)[0]

            found = [(result.names[int(c)], float(s))
                     for c, s in zip(result.boxes.cls, result.boxes.conf)]
            notes = [f for f in found if f[0] in NOTE_PROMPTS]
            hands = [f for f in found if f[0] in CONTROL_PROMPTS]

            total += 1
            images_with_note += bool(notes)
            images_with_hand += bool(hands)

            print(f"  {class_folder.name:<8} notes: {fmt(notes):<40} hands: {fmt(hands)}")

            out_path = OUTPUT / f"{class_folder.name}_{image.stem[:40]}.jpg"
            cv2.imwrite(str(out_path), result.plot())

    print(f"\n{total} images tested:")
    print(f"  With at least one note box : {images_with_note}")
    print(f"  With at least one hand box : {images_with_hand}  (control)")
    print(f"\nAnnotated images saved to {OUTPUT}")


if __name__ == "__main__":
    main()