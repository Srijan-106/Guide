"""
Train a 6-class currency DETECTOR on the hand-labelled seed set.

Unlike the Day 7 classifier, which judges the whole image, a detector
locates the note and draws a box around it - so it should be less affected
by framing and background, such as the orange room that pulled
classifier guesses towards Rs.200.

This first detector is trained on only 144 hand-labelled images. Its main
job is to act as a SEED model: if it can locate notes reliably, it can
auto-label the remaining ~7,000 training images, which are then checked
and used to train a stronger detector.
"""

from ultralytics import YOLO

# ---------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------

# Detection weights pretrained on COCO. Starting from these rather than
# from scratch matters most when the dataset is small, as it is here.
BASE_MODEL = "models/yolo26n.pt"

DATA = r"D:\GUIDE\data\currency_det\data.yaml"

# Small datasets need more passes; early stopping ends training if
# validation stops improving for PATIENCE epochs.
EPOCHS = 100
PATIENCE = 20

# Detection standard. Classification used 224; detection needs more
# resolution to locate object edges accurately.
IMAGE_SIZE = 640

BATCH_SIZE = 16
WORKERS = 4

PROJECT = r"D:\GUIDE\experiments"
RUN_NAME = "currency_det_v1"


def main():
    model = YOLO(BASE_MODEL)

    model.train(
        data=DATA,
        epochs=EPOCHS,
        patience=PATIENCE,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        workers=WORKERS,
        project=PROJECT,
        name=RUN_NAME,
        device=0,
        # Day 7 showed training was held back by the CPU decoding large
        # JPEGs every epoch. With only 180 images, the resized images fit
        # easily in memory, so they are decoded once and reused.
        cache="ram",
        verbose=True,
    )


if __name__ == "__main__":
    main()