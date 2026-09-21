"""
Fine-tune a YOLO26 classification model on the Indian currency dataset.

This trains a model to identify which denomination a note is, given an
image of it. Classification rather than detection was chosen because the
source dataset has folder labels rather than bounding boxes, and because
it matches the actual use case: a user holds a note up to the camera and
asks what it is, rather than scanning a room for money.
"""

from ultralytics import YOLO

# ---------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------

# Nano classification model, pretrained on ImageNet. Fine-tuning from
# pretrained weights converges far faster than training from scratch.
#
# Nano is sufficient here: six visually distinct classes (Indian notes
# differ strongly in colour) is a much easier task than 80-class
# detection. A larger model would use more memory for little gain.
BASE_MODEL = "models/yolo26n-cls.pt"

DATA_PATH = r"D:\GUIDE\data\currency"

# One epoch is one complete pass through the training data.
EPOCHS = 30

# Images resized to this before training. 224 is the classification
# standard - detection uses 640, which needs far more memory.
IMAGE_SIZE = 224

# How many images are processed at once. Training needs much more memory
# than inference, since it stores gradients and activations for every
# layer. Batch 8 is conservative for an 8 GB card that is also running
# a desktop environment.
BATCH_SIZE = 32

# Loading images uses CPU and RAM. Fewer workers means less system load,
# at some cost to training speed.
WORKERS = 4

PROJECT = r"D:\GUIDE\experiments"
RUN_NAME = "currency_v2"


def main():
    model = YOLO(BASE_MODEL)

    model.train(
        data=DATA_PATH,
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        workers=WORKERS,
        project=PROJECT,
        name=RUN_NAME,
        device=0,          # use the GPU
        patience=10,       # stop early if no improvement for 10 epochs
        cache=False,       # do not hold the dataset in RAM
        verbose=True,
    )


if __name__ == "__main__":
    main()