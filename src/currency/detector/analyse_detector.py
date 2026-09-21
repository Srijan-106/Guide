"""
Text-based error analysis for the currency detector.

1. Prints the confusion matrix as numbers rather than an image.
2. Checks every validation image against its hand-drawn box and lists
   each failure: a wrong denomination, a missed note, or an extra
   detection where no note exists.
"""

from pathlib import Path

from ultralytics import YOLO

MODEL = r"D:\GUIDE\experiments\currency_det_v1\weights\best.pt"
DATA = r"D:\GUIDE\data\currency_det\data.yaml"
VAL_IMAGES = Path(r"D:\GUIDE\data\currency_det\images\val")
VAL_LABELS = Path(r"D:\GUIDE\data\currency_det\labels\val")

# Same cut-off the confusion-matrix image uses
CONFIDENCE = 0.25

# A prediction counts as "the same note" if it overlaps the hand-drawn
# box by at least this much (intersection over union)
MATCH_IOU = 0.5


def to_corners(xc, yc, w, h):
    """YOLO centre/width/height (fractions) -> corner coordinates."""
    return (xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2)


def iou(a, b):
    """Intersection over union: 0 = no overlap, 1 = identical boxes."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = ((a[2] - a[0]) * (a[3] - a[1]) +
             (b[2] - b[0]) * (b[3] - b[1]) - inter)
    return inter / union if union else 0.0


def print_confusion_matrix(model):
    metrics = model.val(data=DATA, plots=False, verbose=False)
    matrix = metrics.confusion_matrix.matrix
    names = list(model.names.values()) + ["background"]

    # Ultralytics stores ROWS as predictions and COLUMNS as the truth.
    # "background" as a column = a detection where no note existed.
    # "background" as a row    = a real note the model missed.
    print("\nConfusion matrix (rows = PREDICTED, columns = TRUE)")
    print(" " * 12 + "".join(n.rjust(11) for n in names))
    for i, row_name in enumerate(names):
        cells = "".join(str(int(v)).rjust(11) for v in matrix[i])
        print(row_name.ljust(12) + cells)


def check_each_image(model):
    names = model.names
    counts = {"correct": 0, "wrong": 0, "missed": 0, "extra": 0}

    print(f"\nFailures on validation images (confidence >= {CONFIDENCE}):")

    for image in sorted(VAL_IMAGES.glob("*.jpg")):
        label = VAL_LABELS / f"{image.stem}.txt"
        truth = []
        for line in label.read_text().splitlines():
            if line.strip():
                parts = line.split()
                truth.append((int(parts[0]), to_corners(*map(float, parts[1:]))))

        result = model.predict(str(image), conf=CONFIDENCE, verbose=False)[0]
        preds = [(int(c), float(s), tuple(b))
                 for c, s, b in zip(result.boxes.cls.tolist(),
                                    result.boxes.conf.tolist(),
                                    result.boxes.xyxyn.tolist())]

        problems = []
        used = set()

        for true_class, true_box in truth:
            # The prediction overlapping this note the most
            best, best_iou = None, 0.0
            for index, (_, _, box) in enumerate(preds):
                overlap = iou(true_box, box)
                if overlap > best_iou:
                    best, best_iou = index, overlap

            if best is None or best_iou < MATCH_IOU:
                counts["missed"] += 1
                problems.append(f"MISSED {names[true_class]}")
                continue

            used.add(best)
            pred_class, pred_conf, _ = preds[best]
            if pred_class == true_class:
                counts["correct"] += 1
            else:
                counts["wrong"] += 1
                problems.append(f"WRONG: true {names[true_class]}, predicted "
                                f"{names[pred_class]} ({pred_conf:.2f})")

        # Any prediction not matching a real note is an extra detection
        for index, (pred_class, pred_conf, _) in enumerate(preds):
            if index not in used:
                counts["extra"] += 1
                problems.append(f"EXTRA {names[pred_class]} ({pred_conf:.2f})")

        if problems:
            print(f"  {image.name[:45]}")
            for p in problems:
                print(f"      {p}")

    print(f"\nSummary: {counts['correct']} correct, {counts['wrong']} wrong "
          f"denomination, {counts['missed']} missed, {counts['extra']} extra")


def main():
    model = YOLO(MODEL)
    print_confusion_matrix(model)
    check_each_image(model)


if __name__ == "__main__":
    main()