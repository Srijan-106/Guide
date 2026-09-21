"""
Evaluate the trained currency classifier on an EXTERNAL dataset - images
from a different source than the training data.

The v2 validation accuracy (99.3%) was measured on images from the same
dataset used for training: same camera, same photographers, similar
backgrounds. That shows the model handles unseen photos from that source,
but not whether it generalises to notes photographed elsewhere. This
script tests exactly that.

It saves two files:
  - external_misclassified.csv : only the wrong predictions, for inspection
  - external_predictions.csv   : every prediction with its confidence, used
                                 to choose a confidence threshold

The external dataset is for testing only. It must never be used for
training, or it can no longer measure generalisation.
"""

import csv
from collections import defaultdict
from pathlib import Path

from ultralytics import YOLO

# ---------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------

MODEL_PATH = r"D:\GUIDE\experiments\currency_v2\weights\best.pt"
EXTERNAL_DATA = Path(r"D:\currenc test\Datasets")
MISCLASSIFIED_CSV = Path(r"D:\GUIDE\experiments\currency_v2\external_misclassified.csv")
ALL_PREDICTIONS_CSV = Path(r"D:\GUIDE\experiments\currency_v2\external_predictions.csv")


# The external folders are named "10", "100"... while the model's classes
# are named "Rs.10", "Rs.100"... These helpers translate between them.
def to_class_name(folder_name):
    return f"Rs.{folder_name}"


def value(class_name):
    return int(class_name.replace("Rs.", ""))


# ---------------------------------------------------------------------

def main():
    model = YOLO(MODEL_PATH)
    print(f"Model classes: {list(model.names.values())}\n")

    totals = defaultdict(int)
    correct = defaultdict(int)
    confusion = defaultdict(int)      # (true, predicted) -> count
    misclassified = []
    all_predictions = []

    folders = [f for f in EXTERNAL_DATA.iterdir() if f.is_dir()]
    folders.sort(key=lambda f: int(f.name))

    for folder in folders:
        true_class = to_class_name(folder.name)
        print(f"Testing {true_class}...")

        # stream=True processes one image at a time rather than loading
        # the whole folder into memory at once
        for result in model.predict(source=str(folder), imgsz=224, device=0,
                                    stream=True, verbose=False):
            predicted = model.names[result.probs.top1]
            confidence = float(result.probs.top1conf)

            # Keep every prediction, right or wrong, for threshold analysis
            all_predictions.append(
                (result.path, true_class, predicted, f"{confidence:.4f}")
            )

            totals[true_class] += 1
            confusion[(true_class, predicted)] += 1

            if predicted == true_class:
                correct[true_class] += 1
            else:
                misclassified.append(
                    (result.path, true_class, predicted, f"{confidence:.3f}")
                )

    classes = sorted(totals, key=value)

    # --- Overall accuracy ---
    total_all = sum(totals.values())
    correct_all = sum(correct.values())
    print(f"\nOverall accuracy: {correct_all}/{total_all} = "
          f"{correct_all / total_all * 100:.2f}%\n")

    # --- Per-class recall ---
    print("Per-class recall:")
    for c in classes:
        print(f"  {c:<8} {correct[c]:>4}/{totals[c]:<4} = "
              f"{correct[c] / totals[c] * 100:6.2f}%")

    # --- Confusion matrix ---
    # Rows are the TRUE note, columns the PREDICTION. Note this is the
    # opposite orientation to the Ultralytics chart, which puts
    # predictions on the rows.
    print("\nConfusion matrix (rows = true, columns = predicted):")
    print("".ljust(10) + "".join(c.rjust(8) for c in classes))
    for t in classes:
        row = "".join(str(confusion[(t, p)]).rjust(8) for p in classes)
        print(t.ljust(10) + row)

    # --- Overestimates: the safety-relevant error ---
    # Predicting a HIGHER value than the true note is the costly mistake
    # for a blind user - e.g. being handed Rs.100 in change and told it
    # is Rs.500.
    over = [m for m in misclassified if value(m[2]) > value(m[1])]
    into_500 = [m for m in over if m[2] == "Rs.500"]
    print(f"\nErrors that OVERESTIMATED the note's value: {len(over)} "
          f"(of which {len(into_500)} predicted Rs.500)")

    # --- Save misclassified images for inspection ---
    with open(MISCLASSIFIED_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "true", "predicted", "confidence"])
        writer.writerows(misclassified)

    print(f"\n{len(misclassified)} misclassified images listed in:")
    print(f"  {MISCLASSIFIED_CSV}")

    # --- Save every prediction, for choosing a confidence threshold ---
    with open(ALL_PREDICTIONS_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "true", "predicted", "confidence"])
        writer.writerows(all_predictions)

    print(f"All {len(all_predictions)} predictions saved to:")
    print(f"  {ALL_PREDICTIONS_CSV}")


if __name__ == "__main__":
    main()