"""
Choose a confidence threshold for the currency classifier from data.

The classifier must always name one of six notes - it cannot say "I don't
know". A confidence threshold lets it abstain: below the threshold, the
app says "not sure" instead of naming a note.

This is a trade-off:
  - too low  -> wrong answers are still spoken
  - too high -> the app says "not sure" even when it would have been right

Caveat: the threshold is chosen using the external dataset, which is also
where results are reported. That makes the reported figures somewhat
optimistic. A cleaner method would choose the threshold on one part of the
data and report results on another.
"""

import csv
from pathlib import Path
from statistics import median

PREDICTIONS = Path(r"D:\GUIDE\experiments\currency_v2\external_predictions.csv")

# 0.0 means "never abstain" - the current behaviour, shown as a baseline
THRESHOLDS = [0.0, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98, 0.99]


def value(class_name):
    return int(class_name.replace("Rs.", ""))


def main():
    with open(PREDICTIONS, newline="") as f:
        rows = list(csv.DictReader(f))

    for r in rows:
        r["confidence"] = float(r["confidence"])
        r["correct"] = r["true"] == r["predicted"]

    total = len(rows)
    print(f"{total} predictions loaded\n")

    # If confidence is useful at all, wrong answers should score lower
    correct_conf = [r["confidence"] for r in rows if r["correct"]]
    wrong_conf = [r["confidence"] for r in rows if not r["correct"]]
    print(f"Median confidence when CORRECT: {median(correct_conf):.3f}")
    print(f"Median confidence when WRONG:   {median(wrong_conf):.3f}\n")

    # The five classes the model handles well. Abstaining on these is the
    # COST of a threshold - it says "not sure" when it would have been right.
    others = [r for r in rows if r["true"] != "Rs.20"]

    print(f"{'Threshold':>9}  {'Coverage':>8}  {'Accuracy':>8}  "
          f"{'Wrong':>5}  {'Over':>5}  {'Coverage excl. Rs.20':>20}")

    for t in THRESHOLDS:
        answered = [r for r in rows if r["confidence"] >= t]
        n = len(answered)
        right = sum(r["correct"] for r in answered)
        wrong = n - right
        over = sum(1 for r in answered
                   if not r["correct"] and value(r["predicted"]) > value(r["true"]))
        others_answered = sum(1 for r in others if r["confidence"] >= t)

        coverage = n / total * 100
        accuracy = right / n * 100 if n else 0.0
        others_cov = others_answered / len(others) * 100

        print(f"{t:>9.2f}  {coverage:>7.1f}%  {accuracy:>7.1f}%  "
              f"{wrong:>5}  {over:>5}  {others_cov:>19.1f}%")


if __name__ == "__main__":
    main()