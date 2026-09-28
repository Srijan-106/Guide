"""
Record evaluation results for VisionAssist, one test at a time.

The application needs the camera, so this runs as a separate program in
a second terminal: the app in one window, this in the other. It shows
each test in a fixed order, you enter a verdict, and everything is
written to a CSV as you go - so results are recorded rather than
remembered.

The test list is fixed before testing begins. Changing it after seeing
results - swapping in easier objects, dropping awkward texts - is how
an evaluation stops meaning anything.

    python src\\evaluation\\record_results.py

Progress is saved after every answer, so it can be stopped and resumed.
"""

import csv
from datetime import datetime
from pathlib import Path

OUTPUT = Path("evaluation/results.csv")

OBJECTS = [
    "bottle", "cup", "book", "phone", "chair",
    "cream tube", "spray bottle", "pen", "headphones", "keyboard",
]

TEXTS = [
    "BOLDFIT bottle (large, metallic)",
    "Book cover (large, multi-line)",
    "Cream tube label (small print)",
    "Product back label (small, dense)",
    "Handwritten note - neat",
    "Handwritten note - normal",
    "Laptop or phone screen (backlit)",
    "Drink can (curved, shiny)",
]

LIGHTING = ["bright", "normal", "dim"]

# Verdicts offered for each kind of test
SCALES = {
    "detection": {
        "1": "correct",       # announced the right object
        "2": "wrong",         # announced something else
        "3": "missed",        # said nothing about it
    },
    "identify": {
        "1": "correct",       # named the object correctly
        "2": "wrong",         # named the wrong object
        "3": "no label",      # said nothing about the object (text only)
        "4": "hedged right",  # "this might be X" and X was right
    },
    "reading": {
        "1": "exact",         # every word correct
        "2": "partial",       # some words right, some wrong or missing
        "3": "wrong",         # read something unrelated
        "4": "nothing",       # "no readable text found"
    },
}


def build_test_plan():
    """The fixed list of tests, in the order they will be run."""
    tests = []

    # Detection: each object under three lighting conditions, held close
    for light in LIGHTING:
        for name in OBJECTS:
            tests.append(("detection", name, light, "close"))

    # Identification: each object once, in normal light
    for name in OBJECTS:
        tests.append(("identify", name, "normal", "close"))

    # Reading: each text in normal light, then dim
    for light in ["normal", "dim"]:
        for name in TEXTS:
            tests.append(("reading", name, light, "close"))

    return tests


def already_done():
    """Which tests are already recorded, so a session can be resumed."""
    if not OUTPUT.exists():
        return set()
    with open(OUTPUT, newline="", encoding="utf-8") as f:
        return {(row["kind"], row["item"], row["lighting"])
                for row in csv.DictReader(f)}


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    tests = build_test_plan()
    done = already_done()
    remaining = [t for t in tests
                 if (t[0], t[1], t[2]) not in done]

    if not remaining:
        print(f"All {len(tests)} tests already recorded in {OUTPUT}")
        return

    print(f"\n{len(tests)} tests in total, {len(remaining)} remaining.")
    print("Run the application in another terminal, then work through these.")
    print("Type 's' to skip a test, or 'q' to stop and continue later.\n")

    new_file = not OUTPUT.exists() or OUTPUT.stat().st_size == 0
    with open(OUTPUT, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["timestamp", "kind", "item", "lighting",
                             "distance", "verdict", "heard", "notes"])

        for number, (kind, item, light, distance) in enumerate(remaining, 1):
            scale = SCALES[kind]

            print("-" * 62)
            print(f"Test {number} of {len(remaining)}   [{kind}]")
            print(f"  Item:      {item}")
            print(f"  Lighting:  {light}")
            print(f"  Action:    ", end="")
            if kind == "detection":
                print("hold it in view and wait for the ambient announcement")
            elif kind == "identify":
                print("hold it centred, press I, listen to the object name")
            else:
                print("point at the text, press SPACE, listen to the reading")

            options = "   ".join(f"{key}={value}" for key, value in scale.items())
            answer = input(f"  Verdict ({options}, s=skip, q=quit): ").strip().lower()

            if answer == "q":
                print(f"\nStopped. {len(remaining) - number + 1} tests left.")
                print(f"Run this again to continue. Saved in {OUTPUT}")
                return
            if answer == "s" or answer not in scale:
                print("  skipped")
                continue

            heard = input("  What did it actually say? (enter to skip): ").strip()
            notes = input("  Notes (enter to skip): ").strip()

            writer.writerow([datetime.now().isoformat(timespec="seconds"),
                             kind, item, light, distance,
                             scale[answer], heard, notes])
            f.flush()          # write immediately, so nothing is lost
            print(f"  recorded: {scale[answer]}")

    print(f"\nFinished. Results in {OUTPUT}")


if __name__ == "__main__":
    main()