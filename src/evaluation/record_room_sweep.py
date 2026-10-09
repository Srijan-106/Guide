"""
Room sweep: a second evaluation of object detection, asking a different
question from the Day 9 tests.

    Day 9 asked:  "of these 10 chosen objects, how many does it name?"
                  Three of the ten were known weak cases, chosen on purpose.

    This asks:    "of the objects actually in a room, how many does it name?"

Both are honest; they measure different things, and the two numbers are
reported side by side rather than one replacing the other. Nothing here
touches evaluation/results.csv - this writes its own file.

WHY THE LIST IS WRITTEN FIRST
The list of objects is typed in BEFORE the application is started, in step 1,
and cannot be added to afterwards. Choosing objects while testing - even
without meaning to - means choosing the ones that work, and the number then
measures the chooser rather than the detector.

    python src\\evaluation\\record_room_sweep.py

Step 1 is done once and saved. Running the script again goes straight to
step 2 and continues where it stopped, so a sweep can be done in several
sittings.
"""

import csv
from datetime import datetime
from pathlib import Path

PLAN = Path("evaluation/room_sweep_plan.csv")
RESULTS = Path("evaluation/room_sweep.csv")

# What can be recorded for each object. "wrong name" and "nothing" are kept
# apart because they are very different for a blind user: a wrong name is
# misleading, silence is only unhelpful.
VERDICTS = {
    "1": "correct",       # named it, or gave a fair synonym ("canned" for a can)
    "2": "wrong name",    # named something else
    "3": "nothing",       # said nothing about it
}


# ---------------------------------------------------------------------
# Step 1 - write the list, before the application is started
# ---------------------------------------------------------------------

def write_plan():
    """Type the objects in, in the order they will be shown."""
    print("\n" + "=" * 62)
    print("STEP 1 - LIST THE OBJECTS")
    print("=" * 62)
    print("""
Before starting the application, walk through the room and type the objects
you see, in the order you see them, one per line.

  - Write down EVERY object you notice, including ones you expect to fail.
    Skipping those is what makes a number meaningless.
  - Use the everyday name ("mug", "slipper"), not the name you think the
    model uses.
  - Aim for 25-30. Press Enter on an empty line when finished.
""")

    objects = []
    while True:
        name = input(f"  {len(objects) + 1:2}. ").strip()
        if not name:
            break
        objects.append(name)

    if len(objects) < 5:
        print("\nToo few to measure anything. Nothing was saved.")
        return None

    room = input("\n  Which room is this? ").strip() or "unnamed"
    light = input("  Lighting (bright / normal / dim): ").strip() or "normal"

    PLAN.parent.mkdir(parents=True, exist_ok=True)
    with open(PLAN, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["order", "object", "room", "lighting"])
        for number, name in enumerate(objects, 1):
            writer.writerow([number, name, room, light])

    print(f"\n  {len(objects)} objects saved to {PLAN}.")
    print("  This list is now fixed. Start the application and run this "
          "script again.")
    return objects


def read_plan():
    if not PLAN.exists():
        return []
    with open(PLAN, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------
# Step 2 - show each object and record what was said
# ---------------------------------------------------------------------

def already_done():
    if not RESULTS.exists():
        return set()
    with open(RESULTS, newline="", encoding="utf-8") as f:
        return {row["object"] for row in csv.DictReader(f)}


def record(plan):
    done = already_done()
    remaining = [row for row in plan if row["object"] not in done]

    if not remaining:
        print(f"\nAll {len(plan)} objects already recorded in {RESULTS}")
        return

    print("\n" + "=" * 62)
    print("STEP 2 - SHOW EACH OBJECT")
    print("=" * 62)
    print(f"""
{len(plan)} objects in the list, {len(remaining)} left.

For each one:
  - Cover the camera for a few seconds first, so the object counts as newly
    appeared. Without this, something already in view is never announced.
  - Hold it in front of the camera, centred.
  - Listen, and read the ANNOUNCING line in the application's terminal.
  - Type what the application said, then the verdict.

Type 's' to skip, 'q' to stop and continue later.
""")

    new_file = not RESULTS.exists() or RESULTS.stat().st_size == 0
    with open(RESULTS, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["timestamp", "order", "object", "room",
                             "lighting", "said", "verdict", "notes"])

        for number, row in enumerate(remaining, 1):
            print("-" * 62)
            print(f"Object {number} of {len(remaining)}:  {row['object']}"
                  f"   ({row['lighting']} light)")

            said = input("  What did it say? (enter if nothing): ").strip()
            options = "   ".join(f"{k}={v}" for k, v in VERDICTS.items())
            answer = input(f"  Verdict ({options}, s=skip, q=quit): "
                           ).strip().lower()

            if answer == "q":
                print(f"\nStopped. {len(remaining) - number + 1} left. "
                      f"Run this again to continue.")
                return
            if answer == "s" or answer not in VERDICTS:
                print("  skipped")
                continue

            notes = input("  Notes (enter to skip): ").strip()

            writer.writerow([datetime.now().isoformat(timespec="seconds"),
                             row["order"], row["object"], row["room"],
                             row["lighting"], said, VERDICTS[answer], notes])
            f.flush()                      # written immediately, never lost
            print(f"  recorded: {VERDICTS[answer]}")

    print(f"\nFinished. Results in {RESULTS}")
    print("Now run:  python src\\evaluation\\analyse_room_sweep.py")


def main():
    plan = read_plan()
    if not plan:
        if write_plan() is None:
            return
        return                             # list written; app not running yet
    record(plan)


if __name__ == "__main__":
    main()
