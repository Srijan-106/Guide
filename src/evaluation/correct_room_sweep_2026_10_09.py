"""
Corrections to evaluation/room_sweep.csv, made on 9 October 2026.

Every recorded row was checked against the application's own log from the
same session. Where a verdict contradicted the log, it is corrected here -
in code, with the reason written into the row's notes - rather than edited
by hand, so every change is visible and explained. Same method as
correct_results_2026_10_06.py and correct_results_2026_10_07.py.

WHAT THE CHECK FOUND
Three objects were recorded as "nothing" or left unrecorded, but the log
shows the application announced all three. The tester was reading the
recorder's terminal rather than listening, and announcements are merged
into one sentence with other objects, so an individual name goes past
quickly. The detector named them; the person missed them. That is a
usability finding, not a detection failure, and it is kept as a finding -
see docs/devlog.md.

Safe to run more than once: changes already made are skipped.

    python src\\evaluation\\correct_room_sweep_2026_10_09.py
"""

import csv
import shutil
from datetime import datetime
from pathlib import Path

RESULTS = Path("evaluation/room_sweep.csv")
PLAN = Path("evaluation/room_sweep_plan.csv")
BACKUP = Path("evaluation/room_sweep_before_2026-10-09_corrections.csv")
TAG = "corrected 2026-10-09"

# object -> (field to change, new value, reason)
CORRECTIONS = {
    # The log line "ANNOUNCING: hanger" is followed by
    # "[speech] merged 2 announcements: hat, hanger" - it was detected,
    # named and spoken. It was not heard because it arrived inside a
    # merged sentence while the tester was reading the other terminal.
    "hanger": ("verdict", "correct",
               "log shows ANNOUNCING: hanger, spoken in a merged "
               "announcement; not heard at the time"),

    # "hat" is the Objects365 class for a cap, and the log announces it
    # throughout the session. The tester confirms cap and hat are the
    # same object here.
    "cap": ("verdict", "correct",
            "announced as 'hat', the vocabulary's name for a cap; "
            "log shows it announced repeatedly"),
}

# object -> extra fact, added to the notes without changing the verdict
NOTES = {
    # Kept as a failure although the name given was not wrong, only
    # unspecific. The rule applied: the name must identify the object.
    # A green bottle is a bottle, so "bottle" identifies it; a spray
    # bottle is a different kind of object, and "bottle" does not.
    "spray bottle": "said 'bottle' - true but not specific enough to "
                    "identify the object; 'green bottle' was counted "
                    "correct because a green bottle IS a bottle",

    # The log announces "toy" three times. One may have been this, but
    # the Hot Wheels cars were also announced as "toy", and the log has
    # no timestamps, so it cannot be told which was which. The harsher
    # verdict is kept where the evidence does not decide.
    "bike miniature": "log announces 'toy' 3 times but it cannot be told "
                      "which was this and which was the Hot Wheels cars; "
                      "kept as a failure because unverifiable",
}

# Objects skipped during the sweep, which the log shows were announced.
# The row is added with what the log recorded.
ADDITIONS = {
    "power socket": ("power outlet", "correct",
                     "skipped during the sweep; log shows "
                     "ANNOUNCING: power outlet"),
}


def plan_details(name):
    """order, room and lighting for an object, from the fixed list."""
    with open(PLAN, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["object"].strip().lower() == name.lower():
                return row["order"], row["room"], row["lighting"]
    return "", "", ""


def add_note(row, text):
    note = f"{TAG}: {text}"
    row["notes"] = f"{row['notes']} | {note}" if row["notes"] else note


def main():
    with open(RESULTS, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        rows = list(reader)

    # Keep an untouched copy, made only the first time this runs
    if not BACKUP.exists():
        shutil.copy(RESULTS, BACKUP)
        print(f"Backup saved: {BACKUP}")

    present = {row["object"].strip().lower() for row in rows}

    for row in rows:
        name = row["object"].strip().lower()

        if name in CORRECTIONS and TAG not in row["notes"]:
            field, new_value, reason = CORRECTIONS[name]
            old_value = row[field]
            row[field] = new_value
            add_note(row, f"{field} was '{old_value}' - {reason}")
            print(f"CORRECTED  {row['object']:<16} {field}: '{old_value}' "
                  f"-> '{new_value}'")

        if name in NOTES and TAG not in row["notes"]:
            add_note(row, NOTES[name])
            print(f"NOTE ADDED {row['object']:<16} {NOTES[name][:48]}...")

    for name, (said, verdict, reason) in ADDITIONS.items():
        if name.lower() in present:
            continue
        order, room, lighting = plan_details(name)
        row = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "order": order, "object": name, "room": room,
            "lighting": lighting, "said": said, "verdict": verdict,
            "notes": f"{TAG}: {reason}",
        }
        rows.append(row)
        print(f"ADDED      {name:<16} {verdict} - {reason}")

    # Back into the order the objects were shown in
    rows.sort(key=lambda r: int(r["order"]) if r["order"].isdigit() else 999)

    with open(RESULTS, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n{len(rows)} rows now in {RESULTS}")
    print("Run:  python src\\evaluation\\analyse_room_sweep.py")


if __name__ == "__main__":
    main()
