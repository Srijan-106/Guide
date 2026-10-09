"""
Turn the room sweep into numbers, and put them beside the Day 9 results.

Reports:
  1. How many of the objects actually in the room the application named
  2. The split that matters: a wrong name misleads, silence only unhelps
  3. Every object it could not name, and what it said instead
  4. The Day 9 number next to it, so neither replaces the other

    python src\\evaluation\\analyse_room_sweep.py
"""

import csv
from collections import Counter, defaultdict
from pathlib import Path

SWEEP = Path("evaluation/room_sweep.csv")
DAY9 = Path("evaluation/results.csv")

SUCCESS = {"correct"}


def percent(part, whole):
    return f"{part / whole * 100:.1f}%" if whole else "-"


def day9_detection():
    """The Day 9 detection result, for comparison. None if unavailable."""
    if not DAY9.exists():
        return None
    with open(DAY9, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["kind"] == "detection"]
    if not rows:
        return None
    good = sum(r["verdict"] == "correct" for r in rows)
    return good, len(rows)


def main():
    if not SWEEP.exists():
        print(f"No results yet at {SWEEP}. Run record_room_sweep.py first.")
        return

    with open(SWEEP, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        print("No results recorded yet.")
        return

    good = sum(r["verdict"] in SUCCESS for r in rows)
    rooms = sorted({r["room"] for r in rows})

    print("=" * 64)
    print(f"ROOM SWEEP   {good}/{len(rows)} = {percent(good, len(rows))}")
    print("=" * 64)
    print(f"  Rooms: {', '.join(rooms)}")

    print("\n  Verdicts:")
    for verdict, count in Counter(r["verdict"] for r in rows).most_common():
        print(f"    {verdict:<12} {count:>3}   {percent(count, len(rows))}")

    # The kind of failure matters more than the rate: for someone who cannot
    # check, a wrong name is misleading, while silence is merely unhelpful.
    wrong = sum(r["verdict"] == "wrong name" for r in rows)
    nothing = sum(r["verdict"] == "nothing" for r in rows)
    print(f"\n  Of {len(rows) - good} failures: {wrong} wrong name, "
          f"{nothing} silent.")

    by_room = defaultdict(list)
    for r in rows:
        by_room[r["room"]].append(r)
    if len(by_room) > 1:
        print("\n  By room:")
        for room, group in sorted(by_room.items()):
            ok = sum(r["verdict"] in SUCCESS for r in group)
            print(f"    {room:<16} {ok:>2}/{len(group):<3} "
                  f"{percent(ok, len(group))}")

    named = [r for r in rows if r["verdict"] in SUCCESS]
    if named:
        print("\n  Named correctly:")
        print("    " + ", ".join(sorted(r["object"] for r in named)))

    missed = [r for r in rows if r["verdict"] not in SUCCESS]
    if missed:
        print("\n  Not named, and what it said instead:")
        for r in sorted(missed, key=lambda r: r["verdict"]):
            said = r["said"] or "(nothing)"
            print(f"    {r['object']:<22} {r['verdict']:<12} -> {said}")

    day9 = day9_detection()
    print("\n" + "=" * 64)
    print("THE TWO MEASUREMENTS")
    print("=" * 64)
    if day9:
        print(f"  Chosen list (Day 9)   {day9[0]}/{day9[1]} = "
              f"{percent(day9[0], day9[1])}   ten objects fixed in advance,")
        print("                        three of them known weak cases")
    print(f"  Room sweep            {good}/{len(rows)} = "
          f"{percent(good, len(rows))}   the objects actually in a room,")
    print("                        listed before the application was started")
    print("\n  Different questions, not a before and after. Report both.")


if __name__ == "__main__":
    main()
