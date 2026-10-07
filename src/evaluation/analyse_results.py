"""
Turn the recorded evaluation results into numbers.

Reports accuracy for each feature, then breaks it down by lighting and by
item, so the weak spots are visible rather than hidden in an average.
"""

import csv
from collections import Counter, defaultdict
from pathlib import Path

RESULTS = Path("evaluation/results.csv")

# Which verdicts count as a success for each kind of test.
# "hedged right" counts: the app said "this might be X" and X was right -
# honest uncertainty about a correct answer is not a failure.
SUCCESS = {
    "detection": {"correct"},
    "identify": {"correct", "hedged right"},
    "reading": {"exact", "partial"},

    # Day 12 tests. A voice command succeeds only if the right action ran.
    "voice_en": {"correct"},
    "voice_hi": {"correct"},

    # A visual answer succeeds only if it is fully correct. "partly
    # correct" is NOT counted: unlike a partial reading, a half-right
    # answer ("bottle in your left hand" when it is in the right) can
    # mislead someone who cannot check it. It still shows in the verdicts.
    "question_en": {"correct"},
    "question_hi": {"correct"},

    # A small slip that leaves the meaning clear still does the job
    "translation": {"meaning kept", "small error"},
}

# The order the features are reported in
KINDS = ("detection", "identify", "reading",
         "voice_en", "voice_hi", "question_en", "question_hi", "translation")


def percent(part, whole):
    return f"{part / whole * 100:5.1f}%" if whole else "    -"


def main():
    if not RESULTS.exists():
        print(f"No results file at {RESULTS}")
        return

    with open(RESULTS, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    print(f"{len(rows)} tests recorded\n")

    for kind in KINDS:
        tests = [r for r in rows if r["kind"] == kind]
        if not tests:
            continue

        good = sum(r["verdict"] in SUCCESS[kind] for r in tests)

        print("=" * 64)
        print(f"{kind.upper()}   {good}/{len(tests)} = {percent(good, len(tests))}")
        print("=" * 64)

        # Every verdict, so failures are visible rather than averaged away
        print("\n  Verdicts:")
        for verdict, count in Counter(r["verdict"] for r in tests).most_common():
            print(f"    {verdict:<18} {count:>3}   {percent(count, len(tests))}")

        # By lighting - does it hold up in dim light?
        by_light = defaultdict(list)
        for r in tests:
            by_light[r["lighting"]].append(r)

        if len(by_light) > 1:
            print("\n  By lighting:")
            for light in ("bright", "normal", "dim"):
                group = by_light.get(light, [])
                if group:
                    ok = sum(r["verdict"] in SUCCESS[kind] for r in group)
                    print(f"    {light:<8} {ok:>2}/{len(group):<3} {percent(ok, len(group))}")

        # By item, worst first - this is where the useful detail is
        by_item = defaultdict(list)
        for r in tests:
            by_item[r["item"]].append(r)

        print("\n  By item (worst first):")
        scored = []
        for item, group in by_item.items():
            ok = sum(r["verdict"] in SUCCESS[kind] for r in group)
            scored.append((ok / len(group), item, ok, len(group)))

        for rate, item, ok, total in sorted(scored):
            failures = [r["verdict"] for r in by_item[item]
                        if r["verdict"] not in SUCCESS[kind]]
            note = f"   ({', '.join(failures)})" if failures else ""
            print(f"    {item:<36} {ok}/{total} {percent(ok, total)}{note}")

        print()

    # Trap questions ask about something that is NOT there. They are the
    # direct test of whether the vision model invents things - the most
    # dangerous failure for a user who cannot check the answer.
    traps = [r for r in rows
             if r["kind"].startswith("question_") and "TRAP" in r["item"]]
    if traps:
        passed = sum(r["verdict"] == "correct" for r in traps)
        made_up = sum(r["verdict"] == "made up" for r in traps)
        print("=" * 64)
        print(f"TRAP QUESTIONS   {passed}/{len(traps)} passed, "
              f"{made_up} made up")
        print("=" * 64)
        for r in traps:
            print(f"    {r['verdict']:<14} {r['item']}")
        print()

    # What the app said when it got things wrong - the most useful detail.
    # .get() so a kind added later without a SUCCESS entry cannot crash this.
    wrong = [r for r in rows
             if r["verdict"] not in SUCCESS.get(r["kind"], set())
             and r.get("heard")]
    if wrong:
        print("=" * 64)
        print("WHAT IT SAID WHEN WRONG")
        print("=" * 64)
        for r in wrong:
            print(f"  {r['item']:<32} ({r['lighting']:<6}) -> {r['heard']}")


if __name__ == "__main__":
    main()