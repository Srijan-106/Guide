"""
Corrections to evaluation/results.csv, made on 6 October 2026.

Each recorded test was checked against the application's own log from
the same session. Where a verdict contradicted the log, or the tester's
own note on that row, it is corrected here - in code, with the reason
written into the row's notes - rather than edited by hand, so every
change is visible and explained.

Three kinds of change:

  CORRECTED  the test happened, but the verdict was entered wrongly.
             The verdict is fixed and the reason added to the notes.

  REMOVED    the test did not actually test what it claims, so it is
             deleted and will be run again with the recorder. Removing
             a test is only done when the log shows the test itself was
             invalid - never because the result was disappointing.

  NOTE       the verdict stands, but a fact from the app log that
             matters for reading it (such as the number of presses) is
             added to the notes.

Safe to run more than once: changes already made are skipped.

    python src\\evaluation\\correct_results_2026_10_06.py
"""

import csv
import shutil
from pathlib import Path

RESULTS = Path("evaluation/results.csv")
BACKUP = Path("evaluation/results_before_2026-10-06_corrections.csv")
TAG = "corrected 2026-10-06"

# timestamp of the row -> (field to change, new value, reason)
CORRECTIONS = {
    # Detection means the app SAYS the object's name. The tester's note
    # on this row records that it only read the label text ("Glyco 6").
    "2026-10-06T18:09:45": ("verdict", "missed",
                            "the app read the label text but never "
                            "named the object"),

    # The app log for this press of I shows "PROMINENT: none" - the
    # chair was judged too far from the centre, so no object was named.
    "2026-10-06T18:10:42": ("verdict", "no label",
                            "app log shows PROMINENT: none - "
                            "no object was named"),

    # The "heard" column held the label drawn on screen, but the note
    # says nothing was spoken - so nothing was heard.
    "2026-09-29T00:38:08": ("heard", "",
                            "'heard' held the on-screen label; "
                            "nothing was spoken"),

    # --- Third pass: the re-run at 18:53-18:58 ------------------------

    # Cream tube: the log shows the app said "glycol Acid cream 69 w" -
    # the spell checker turned "Glycolic" into "glycol" and "6%" came
    # out as "69". Some words right, some wrong = partial.
    "2026-10-06T18:55:33": ("verdict", "partial",
                            "app said 'glycol Acid cream 69 w': "
                            "Glycolic and 6% were wrong"),

    # Handwriting: every word read below 0.15 confidence, so nothing
    # was spoken (SPEAKING IN ORDER: []). "wrong" means it read out
    # something unrelated; saying nothing is "nothing".
    "2026-10-06T18:56:47": ("verdict", "nothing",
                            "log shows nothing was spoken"),
    "2026-10-06T18:56:50": ("verdict", "nothing",
                            "log shows nothing was spoken"),

    # Drink can: the can says "Pepsi Zero Sugar"; the app said
    # "PePSI, Sugar" and then "Pepsi" - "Zero" was never spoken.
    "2026-10-06T18:58:04": ("verdict", "partial",
                            "app said 'Pepsi' / 'PePSI, Sugar'; "
                            "'Zero' was never spoken"),
}

# timestamp of the row -> extra fact from the app log, added to notes
NOTES = {
    # Re-aiming and pressing again is allowed - a blind user re-aims
    # too - but the number of presses is recorded
    "2026-10-06T18:54:20": "read on the 4th press; presses 1-3 read "
                           "only '1', '1', '8'",

    # The chair was detected and announced at startup, but the speech
    # queue was busy and the announcement was dropped, so it was not
    # heard - a real miss from the user's point of view
    "2026-10-06T18:53:32": "app announced chair at startup but the "
                           "speech was dropped as too late (log)",

    # What was spoken, from the log. Two words were damaged by the
    # spell checker: "Hazrat" became "hazard", "gmail.com" "malcolm".
    "2026-10-06T19:05:15": "app said '06.55, Kota Jn, Passenger "
                           "Information, You can now book' - and the spell "
                           "checker changed Hazrat to hazard, gmail.com to "
                           "malcolm",
}

# timestamp of the row -> reason it is removed
REMOVALS = {
    # The chair was in view from the moment the app started, so it was
    # announced once at startup and never left the frame. These two
    # tests could not show whether it is announced when it comes into
    # view. They are re-run with the camera covered first.
    "2026-10-06T18:07:28": "chair never left the frame during the test",
    "2026-10-06T18:09:00": "chair never left the frame during the test",

    # The app log has no reading-mode switch and no BOLDFIT text in
    # this session, and the "heard" column says "cup" - the reading
    # was never actually performed.
    "2026-10-06T18:11:00": "no reading appears in the app log",

    # --- Second pass: the re-run at 18:33-18:37 -----------------------

    # The camera was not covered first: the app log has no
    # "FORGOTTEN: chair" before these tests, so the chair again never
    # left the frame. The three were also saved within 7 seconds, too
    # quickly to change the light between them.
    "2026-10-06T18:33:30": "chair never left the frame; 3 tests in 7 s",
    "2026-10-06T18:33:34": "chair never left the frame; 3 tests in 7 s",
    "2026-10-06T18:33:37": "chair never left the frame; 3 tests in 7 s",

    # Reading tests are defined as SPACE in reading mode (up to 20
    # words). The app log shows reading mode was never switched on:
    # all text was read with I (up to 6 words), and there are only 6
    # readings for these 8 tests - so at least two were never run, and
    # no "BOLDFIT" text appears at all.
    "2026-10-06T18:34:29": "read with I, not SPACE; no BOLDFIT text in log",
    "2026-10-06T18:34:57": "read with I, not SPACE in reading mode",
    "2026-10-06T18:35:32": "read with I, not SPACE in reading mode",
    "2026-10-06T18:35:59": "read with I, not SPACE in reading mode",
    "2026-10-06T18:36:25": "read with I, not SPACE in reading mode",
    "2026-10-06T18:36:29": "read with I, not SPACE in reading mode",
    "2026-10-06T18:37:01": "read with I, not SPACE in reading mode",
    "2026-10-06T18:37:31": "read with I, not SPACE in reading mode",

    # --- Third pass ---------------------------------------------------

    # Saved 2 and 4 seconds after the bright chair test - the light
    # cannot have been changed, so these were never tested separately
    "2026-10-06T18:53:36": "saved 4 s after the bright test; not tested",
    "2026-10-06T18:53:38": "saved 6 s after the bright test; not tested",

    # The log has no reading between BOLDFIT and the cream tube - the
    # app had been switched back to ambient mode, where SPACE only
    # says "switch to reading mode". The book cover was never read.
    "2026-10-06T18:55:11": "no reading in the log for this test",

    # --- Fourth pass --------------------------------------------------

    # The log shows the title WAS read ("Silberschatzs Operating System
    # Concepts"), but the next line is a speech error: the internet
    # dropped, so nothing was spoken. This test is about reading, not
    # the network - it is redone with a steady connection.
    "2026-10-06T19:12:35": "title was read, but speech failed: internet "
                           "dropped (speech error in log)",
}


def main():
    with open(RESULTS, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        rows = list(reader)

    # Keep an untouched copy, made only the first time this runs
    if not BACKUP.exists():
        shutil.copy(RESULTS, BACKUP)
        print(f"Backup saved: {BACKUP}")

    kept = []
    seen_tests = set()

    for row in rows:
        stamp = row["timestamp"]

        if stamp in REMOVALS:
            print(f"REMOVED    {stamp}  {row['kind']:<9} {row['item']}"
                  f" ({row['lighting']}) - {REMOVALS[stamp]}")
            continue

        # The same test recorded twice (the duplicate bottle row from
        # Day 9): keep the first recording, drop the later copy
        test = (row["kind"], row["item"], row["lighting"])
        if test in seen_tests:
            print(f"REMOVED    {stamp}  {row['kind']:<9} {row['item']}"
                  f" ({row['lighting']}) - duplicate of an earlier row")
            continue
        seen_tests.add(test)

        if stamp in CORRECTIONS and TAG not in row["notes"]:
            field, new_value, reason = CORRECTIONS[stamp]
            old_value = row[field]
            row[field] = new_value
            note = f"{TAG}: {field} was '{old_value}' - {reason}"
            row["notes"] = f"{row['notes']} | {note}" if row["notes"] else note
            print(f"CORRECTED  {stamp}  {row['kind']:<9} {row['item']}"
                  f" ({row['lighting']}) - {field}: '{old_value}' -> "
                  f"'{new_value}'")

        if stamp in NOTES and TAG not in row["notes"]:
            note = f"{TAG}: {NOTES[stamp]}"
            row["notes"] = f"{row['notes']} | {note}" if row["notes"] else note
            print(f"NOTE ADDED {stamp}  {row['kind']:<9} {row['item']}"
                  f" ({row['lighting']}) - {NOTES[stamp]}")

        kept.append(row)

    with open(RESULTS, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(kept)

    print(f"\n{len(rows)} rows before, {len(kept)} rows now.")


if __name__ == "__main__":
    main()