"""
Corrections to evaluation/results.csv, made on 7 October 2026.

The Day 12 tests (voice commands, visual questions, Hindi translation)
were checked against the application's own log from the same session,
and against what the tester confirmed afterwards. Where a verdict
contradicted the log it is corrected here - in code, with the reason
written into the row's notes - rather than edited by hand.

Same method as correct_results_2026_10_06.py:

  CORRECTED  the test happened, but the verdict was entered wrongly.
  REMOVED    the test did not test what it claims; it is run again.
  NOTE       the verdict stands; a fact from the log is added.

Rule for repeated attempts: the FIRST attempt where the test was set
up correctly counts. If the app mishears a question and confidently
answers a different one, that is exactly what a blind user would
experience, so it is counted - not retried away.

Safe to run more than once: changes already made are skipped.

    python src\\evaluation\\correct_results_2026_10_07.py
"""

import csv
import shutil
from pathlib import Path

RESULTS = Path("evaluation/results.csv")
BACKUP = Path("evaluation/results_before_2026-10-07_corrections.csv")
TAG = "corrected 2026-10-07"

# timestamp of the row -> (field to change, new value, reason)
CORRECTIONS = {
    # Whisper heard "आस पास का दिख रहा है" ("का" for "क्या", and
    # "आस पास" as two words), so no command matched. The app sent it to
    # the vision model as a question, which described the scene.
    "2026-10-07T14:57:46": ("verdict", "asked as question",
                            "heard 'आस पास का दिख रहा है' - no command "
                            "matched; the vision model answered instead"),

    # Whisper heard a completely different phrase, so "repeat" never ran
    "2026-10-07T14:58:07": ("verdict", "asked as question",
                            "heard 'मेरा ब्रेसलेट करो' - no command "
                            "matched; the vision model answered instead"),

    # First try: heard "किसर्ट किस कर रंग की है", and the answer was
    # about the wall ("The wall behind you is orange"). The second try
    # was also misheard. Only the third try answered "grey".
    "2026-10-07T15:08:16": ("verdict", "wrong",
                            "1st try heard 'किसर्ट किस कर रंग की है' and "
                            "answered about the wall; grey only on the "
                            "3rd try"),

    # The translation is judged on the same (first) answer as the
    # question. That one was translated faithfully: "कैमरा आपको इंगित कर
    # रहा है, दीवार नहीं। दीवार आपके पीछे नारंगी है।" The "small error"
    # entered was about the 3rd answer, where "turtleneck" became
    # "टर्टलगेक".
    "2026-10-07T15:08:51": ("verdict", "meaning kept",
                            "judged on the 1st answer, which was "
                            "translated faithfully; the 3rd answer "
                            "garbled 'turtleneck' as 'टर्टलगेक'"),

    # --- Second pass: the last three tests ----------------------------
    # A translation test asks one thing: does the Hindi sentence mean
    # the same as the English one? Both verdicts below had been scored
    # on how the app reacted to the words "translation of" being spoken
    # aloud - which is not what the test measures.

    # English: "a smartphone with a yellow case". Hindi: "पीला बैग"
    # (yellow bag) - the translator changed what the object is.
    "2026-10-07T17:15:59": ("verdict", "meaning changed",
                            "the 1st answer said 'yellow case'; the Hindi "
                            "said 'पीला बैग' (yellow bag)"),

    # English: "No dog is visible in the image." Hindi: "चित्र में कोई
    # कुत्ता दिखाई नहीं देता है।" - the same meaning, word for word.
    "2026-10-07T17:18:06": ("verdict", "meaning kept",
                            "'No dog is visible in the image' -> 'चित्र "
                            "में कोई कुत्ता दिखाई नहीं देता है' is faithful"),
}

# timestamp of the row -> reason it is removed
REMOVALS = {
    # The dog question was never asked. "क्या मेरे कमरे में कोई और
    # आदमी है" (is there another man in my room) was asked instead -
    # and there was one, in a red t-shirt, so that unplanned answer was
    # right. The trap question is asked again.
    "2026-10-07T15:09:28": "the dog question was never asked; 'is there "
                           "another man in my room' was asked instead",
    "2026-10-07T15:09:56": "translation of the removed question",
}

# timestamp of the row -> extra fact from the log or the tester
NOTES = {
    "2026-10-07T14:51:51": "Classmate project paper sheets, not a book; "
                           "full text on the 2nd press",
    "2026-10-07T14:54:42": "heard 'What, repeat.' - the right action ran",
    "2026-10-07T15:01:07": "1st ask: 'your hands are not visible' - true, "
                           "the hand was not up yet; 2 fingers on the "
                           "2nd ask",
    "2026-10-07T15:01:36": "a different bottle was used, not BOLDFIT; its "
                           "sticker says 'WHEY BEFORE BAE', read correctly",
    "2026-10-07T15:06:56": "1st answer named only the phone ('yellow "
                           "case'); the 2nd named phone and bottle but put "
                           "them in different hands",
}


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

    kept = []
    for row in rows:
        stamp = row["timestamp"]
        label = f"{stamp}  {row['kind']:<11} {row['item'][:40]}"

        if stamp in REMOVALS:
            print(f"REMOVED    {label} - {REMOVALS[stamp]}")
            continue

        already = TAG in row["notes"]

        if stamp in CORRECTIONS and not already:
            field, new_value, reason = CORRECTIONS[stamp]
            old_value = row[field]
            row[field] = new_value
            add_note(row, f"{field} was '{old_value}' - {reason}")
            print(f"CORRECTED  {label} - {field}: '{old_value}' -> "
                  f"'{new_value}'")

        if stamp in NOTES and not already:
            add_note(row, NOTES[stamp])
            print(f"NOTE ADDED {label}")

        kept.append(row)

    with open(RESULTS, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(kept)

    print(f"\n{len(rows)} rows before, {len(kept)} rows now.")


if __name__ == "__main__":
    main()