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

Day 9 tests: detection, identification and reading (56 tests).
Day 12 tests, fixed on 6 October 2026 before any of them were run:
voice commands in English and Hindi, visual questions in English and
Hindi (including trap questions about things that are not there), and
the Hindi translation of each Hindi answer (30 tests).
"""

import csv
from datetime import datetime
from pathlib import Path

OUTPUT = Path("evaluation/results.csv")

# ----------------------------------------------------------------------
# Day 9 tests
# ----------------------------------------------------------------------

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

# ----------------------------------------------------------------------
# Day 12 tests
# ----------------------------------------------------------------------
# Each voice command is "phrase -> the action it should trigger". The
# phrases are natural ways of asking, chosen before testing - not copied
# from the app's keyword lists.
#
# The order matters: the English list ends by switching to Hindi, and
# the Hindi list ends by switching back to English.

VOICE_EN = [
    "what is this -> identify",
    "what do you see around me -> describe",
    "say that again -> repeat",
    "read this for me -> read",
    "go back to ambient mode -> ambient mode",
    "switch to reading mode -> reading mode",
    "speak in Hindi -> switch to Hindi",
]

VOICE_HI = [
    "यह क्या है (yeh kya hai) -> identify",
    "आसपास क्या दिख रहा है (aaspaas kya dikh raha hai) -> describe",
    "फिर से बोलो (phir se bolo) -> repeat",
    "इसे पढ़ो (ise padho) -> read",
    "देखने का मोड (dekhne ka mode) -> ambient mode",
    "पढ़ने का मोड चालू करो (padhne ka mode chalu karo) -> reading mode",
    "इंग्लिश में बोलो (English mein bolo) -> switch to English",
]

# The last three are traps: they ask about something that is NOT there.
# A model that invents an answer fails them.
QUESTIONS_EN = [
    "What am I wearing?",
    "What colour is my shirt?",
    "What is in my right hand? (hold a bottle)",
    "How many fingers am I holding up?",
    "What is written on this bottle? (hold BOLDFIT)",
    "Is there a cup on the table? (TRAP: no cup in view)",
    "Is there a dog in this room? (TRAP)",
    "What colour is my watch? (TRAP: no watch on)",
]

QUESTIONS_HI = [
    "मैंने क्या पहना है? (maine kya pehna hai)",
    "मेरे हाथ में क्या है? (mere haath mein kya hai) (hold a bottle)",
    "मेरी शर्ट किस रंग की है? (meri shirt kis rang ki hai)",
    "क्या यहाँ कोई कुत्ता है? (kya yahan koi kutta hai) (TRAP)",
]

# ----------------------------------------------------------------------
# Verdicts offered for each kind of test
# ----------------------------------------------------------------------

VOICE_SCALE = {
    "1": "correct",         # the right action happened
    "2": "wrong action",    # a different command ran
    "3": "asked as question",  # no command matched - sent to the vision model
    "4": "nothing",         # nothing happened, or "I did not hear that"
}

QUESTION_SCALE = {
    "1": "correct",         # the answer was right
    "2": "partly correct",  # some of it right, some wrong or missing
    "3": "wrong",           # the answer was wrong
    "4": "made up",         # described something that is not there
    "5": "no answer",       # an error, a timeout, or nothing at all
}

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
    "voice_en": VOICE_SCALE,
    "voice_hi": VOICE_SCALE,
    "question_en": QUESTION_SCALE,
    "question_hi": QUESTION_SCALE,
    "translation": {
        "1": "meaning kept",    # the Hindi says what the English said
        "2": "small error",     # a slip, but the meaning is clear
        "3": "meaning changed", # the Hindi says something different
        "4": "not translated",  # the answer was spoken in English
    },
}

# What to do for each kind of test, shown before the verdict
ACTIONS = {
    "detection": "cover the camera for 35 s, uncover, listen for the name",
    "identify": "hold it centred, press I, listen to the object name",
    "reading": "press M (check 'Switched to READING mode'), point at "
               "the text, press SPACE, listen",
    "voice_en": "press V, say the phrase (before the ->), press V. "
                "Check the 'heard:' line in the app's terminal",
    "voice_hi": "press V, say the Hindi phrase, press V. "
                "Check the 'heard:' line in the app's terminal",
    "question_en": "press V, ask the question, press V, listen to the answer",
    "question_hi": "app in Hindi (L). Press V, ask in Hindi, press V, "
                   "listen to the answer",
    "translation": "for the Hindi answer you just heard: compare it with "
                   "the '[vision] English answer:' line in the app's terminal",
}


def build_test_plan():
    """The fixed list of tests, in the order they will be run."""
    tests = []

    # --- Day 9 ---------------------------------------------------------

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

    # --- Day 12 (all in normal light) ----------------------------------

    for phrase in VOICE_EN:
        tests.append(("voice_en", phrase, "normal", "-"))

    for phrase in VOICE_HI:
        tests.append(("voice_hi", phrase, "normal", "-"))

    for question in QUESTIONS_EN:
        tests.append(("question_en", question, "normal", "-"))

    # Each Hindi answer is judged twice, straight after it is heard:
    # once for whether it is right, once for the translation
    for question in QUESTIONS_HI:
        tests.append(("question_hi", question, "normal", "-"))
        tests.append(("translation", "translation of: " + question,
                      "normal", "-"))

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

    # An empty file counts as new: the header must be written, and
    # checking only for existence missed the case of an empty file
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
            print(f"  Action:    {ACTIONS[kind]}")

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