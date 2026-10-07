"""
Check the OCR spelling correction against every case from testing.

Runs the real spell checker through src/language/spelling.py - the same
code main.py uses - on every word it changed in the 6-7 October logs,
with the confidence OCR gave each one. No camera or model is needed.

    python src\\evaluation\\check_spelling.py

Every line must say PASS. A FAIL means a change to spelling.py has
broken a case that was decided on evidence: either a damaging correction
is back, or a helpful one was lost.
"""

import sys

sys.path.insert(0, "src")          # find the app's modules, as main.py does

from spellchecker import SpellChecker

from language.spelling import check_spelling

# The same setting as in main.py
SPELLCHECK_BELOW = 0.90

# (word as OCR read it, OCR confidence, what should be spoken)
CASES = [
    # Helpful corrections: must still happen
    ("Ycstcrday",   0.80, "yesterday"),
    ("Retund",      0.83, "refund"),
    ("classmale",   0.69, "classmate"),
    ("BROJECT",     0.41, "project"),
    ("PRQJECT",     0.46, "project"),
    ("LDITION",     0.67, "edition"),
    ("Crodit",      0.31, "credit"),
    ("Teachor",     0.67, "teacher"),
    ("PERFORMANGE", 0.50, "performance"),

    # Damaging or wrong corrections: must be blocked, the word kept
    ("Glycolic",    0.33, "Glycolic"),     # not "glycol"
    ("Glyco",       0.74, "Glyco"),        # brand name, not "glycol"
    ("Glycolc",     0.45, "Glycolc"),      # was "Glycolic"; "glycol" is wrong
    ("Hazrat",      0.78, "Hazrat"),       # place name, not "hazard"
    ("RivEST",      0.25, "RivEST"),       # author's name, not "rivets"
    ("Hoighl",      0.42, "Hoighl"),       # was "Height"; "high" is wrong
    ("gmalcom",     0.85, "gmalcom"),      # was "gmail.com"; not "malcolm"

    # Small helpful corrections given up, by design (see spelling.py)
    ("Telephong",   0.57, "Telephong"),
    ("Creamn",      0.45, "Creamn"),
    ("Issuedho",    0.53, "Issuedho"),

    # Never checked at all
    ("BOLDFIT",     0.92, "BOLDFIT"),      # confident reading: trusted
    ("Pepsi",       0.58, "Pepsi"),        # a known word
    ("Wame",        0.03, "Wame"),         # too short to correct
]


def main():
    spell = SpellChecker()
    passed = 0

    for word, confidence, expected in CASES:
        spoken, suggestion, accepted = check_spelling(
            word, confidence, spell, SPELLCHECK_BELOW)
        ok = spoken == expected
        passed += ok

        if suggestion is None:
            what = "not checked"
        elif accepted:
            what = f"corrected (look-alike letters)"
        else:
            what = f"blocked: would have become '{suggestion}'"

        print(f"{'PASS' if ok else 'FAIL'}  {word:<12} -> {spoken:<12} {what}")
        if not ok:
            print(f"      expected: {expected}")

    print(f"\n{passed}/{len(CASES)} PASS")


if __name__ == "__main__":
    main()