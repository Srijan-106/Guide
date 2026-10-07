"""
Check the translation glossary after changing it.

Three parts:

  1. THE GLOSSARY ITSELF - how many words, and that it loaded (a word
     listed twice, or in upper case, stops it loading - see translator.py).

  2. REAL ANSWERS from the 7 October tests, translated with the current
     glossary next to the Hindi the app produced on the day: did the
     change fix the errors, and did it break anything already right?

  3. WORDS LEFT OUT ON PURPOSE - words whose Hindi depends on the words
     around them ("old man" = बूढ़ा आदमी, "old bag" = पुराना बैग). These
     are left to NLLB, which reads the whole sentence. Each sentence
     lists the Hindi words it should contain; the script marks it PASS
     if one is there, or CHECK if a person needs to look.

  4. POLISHING - the step that fixes NLLB's grammar slips (feminine
     agreement, तुम -> आप), tested on sentences NLLB actually produced
     on 7 October. These need no model and must all PASS.

    python src\\evaluation\\check_glossary.py

PASS/CHECK is a quick filter, not the judgement: read every line.
"""

import sys
import time
import unicodedata

sys.path.insert(0, "src")          # find the app's modules, as main.py does

from language.translator import (GLOSSARY, HindiTranslator,
                                 apply_glossary, polish_hindi)

# --- Part 2: (English answer, Hindi from the app's log on 7 October) ------
REAL_ANSWERS = [
    # The error to fix: "yellow case" became "पीला बैग" (yellow bag)
    ("You are holding a smartphone with a yellow case and a black back.",
     "आपके हाथ में एक पीला बैग और एक काला पीठ वाला स्मार्टफोन है"),

    # The garbled word: "turtleneck" became "टर्टलगेक"
    ("You are wearing a grey turtleneck sweater.",
     "आप एक ग्रे टर्टलगेक स्वेटर पहन रहे हैं।"),

    # Already right - must stay right
    ("You are wearing a grey t-shirt, glasses, and white earbuds.",
     "आप ग्रे टी-शर्ट, चश्मे और सफेद ईयरबड्स पहनते हैं।"),
    ("No dog is visible in the image.",
     "चित्र में कोई कुत्ता दिखाई नहीं देता है।"),
    ("You have dark, curly hair.",
     "आपके काले, घुंघराले बाल हैं।"),
]

# --- Part 3: (English, Hindi words it should contain - any one) -----------
CONTEXT_WORDS = [
    # "old": the person or the thing decides the word
    ("An old man is sitting on the chair.", ["बूढ़ा", "बुजुर्ग", "वृद्ध"]),
    ("There is an old bag on the bed.", ["पुराना"]),

    # "light": a lamp, or not heavy
    ("There is a light on the wall.", ["लाइट", "रोशनी", "बत्ती", "प्रकाश"]),
    ("This bag is very light.", ["हल्का"]),

    # "orange": the fruit, or the colour
    # Known NLLB slip, accepted: it picks the fruit correctly but writes
    # "एक संतरे" for "एक संतरा". Adding "an orange" to the glossary would
    # break "an orange t-shirt", so this one stays a CHECK.
    ("There is an orange on the plate.", ["संतरा"]),
    ("You are wearing an orange t-shirt.", ["नारंगी", "ऑरेंज"]),

    # Colours must agree with the noun - even a noun from the glossary
    ("You are wearing a black shirt.", ["काली"]),
    ("You are wearing black shoes.", ["काले"]),

    # "right" and "back"
    ("The bottle is on your right.", ["दाएं", "दाईं", "दायें", "दाहिने",
                                      "दाहिनी", "दाएँ"]),
    ("The back of the phone is black.", ["पिछला", "पिछले", "पीछे"]),

    # Plurals are not in the glossary; NLLB should still translate them
    ("There are two pillows on the bed.", ["तकिए", "तकिये"]),
]


# --- Part 4: (what NLLB wrote on 7 October, what it must become) ----------
POLISH_CASES = [
    # Feminine agreement - with and without the glossary
    ("आप एक काले शर्ट पहन रहे हैं।", "आप एक काली शर्ट पहन रहे हैं।"),
    ("आप एक सफेद शर्ट और एक काला घड़ी पहनते हैं।",
     "आप एक सफेद शर्ट और एक काली घड़ी पहनते हैं।"),
    # Informal तुम -> आप, together with agreement
    ("तुम एक काला शर्ट पहन रहे हो।", "आप एक काली शर्ट पहन रहे हैं।"),
    ("तुम काले जूते पहन रहे हो।", "आप काले जूते पहन रहे हैं।"),
    # Must NOT change: masculine nouns, plural hair, a describing word
    # that is not before a noun, and "हो" that is not the verb for तुम
    ("आपके काले, घुंघराले बाल हैं।", "आपके काले, घुंघराले बाल हैं।"),
    ("बिस्तर पर एक पुराना बैग है।", "बिस्तर पर एक पुराना बैग है।"),
    ("फोन का पीछे का रंग काला है।", "फोन का पीछे का रंग काला है।"),
    ("यह हो सकता है कि तुम थके हो।", "यह हो सकता है कि आप थके हैं।"),
]


def normalise(text):
    """Ignore the nukta dot, so सफ़ेद and सफेद count as the same word."""
    return unicodedata.normalize("NFD", text).replace("़", "")


def main():
    # --- Part 1 -----------------------------------------------------------
    print("=" * 64)
    print(f"1. GLOSSARY: {len(GLOSSARY)} words, loaded without errors")
    print("=" * 64)

    translator = HindiTranslator()

    # --- Part 2 -----------------------------------------------------------
    print("\n" + "=" * 64)
    print("2. REAL ANSWERS FROM TESTING - before and now")
    print("=" * 64)
    for english, before in REAL_ANSWERS:
        started = time.time()
        now = translator.translate(english)
        print("-" * 64)
        print(f"EN:      {english}")
        print(f"swapped: {apply_glossary(english)}")
        print(f"before:  {before}")
        print(f"now:     {now}   ({time.time() - started:.1f} s)")

    # --- Part 3 -----------------------------------------------------------
    print("\n" + "=" * 64)
    print("3. WORDS LEFT TO NLLB ON PURPOSE")
    print("=" * 64)
    passed = 0
    for english, expected in CONTEXT_WORDS:
        now = translator.translate(english) or ""
        ok = any(normalise(word) in normalise(now) for word in expected)
        passed += ok
        print("-" * 64)
        print(f"{'PASS ' if ok else 'CHECK'}  EN: {english}")
        print(f"       HI: {now}")
        if not ok:
            print(f"       expected one of: {', '.join(expected)}")

    print("-" * 64)
    print(f"\n{passed}/{len(CONTEXT_WORDS)} PASS. Look at every CHECK line: "
          f"if NLLB really chose the wrong word, add the PHRASE to the "
          f"glossary (rule 5 in translator.py), never the single word.")

    # --- Part 4 -----------------------------------------------------------
    print("\n" + "=" * 64)
    print("4. POLISHING - must all PASS")
    print("=" * 64)
    polished_ok = 0
    for nllb_wrote, expected in POLISH_CASES:
        result = polish_hindi(nllb_wrote)
        ok = result == expected
        polished_ok += ok
        print(f"{'PASS' if ok else 'FAIL'}  {nllb_wrote}")
        print(f"      -> {result}")
        if not ok:
            print(f"      expected: {expected}")
    print(f"\n{polished_ok}/{len(POLISH_CASES)} PASS")


if __name__ == "__main__":
    main()