"""
Test whether a glossary fixes NLLB's object-name errors.

Plain NLLB translated "a red bracelet on your right wrist" as "a red
wrist on your right wrist" - the object was lost. Object names are now
swapped for their spoken Hindi form BEFORE translation, so the
translator sees them already in Devanagari and leaves them alone. This
also enforces the Hinglish style chosen for Hindi mode.
"""

import re
import time

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

MODEL = "facebook/nllb-200-distilled-600M"

# English object name -> how a Hindi speaker says it. Mostly English
# words in Devanagari (Hinglish); a few everyday items use the Hindi word.
GLOSSARY = {
    "t-shirt": "टी-शर्ट", "shirt": "शर्ट", "bracelet": "ब्रेसलेट",
    "headphones": "हेडफ़ोन", "earphones": "ईयरफ़ोन", "glasses": "चश्मा",
    "slippers": "चप्पल", "slipper": "चप्पल", "cap": "कैप", "hat": "टोपी",
    "watch": "घड़ी", "ring": "अंगूठी", "bottle": "बोतल", "phone": "फ़ोन",
    "cup": "कप", "chair": "कुर्सी", "bed": "बेड", "laptop": "लैपटॉप",
    "book": "किताब", "pillow": "तकिया", "lamp": "लैंप",
}

# Longest first, so "t-shirt" is replaced before "shirt" can match inside it
PATTERN = re.compile(
    r"\b(" + "|".join(sorted(map(re.escape, GLOSSARY), key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)


def apply_glossary(text):
    return PATTERN.sub(lambda match: GLOSSARY[match.group(0).lower()], text)


SENTENCES = [
    "You are wearing a red bracelet on your right wrist.",
    "You are wearing a light pink long-sleeved shirt, glasses, and over-ear headphones.",
    "The wall behind you is orange, and there is a white wall to the left.",
    "A dark-colored slipper is visible on the floor near the left wall.",
]

tokenizer = AutoTokenizer.from_pretrained(MODEL, src_lang="eng_Latn")
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL)
hindi = tokenizer.convert_tokens_to_ids("hin_Deva")


def translate(text):
    inputs = tokenizer(text, return_tensors="pt")
    output = model.generate(**inputs, forced_bos_token_id=hindi,
                            max_new_tokens=128)
    return tokenizer.batch_decode(output, skip_special_tokens=True)[0]


for sentence in SENTENCES:
    swapped = apply_glossary(sentence)
    started = time.time()
    result = translate(swapped)

    print(f"EN:      {sentence}")
    print(f"swapped: {swapped}")
    print(f"HI:      {result}")
    print(f"         ({time.time() - started:.1f} s)\n")