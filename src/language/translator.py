"""
English to Hindi translation, running locally, for Hindi mode.

Used to translate the vision model's answers. The vision model answers
accurately in English but becomes inaccurate when asked to answer in
Hindi directly, so its English answers are translated instead.

WHY THIS TRANSLATOR (tested 5 October):
  the vision model itself   changed facts: "a red bracelet on the right
                            wrist" became "on the knee of the right hand"
  Google Translate (free)   refused every request - four out of four
                            returned "too many requests", though only one
                            request was made each time
  NLLB-200, distilled 600M  correct on all four test sentences, 1.2-1.9 s
                            each on the CPU - the one in use here

NLLB-200 ("No Language Left Behind", Meta) is built for translation
across 200 languages. It runs on the CPU, so it does not compete with
detection and the vision model for GPU memory. Licence: CC-BY-NC 4.0 -
non-commercial use only.

THE GLOSSARY: on its own, NLLB translated "a red bracelet" as "a red
wrist", losing the object entirely. Object names are therefore swapped
for their spoken Hindi form before translation. The translator sees a
word already in Devanagari and leaves it alone - a standard technique
called a terminology constraint. It also enforces the Hinglish style of
Hindi mode: object names stay as Hindi speakers actually say them.

Loading takes a few seconds and about 2.5 GB of memory, so the model is
loaded only when Hindi mode is first used - English-only sessions never
pay that cost.
"""

import re
import threading
import time
import unicodedata

MODEL = "facebook/nllb-200-distilled-600M"

# NLLB names languages by language and script
SOURCE_LANGUAGE = "eng_Latn"
TARGET_LANGUAGE = "hin_Deva"

# Two short sentences translate well within this
MAX_NEW_TOKENS = 128

# ---------------------------------------------------------------------
# THE GLOSSARY: English word -> how a Hindi speaker says it
# ---------------------------------------------------------------------
# Mostly English words in Devanagari (Hinglish); a few everyday items
# use the Hindi word.
#
# HOW IT WORKS: each listed word is swapped for its Hindi form INSIDE
# the English sentence, before NLLB translates it. NLLB then sees the
# word already in Devanagari and keeps it. That is why only some kinds
# of word can go here.
#
# RULES - learned in testing, 5-7 October:
#
#   1. NOUNS for things a person wears, holds, or has around them.
#
#   2. NEVER a word whose Hindi depends on the words around it. One
#      entry gives one fixed form, everywhere:
#        "old"    old man = बूढ़ा आदमी, old bag = पुराना बैग
#        "light"  a light on the wall = लाइट, a light bag = हल्का बैग
#        "orange" an orange (fruit) = संतरा, an orange shirt = नारंगी
#        "back"   behind = पीछे, the back of a phone = पिछला हिस्सा
#        "right"  the side = दायाँ, correct = सही
#      NLLB reads the whole sentence and chooses correctly on its own.
#      The glossary would take that choice away. check_glossary.py tests
#      that NLLB gets these right.
#
#   3. NO colours or describing words: Hindi changes them to agree with
#      the noun - काला चश्मा, काली शर्ट, काले जूते.
#
#   4. NO action words ("walking" -> "चल रहा है" scrambles the
#      sentence, and is masculine only) and NO direction words (Hindi
#      puts them after the noun: "मेज़ के नीचे").
#
#   5. If NLLB gets one meaning of a word wrong, add the PHRASE, never
#      the word - e.g. ("old man", "बूढ़ा आदमी"). Longer entries are
#      matched first, so a phrase always wins over a single word.
#
#   6. Only add what NLLB gets wrong, or what Hindi speakers say in
#      English. Animals, food, places and colours are already translated
#      correctly - every extra entry is one more way to break a sentence.
#
# A list of pairs rather than a dictionary: in a Python dictionary a
# repeated word is SILENTLY overwritten by its last entry (the 7 October
# draft had six such repeats). _build_glossary() refuses repeats instead.

_GLOSSARY_PAIRS = [
    # --- the original list (5 October) -----------------------------------
    ("t-shirt", "टी-शर्ट"), ("shirt", "शर्ट"), ("bracelet", "ब्रेसलेट"),
    ("headphones", "हेडफ़ोन"), ("earphones", "ईयरफ़ोन"), ("glasses", "चश्मा"),
    ("slippers", "चप्पल"), ("slipper", "चप्पल"), ("cap", "कैप"),
    ("hat", "टोपी"), ("watch", "घड़ी"), ("ring", "अंगूठी"),
    ("bottle", "बोतल"), ("phone", "फ़ोन"), ("cup", "कप"),
    ("chair", "कुर्सी"), ("bed", "बेड"), ("laptop", "लैपटॉप"),
    ("book", "किताब"), ("pillow", "तकिया"), ("lamp", "लैंप"),

    # Plural added 7 October: NLLB wrote "दो तकिया" (two pillow)
    ("pillows", "तकिए"),

    # The one describing word: NLLB translated "curly hair" as "गुलदस्ता
    # बाल" (bouquet hair). It only ever describes hair, which is plural
    # in Hindi, so the fixed form always fits.
    ("curly", "घुंघराले"),

    # --- added 7 October: everyday items ------------------------------------

    # Clothing. "turtleneck" came out as "टर्टलगेक" in testing.
    ("turtleneck", "टर्टलनेक"), ("sweater", "स्वेटर"), ("hoodie", "हुडी"),
    ("jacket", "जैकेट"), ("jeans", "जींस"), ("pants", "पैंट"),
    ("trousers", "पैंट"), ("shorts", "शॉर्ट्स"), ("kurta", "कुर्ता"),
    ("saree", "साड़ी"), ("dress", "ड्रेस"), ("skirt", "स्कर्ट"),
    ("scarf", "स्कार्फ़"), ("gloves", "दस्ताने"), ("belt", "बेल्ट"),
    ("socks", "मोज़े"), ("shoes", "जूते"), ("shoe", "जूता"),
    ("sandals", "सैंडल"),

    # Worn or carried
    ("earbuds", "ईयरबड्स"), ("sunglasses", "सनग्लासेस"),
    ("necklace", "नेकलेस"), ("earrings", "ईयररिंग्स"),
    ("earring", "ईयररिंग"), ("wallet", "वॉलेट"), ("purse", "पर्स"),
    ("handbag", "हैंडबैग"), ("backpack", "बैग"), ("bag", "बैग"),
    ("umbrella", "छाता"),

    # Phone and electronics. "a yellow case" became "पीला बैग" (yellow
    # bag) in testing - when describing a person, a case is a phone cover.
    ("smartphone", "स्मार्टफ़ोन"), ("mobile", "मोबाइल"), ("case", "कवर"),
    ("charger", "चार्जर"), ("cable", "केबल"), ("remote", "रिमोट"),
    ("keyboard", "कीबोर्ड"), ("mouse", "माउस"), ("computer", "कंप्यूटर"),
    ("camera", "कैमरा"), ("microphone", "माइक"), ("headset", "हेडसेट"),
    ("tv", "टीवी"), ("television", "टीवी"),

    # Desk and kitchen
    ("notebook", "नोटबुक"), ("pen", "पेन"), ("pencil", "पेंसिल"),
    ("eraser", "रबर"), ("calendar", "कैलेंडर"), ("clock", "घड़ी"),
    ("key", "चाबी"), ("keys", "चाबियाँ"), ("spoon", "चम्मच"),
    ("fork", "कांटा"), ("knife", "चाकू"), ("plate", "प्लेट"),
    ("bowl", "कटोरी"),

    # Room and home
    ("table", "टेबल"), ("desk", "डेस्क"), ("sofa", "सोफ़ा"),
    ("couch", "सोफ़ा"), ("shelf", "शेल्फ़"), ("blanket", "कंबल"),
    ("mattress", "गद्दा"), ("fan", "पंखा"), ("mirror", "आईना"),
    ("door", "दरवाज़ा"), ("window", "खिड़की"), ("curtain", "पर्दा"),
    ("stairs", "सीढ़ियाँ"), ("towel", "तौलिया"), ("bucket", "बाल्टी"),
    ("broom", "झाड़ू"), ("toothbrush", "टूथब्रश"),
    ("toothpaste", "टूथपेस्ट"), ("soap", "साबुन"),
    ("shampoo", "शैम्पू"), ("comb", "कंघी"),

    # On the road - for outdoor use
    ("car", "कार"), ("bus", "बस"), ("truck", "ट्रक"), ("bike", "बाइक"),
    ("bicycle", "साइकिल"), ("motorcycle", "मोटरसाइकिल"),
    ("scooter", "स्कूटर"), ("rickshaw", "रिक्शा"),
]


def _build_glossary(pairs):
    """Turn the pairs into a lookup table, refusing any mistakes.

    Raises an error at startup rather than translating wrongly later:
    a repeated word would otherwise keep only its last meaning, and an
    upper-case word would never be found (look-ups are lower-case).
    """
    glossary = {}
    for english, hindi in pairs:
        if english != english.lower():
            raise ValueError(f"glossary word must be lower-case: {english!r}")
        if english in glossary:
            raise ValueError(f"glossary word listed twice: {english!r}")
        glossary[english] = hindi
    return glossary


GLOSSARY = _build_glossary(_GLOSSARY_PAIRS)

# Longest first, so "t-shirt" is replaced before "shirt" can match inside
# it, and a phrase such as "old man" before any single word in it
_PATTERN = re.compile(
    r"\b(" + "|".join(sorted(map(re.escape, GLOSSARY), key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)


def apply_glossary(text):
    """Replace each known object name with its spoken Hindi form."""
    return _PATTERN.sub(lambda match: GLOSSARY[match.group(0).lower()], text)


# ---------------------------------------------------------------------
# POLISHING: two grammar slips NLLB makes, fixed after translation
# ---------------------------------------------------------------------
# Tested 7 October. Both happen WITHOUT the glossary too - they are
# NLLB's own errors, not caused by swapping words:
#
#   1. Describing words not agreeing with a feminine noun:
#        "एक काला शर्ट", "एक काले शर्ट", "एक काला घड़ी"
#      should be काली शर्ट, काली घड़ी. In Hindi a describing word that
#      ends in -आ/-े takes -ई before a feminine noun, singular or plural.
#
#   2. Switching to the informal तुम: "तुम एक काला शर्ट पहन रहे हो"
#      The assistant always addresses the user as आप.
#
# Deliberately narrow: only words directly before a noun in the list
# below are changed, and only the forms listed. Anything else is left
# exactly as NLLB wrote it.

# Describing words that change for a feminine noun: masculine forms ->
# feminine form
_FEMININE_FORM = {
    "काला": "काली", "काले": "काली",
    "नीला": "नीली", "नीले": "नीली",
    "पीला": "पीली", "पीले": "पीली",
    "हरा": "हरी", "हरे": "हरी",
    "भूरा": "भूरी", "भूरे": "भूरी",
    "बड़ा": "बड़ी", "बड़े": "बड़ी",
    "छोटा": "छोटी", "छोटे": "छोटी",
    "लंबा": "लंबी", "लंबे": "लंबी",
    "पुराना": "पुरानी", "पुराने": "पुरानी",
    "नया": "नई", "नए": "नई",
}

# Feminine nouns, as they appear after the glossary or from NLLB. Only
# nouns whose gender is not in doubt in everyday Hindi.
_FEMININE_NOUNS = [
    "शर्ट", "टी-शर्ट", "घड़ी", "बोतल", "कुर्सी", "टोपी", "चप्पल",
    "किताब", "अंगूठी", "जींस", "पैंट", "जैकेट", "ड्रेस", "स्कर्ट",
    "साड़ी", "बेल्ट", "हुडी", "टेबल", "मेज़", "चाबी", "कटोरी",
    "प्लेट", "पेंसिल", "नोटबुक", "कंघी", "बाल्टी", "झाड़ू", "कार",
    "बस", "बाइक", "साइकिल", "मोटरसाइकिल", "दीवार", "खिड़की", "लाइट",
]

# Devanagari letters and signs: used instead of \b, because Python does
# not count Hindi vowel signs (ा, ी, े) as word characters
_DEV = "\u0900-\u097F"


def _flexible(word):
    """A pattern for a word, matching it with or without the nukta dot."""
    word = unicodedata.normalize("NFC", word)
    return re.escape(word).replace("\u093c", "\u093c?")


_AGREEMENT = re.compile(
    f"(?<![{_DEV}])("
    + "|".join(_flexible(w) for w in sorted(_FEMININE_FORM, key=len, reverse=True))
    + f")(\\s+)("
    + "|".join(_flexible(w) for w in sorted(_FEMININE_NOUNS, key=len, reverse=True))
    + f")(?![{_DEV}])"
)

# Informal -> respectful. Longer forms first, so तुम्हारे is not read as तुम
_RESPECTFUL = [
    ("तुम्हारे", "आपके"), ("तुम्हारी", "आपकी"), ("तुम्हारा", "आपका"),
    ("तुम्हें", "आपको"), ("तुमको", "आपको"), ("तुमने", "आपने"),
    ("तुम", "आप"),
]


def polish_hindi(text):
    """Fix NLLB's two common grammar slips. Changes nothing else."""
    text = unicodedata.normalize("NFC", text)

    # 1. Feminine agreement: "काले शर्ट" -> "काली शर्ट"
    def feminine(match):
        adjective = unicodedata.normalize("NFC", match.group(1))
        adjective = adjective.replace("\u093c", "")  # compare without nukta
        for masculine, female in _FEMININE_FORM.items():
            if masculine.replace("\u093c", "") == adjective:
                return female + match.group(2) + match.group(3)
        return match.group(0)

    text = _AGREEMENT.sub(feminine, text)

    # 2. तुम -> आप, and the verb "हो" that goes with तुम -> "हैं"
    informal = False
    for old, new in _RESPECTFUL:
        pattern = re.compile(f"(?<![{_DEV}]){old}(?![{_DEV}])")
        if pattern.search(text):
            informal = True
            text = pattern.sub(new, text)
    if informal:
        # Only "हो" ending a clause: "पहन रहे हो।" -> "पहन रहे हैं।"
        text = re.sub(f"(?<![{_DEV}])हो(?=\\s*[।,?!]|\\s*$)", "हैं", text)

    return text


class HindiTranslator:
    def __init__(self):
        self._tokenizer = None
        self._model = None
        self._hindi_id = None
        self._lock = threading.Lock()
        self._started_loading = False
        self.available = True

    def preload(self):
        """Begin loading in the background, so the first answer is not delayed.

        Called when the user switches to Hindi.
        """
        if self._started_loading:
            return
        self._started_loading = True
        threading.Thread(target=self._load, daemon=True).start()

    def _load(self):
        """Load the model. Safe to call from several threads at once."""
        with self._lock:
            if self._model is not None or not self.available:
                return
            try:
                # Imported here rather than at the top, so that simply
                # starting the application does not load transformers
                from transformers import (AutoModelForSeq2SeqLM,
                                          AutoTokenizer, logging)
                logging.set_verbosity_error()

                print("  [translate] loading the Hindi translation model...")
                started = time.time()

                try:
                    # Use the copy already on disk, without contacting
                    # the internet - so Hindi mode works offline
                    tokenizer = AutoTokenizer.from_pretrained(
                        MODEL, src_lang=SOURCE_LANGUAGE, local_files_only=True)
                    model = AutoModelForSeq2SeqLM.from_pretrained(
                        MODEL, local_files_only=True)
                except Exception:
                    # Not downloaded yet: fetch it once (about 2.4 GB)
                    print("  [translate] first use - downloading about 2.4 GB")
                    tokenizer = AutoTokenizer.from_pretrained(
                        MODEL, src_lang=SOURCE_LANGUAGE)
                    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL)

                model.eval()
                hindi_id = tokenizer.convert_tokens_to_ids(TARGET_LANGUAGE)
                print(f"  [translate] loaded in {time.time() - started:.1f} s")

                # The first translation after loading is several times
                # slower than the rest - 9.5 and 10.6 s against about 1 s,
                # measured on 7 October - while the model sets itself up on
                # its first run. One short translation here, still in the
                # background, means the user's first Hindi answer does not
                # pay that cost. It runs BEFORE the model is handed over
                # below, so a real request waits for it instead of running
                # alongside it.
                self._warm_up(tokenizer, model, hindi_id)

                self._tokenizer = tokenizer
                self._model = model
                self._hindi_id = hindi_id
                print(f"  [translate] ready in {time.time() - started:.1f} s")

            except Exception as error:
                print(f"  [translate] unavailable: {error}")
                self.available = False

    def _warm_up(self, tokenizer, model, hindi_id):
        """Run one short translation so the first real one is fast.

        A failure here is only printed: a slow first answer is better than
        switching Hindi translation off.
        """
        try:
            import torch

            started = time.time()
            with torch.no_grad():
                inputs = tokenizer("Hello.", return_tensors="pt")
                model.generate(**inputs, forced_bos_token_id=hindi_id,
                               max_new_tokens=8)
            print(f"  [translate] warmed up in {time.time() - started:.1f} s")
        except Exception as error:
            print(f"  [translate] warm-up skipped: {error}")

    def translate(self, english):
        """Translate English into Hindi. Returns None if that is not possible.

        None rather than an exception, so the caller can fall back to
        speaking the English answer - an accurate answer in English is
        better than none at all.
        """
        if not self.available:
            return None

        if self._model is None:
            # Waits here if a background load is already under way
            self._load()
            if self._model is None:
                return None

        import torch

        swapped = apply_glossary(english)
        with torch.no_grad():
            inputs = self._tokenizer(swapped, return_tensors="pt")
            output = self._model.generate(
                **inputs,
                forced_bos_token_id=self._hindi_id,
                max_new_tokens=MAX_NEW_TOKENS,
            )
        hindi = self._tokenizer.batch_decode(output, skip_special_tokens=True)[0]
        return polish_hindi(hindi.strip()) or None