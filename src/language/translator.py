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

MODEL = "facebook/nllb-200-distilled-600M"

# NLLB names languages by language and script
SOURCE_LANGUAGE = "eng_Latn"
TARGET_LANGUAGE = "hin_Deva"

# Two short sentences translate well within this
MAX_NEW_TOKENS = 128

# English object name -> how a Hindi speaker says it. Mostly English
# words in Devanagari (Hinglish); a few everyday items use the Hindi word.
GLOSSARY = {
    "t-shirt": "टी-शर्ट", "shirt": "शर्ट", "bracelet": "ब्रेसलेट",
    "headphones": "हेडफ़ोन", "earphones": "ईयरफ़ोन", "glasses": "चश्मा",
    "slippers": "चप्पल", "slipper": "चप्पल", "cap": "कैप", "hat": "टोपी",
    "watch": "घड़ी", "ring": "अंगूठी", "bottle": "बोतल", "phone": "फ़ोन",
    "cup": "कप", "chair": "कुर्सी", "bed": "बेड", "laptop": "लैपटॉप",
    "book": "किताब", "pillow": "तकिया", "lamp": "लैंप", 
    # Describing words NLLB got wrong in testing:
    # "curly" came out as "गुलदस्ता" (bouquet), so it is fixed here too.
    "curly": "घुंघराले",
}

# Longest first, so "t-shirt" is replaced before "shirt" can match inside it
_PATTERN = re.compile(
    r"\b(" + "|".join(sorted(map(re.escape, GLOSSARY), key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)


def apply_glossary(text):
    """Replace each known object name with its spoken Hindi form."""
    return _PATTERN.sub(lambda match: GLOSSARY[match.group(0).lower()], text)


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
                self._tokenizer = tokenizer
                self._model = model
                self._hindi_id = tokenizer.convert_tokens_to_ids(TARGET_LANGUAGE)
                print(f"  [translate] ready in {time.time() - started:.1f} s")

            except Exception as error:
                print(f"  [translate] unavailable: {error}")
                self.available = False

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
        return hindi.strip() or None