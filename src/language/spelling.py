"""
Spelling correction for OCR readings - limited to OCR's own mistakes.

OCR misreads letters that LOOK alike: "c" for "e", "l" for "t", "B" for
"P". A spell checker undoes those well. But it also "corrects" words it
simply does not know - names, brands, chemical names - into ordinary
words, and then the reading is confidently wrong.

Every correction in the 6-7 October test logs, checked with the same
spell checker (pyspellchecker 0.9.0):

  helped                          damaged a correct reading
  Ycstcrday  -> yesterday         Glycolic -> glycol    (a cream's ingredient)
  Retund     -> refund            Glyco    -> glycol    (a brand name)
  classmale  -> classmate         Hazrat   -> hazard    (a place name)
  BROJECT    -> project           RivEST   -> rivets    (an author's name)
  PRQJECT    -> project
  LDITION    -> edition           wrong either way
  Crodit     -> credit            Hoighl   -> high      (it said "Height")
  Teachor    -> teacher           gmalcom  -> malcolm   (it said "gmail.com")
  PERFORMANGE -> performance      Glycolc  -> glycol    (it said "Glycolic")

Every correction that helped swapped letters that look alike, and kept
the word's length. Every damaging one did something else: dropped or
added letters (Glycolic -> glycol), or swapped letters that do not look
alike (Hazrat -> hazard). So a correction is now accepted only when it is
the same length and every changed letter is a look-alike pair.

On the words above, this keeps 9 helpful corrections and blocks all 7
damaging or wrong ones. It loses three small helpful ones - Telephong,
Creamn and Issuedho are now spoken as read - which are still
understandable; a wrong word stated confidently is not.

Tested by src/evaluation/check_spelling.py.
"""

# Pairs of letters OCR confuses because they look alike (compared in
# lower case). The first group was seen in testing; the second is a few
# well-known OCR confusions. Keep this list short: every pair added is
# another way for a correct word to be changed.
LOOK_ALIKE_PAIRS = [
    # Seen in the 6-7 October logs
    ("c", "e"),     # Ycstcrday -> yesterday
    ("o", "e"),     # Crodit -> credit, Teachor -> teacher
    ("t", "f"),     # Retund -> refund
    ("l", "t"),     # classmale -> classmate
    ("b", "p"),     # BROJECT -> project
    ("q", "o"),     # PRQJECT -> project
    ("l", "e"),     # LDITION -> edition (capital L and E)
    ("c", "g"),     # PERFORMANGE -> performance (capital C and G)

    # Common OCR confusions
    ("i", "l"), ("i", "j"), ("u", "v"), ("n", "h"),
]

_LOOK_ALIKE = {frozenset(pair) for pair in LOOK_ALIKE_PAIRS}


def is_look_alike_slip(original, correction):
    """True if correction differs from original only by look-alike letters.

    Both words are compared in lower case, letter by letter. Different
    lengths mean letters were added or removed, which is not how OCR
    misreads a single letter - so that is never accepted.
    """
    original, correction = original.lower(), correction.lower()
    if len(original) != len(correction) or original == correction:
        return False

    for a, b in zip(original, correction):
        if a != b and frozenset((a, b)) not in _LOOK_ALIKE:
            return False
    return True


def check_spelling(word, confidence, spell, below):
    """Decide whether to correct one OCR word.

    word       - one word as OCR read it
    confidence - OCR's confidence for the text it came from
    spell      - a SpellChecker, or None if spelling correction is off
    below      - only words read with less confidence than this are
                 checked; a confident reading such as BOLDFIT at 0.92 is
                 trusted as it is

    Returns (word_to_speak, suggestion, accepted):
      word_to_speak - the word to use
      suggestion    - what the spell checker proposed, or None if it was
                      not asked (confident reading, short word, known word)
      accepted      - True if the suggestion was used
    """
    if (spell is None or confidence >= below or not word.isalpha()
            or len(word) < 5):
        return word, None, False

    lower = word.lower()
    if spell.known([lower]):
        return word, None, False

    suggestion = spell.correction(lower)
    if not suggestion or suggestion == lower:
        return word, None, False

    if is_look_alike_slip(lower, suggestion):
        return suggestion, suggestion, True

    return word, suggestion, False