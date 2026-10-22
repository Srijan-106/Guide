"""
How often does Whisper mishear Hindi - and could the app tell?

Before changing anything about voice input, this measures two things:

  1. How many Hindi sentences Whisper hears wrongly, in this room, with
     this microphone.
  2. Whether Whisper's own confidence is LOWER when it hears wrongly. If
     it is, the app can say "I did not catch that, please say it again"
     instead of answering a misheard question - which is what produced
     the random answer on 22 October ('यह बाट कर' for "मैंने यहाँ क्या
     पहना है"). If confidence is the same either way, that fix would not
     work, and a different one is needed.

Whisper is loaded and called exactly as the application does it
(src/voice/listener.py): the same model size, vocabulary prompt, beam size
and settings. Only the confidence numbers are kept, which the application
currently discards.

The list of sentences is fixed below, before testing - the same rule as
the other evaluations. Close the application first: this loads Whisper on
the GPU by itself.

    python src\\evaluation\\check_hindi_hearing.py            record
    python src\\evaluation\\check_hindi_hearing.py --from 3   start at no. 3
    python src\\evaluation\\check_hindi_hearing.py --summary  results only

Results are added to evaluation/hindi_hearing.csv after every sentence, so
running it twice gives twice as much data, and a run stopped halfway can
continue with --from.

THE TERMINAL AND HINDI: the VS Code terminal draws each character on its
own and cannot join Devanagari, so vowel signs land in the wrong place and
the text is hard to read. Every sentence, and what Whisper heard, is
therefore also shown in Roman letters. A hearing that matches the sentence
exactly (ignoring punctuation) is marked right automatically; only the
others are left for a person to judge.
"""

import csv
import datetime
import os
import sys
import unicodedata

sys.path.insert(0, "src")          # find the app's modules, as main.py does

import numpy as np

RESULTS = os.path.join("evaluation", "hindi_hearing.csv")

# Fixed before testing. The first seven are questions for the vision
# model; the last three are commands. Number 5 is the one misheard on
# 22 October. Each has the way it is said, in Roman letters, beside it.
SENTENCES = [
    ("मैंने क्या पहना है?",        "maine kya pehna hai?"),
    ("मेरी शर्ट किस रंग की है?",   "meri shirt kis rang ki hai?"),
    ("मेरे हाथ में क्या है?",      "mere haath mein kya hai?"),
    ("क्या यहाँ कोई कुत्ता है?",    "kya yahan koi kutta hai?"),
    ("मैंने यहाँ क्या पहना है?",   "maine yahan kya pehna hai?"),
    ("मेरा बैग कहाँ है?",          "mera bag kahan hai?"),
    ("दीवार किस रंग की है?",       "deewar kis rang ki hai?"),
    ("आसपास क्या दिख रहा है?",     "aaspaas kya dikh raha hai?"),
    ("फिर से बोलो",                "phir se bolo"),
    ("यह क्या है?",                "yeh kya hai?"),
]

# --- Devanagari in Roman letters, for reading in the terminal ----------
# A simple letter-by-letter conversion, good enough to read and judge what
# Whisper heard - not a full transliteration (it keeps vowels Hindi drops
# in speech, so पहना reads "pahanaa", not "pehna").
_VOWELS = {"अ": "a", "आ": "aa", "इ": "i", "ई": "ee", "उ": "u", "ऊ": "oo",
           "ऋ": "ri", "ए": "e", "ऐ": "ai", "ओ": "o", "औ": "au",
           "ऍ": "e", "ऑ": "o"}
_SIGNS = {"ा": "aa", "ि": "i", "ी": "ee", "ु": "u", "ू": "oo", "ृ": "ri",
          "े": "e", "ै": "ai", "ो": "o", "ौ": "au", "ॅ": "e", "ॉ": "o"}
_CONSONANTS = {"क": "k", "ख": "kh", "ग": "g", "घ": "gh", "ङ": "ng",
               "च": "ch", "छ": "chh", "ज": "j", "झ": "jh", "ञ": "ny",
               "ट": "t", "ठ": "th", "ड": "d", "ढ": "dh", "ण": "n",
               "त": "t", "थ": "th", "द": "d", "ध": "dh", "न": "n",
               "प": "p", "फ": "ph", "ब": "b", "भ": "bh", "म": "m",
               "य": "y", "र": "r", "ल": "l", "व": "v", "श": "sh",
               "ष": "sh", "स": "s", "ह": "h"}
# A consonant with the nukta dot below it is a different sound
_WITH_NUKTA = {"क": "q", "ख": "kh", "ग": "g", "ज": "z", "ड": "r",
               "ढ": "rh", "फ": "f"}
_OTHER = {"ं": "n", "ँ": "n", "ः": "h", "।": ".", "॥": "."}


def roman(text):
    """Write Devanagari text in Roman letters. Anything else is unchanged."""
    text = unicodedata.normalize("NFD", text)   # क़ -> क + nukta
    out = []
    i = 0
    while i < len(text):
        ch = text[i]
        if ch in _CONSONANTS:
            sound = _CONSONANTS[ch]
            if i + 1 < len(text) and text[i + 1] == "़":     # nukta
                sound = _WITH_NUKTA.get(ch, sound)
                i += 1
            nxt = text[i + 1] if i + 1 < len(text) else ""
            if nxt in _SIGNS:
                out.append(sound + _SIGNS[nxt])
                i += 1
            elif nxt == "्":                                # virama
                out.append(sound)
                i += 1
            elif nxt and (nxt in _CONSONANTS or nxt in "ंँः"
                          or unicodedata.category(nxt).startswith("M")):
                out.append(sound + "a")
            else:
                out.append(sound)        # end of a word: no final "a"
        elif ch in _VOWELS:
            out.append(_VOWELS[ch])
        elif ch in _OTHER:
            out.append(_OTHER[ch])
        else:
            out.append(ch)
        i += 1
    return "".join(out)


def same_words(said, heard):
    """True if heard is the sentence exactly, ignoring punctuation, the
    nukta dot, and the difference between ँ and ं (यहाँ / यहां)."""
    from voice.listener import normalise
    def clean(text):
        return normalise(text).replace("ँ", "ं").replace("?", "")
    return clean(said) == clean(heard)

COLUMNS = ["timestamp", "said", "heard", "verdict", "avg_logprob",
           "worst_logprob", "no_speech_prob", "compression_ratio",
           "seconds_of_audio"]

VERDICTS = {"1": "right", "2": "spelling only", "3": "wrong"}


def transcribe_with_confidence(listener, audio):
    """Transcribe exactly as VoiceListener.transcribe() does, but also
    return Whisper's confidence numbers.

    avg_logprob   - how sure Whisper was of the words it chose: 0 is
                    certain, more negative is less sure. Averaged over
                    the segments, weighted by length.
    worst_logprob - the least sure segment
    no_speech_prob - how likely Whisper thought the clip was silence
    compression_ratio - high when the text repeats itself, a sign of
                    Whisper looping
    """
    from voice.listener import (BEAM_SIZE, VOCABULARY_PROMPTS,
                                collapse_repeats)

    peak = float(np.abs(audio).max())
    if peak < 0.01:
        return "", None
    audio = audio / peak * 0.7                     # as the app does

    result = listener.model.transcribe(           # same arguments as the app
        audio,
        language="hi",
        fp16=True,
        temperature=0,
        beam_size=BEAM_SIZE,
        initial_prompt=VOCABULARY_PROMPTS.get("hi"),
        no_speech_threshold=0.6,
        condition_on_previous_text=False,
    )
    text = collapse_repeats(result["text"].strip())

    segments = result.get("segments") or []
    if not segments:
        return text, None
    lengths = [max(len(s["tokens"]), 1) for s in segments]
    confidence = {
        "avg_logprob": sum(s["avg_logprob"] * n
                           for s, n in zip(segments, lengths)) / sum(lengths),
        "worst_logprob": min(s["avg_logprob"] for s in segments),
        "no_speech_prob": max(s["no_speech_prob"] for s in segments),
        "compression_ratio": max(s["compression_ratio"] for s in segments),
    }
    return text, confidence


def record(listener):
    """Record one sentence: Enter to start, Enter to stop."""
    input("    Press Enter, then say it...")
    listener.start_recording()
    input("    ...and press Enter when you have finished.")
    return listener.stop_recording()


def save(row):
    new_file = not os.path.exists(RESULTS)
    with open(RESULTS, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def run_test(start=1):
    from voice.listener import VoiceListener

    print(__doc__.split("\n\n")[1])
    listener = VoiceListener()
    print(f"\n{len(SENTENCES)} sentences. Say each one once, at your normal "
          f"speed and volume - the way you speak to the app.\n")

    try:
        for number, (sentence, spoken) in enumerate(SENTENCES, 1):
            if number < start:
                continue
            print(f"[{number}/{len(SENTENCES)}]  SAY:  {spoken}")
            print(f"            ({sentence})")
            while True:
                audio = record(listener)
                if audio is not None:
                    break
                print("    Nothing recorded (too short). Try again.")

            text, confidence = transcribe_with_confidence(listener, audio)
            print(f"    heard:  {roman(text)}")
            print(f"            ({text})")
            if confidence:
                print(f"    confidence: {confidence['avg_logprob']:.2f}  "
                      f"(0 = certain, more negative = less sure)")

            if same_words(sentence, text):
                answer = "1"
                print("    Exactly the sentence - marked right.")
            else:
                print(f"    expected: {roman(sentence)}")
                answer = ""
            while answer not in VERDICTS:
                answer = input("    Was it heard right?  1 = right   "
                               "2 = only spelling differs   3 = wrong:  ").strip()

            row = {"timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
                   "said": sentence, "heard": text,
                   "verdict": VERDICTS[answer],
                   "seconds_of_audio": f"{len(audio) / 16000:.1f}"}
            for key in COLUMNS[4:8]:
                row[key] = f"{confidence[key]:.3f}" if confidence else ""
            save(row)
            print()
    finally:
        listener.close()

    summarise()


def summarise():
    """Report the results, and whether a confidence cut-off would help."""
    if not os.path.exists(RESULTS):
        print(f"No results yet: {RESULTS} does not exist.")
        return
    with open(RESULTS, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print("No results yet.")
        return

    understood = [r for r in rows if r["verdict"] in ("right", "spelling only")]
    wrong = [r for r in rows if r["verdict"] == "wrong"]
    print("=" * 64)
    print(f"HEARD RIGHT: {len(understood)}/{len(rows)} "
          f"= {100 * len(understood) / len(rows):.0f}%   "
          f"(spelling-only differences count as right)")
    print("=" * 64)

    if wrong:
        print("\nMisheard:")
        for r in wrong:
            print(f"  said  {roman(r['said'])}\n  heard {roman(r['heard'])}   "
                  f"confidence {r['avg_logprob'] or '-'}\n")

    scored = [(float(r["avg_logprob"]), r["verdict"] == "wrong")
              for r in rows if r["avg_logprob"]]
    if not wrong or not understood or len(scored) < len(rows):
        print("\nNeed both right and wrong hearings, with confidence, to test a "
              "cut-off.")
        return

    print("\nEvery hearing, least sure first:")
    for value, is_wrong in sorted(scored):
        print(f"  {value:6.2f}  {'WRONG' if is_wrong else 'right'}")

    # A cut-off rejects everything below it: the app would say "please
    # say that again" instead of answering. Good cut-offs catch many wrong
    # hearings while rejecting few right ones.
    print("\nIf the app asked to repeat below a cut-off:")
    print("  cut-off   wrong caught   right ones asked to repeat")
    for cut in sorted({round(v, 2) for v, _ in scored} | {0.0}):
        caught = sum(1 for v, w in scored if w and v < cut)
        annoyed = sum(1 for v, w in scored if not w and v < cut)
        print(f"  {cut:6.2f}    {caught}/{len(wrong)}             "
              f"{annoyed}/{len(understood)}")

    print("\nA useful cut-off catches most WRONG hearings while asking the user "
          "to repeat few RIGHT ones. If no line does that, confidence does not "
          "separate them, and a different fix is needed. Read the whole table "
          "before deciding - with 10 or 20 sentences, it is a first look, not "
          "a proof.")


if __name__ == "__main__":
    if "--summary" in sys.argv:
        summarise()
    elif "--from" in sys.argv:
        run_test(start=int(sys.argv[sys.argv.index("--from") + 1]))
    else:
        run_test()