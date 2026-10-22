"""
Voice commands for Guide, using OpenAI Whisper locally.

Whisper runs on the GPU, offline and free, and understands both English
and Hindi. Push-to-talk: V starts recording and V again stops it, so
there is no need to detect when speech has ended.

Commands are matched by keyword rather than exact phrase, so "can you
read this please" still matches "read". Anything that matches no command
is passed on as a question for the vision model.

HINDI: Whisper is told explicitly which language to expect. Spoken Hindi
and Urdu are nearly identical, and when left to detect the language
itself Whisper often writes Hindi speech in Urdu script - which no Hindi
keyword would ever match.
"""

import queue
import re
import threading
import time
import unicodedata

import numpy as np
import sounddevice as sd
import whisper

# Whisper expects audio at 16 kHz
SAMPLE_RATE = 16000

# Model sizes: tiny, base, small, medium, large. "base" struggled with an
# Indian English accent unless the speaker talked loudly and slowly;
# "small" is roughly three times larger, much better with accents, and
# transcribed commands in 0.4-0.6 seconds.
MODEL_SIZE = "small"

# Recordings shorter than this are treated as an accidental key press
MIN_SECONDS = 0.4

# Told to Whisper before it listens, so that when a word is unclear it
# leans towards the words this application actually uses. This matters
# most for accents the model saw less of in training.
#
# The prompt can also be COPIED: when Whisper is unsure, it may borrow
# words from it. On 7 October "फिर से बोलो" was transcribed as "मेरा
# ब्रेसलेट करो" - "मेरा ब्रेसलेट" taken straight from the example question
# that used to end the Hindi prompt. So the Hindi prompt now holds every
# command phrase the user is likely to say, and only one example question,
# with no specific object in it. (The English prompt scored 7/7 in the
# same tests and is unchanged.)
#
# A colour question was added on 22 October. In 20 test sentences
# (src/evaluation/check_hindi_hearing.py), "किस रंग की" was misheard all
# four times it was said - as "की सरंका", "रहंका" and twice "रन्की" -
# while the other sentences were heard correctly. It has no object in it,
# for the reason above.
VOCABULARY_PROMPTS = {
    "en": (
        "Commands: read this, what is this, identify, describe, repeat, "
        "reading mode, ambient mode, Hindi, stop. Questions such as: what "
        "am I wearing, what colour is my bracelet, what does the label say."
    ),
    "hi": (
        "कमांड: पढ़ो, यह क्या है, पहचानो, आसपास क्या दिख रहा है, "
        "फिर से बोलो, दोबारा, पढ़ने का मोड, देखने का मोड, इंग्लिश, "
        "बंद करो। सवाल जैसे: मैंने क्या पहना है, यह किस रंग की है।"
    ),
}

# Whisper normally takes its first guess at each word. Beam search keeps
# several candidate transcriptions and picks the most likely overall -
# slower, but more accurate on unclear speech.
BEAM_SIZE = 5

# Phrases Whisper produces from near-silence or a cut-off clip. It was
# trained partly on video subtitles, where these are common closing
# lines - "subscribe" was transcribed twice in one test session when
# nothing of the sort was said. None is a plausible request here.
HALLUCINATIONS = {
    "subscribe", "please subscribe", "thank you", "thanks",
    "thank you for watching", "thanks for watching", "you", "bye",
    "धन्यवाद", "शुक्रिया", "सब्सक्राइब करें", "सब्सक्राइब",
}


def collapse_repeats(text):
    """Reduce a phrase Whisper has repeated to a single copy.

    With beam search, Whisper sometimes loops on a phrase - "What am I
    wearing" was transcribed five times over from one question. If every
    part of the text is the same phrase, keep just one.
    """
    parts = [part.strip() for part in re.split(r"[,.!?।]+", text)
             if part.strip()]
    if len(parts) > 1 and len({part.lower() for part in parts}) == 1:
        return parts[0]
    return text


def normalise(text):
    """Make spoken text comparable, whatever the spelling variant.

    Lower-cases English, removes punctuation (including the Devanagari
    full stop "।"), and removes the nukta dot - Whisper may write पढ़ो or
    पढो, ज़रा or जरा, and both should match the same keyword.

    Also joins "आस पास" into "आसपास": both spellings are common, and on
    7 October Whisper wrote it as two words, so no keyword matched.
    """
    text = unicodedata.normalize("NFD", text.lower())
    text = text.replace("़", "")          # the nukta dot
    for mark in ".,!?।":
        text = text.replace(mark, " ")
    text = " ".join(text.split())
    return text.replace("आस पास", "आसपास")


class VoiceListener:
    def __init__(self, model_size=MODEL_SIZE):
        print(f"Loading speech model ({model_size})...")
        self.model = whisper.load_model(model_size)

        self._frames = queue.Queue()
        self.recording = False

        # The microphone is opened once and left running; audio is only
        # kept while recording. Opening a stream takes a noticeable
        # fraction of a second, and opening it on each key press meant
        # the first word of a command was often lost - "colour of my
        # bracelet" arrived as "of my bracelet".
        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype="float32",
            callback=self._on_audio,
        )
        self._stream.start()

    # -----------------------------------------------------------------
    # Recording
    # -----------------------------------------------------------------

    def _on_audio(self, indata, frames, time_info, status):
        """Called by the audio driver for every block of samples."""
        if self.recording:
            # copy() because the buffer is reused by the audio driver
            self._frames.put(indata.copy())

    def start_recording(self):
        """Begin keeping audio. Called when V is pressed."""
        if self.recording:
            return
        self._frames = queue.Queue()
        self.recording = True

    def stop_recording(self):
        """Stop keeping audio and return what was recorded."""
        if not self.recording:
            return None

        self.recording = False

        chunks = []
        while not self._frames.empty():
            chunks.append(self._frames.get())

        if not chunks:
            return None

        audio = np.concatenate(chunks, axis=0).flatten()
        if len(audio) < SAMPLE_RATE * MIN_SECONDS:
            return None                      # too short to be speech

        return audio

    def close(self):
        """Release the microphone. Called when the application exits."""
        try:
            self._stream.stop()
            self._stream.close()
        except Exception:
            pass

    # -----------------------------------------------------------------
    # Transcription
    # -----------------------------------------------------------------

    def transcribe(self, audio, language="en"):
        """Turn recorded audio into text, in the given language.

        The audio is normalised first: a clipping microphone flattens the
        waveform and destroys the detail Whisper needs, which makes it
        hallucinate plausible-sounding subtitle phrases.
        """
        peak = float(np.abs(audio).max())

        if peak < 0.01:
            print("  [voice] almost silent - nothing to transcribe")
            return ""

        # Scale to a consistent level, leaving headroom below maximum
        audio = audio / peak * 0.7

        result = self.model.transcribe(
            audio,
            language=language,
            fp16=True,
            temperature=0,
            beam_size=BEAM_SIZE,
            initial_prompt=VOCABULARY_PROMPTS.get(language),
            # Return nothing, rather than an invented phrase, when the
            # clip does not contain speech
            no_speech_threshold=0.6,
            condition_on_previous_text=False,
        )
        text = collapse_repeats(result["text"].strip())

        if normalise(text) in {normalise(h) for h in HALLUCINATIONS}:
            print(f"  [voice] ignored a likely hallucination: {text!r}")
            return ""

        return text

    def transcribe_in_background(self, audio, on_done, language="en"):
        """Transcribe without blocking the video loop."""
        def work():
            try:
                started = time.time()
                text = self.transcribe(audio, language)
                # Timed, so the cost of a larger model is measured
                # rather than guessed
                print(f"  [voice] transcribed in {time.time() - started:.1f} s")
                on_done(text)
            except Exception as error:
                print(f"Transcription error: {error}")
                on_done(None)

        threading.Thread(target=work, daemon=True).start()


# ---------------------------------------------------------------------
# Matching spoken words to commands
# ---------------------------------------------------------------------

# Each command lists the words that trigger it. Order matters: the first
# match wins, so more specific phrases come before general ones -
# "reading mode" must be checked before "read".
#
# Language switching is listed in both, so either language can be
# reached from the other by voice.

COMMANDS = {
    "en": [
        ("language_hindi", ["hindi"]),
        ("reading_mode",   ["reading mode", "read mode"]),
        ("ambient_mode",   ["ambient mode", "ambient"]),
        ("identify",       ["what is this", "what's this", "identify", "what am i holding"]),
        ("describe",       ["describe", "what do you see", "what is around", "what's around"]),
        ("read",           ["read"]),
        ("repeat",         ["repeat", "again", "say that"]),
        ("quit",           ["stop", "quit", "exit", "close"]),
    ],
    "hi": [
        ("language_english", ["इंग्लिश", "अंग्रेजी", "अंग्रेज़ी", "english"]),
        ("reading_mode",     ["पढ़ने का मोड", "पढ़ने वाला मोड", "रीडिंग मोड"]),
        ("ambient_mode",     ["देखने का मोड", "देखने वाला मोड", "एम्बिएंट"]),
        ("identify",         ["यह क्या है", "ये क्या है", "क्या है यह", "क्या है ये",
                              "पहचानो", "पहचानिए", "पहचान"]),
        # "आसपास का" added 7 October: Whisper heard "आसपास क्या दिख रहा है"
        # as "आस पास का दिख रहा है" - "क्या" shortened to "का"
        ("describe",         ["क्या दिख रहा", "आसपास क्या", "आसपास का",
                              "डिस्क्राइब"]),
        ("read",             ["पढ़ो", "पढ़िए", "पढ़ के", "पढ़कर", "रीड"]),
        ("repeat",           ["दोबारा", "फिर से", "दोहराओ", "रिपीट"]),
        ("quit",             ["बंद करो", "बंद कर दो", "रुको", "स्टॉप"]),
    ],
}


def match_command(text, language="en"):
    """Return the command name for spoken text, or None.

    Matching is by keyword rather than exact phrase: "can you read this
    please" contains "read" and so matches. Both the spoken text and the
    keywords are normalised, so spelling variants still match.

    WHOLE WORDS ONLY (22 October). A keyword used to match inside another
    word: "which hand am I raising", heard as "...reading", contained
    "read" and switched to Reading mode - twice in one session - and "is
    the door closed?" contained "close", which quits the application. A
    keyword now matches only as whole words, so "reading" no longer
    contains "read". Hindi forms that were matched as part of a longer
    word are listed in full instead ("पहचानिए").
    """
    if not text:
        return None

    # Spaces at both ends, so " read " is found only as a whole word
    spoken = f" {normalise(text)} "

    for name, phrases in COMMANDS.get(language, COMMANDS["en"]):
        if any(f" {normalise(phrase)} " in spoken for phrase in phrases):
            return name

    return None