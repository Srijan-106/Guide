"""
Voice commands for VisionAssist, using OpenAI Whisper locally.

Whisper runs on the GPU, offline and free. It is markedly more accurate
than the small offline recognisers, especially with accents and
background noise, which matters when a wrong command does the wrong
thing.

Push-to-talk: V starts recording and V again stops it, so there is no
need to detect when speech has ended - a source of errors in
always-listening designs.

Commands are matched by keyword rather than exact phrase, so "can you
read this please" still matches "read". Anything that matches no command
is passed on as a question for the vision model.
"""

import queue
import threading

import numpy as np
import sounddevice as sd
import whisper

# Whisper expects audio at 16 kHz
SAMPLE_RATE = 16000

# Model sizes: tiny, base, small, medium, large. "base" was accurate on
# all five test commands; move up if commands are misheard.
MODEL_SIZE = "base"

# Recordings shorter than this are treated as an accidental key press
MIN_SECONDS = 0.4

# Phrases Whisper produces from near-silence or a cut-off clip. It was
# trained partly on video subtitles, where these are common closing
# lines - "subscribe" was transcribed twice in one test session when
# nothing of the sort was said. None is a plausible request here.
HALLUCINATIONS = {
    "subscribe", "please subscribe", "thank you", "thanks",
    "thank you for watching", "thanks for watching", "you", "bye",
}


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

    def transcribe(self, audio):
        """Turn recorded audio into text. Runs on the GPU when available.

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
            language="en",
            fp16=True,
            temperature=0,
            # Return nothing, rather than an invented phrase, when the
            # clip does not contain speech
            no_speech_threshold=0.6,
            condition_on_previous_text=False,
        )
        text = result["text"].strip()

        if text.lower().strip(" .,!?") in HALLUCINATIONS:
            print(f"  [voice] ignored a likely hallucination: {text!r}")
            return ""

        return text

    def transcribe_in_background(self, audio, on_done):
        """Transcribe without blocking the video loop."""
        def work():
            try:
                on_done(self.transcribe(audio))
            except Exception as error:
                print(f"Transcription error: {error}")
                on_done(None)

        threading.Thread(target=work, daemon=True).start()


# ---------------------------------------------------------------------
# Matching spoken words to commands
# ---------------------------------------------------------------------

# Each command lists the words that trigger it. Order matters: the first
# match wins, so more specific phrases are listed before general ones.
COMMANDS = [
    ("reading_mode",  ["reading mode", "read mode"]),
    ("ambient_mode",  ["ambient mode", "ambient"]),
    ("identify",      ["what is this", "what's this", "identify", "what am i holding"]),
    ("describe",      ["describe", "what do you see", "what is around", "what's around"]),
    ("read",          ["read"]),
    ("repeat",        ["repeat", "again", "say that"]),
    ("quit",          ["stop", "quit", "exit", "close"]),
]


def match_command(text):
    """Return the command name for spoken text, or None.

    Matching is by keyword rather than exact phrase: "can you read this
    please" contains "read" and so matches. Exact matching would fail on
    almost everything a person actually says.
    """
    if not text:
        return None

    spoken = text.lower().strip(" .,!?")

    for name, phrases in COMMANDS:
        if any(phrase in spoken for phrase in phrases):
            return name

    return None