"""
Open-ended visual questions, answered by a local vision model.

Keyword commands (read, identify, describe) answer fixed questions
instantly. This handles everything else - "what am I wearing?", "what
colour is my bracelet?", "where are the objects?" - by sending the
current frame and the question to a vision-language model running
locally through Ollama.

MODEL CHOICE (tested 2-5 October):
  qwen2.5vl:3b      failed outright with a token-repeat error
  llava 7B          wrong shirt colour, missed a held phone, and
                    answered about an image it was never given
  moondream 1.8B    invented text on a bottle that was not there
  qwen3-vl:8b       accurate, but ~6 GB: sharing the 8 GB GPU with
                    detection, OCR and Whisper pushed it onto the CPU,
                    and a single answer took 95 seconds
  qwen3-vl:4b       accurate and fast, but this tag is the THINKING
                    variant: it always reasons before answering, the
                    reasoning used up the length limit and returned
                    empty answers, and it kept the GPU busy long enough
                    to make the video stutter
  qwen3-vl:4b-instruct
                    the same model without the reasoning step - the one
                    in use here

HINDI: asked to look at the image and answer in Hindi at the same time,
the 4B model became inaccurate - it invented objects ("a cooler near the
window", "traffic number boards") that were not there. Its English
answers to the same questions were correct. So Hindi answers are made in
two steps: the model answers in English, which it does accurately, and
that answer is then translated. English is the "pivot" language.

The translation is done by a separate, dedicated translation model
running locally - see src/language/translator.py. When the 4B vision
model translated its own answers it changed facts ("a red bracelet on the
right wrist" became "on the knee of the right hand"), and Google's free
translation endpoint refused every request. If translation is not
possible, the English answer is spoken instead: an accurate answer in
English is better than a wrong one in Hindi.

An earlier version put an example Hindi sentence in the prompt to show
the Hinglish style. The model copied it into every answer, whatever the
question - so no examples are given.

HONESTY: the model sometimes states details it is not certain of. Answers
are spoken as "It looks like..." rather than as fact, and the prompt asks
it to admit uncertainty.
"""

import threading
import time

import cv2
import ollama

MODEL = "qwen3-vl:4b-instruct"

# Always answers in English - see the note on Hindi above.
#
# "Address the user as you": the camera faces the person asking, and
# without this the model described them as "he" or "the person". Its
# translation then became "उसने" or "लोग ने" instead of "आपने".
#
# "Answer only what was asked": asked the colour of the wall, it also
# described the user's clothes.
#
# LEFT AND RIGHT (22 October). Because the camera faces the user, the
# picture is a mirror of their own point of view: things on the left of
# the picture are on the user's right. Tested with things whose side was
# known, the model did two different things:
#   - objects: it gave the side as it appears in the PICTURE, consistently
#     - a backpack on the user's right was "to your left", twice
#   - hands: it said "your right hand" seven times out of seven, whichever
#     hand was raised - a guess, not a reading of the picture
# A prompt explaining the mirror was tried first and failed its test
# (3 of 7 right, 4 wrong). So the prompt asks for object sides as they
# appear in the picture - what the model does anyway - and
# correct_left_right() below turns them into the user's own. The side of
# a hand is never taken from the model: it comes from MediaPipe's hand
# model (src/vision/hands.py), which labelled both hands correctly in 15
# of 15 frames each.
#
# "speaking to the person as 'you'": examples written as "in their hand"
# were copied word for word, and the user was spoken about as "them".
VISION_PROMPT = (
    "You are helping a blind person understand what a camera sees. The "
    "camera is pointed at the person asking, so address them as 'you'. "
    "Describe only what is actually visible. If you cannot tell, say so "
    "plainly. Answer only what was asked, in one or two short sentences, "
    "with no preamble. The question may be in Hindi; always answer in "
    "English. "
    "When you say where something is, give left or right as it appears in "
    "the picture. Speak to the person as 'you' - for example 'in your "
    "hand', 'behind you'."
)

# True while the camera faces the user, as with a laptop webcam
# (CAMERA_FACES_USER in main.py): picture sides are then flipped into the
# user's own. For a camera worn facing outward, the picture's left IS the
# user's left - set this to False, and the prompt above needs reviewing.
FLIP_LEFT_RIGHT = True

_SIDE = r"(?:left|right)"

# Sides of the body. The model guesses these, so its side is always taken
# out - "your right wrist" -> "your wrist" - and the side of a hand is put
# back from MediaPipe instead (see add_hand_side). "left-hand side" is a
# position, not a hand, and is left alone.
_BODY_SIDE = [
    (rf"\b(?:your|the|their|his|her) {_SIDE}[- ]?"
     rf"(hand|wrist|arm|shoulder|ear|eye|leg|foot|fingers?)\b(?![- ]side)", r"your \1"),
    (rf"\b{_SIDE}[- ]?(hand|wrist|arm|shoulder)\b(?![- ]side)", r"\1"),
]

# Asides that only explain the picture - "(left side of the image)",
# ", which appears on the left side of the image from the camera's
# perspective". They follow a hand or an object already described, so
# they are dropped whole. A position stated plainly - "the bottle is on
# the left side of the image" - is NOT an aside: it is kept, and turned
# into the user's own side (see _PICTURE_POSITION).
_PICTURE_ASIDES = [
    rf"\s*\((?:on )?(?:the )?{_SIDE}(?: side)? of the (?:image|picture|frame)\)",
    rf",?\s*(?:as it is|which is|which|that|as|it)?\s*(?:visible|appears|shown|seen)\s*"
    rf"on the {_SIDE}(?: side)? of the (?:image|picture|frame)"
    rf"(?: from the camera['’]s (?:perspective|point of view|view))?",
]

# Positions, flipped from the picture's side to the user's. Only "left"
# and "right" inside a position are touched - "to your left", "on the
# right side", "left of the desk", "far right", "top-left" - never "that's
# right" or "right next to". One pattern, applied in a single pass, so a
# phrase such as "on the left side" is flipped once and not back again.
_POSITION = (
    r"\b(?:to|on|at|in|towards|from) (?:your|the|their|my) "
    r"(?:(?:far|upper|lower|top|bottom) )?(?P<a>left|right)\b"
    r"|\b(?:top|bottom|upper|lower|far)[- ](?P<b>left|right)\b"
    r"|\b(?P<c>left|right)(?=[- ](?:hand[- ])?side\b| of\b| corner\b| edge\b)"
)

# After flipping, "on the right side of the image" means "on your right":
# said in the user's terms, which is what the user can act on
_PICTURE_POSITION = (rf"\bon the (?P<side>left|right)(?: side)? of the "
                     rf"(?:image|picture|frame)\b")


def correct_left_right(answer, hand_sides=None, flip=FLIP_LEFT_RIGHT):
    """Make every left and right in an answer the user's own.

    hand_sides - what MediaPipe saw, e.g. ["right"] (see hands.py)

    Takes out the side of a hand or other body part (the model guesses
    it), drops asides that only explain the picture, flips positions from
    the picture's side to the user's when the camera faces them, and puts
    a hand's side back - from MediaPipe, and only when exactly one hand is
    in view. Each change is printed, so it can be checked in the log.
    """
    import re
    cleaned = answer
    for pattern, replacement in _BODY_SIDE:
        cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)
    for pattern in _PICTURE_ASIDES:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)

    if flip:
        def swap(match):
            name = next(n for n in ("a", "b", "c") if match.group(n))
            side = match.group(name)
            new = "right" if side.lower() == "left" else "left"
            if side[0].isupper():
                new = new.capitalize()
            start = match.start(name) - match.start()
            text = match.group(0)
            return text[:start] + new + text[start + len(side):]
        cleaned = re.sub(_POSITION, swap, cleaned, flags=re.IGNORECASE)

    cleaned = re.sub(_PICTURE_POSITION,
                     lambda m: f"on your {m.group('side').lower()}",
                     cleaned, flags=re.IGNORECASE)

    cleaned = add_hand_side(cleaned, hand_sides)

    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    cleaned = re.sub(r"\s+([.,;!?])", r"\1", cleaned).strip()
    if cleaned != answer:
        print(f"  [vision] sides made yours: {answer!r}")
    return cleaned


def add_hand_side(text, hand_sides):
    """Put the side of the hand back, from MediaPipe.

    Only when exactly ONE hand is in view: with two, "your hand" could be
    either, and with none, MediaPipe has nothing to say. "your hand",
    "a hand" and "one hand" all become "your right hand" (or left).
    """
    import re
    if not hand_sides or len(hand_sides) != 1:
        return text
    side = hand_sides[0]
    return re.sub(r"\b(?:your|a|one) hand\b(?![- ]side)", f"your {side} hand",
                  text, flags=re.IGNORECASE)

# A question that has not been answered within this many seconds is
# abandoned. Without a limit, one slow answer kept the model "busy", and
# every question after it was refused - five in a row in one test.
TIMEOUT_SECONDS = 30

# The first question after a pause is much slower: Ollama unloads the
# model from GPU memory after about 5 idle minutes, and loading 3.5 GB
# back takes time - with detection, OCR and speech recognition already
# using the same GPU. On 7 October the first question of a session failed
# with "timed out" at 30 s. The first request therefore gets its own,
# longer limit, and KEEP_ALIVE then holds the model in memory so the
# questions after it stay fast.
COLD_TIMEOUT_SECONDS = 120

# How long Ollama keeps the model loaded after a question. Longer than
# its 5-minute default, so a demonstration or a conversation does not
# pay the loading cost again halfway through. The cost is that the model
# holds about 3.5 GB of GPU memory for this long after the last question.
KEEP_ALIVE = "20m"

# Maximum length of an answer, in tokens. Two short sentences fit easily.
MAX_ANSWER_TOKENS = 200


class VisionAsker:
    def __init__(self, model=MODEL, translator=None):
        """translator - turns English answers into Hindi; see
        src/language/translator.py. Without one, Hindi mode speaks the
        English answer."""
        self.model = model
        self.translator = translator
        self.busy = threading.Event()
        self.client = ollama.Client(timeout=TIMEOUT_SECONDS)

        # Used until the model has answered once - see COLD_TIMEOUT_SECONDS
        self.cold_client = ollama.Client(timeout=COLD_TIMEOUT_SECONDS)
        self.model_is_loaded = False

        # The side of a hand, from MediaPipe - see hands.py. Optional:
        # without it, answers simply give no side for a hand.
        from vision.hands import HandSides
        self.hands = HandSides()

        # Fail here, at startup, rather than when the user first asks:
        # the application can then report that the feature is unavailable
        # and carry on without it. Checking only that Ollama was running
        # was not enough - with the model missing, the application
        # started normally and failed on every question.
        installed = self._installed_models()
        if model not in installed:
            raise RuntimeError(
                f"the vision model '{model}' is not installed. "
                f"Install it with:  ollama pull {model}"
            )

    def _installed_models(self):
        """Names of the models Ollama has downloaded.

        Newer versions of the ollama package return objects, older ones
        dictionaries, so both are handled.
        """
        response = self.client.list()
        models = getattr(response, "models", None)
        if models is None:
            models = response.get("models", [])

        names = set()
        for entry in models:
            if isinstance(entry, dict):
                name = entry.get("model") or entry.get("name")
            else:
                name = getattr(entry, "model", None)
            if name:
                names.add(name)
        return names

    def _translate_to_hindi(self, english):
        """Translate an answer into Hindi, or return None if that fails."""
        if self.translator is None:
            print("  [vision] no translator - speaking English")
            return None
        try:
            return self.translator.translate(english)
        except Exception as error:
            print(f"  [vision] translation failed, speaking English: {error}")
            return None

    def _chat(self, content, images=None, max_tokens=MAX_ANSWER_TOKENS):
        """One request to the model; returns its reply as plain text."""
        message = {"role": "user", "content": content}
        if images:
            message["images"] = images

        # A cold model may take a minute to load; once it has answered,
        # the shorter limit applies, so a real failure is reported quickly
        client = self.client if self.model_is_loaded else self.cold_client

        started = time.time()
        response = client.chat(
            model=self.model,
            messages=[message],
            options={"num_predict": max_tokens},
            keep_alive=KEEP_ALIVE,
        )
        if not self.model_is_loaded:
            self.model_is_loaded = True
            print(f"  [vision] model loaded and answered in "
                  f"{time.time() - started:.1f} s")

        answer = response["message"]["content"].strip()

        # Safety net, in case reasoning text ever arrives
        if "</think>" in answer:
            answer = answer.split("</think>")[-1].strip()

        return answer

    def ask_in_background(self, frame, question, on_answer, language="en"):
        """Send a frame and question to the model without blocking the loop.

        language - "en" or "hi": the language the answer should be in.

        on_answer is called with (answer, None) on success, or
        (None, problem) when something went wrong. The problem is a short
        code - "busy", "capture", "empty" or "no_response" - rather than
        an English sentence, so the caller can say it in either language.
        """
        if self.busy.is_set():
            on_answer(None, "busy")
            return

        self.busy.set()

        # The main loop replaces the frame constantly, so encode a
        # snapshot of this moment before handing it to the thread
        success, buffer = cv2.imencode(".jpg", frame)
        if not success:
            self.busy.clear()
            on_answer(None, "capture")
            return

        image_bytes = buffer.tobytes()
        snapshot = frame.copy()        # for the hand check, same moment

        def work():
            try:
                # Step 1: look at the image and answer, always in English
                answer = self._chat(f"{VISION_PROMPT}\n\n{question}",
                                    images=[image_bytes])
                if not answer:
                    on_answer(None, "empty")
                    return

                # Left and right from the user's point of view - see
                # VISION_PROMPT. Done on the English answer, before
                # translation, so Hindi follows.
                hand_sides = self.hands.sides(snapshot)
                if hand_sides:
                    print(f"  [hands] in view: {', '.join(hand_sides)}")
                answer = correct_left_right(answer, hand_sides)

                # Step 2, Hindi mode only: translate the English answer.
                # If translation is unavailable, the English answer is
                # spoken - accurate English beats inaccurate Hindi.
                if language == "hi":
                    print(f"  [vision] English answer: {answer}")
                    hindi = self._translate_to_hindi(answer)
                    if hindi:
                        answer = hindi

                on_answer(answer, None)

            except Exception as error:
                print(f"Vision model error: {error}")
                on_answer(None, "no_response")
            finally:
                self.busy.clear()

        threading.Thread(target=work, daemon=True).start()