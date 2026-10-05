"""
VisionAssist - main application.

Two modes, switched with the M key or by voice:

  AMBIENT MODE (default)
    Object detection runs continuously and announces newly-appeared
    objects. This is passive awareness - "what is around me".

  READING MODE
    Object announcements are silenced. The user points the camera at
    text and presses SPACE to have it read aloud. This is deliberate
    inquiry - "what does this say".

Speech recognition, text reading and visual questions all run on
background threads, so the video and object detection keep running while
they work.

Speech is prioritised: anything the user asks for jumps ahead of ambient
announcements, and ambient announcements that cannot be spoken promptly
are dropped rather than played late. If the internet drops, speech falls
back to the Windows built-in voice; if speech fails entirely, the
application beeps, since a failure of speech cannot be announced by
speaking.

Controls:
  M     - switch mode
  SPACE - read all text (Reading mode)
  D     - describe everything currently visible
  I     - identify the object being held up, and read any text on it
  R     - repeat the last thing said
  V     - speak: a command, or any question about what the camera sees
  Q     - quit
"""

import math
import re
import sys
import threading
import time

import cv2
from ultralytics import YOLO

sys.path.insert(0, "src")

# easyocr must be imported after the line above: it does
# `from bidi import get_display`, which is satisfied by the stub in
# src/bidi/. See that file for why the real package was removed.
import easyocr

from speech.tts import Speaker

# Voice commands need Whisper, which needs numba and llvmlite. Imported
# optionally so that a failure in an optional feature cannot stop the
# whole application from starting.
try:
    from voice.listener import VoiceListener, match_command
    VOICE_AVAILABLE = True
except Exception as error:
    print(f"Voice commands unavailable: {error}")
    VOICE_AVAILABLE = False

# Visual questions need Ollama running with a vision model. Also optional.
try:
    from vision.asker import VisionAsker
    VISION_AVAILABLE = True
except Exception as error:
    print(f"Visual questions unavailable: {error}")
    VISION_AVAILABLE = False

# Spelling correction for OCR readings. Optional: without it, text is
# spoken exactly as read.
try:
    from spellchecker import SpellChecker
    spell = SpellChecker()
except Exception as error:
    print(f"Spelling correction unavailable: {error}")
    spell = None


# ---------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------

# --- Models ---------------------------------------------------------

# Object detection: YOLO26 pretrained on Objects365 (365 classes)
DETECTION_MODEL = "models/yolo26m-objv1-150.pt"

# Speech recognition: Whisper, running locally on the GPU. "base" was
# accurate on all five test commands; move to "small" if that changes.
VOICE_MODEL = "base"

# --- Diagnostics -----------------------------------------------------

# Every detection, with the reason it was kept or skipped, and every OCR
# word with its confidence. Invaluable while developing, distracting in
# a demonstration - set to False before recording one.
DEBUG = True

# --- Detection thresholds -------------------------------------------

# Threshold for continuous ambient detection. Deliberately high, to stop
# weak detections flickering in and out and being announced repeatedly.
DETECTION_CONFIDENCE = 0.65

# When the user asks about one held-up item, detection runs again on that
# frame at this lower threshold. A deliberate question can accept weaker
# detections - with honest wording.
IDENTIFY_CONFIDENCE = 0.30

# A label below this confidence is not worth announcing at all - better
# to say nothing about the object and just read the text.
MIN_LABEL_CONFIDENCE = 0.40

# At or above this, the label is stated plainly ("I see cup"); below it,
# the label is hedged ("This might be cup").
CONFIDENT_LABEL = 0.50

# --- Announcements ---------------------------------------------------

# How many consecutive frames an object must be missing before it is
# forgotten and becomes eligible to be announced again.
FRAMES_BEFORE_GONE = 150

# Words spoken per reading. I answers "what is this?", where a short
# summary is better; SPACE reads a document or sign, where more text
# matters and an object name adds nothing.
MAX_SPOKEN_WORDS = 6
MAX_WORDS_READING = 20

# Below this OCR confidence, a word that is not in the dictionary is
# corrected to the nearest real word ("PERFORMANGE" -> "performance").
# Confident readings are left alone, so brand names read correctly -
# BOLDFIT at 0.92 - are not "corrected" into ordinary words.
SPELLCHECK_BELOW = 0.90

# OCR reads "1" as "I" or "l" before a unit: "Ikg" for "1kg"
UNIT_MISREAD = re.compile(r"^[Il|](kg|g|mg|ml|l|L)$")

# Ambient announcements are dropped if they cannot be spoken within this
# many seconds - a late announcement describes a scene that has changed.
AMBIENT_EXPIRY = 6.0

# The same object is not announced again within this many seconds, even
# if it leaves the frame and comes back. This stops objects at the edge
# of the frame being announced repeatedly as detection flickers.
ANNOUNCE_COOLDOWN = 30.0

# --- Choosing the held-up object -------------------------------------

# A held-up object covers the centre of the frame and is not enormous.
# Anything larger than this share of the frame is background - a wall,
# a bed, a shelf - rather than the thing being shown to the camera.
MAX_OBJECT_AREA = 0.55

# If nothing covers the centre, an object is still accepted when its own
# centre is within this distance of the frame centre (0 = centre, 1 = corner)
MAX_OBJECT_DISTANCE = 0.30

# True while testing with a laptop webcam that faces the user: then the
# user's own body and worn items are always in view, and must not be
# mistaken for the item being held up. For a camera worn facing outward
# - the real use case - set this to False.
CAMERA_FACES_USER = True

# Things worn or carried by the user. Only used when CAMERA_FACES_USER is
# True, and only for choosing the held-up item - ambient announcements
# and the D key still report all of these normally.
IGNORE_FOR_IDENTIFY = {
    "person", "glasses", "hat", "head phone", "headphone", "earphone",
    "watch", "bracelet", "necklace", "ring", "tie", "belt", "helmet",
    "mask", "gloves", "backpack", "handbag",
}

# An object tracked for longer than this counts as established
# background rather than something just brought into view.
RECENT_SECONDS = 20.0

# How long the chosen object stays highlighted on screen
HIGHLIGHT_SECONDS = 3.0

# --- Failure handling ------------------------------------------------

# If the camera stops returning frames, keep retrying for this long
# before giving up. A brief glitch should not end the session.
CAMERA_RETRY_SECONDS = 3.0

# How often to beep while speech is broken, so the user is not left in
# silence without knowing why. A beep needs no network.
BEEP_INTERVAL = 10.0

# While using the offline voice, retry the online one this often - the
# network may have come back.
ONLINE_RETRY_INTERVAL = 60.0

# --- Camera and modes ------------------------------------------------

CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

MODE_AMBIENT = "ambient"
MODE_READING = "reading"


# ---------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------

def fail(message, detail=""):
    """Report a startup problem in plain language, then exit.

    Startup failures are printed rather than spoken: speech may be the
    thing that is broken, and the person setting the project up is
    reading the terminal anyway.
    """
    print(f"\nCannot start: {message}")
    if detail:
        print(f"  {detail}")
    sys.exit(1)


print("Loading models...")

try:
    detector = YOLO(DETECTION_MODEL)
except Exception as error:
    fail(f"the detection model could not be loaded ({DETECTION_MODEL}).",
         f"Check the file exists, and run from the project folder. [{error}]")

try:
    reader = easyocr.Reader(['en'], gpu=True)
except Exception as error:
    fail("the text reader could not be loaded.",
         f"EasyOCR downloads its models on first run, so this may be a "
         f"network problem. [{error}]")

try:
    speaker = Speaker()
except Exception as error:
    fail("the speech system could not start.",
         f"Check that audio output is available. [{error}]")

# Voice commands are optional: if the microphone or the model is
# unavailable, the application still works with the keyboard.
if VOICE_AVAILABLE:
    try:
        voice = VoiceListener(VOICE_MODEL)
    except Exception as error:
        print(f"Voice commands unavailable: {error}")
        voice = None
else:
    voice = None

# Visual questions are optional: without Ollama running, every other
# feature still works.
if VISION_AVAILABLE:
    try:
        asker = VisionAsker()
    except Exception as error:
        print(f"Visual questions unavailable: {error}")
        asker = None
else:
    asker = None

camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

if not camera.isOpened():
    speaker.shutdown()
    fail("the camera could not be opened.",
         "It may be unplugged, or in use by another program "
         "(video call, camera app).")

mode = MODE_AMBIENT

# Remembers the last thing spoken, so the user can ask to hear it again
last_announcement = ""

# Set while a text reading is running on the background thread, so a
# second reading cannot start on top of it
ocr_busy = threading.Event()

# Speech health, tracked so messages are printed once per change rather
# than every frame
speech_broken = False
was_offline = False
last_beep = 0.0
last_online_retry = 0.0

# Voice command state
voice_active = False        # True while recording
pending_command = None      # set by the voice thread, run by the main loop
current_frame = None        # newest frame, for voice-triggered commands

print("Ready.")
print("  M     - switch between Ambient and Reading mode")
print("  SPACE - read all text (Reading mode only)")
print("  D     - describe what is currently visible")
print("  I     - identify the held-up object and read its text")
print("  R     - repeat last announcement")
if voice is not None:
    print("  V     - speak (press to start, press again to stop)")
    if asker is not None:
        print("            a command, or any question: \"what am I wearing?\"")
print("  Q     - quit")


# ---------------------------------------------------------------------
# Object tracking state
# ---------------------------------------------------------------------

currently_present = set()   # objects believed to be in view right now
missing_counts = {}         # label -> consecutive frames not seen
last_announced = {}         # label -> time it was last spoken
first_seen = {}             # label -> time it (re)appeared in view
previous_time = 0

# The object chosen by the last identification, highlighted on screen
prominent_box = None
prominent_until = 0.0


# ---------------------------------------------------------------------
# Speech helpers
# ---------------------------------------------------------------------

def say(text, priority=False, expires_in=None):
    """Speak text and remember it, so the R key can repeat it.

    priority   - jumps ahead of queued ambient chatter and stops whatever
                 is playing. Used for anything the user asked for.
    expires_in - drop the announcement if it has not been spoken within
                 this many seconds.
    """
    global last_announcement
    if not text or not text.strip():
        return
    last_announcement = text
    speaker.speak(text, priority=priority, expires_in=expires_in)


def beep():
    """A short tone the application generates itself.

    Used to signal that speech has stopped working, and to confirm that
    voice recording has started. It needs no network and no speech
    engine, so it still works when everything else has failed - which is
    exactly when the user needs to be told.
    """
    try:
        import winsound
        winsound.Beep(880, 250)
    except Exception:
        # Not Windows, or no sound hardware: fall back to the terminal bell
        print("\a", end="", flush=True)


# ---------------------------------------------------------------------
# Text reading
# ---------------------------------------------------------------------

def clean_word(text, confidence):
    """Make an OCR reading easier to understand when spoken.

    Three fixes, all found testing a protein box labelled "Beast Life
    Performance Protein 1kg", which was read as "BEAST, PERFORMANGE,
    PROTEINa, Ikg":
      - "Ikg" -> "1 kg": OCR confuses 1 with I or l before a unit
      - "PERFORMANGE" -> "performance": a low-confidence word that is
        not in the dictionary is corrected to the nearest real word
      - "BEAST" -> "Beast": all capitals can make the voice spell a
        word out letter by letter
    """
    match = UNIT_MISREAD.match(text)
    if match:
        return "1 " + match.group(1)

    words = []
    for word in text.split():
        if (spell is not None and confidence < SPELLCHECK_BELOW
                and word.isalpha() and len(word) >= 5
                and not spell.known([word.lower()])):
            fixed = spell.correction(word.lower())
            if fixed and fixed != word.lower():
                if DEBUG:
                    print(f"  [spelling] {word} -> {fixed}")
                word = fixed

        if word.isupper() and len(word) > 3:
            word = word.capitalize()

        words.append(word)

    return " ".join(words)


def read_text_from_frame(frame, max_words=MAX_SPOKEN_WORDS):
    """Run OCR on a frame and return the text worth speaking,
    in approximate reading order."""
    results = reader.readtext(frame)

    if DEBUG:
        print("Extracted text:")

    kept = []

    for result in results:
        bounding_box, text, confidence = result
        if DEBUG:
            print(f"  {text}  (confidence: {confidence:.2f})")

        cleaned = text.strip()

        # Longer text naturally scores lower in confidence, since more
        # characters means more chances for error. A flat threshold would
        # discard exactly the most informative readings.
        if len(cleaned) >= 12:
            threshold = 0.25
        elif len(cleaned) >= 5:
            threshold = 0.40
        elif len(cleaned) == 4:
            # Four-letter words like "LIFE" read at 0.58-0.64 were being
            # dropped by the 0.75 bar meant for three-letter fragments,
            # while four-letter junk ("Hikg", "CRuY", "Fepe") scored
            # below 0.20 - so 0.50 keeps the words and still rejects junk
            threshold = 0.50
        else:
            threshold = 0.75

        if confidence > threshold and len(cleaned) >= 3:
            kept.append(result)

    # Sort surviving results into reading order: top to bottom, then left
    # to right. Vertical position is rounded into bands of 30 pixels so
    # words on the same visual line group together.
    def reading_order(result):
        box = result[0]
        top = min(point[1] for point in box)
        left = min(point[0] for point in box)
        return (round(top / 30), left)

    kept.sort(key=reading_order)

    final_words = [clean_word(result[1].strip(), result[2])
                   for result in kept[:max_words]]
    print(f"SPEAKING IN ORDER: {final_words}")

    return final_words


def start_reading(frame, start_message, on_done, max_words=MAX_SPOKEN_WORDS):
    """Read text on a background thread, so the video keeps running.

    frame         - the frame to read, copied so the loop can move on
    start_message - spoken immediately, e.g. "Reading"
    on_done       - called with the list of words once reading finishes
    """
    if frame is None:
        return

    if ocr_busy.is_set():
        say("Still reading, please wait", priority=True)
        return

    ocr_busy.set()
    say(start_message, priority=True)

    # The main loop replaces `frame` on every iteration, so the thread
    # must work on its own snapshot taken at the moment of the request
    snapshot = frame.copy()

    def work():
        try:
            words = read_text_from_frame(snapshot, max_words)
            on_done(words)
        except Exception as error:
            print(f"OCR error: {error}")
            say("Sorry, reading failed", priority=True)
        finally:
            # Always clear the flag - otherwise one error would block
            # every future reading
            ocr_busy.clear()

    # daemon=True means this thread will not keep the program alive
    # after the user quits
    threading.Thread(target=work, daemon=True).start()


def speak_identity(words, label, confidence):
    """Announce the object and the text on it.

    Kept as separate sentences, so a wrong object label cannot corrupt
    the text - which is the more reliable of the two. Weak detections are
    phrased as uncertain rather than stated as fact.
    """
    if label and confidence < CONFIDENT_LABEL:
        object_part = f"This might be {label}."
    elif label:
        object_part = f"I see {label}."
    else:
        object_part = None

    if words and object_part:
        say(f"{object_part} It says: " + ", ".join(words), priority=True)
    elif words:
        say("It says: " + ", ".join(words), priority=True)
    elif object_part:
        say(f"{object_part} No readable text.", priority=True)
    else:
        say("I cannot identify anything", priority=True)


# ---------------------------------------------------------------------
# Choosing the held-up object
# ---------------------------------------------------------------------

def most_prominent_object(result):
    """Pick the object most likely to be the one held up to the camera.

    Returns (label, box, confidence), where box is [x1, y1, x2, y2] in
    pixels, or (None, None, None) if nothing suitable was found.

    Rules, in order:
      1. Ignore worn items (when the camera faces the user), detections
         too uncertain to name, and anything filling most of the frame.
      2. Prefer the SMALLEST box that contains the centre of the frame -
         the most specific thing under the centre point - with confidence
         deciding between boxes of near-identical size.
      3. Otherwise take the nearest object to the centre, preferring
         anything that is not established background.

    With DEBUG on, every detection is printed with the reason it was kept
    or skipped, so the choice can be checked rather than guessed.
    """
    now = time.time()
    covering_centre = []
    nearby = []

    if DEBUG:
        print(f"IDENTIFY: {len(result.boxes)} detections in this frame")

    for box_n, box_px, class_id, conf in zip(result.boxes.xywhn,
                                             result.boxes.xyxy,
                                             result.boxes.cls,
                                             result.boxes.conf):
        label = result.names[int(class_id)]
        confidence = float(conf)
        x, y, width, height = box_n.tolist()
        area = width * height
        distance = math.hypot(x - 0.5, y - 0.5)
        info = (f"  {label:<15} conf {confidence:.2f}   "
                f"area {area * 100:3.0f}%   distance {distance:.2f}")

        # Note on indentation: each `continue` sits OUTSIDE its `if DEBUG`
        # block. Indenting it inside would mean the object is only skipped
        # when debugging is on - so turning DEBUG off would silently change
        # which object gets chosen.

        if CAMERA_FACES_USER and label.lower() in IGNORE_FOR_IDENTIFY:
            if DEBUG:
                print(info + "   -> skipped: worn item")
            continue

        if confidence < MIN_LABEL_CONFIDENCE:
            if DEBUG:
                print(info + "   -> skipped: too uncertain to name")
            continue

        if area > MAX_OBJECT_AREA:
            if DEBUG:
                print(info + "   -> skipped: too large")
            continue

        pixels = [int(v) for v in box_px.tolist()]

        # Background = something ambient tracking has known about for a
        # while. A held item is usually new - or too faint for ambient
        # tracking, at its higher threshold, to have registered at all.
        is_background = (label in currently_present and
                         now - first_seen.get(label, now) > RECENT_SECONDS)

        if abs(x - 0.5) <= width / 2 and abs(y - 0.5) <= height / 2:
            if DEBUG:
                print(info + "   -> candidate: covers centre")
            # Area is rounded to 2 decimals (1% of the frame) so boxes of
            # near-identical size tie, letting confidence decide. Two
            # boxes often land on one object - "canned" and "bottle" on a
            # single can - and without this the smaller, less confident
            # one wins. -confidence sorts higher values first, since
            # min() takes the smallest number.
            covering_centre.append((round(area, 2), -confidence, label,
                                    pixels, confidence))
        else:
            if DEBUG:
                print(info + "   -> candidate: nearby")
            nearby.append((is_background, round(distance, 2), -confidence,
                           label, pixels, confidence))

    # The PROMINENT lines always print: one line, and useful even in a
    # demonstration, since it shows what the announcement was based on.
    if covering_centre:
        area, _, label, pixels, confidence = min(covering_centre)
        print(f"PROMINENT: {label} (covers centre, conf {confidence:.2f})")
        return label, pixels, confidence

    if nearby:
        background, distance, _, label, pixels, confidence = min(nearby)
        if distance <= MAX_OBJECT_DISTANCE:
            kind = "background" if background else "new"
            print(f"PROMINENT: {label} ({kind}, distance {distance:.2f}, "
                  f"conf {confidence:.2f})")
            return label, pixels, confidence
        print(f"PROMINENT: none (nearest, {label}, is too far: {distance:.2f})")
        return None, None, None

    print("PROMINENT: none")
    return None, None, None


def identify_and_read(frame):
    """Name the held-up object and read a short amount of its text.

    Answers "what is this?" - so the object name matters, and a brief
    reading is better than a long one.
    """
    global prominent_box, prominent_until

    if frame is None:
        return

    try:
        result = detector.predict(source=frame, conf=IDENTIFY_CONFIDENCE,
                                  verbose=False)[0]
        label, prominent_box, confidence = most_prominent_object(result)
    except Exception as error:
        print(f"Detection error: {error}")
        label, prominent_box, confidence = None, None, None

    prominent_until = time.time() + HIGHLIGHT_SECONDS

    start_reading(frame, "Identifying",
                  lambda words: speak_identity(words, label, confidence))


def ask_vision(question):
    """Answer an open-ended question about what the camera sees.

    Keyword commands answer fixed questions instantly. This handles
    everything else - "what am I wearing?", "what is on the bed?" - using
    a vision-language model running locally. It takes a few seconds, so
    it is only ever used for a deliberate question.
    """
    if asker is None:
        say("Visual questions are not available", priority=True)
        return

    if current_frame is None:
        return

    say("Let me look", priority=True)

    def on_answer(answer, problem):
        if problem:
            say(f"Sorry, {problem}", priority=True)
            return
        print(f"VISION: {answer}")
        # "It looks like" signals the model's impression rather than
        # certainty - it sometimes states details it is unsure of
        say("It looks like " + answer, priority=True)

    asker.ask_in_background(current_frame, question, on_answer)


def read_document(frame):
    """Read the text only, with no object name.

    Answers "read this to me" - for documents, signs and letters, where
    the object is obvious and the text is the point. No detection runs,
    and the word limit is higher.
    """
    def speak_text(words):
        if words:
            say(", ".join(words), priority=True)
        else:
            say("No readable text found", priority=True)

    start_reading(frame, "Reading", speak_text,
                  max_words=MAX_WORDS_READING)


# ---------------------------------------------------------------------
# Object announcements
# ---------------------------------------------------------------------

def describe_scene():
    """Speak a summary of everything currently believed to be visible."""
    if not currently_present:
        say("I do not see anything I recognise", priority=True)
        return

    say("I see: " + ", ".join(sorted(currently_present)), priority=True)


def track_objects(detected_now, announce):
    """Update which objects are considered present.

    Newly-appeared objects are announced together in one sentence rather
    than one at a time: seven separate announcements meant seven network
    round trips, which put the speech queue five seconds behind.

    When announce is False the tracking still runs, so that switching
    back to ambient mode does not immediately re-announce everything
    that was already in view.
    """
    new_objects = detected_now - currently_present
    now = time.time()
    to_announce = []

    for label in sorted(new_objects):
        currently_present.add(label)
        first_seen[label] = now

        # Skip anything announced recently, even if it left and returned
        if now - last_announced.get(label, 0) < ANNOUNCE_COOLDOWN:
            continue

        last_announced[label] = now
        to_announce.append(label)

    # While speech is broken, ambient announcements are pointless: each
    # one attempts a network call that fails. Deliberate requests are
    # still attempted, since the user is actively asking.
    if announce and to_announce and speaker.is_working():
        print(f"ANNOUNCING: {', '.join(to_announce)}")
        say(", ".join(to_announce), expires_in=AMBIENT_EXPIRY)

    for label in detected_now:
        missing_counts[label] = 0

    for label in list(currently_present):
        if label not in detected_now:
            missing_counts[label] = missing_counts.get(label, 0) + 1

            if missing_counts[label] >= FRAMES_BEFORE_GONE:
                currently_present.remove(label)
                missing_counts.pop(label, None)
                if announce:
                    print(f"FORGOTTEN: {label}")


# ---------------------------------------------------------------------
# Commands, from keys or from speech
# ---------------------------------------------------------------------

def run_command(name):
    """Run a command. Returns False if the application should stop."""
    global mode

    if name == "read":
        if mode != MODE_READING:
            mode = MODE_READING
            print("Switched to READING mode")
        read_document(current_frame)

    elif name == "identify":
        identify_and_read(current_frame)

    elif name == "describe":
        describe_scene()

    elif name == "repeat":
        if last_announcement:
            print(f"REPEATING: {last_announcement}")
            speaker.speak(last_announcement, priority=True)
        else:
            speaker.speak("Nothing to repeat", priority=True)

    elif name == "reading_mode":
        mode = MODE_READING
        print("Switched to READING mode")
        say("Reading mode", priority=True)

    elif name == "ambient_mode":
        mode = MODE_AMBIENT
        print("Switched to AMBIENT mode")
        say("Ambient mode", priority=True)

    elif name == "quit":
        say("Stopping", priority=True)
        time.sleep(1.5)           # let the announcement finish
        return False

    return True


def handle_voice_key():
    """Start or stop voice recording, depending on the current state.

    OpenCV cannot detect a key being released, so V toggles: press to
    start, press again to stop.
    """
    global voice_active

    if voice is None:
        say("Voice commands are not available", priority=True)
        return

    if not voice.recording:
        beep()                       # instant, unlike a spoken prompt
        voice.start_recording()
        voice_active = True
        print("LISTENING...")
        return

    audio = voice.stop_recording()
    voice_active = False

    if audio is None:
        print("  nothing recorded")
        return

    print("  transcribing...")

    def on_transcribed(text):
        global pending_command
        if not text:
            say("I did not hear anything", priority=True)
            return

        print(f"  heard: {text!r}")
        command = match_command(text)

        if command:
            # The command is run by the main loop, not here: this runs on
            # a background thread, and the commands use camera frames and
            # change mode state that belong to the loop.
            pending_command = command
        elif asker is not None:
            # Not a known command, so treat it as a question about the
            # scene - this is what makes "what am I wearing?" work
            ask_vision(text)
        else:
            say(f"I heard {text}. I do not know that command.", priority=True)

    voice.transcribe_in_background(audio, on_transcribed)


# ---------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------

say("Ambient mode", priority=True)

try:
    # Set to the time the camera first stopped responding, so a brief
    # glitch can be tolerated while a real disconnection ends the session
    lost_camera_since = None

    while True:
        success, frame = camera.read()

        if not success:
            if lost_camera_since is None:
                lost_camera_since = time.time()
                print("Camera stopped responding - retrying...")
                say("Camera problem", priority=True)

            if time.time() - lost_camera_since > CAMERA_RETRY_SECONDS:
                print("Camera did not recover.")
                say("Camera disconnected. Stopping.", priority=True)
                time.sleep(2)          # let the announcement finish playing
                break

            time.sleep(0.1)
            continue

        if lost_camera_since is not None:
            lost_camera_since = None
            print("Camera recovered.")
            say("Camera working again", priority=True)

        # One bad frame should not end the session
        try:
            results = detector.predict(
                source=frame,
                conf=DETECTION_CONFIDENCE,
                verbose=False
            )
        except Exception as error:
            print(f"Detection error: {error}")
            continue

        annotated_frame = results[0].plot()

        # Keep the newest frame available to voice commands and visual
        # questions, which arrive from a background thread rather than
        # from a keypress
        current_frame = frame

        detected_now = set()
        for box in results[0].boxes:
            class_id = int(box.cls[0])
            detected_now.add(results[0].names[class_id])

        track_objects(detected_now, announce=(mode == MODE_AMBIENT))

        current_time = time.time()
        fps = 1 / (current_time - previous_time) if previous_time else 0
        previous_time = current_time

        cv2.putText(
            annotated_frame, f"FPS: {int(fps)}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
        )

        cv2.putText(
            annotated_frame, f"MODE: {mode.upper()}", (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2
        )

        # Highlight the object chosen by the last identification
        if prominent_box and time.time() < prominent_until:
            x1, y1, x2, y2 = prominent_box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 140, 255), 4)
            cv2.putText(annotated_frame, "CHOSEN", (x1, max(30, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 140, 255), 2)

        # Visible sign that a reading is in progress
        if ocr_busy.is_set():
            cv2.putText(
                annotated_frame, "READING TEXT...", (10, 310),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2
            )

        # Visible sign that the vision model is working on a question
        if asker is not None and asker.busy.is_set():
            cv2.putText(
                annotated_frame, "THINKING...", (10, 415),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 160, 60), 2
            )

        # --- Speech health ---------------------------------------------
        # Three states: working online, working offline with the Windows
        # voice, or not working at all - which cannot be announced by
        # speaking, so a beep is used instead.

        if speaker.is_offline():
            if not was_offline:
                was_offline = True
                # Start the retry countdown now. Setting it only at
                # startup meant that if the application had already been
                # running longer than the interval, the first retry fired
                # instantly.
                last_online_retry = time.time()
                print("Using the offline voice (no internet).")

            cv2.putText(
                annotated_frame, "OFFLINE VOICE", (10, 345),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2
            )

            if time.time() - last_online_retry > ONLINE_RETRY_INTERVAL:
                last_online_retry = time.time()
                speaker.try_online_again()

        elif was_offline:
            was_offline = False
            print("Back to the online voice.")
            say("Back online", priority=True)

        if not speaker.is_working():
            if not speech_broken:
                speech_broken = True
                print(f"SPEECH FAILING: {speaker.failure_reason()}")

            if time.time() - last_beep > BEEP_INTERVAL:
                last_beep = time.time()
                beep()

            cv2.putText(
                annotated_frame, "SPEECH NOT WORKING", (10, 345),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2
            )
        elif speech_broken:
            speech_broken = False
            print("Speech working again.")

        # Visible sign that voice recording is in progress
        if voice_active:
            cv2.putText(
                annotated_frame, "LISTENING...", (10, 380),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2
            )

        # On-screen control reminder. Drawn one line at a time because
        # OpenCV has no multi-line text support. Only features that are
        # actually available are listed.
        controls = [
            "M - switch mode",
            "SPACE - read all text",
            "D - describe scene",
            "I - what is this?",
            "R - repeat",
        ]
        if voice is not None:
            controls.append("V - speak: command or question")
        controls.append("Q - quit")

        for index, line in enumerate(controls):
            cv2.putText(
                annotated_frame, line, (10, 110 + index * 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1
            )

        cv2.imshow("VisionAssist", annotated_frame)

        # Commands recognised from speech are run here, on the main loop,
        # because they use the camera frame and change the mode
        if pending_command:
            command, pending_command = pending_command, None
            if not run_command(command):
                break

        key = cv2.waitKey(1) & 0xFF

        # --- M = switch mode ---
        if key == ord('m'):
            if mode == MODE_AMBIENT:
                mode = MODE_READING
                print("Switched to READING mode")
                say("Reading mode. Press space to read text.", priority=True)
            else:
                mode = MODE_AMBIENT
                print("Switched to AMBIENT mode")
                say("Ambient mode", priority=True)

        # --- SPACE = read all text (Reading mode only) ---
        elif key == ord(' '):
            if mode == MODE_READING:
                read_document(frame)
            else:
                say("Switch to reading mode first", priority=True)

        # --- D = describe what is visible ---
        elif key == ord('d'):
            describe_scene()

        # --- I = identify the held-up object and its text ---
        elif key == ord('i'):
            identify_and_read(frame)

        # --- R = repeat last announcement ---
        elif key == ord('r'):
            if last_announcement:
                print(f"REPEATING: {last_announcement}")
                speaker.speak(last_announcement, priority=True)
            else:
                speaker.speak("Nothing to repeat", priority=True)

        # --- V = speak a command or a question ---
        elif key == ord('v'):
            handle_voice_key()

        elif key == ord('q'):
            break

except KeyboardInterrupt:
    print("\nStopped by user (Ctrl+C).")

finally:
    print("Shutting down...")
    if voice is not None:
        voice.stop_recording()
        voice.close()
    camera.release()
    cv2.destroyAllWindows()
    speaker.shutdown()