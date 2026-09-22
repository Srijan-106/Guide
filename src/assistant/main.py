"""
VisionAssist - main application.

Two modes, switched with the M key:

  AMBIENT MODE (default)
    Object detection runs continuously and announces newly-appeared
    objects. This is passive awareness - "what is around me".

  READING MODE
    Object announcements are silenced. The user points the camera at
    text and presses SPACE to have it read aloud. This is deliberate
    inquiry - "what does this say".

Modes exist because the two features serve different intentions.
Ambient chatter interrupting a deliberate reading is confusing, and a
user focused on a product label does not want to hear about the
furniture behind them.

Controls:
  M     - switch mode
  SPACE - read text (Reading mode)
  D     - describe everything currently visible
  I     - identify: object label plus any text on it
  R     - repeat the last thing said
  Q     - quit
"""

import sys
import time

import cv2
import easyocr
from ultralytics import YOLO

sys.path.insert(0, "src")
from speech.tts import Speaker


# ---------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------

# Object detection: YOLO26 pretrained on Objects365 (365 classes)
DETECTION_MODEL = "models/yolo26m-objv1-150.pt"
DETECTION_CONFIDENCE = 0.65

# How many consecutive frames an object must be missing before it is
# forgotten and becomes eligible to be announced again.
FRAMES_BEFORE_GONE = 150

# Maximum number of text fragments spoken per OCR capture.
MAX_SPOKEN_WORDS = 6

CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

MODE_AMBIENT = "ambient"
MODE_READING = "reading"


# ---------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------

print("Loading models...")

detector = YOLO(DETECTION_MODEL)
reader = easyocr.Reader(['en'], gpu=True)
speaker = Speaker()

camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

if not camera.isOpened():
    print("Camera not available")
    sys.exit(1)

mode = MODE_AMBIENT

# Remembers the last thing spoken, so the user can ask to hear it again
last_announcement = ""

print("Ready.")
print("  M     - switch between Ambient and Reading mode")
print("  SPACE - read text (Reading mode only)")
print("  D     - describe what is currently visible")
print("  I     - identify object and read its text")
print("  R     - repeat last announcement")
print("  Q     - quit")


# ---------------------------------------------------------------------
# Object tracking state
# ---------------------------------------------------------------------

currently_present = set()
missing_counts = {}
previous_time = 0


# ---------------------------------------------------------------------
# Speech helper
# ---------------------------------------------------------------------

def say(text):
    """Speak text and remember it, so the R key can repeat it."""
    global last_announcement
    if not text or not text.strip():
        return
    last_announcement = text
    speaker.speak(text)


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------

def read_text_from_frame(frame):
    """Run OCR on a frame and return the text worth speaking,
    in approximate reading order."""
    results = reader.readtext(frame)

    print("Extracted text:")
    kept = []

    for result in results:
        bounding_box, text, confidence = result
        print(f"  {text}  (confidence: {confidence:.2f})")

        cleaned = text.strip()

        # Longer text naturally scores lower in confidence, since more
        # characters means more chances for error. A flat threshold would
        # discard exactly the most informative readings.
        if len(cleaned) >= 12:
            threshold = 0.25
        elif len(cleaned) >= 5:
            threshold = 0.40
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

    final_words = [result[1].strip() for result in kept[:MAX_SPOKEN_WORDS]]
    print(f"SPEAKING IN ORDER: {final_words}")

    return final_words


def describe_scene():
    """Speak a summary of everything currently believed to be visible."""
    if not currently_present:
        say("I do not see anything I recognise")
        return

    objects = sorted(currently_present)
    say("I see: " + ", ".join(objects))


def identify(frame, detected_now):
    """Announce any text found on an object, falling back to the object
    label only when no text is found.

    Object labels from a general-purpose model are often wrong for
    specific products, while text is far more reliable - so the label is
    only used when the reliable source produces nothing.
    """
    words = read_text_from_frame(frame)

    if words:
        say("It says: " + ", ".join(words))
    elif detected_now:
        objects = sorted(detected_now)
        say("I see " + objects[0] + ", but no readable text")
    else:
        say("I cannot identify anything")


def track_objects(detected_now, announce):
    """Update which objects are considered present.

    When announce is False the tracking still runs, so that switching
    back to ambient mode does not immediately re-announce everything
    that was already in view.
    """
    new_objects = detected_now - currently_present

    for label in new_objects:
        currently_present.add(label)
        if announce:
            print(f"ANNOUNCING: {label}")
            say(label)

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
# Main loop
# ---------------------------------------------------------------------

say("Ambient mode")

try:
    while True:
        success, frame = camera.read()
        if not success:
            break

        results = detector.predict(
            source=frame,
            conf=DETECTION_CONFIDENCE,
            verbose=False
        )
        annotated_frame = results[0].plot()

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

        # On-screen control reminder. Drawn one line at a time because
        # OpenCV has no multi-line text support.
        controls = [
            "M - switch mode",
            "SPACE - read text",
            "D - describe scene",
            "I - identify object",
            "R - repeat",
            "Q - quit",
        ]

        for index, line in enumerate(controls):
            cv2.putText(
                annotated_frame, line, (10, 110 + index * 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1
            )

        cv2.imshow("VisionAssist", annotated_frame)

        key = cv2.waitKey(1) & 0xFF

        # --- M = switch mode ---
        if key == ord('m'):
            if mode == MODE_AMBIENT:
                mode = MODE_READING
                print("Switched to READING mode")
                say("Reading mode. Press space to read text.")
            else:
                mode = MODE_AMBIENT
                print("Switched to AMBIENT mode")
                say("Ambient mode")

        # --- SPACE = read text (Reading mode only) ---
        elif key == ord(' '):
            if mode == MODE_READING:
                say("Reading")
                words = read_text_from_frame(frame)
                if words:
                    say(", ".join(words))
                else:
                    say("No readable text found")
            else:
                say("Switch to reading mode first")

        # --- D = describe what is visible ---
        elif key == ord('d'):
            describe_scene()

        # --- I = identify object and its text ---
        elif key == ord('i'):
            say("Identifying")
            identify(frame, detected_now)

        # --- R = repeat last announcement ---
        elif key == ord('r'):
            if last_announcement:
                print(f"REPEATING: {last_announcement}")
                speaker.speak(last_announcement)
            else:
                speaker.speak("Nothing to repeat")

        elif key == ord('q'):
            break

finally:
    camera.release()
    cv2.destroyAllWindows()
    speaker.shutdown()