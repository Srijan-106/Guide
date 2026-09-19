import sys
import cv2
import easyocr

# Allow importing from the src folder, where the speech module lives
sys.path.insert(0, "src")
from speech.tts import Speaker

# Load the OCR engine for English, using the GPU for faster processing
reader = easyocr.Reader(['en'], gpu=True)

# Create the speech engine (one instance, reused for all announcements)
speaker = Speaker()

# Open the default webcam
camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("Press SPACE to capture and read text. Press 'q' to quit.")

while True:
    success, frame = camera.read()
    if not success:
        break

    cv2.imshow("EasyOCR - Press SPACE to read text", frame)

    # waitKey(1) checks for a keypress with a 1 millisecond window.
    # A larger value here makes the program miss key presses, since it
    # only listens during that brief window each loop.
    key = cv2.waitKey(1) & 0xFF

    # SPACE = capture the current frame and read any text in it
    if key == ord(' '):
        # Let the user know the request was received, since OCR takes
        # 1-2 seconds and silence would be ambiguous for a blind user
        speaker.speak("Reading")

        results = reader.readtext(frame)

        print("Extracted text:")
        detected_words = []

        for (bounding_box, text, confidence) in results:
            # Print everything for diagnostics, including low-confidence noise
            print(f"  {text}  (confidence: {confidence:.2f})")

            # Only speak results the model is reasonably confident about,
            # so the user doesn't hear garbage readings
                       # Longer text naturally scores lower in confidence, since more
            # characters means more chances for error. A flat threshold
            # therefore discards exactly the most informative readings.
            # Longer strings are given a lower bar accordingly.
            if len(text.strip()) >= 12:
                threshold = 0.25
            elif len(text.strip()) >= 5:
                threshold = 0.40
            else:
                threshold = 0.60

            if confidence > threshold:
                detected_words.append(text)

        if detected_words:
            # Join fragments into one sentence rather than speaking each
            # separately, so it flows as a single announcement
            sentence = ", ".join(detected_words)
            speaker.speak(sentence)
        else:
            # Explicit feedback matters: silence could mean failure,
            # still processing, or genuinely no text
            speaker.speak("No readable text found")

    # q = quit
    if key == ord('q'):
        break

# Release the camera and close windows cleanly
camera.release()
cv2.destroyAllWindows()

# Shut down the audio system
speaker.shutdown()