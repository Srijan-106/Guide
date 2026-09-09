import sys
import cv2
import easyocr

sys.path.insert(0, "src")
from speech.tts import Speaker

reader = easyocr.Reader(['en'], gpu=True)
speaker = Speaker()

camera = cv2.VideoCapture(0)

print("Press SPACE to capture and read text. Press 'q' to quit.")

while True:
    success, frame = camera.read()
    if not success:
        break

    cv2.imshow("EasyOCR - Press SPACE to read text", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord(' '):
        results = reader.readtext(frame)

        print("Extracted text:")
        detected_words = []

        for (bounding_box, text, confidence) in results:
            print(f"  {text}  (confidence: {confidence:.2f})")
            if confidence > 0.5:
                detected_words.append(text)

        if detected_words:
            sentence = ", ".join(detected_words)
            speaker.speak(sentence)
        else:
            speaker.speak("No readable text found")

    if key == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()