import cv2
import easyocr

reader = easyocr.Reader(['en'], gpu=True)

camera = cv2.VideoCapture(0)

print("Press SPACE to capture and read text. Press 'q' to quit.")

while True:
    success, frame = camera.read()
    if not success:
        break

    cv2.imshow("EasyOCR Test - Press SPACE to read text", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord(' '):
        results = reader.readtext(frame)
        print("Extracted text:")
        for (bounding_box, text, confidence) in results:
            print(f"  {text}  (confidence: {confidence:.2f})")

    if key == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
