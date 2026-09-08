import cv2
import pytesseract
import os

pytesseract.pytesseract.tesseract_cmd = r"D:\softwares\Tesseract\tesseract.exe"
os.environ["TESSDATA_PREFIX"] = r"D:\softwares\Tesseract\tessdata"

camera = cv2.VideoCapture(0)

print("Press SPACE to capture and read text. Press 'q' to quit.")

while True:
    success, frame = camera.read()
    if not success:
        break

    cv2.imshow("OCR Test - Press SPACE to read text", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord(' '):
        extracted_text = pytesseract.image_to_string(frame)
        print("Extracted text:")
        print(extracted_text)

    if key == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()