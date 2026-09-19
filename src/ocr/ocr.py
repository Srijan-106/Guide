
"""
DEPRECATED - Tesseract-based OCR. Kept for reference only.

This was the first OCR implementation (Day 4). It was replaced by
ocr_easy.py after testing showed Tesseract failed on curved surfaces,
metallic/low-contrast labels, and angled stylized text - all of which
EasyOCR handled correctly. See docs/devlog.md (Day 4) for the full
comparison and results.

This file is NOT part of the working application and requires a
separate Tesseract installation to run. The path below is specific to
the original development machine and will need changing to run elsewhere.
"""

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