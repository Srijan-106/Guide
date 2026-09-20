# Guide (VisionAssist)

A software-first AI accessibility assistant for blind and visually impaired users.

## Project Status
**In active development —**
- **Day 0** — Research and planning phase completed. No functional code yet.
- **Day 1** — Python virtual environment set up. GPU-accelerated PyTorch (CUDA 12.8) verified working with RTX 5060. No functional application code yet.
- **Day 2** — First working object detection module built and tested on a live webcam feed using YOLOv8n, running on GPU.
- **Day 3** — Added real-time FPS measurement (~30 FPS confirmed end-to-end) and conducted a structured confidence-threshold experiment, documenting a genuine detection failure case for honest evaluation.
- **Day 4** — Implemented OCR text reading. Tested Tesseract, identified real-world reliability failures on curved, low-contrast, and stylized text, then switched to EasyOCR — verified through direct before/after comparison that all previously-failing test objects now read correctly.
- **Day 5** — Added offline text-to-speech output for both OCR and object detection, making results audible rather than terminal-only. Implemented change-detection with flicker tolerance so objects are announced on appearance rather than every frame, and resolved four distinct speech-engine failures including a Windows thread-affinity limitation.
- **Day 5 (Part A)** — Upgraded to YOLO26; documented two failed experiments proving vocabulary size does not fix missed detections; replaced pyttsx3 with edge-tts after five distinct failures; improved OCR filtering with length-aware confidence thresholds; fixed portability issues preventing the project running on other machines.
- **Day 6** — Integrated detection and OCR into a single application with mode switching, and fixed OCR reading order by sorting results on bounding-box position rather than detection order.
- **Day 6 (Part A)** — Added describe, identify and repeat commands; upgraded to a YOLO26 model pretrained on Objects365 (365 classes), which detects everyday objects such as glasses, bracelets and pillows that COCO's 80-class vocabulary cannot.

## Development Log
See full day-by-day progress in `docs/devlog.md`.

## Problem
Blind and visually impaired individuals often lack affordable, real-time,
privacy-respecting tools to understand their immediate surroundings — including
identifying objects, reading printed text, recognizing currency, and getting
natural spoken descriptions of a scene.

## Planned Core Features
- Real-time object detection (what is around the user)
- Optical Character Recognition, meaning automated text reading from images (reading signs, labels, documents)
- Currency recognition, trained specifically for Indian currency notes
- Local, offline large language model integration for natural spoken scene descriptions

## Why This Project
Built as a first major end-to-end Data Science / Machine Learning / Computer Vision
project, with an emphasis on real evaluation, genuine understanding of every
component, and honest documentation of the development process — including
failures and debugging, not just final results.

## Research Foundation
This project was built following a structured review of 2024-2026 academic
literature and existing open-source/commercial systems in the assistive
technology space, to identify genuine, honest gaps rather than assuming novelty.


**Note:** `src/ocr/ocr.py` is a deprecated Tesseract implementation, kept
for reference. The active OCR module is `src/ocr/ocr_easy.py`, which uses
EasyOCR. See `docs/devlog.md` (Day 4) for the comparison that motivated
the switch.


## Development Log
See day-by-day progress in `docs/devlog.md`.
