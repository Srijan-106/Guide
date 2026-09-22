# Guide (VisionAssist)

A software-first AI accessibility assistant for blind and visually impaired users. Point a camera at the world and hear what is there: the objects around you, and the text on labels, signs and packaging.

Built as a first major end-to-end machine learning project, with an emphasis on honest evaluation: documented failures, measured results, and the reasoning behind every decision.

---

## What it does

| Feature | Status |
|---|---|
| Real-time object detection (365 object types), announced aloud | Working |
| Text reading (OCR), announced in reading order | Working |
| Spoken summary of everything in view, on demand | Working |
| Identify an object and read the text on it | Working |
| Repeat the last announcement | Working |

The application runs in two modes:

- **Ambient mode** (default) - announces objects as they appear. Passive awareness: *what is around me?*
- **Reading mode** - object announcements are silenced; press SPACE to read text aloud. Deliberate inquiry: *what does this say?*

Modes exist because testing showed object announcements interrupting text readings, which was confusing in practice.

## Controls

| Key | Action |
|---|---|
| M | Switch between Ambient and Reading mode |
| SPACE | Read text in view (Reading mode) |
| D | Describe everything currently visible |
| I | Identify an object and read any text on it |
| R | Repeat the last announcement |
| Q | Quit |

The video window must be focused (clicked) for keys to register.

---

## Evaluation highlights

**OCR engine.** Tesseract failed on curved, metallic and stylised text. EasyOCR read all the same test objects correctly - for example "BOLDFIT" on a metallic bottle at 0.99 confidence, where Tesseract returned garbage.

**OCR reading order.** EasyOCR returns text in its own detection order. Sorting results by position turned `USA, Edition, Global, Silberschatzs, OPERATING, SYSTEM, CONCEPTS` into `Edition, Silberschatzs, OPERATING, SYSTEM, CONCEPTS`.

**OCR filtering.** A flat confidence threshold discarded the most useful readings: "Glycolic Acid Cream 6% w/w" scored 0.35 and was dropped, while the fragment "Glyco 6" scored 0.80. Length-aware thresholds fixed this.

**Detection model.** Three models were compared on everyday objects:

| Model | Result |
|---|---|
| YOLO26, COCO (80 classes) | Pen detected as "toothbrush"; no glasses, pillows or slippers |
| YOLOv8, Open Images (600 classes) | Rejected: announced "Man", "Person" and "Human face" for one person |
| **YOLO26, Objects365 (365 classes)** | **Chosen:** glasses (0.91), bracelet (0.76), pillow, slippers - one label per object |

**Performance.** 31-43 FPS with detection and OCR loaded, at 1280x720, using about 300 MB of GPU memory on an RTX 5060 Laptop GPU.

Full details: [docs/devlog.md](docs/devlog.md)

---

## Known limitations

- **Speech requires an internet connection** (Microsoft Edge voices via `edge-tts`). Images are processed locally; only the text to be spoken is sent to the speech service.
- **OCR pauses the video** for 1-2 seconds per reading.
- **Objects at the edge of the frame** can be announced more than once as detection confidence fluctuates.
- **Printed faces** on posters or photographs can be announced as "person".
- **No error handling yet** - if the camera disconnects, the program stops without a spoken explanation.
- Tested only on Windows 11.

---

## Installation

Requires Python 3.13 and Windows. An NVIDIA GPU is strongly recommended.

    git clone https://github.com/Srijan-106/Guide.git
    cd Guide
    python -m venv .venv
    .venv\Scripts\activate

Install PyTorch with GPU support first (it is not on PyPI's default index):

    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

Then the remaining dependencies:

    pip install -r requirements.txt

GPU setup, RTX 50-series notes, and Windows Smart App Control issues: [docs/gpu-setup.md](docs/gpu-setup.md)

## Running

    python src\assistant\main.py

---

## Project structure

    Guide/
    ├── src/
    │   ├── assistant/main.py      the application
    │   └── speech/tts.py          text-to-speech module
    ├── archive/                   deprecated standalone scripts, kept for reference
    │   ├── detection/detect.py    first detection program (Days 2-5)
    │   └── ocr/                   first OCR programs (Days 4-5)
    ├── models/                    pretrained model files
    ├── docs/                      development log, GPU setup, images
    └── requirements.txt

The files in `archive/` were superseded by `src/assistant/main.py`. They are kept because the development log describes how they were built.

---

## Project status

- **Day 0** - Research and planning completed.
- **Day 1** - Python environment set up; GPU-accelerated PyTorch verified on an RTX 5060.
- **Day 2** - First object detection module, tested on a live webcam feed.
- **Day 3** - Real-time FPS measurement (~30 FPS) and a structured confidence-threshold experiment, documenting a genuine failure case.
- **Day 4** - OCR implemented; Tesseract failed on real-world objects and was replaced with EasyOCR after direct comparison.
- **Day 5** - Text-to-speech for detection and OCR; change detection so objects are announced once rather than every frame.
- **Day 5 (Part A)** - Upgraded to YOLO26; replaced pyttsx3 with edge-tts after five failures; length-aware OCR filtering; portability fixes.
- **Day 6** - Detection and OCR integrated into one application with mode switching; OCR reading order fixed.
- **Day 6 (Part A)** - Describe, identify and repeat commands; upgraded to an Objects365-pretrained model (365 classes).
- **Day 7** - Diagnosed a Windows Smart App Control block, migrated to stable PyTorch 2.11 (CUDA 12.8) - correcting the Day 1 nightly-build choice - and tidied the repository.

Full day-by-day record: [docs/devlog.md](docs/devlog.md)

---

## Models and attribution

- **Object detection:** [Ultralytics YOLO26](https://docs.ultralytics.com/), pretrained on Objects365
- **Text reading:** [EasyOCR](https://github.com/JaidedAI/EasyOCR)
- **Speech:** [edge-tts](https://pypi.org/project/edge-tts/), using Microsoft Edge's online voices

---

## Research foundation

The project began with a structured review of 2024-2026 academic literature and existing open-source and commercial assistive systems, to identify genuine gaps rather than assume novelty. This project does not claim novelty; its aim is a working system that is honestly evaluated and fully understood.

## Roadmap

- Non-blocking OCR, so reading text no longer pauses the video
- Error handling with spoken explanations
- Fewer repeated announcements for flickering objects
- Systematic evaluation of the finished app, and a demo
- Later: voice commands, offline text-to-speech, LLM scene descriptions, Hindi support