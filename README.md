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

- **Ambient mode** (default) — announces objects as they appear. Passive awareness: *what is around me?*
- **Reading mode** — object announcements are silenced; press SPACE to read text aloud. Deliberate inquiry: *what does this say?*

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

**OCR engine.** Tesseract failed on curved, metallic and stylised text. EasyOCR read all the same test objects correctly — for example "BOLDFIT" on a metallic bottle at 0.99 confidence, where Tesseract returned garbage.

**OCR reading order.** EasyOCR returns text in its own detection order. Sorting results by position turned `USA, Edition, Global, Silberschatzs, OPERATING, SYSTEM, CONCEPTS` into `Edition, Silberschatzs, OPERATING, SYSTEM, CONCEPTS`.

**OCR filtering.** A flat confidence threshold discarded the most useful readings: "Glycolic Acid Cream 6% w/w" scored 0.35 and was dropped, while the fragment "Glyco 6" scored 0.80. Length-aware thresholds fixed this.

**Detection model.** Three models were compared on everyday objects:

| Model | Result |
|---|---|
| YOLO26, COCO (80 classes) | Pen detected as "toothbrush"; no glasses, pillows or slippers |
| YOLOv8, Open Images (600 classes) | Rejected: announced "Man", "Person" and "Human face" for one person |
| **YOLO26, Objects365 (365 classes)** | **Chosen:** glasses (0.91), bracelet (0.76), pillow, slippers — one label per object |

**Performance.** 31–43 FPS with detection and OCR loaded, at 1280x720, using about 300 MB of GPU memory on an RTX 5060 Laptop GPU. Text reading runs on a background thread, so the video does not pause while it works.

**Failure behaviour.** A missing model or busy camera is reported in plain language at startup. A camera lost mid-session is announced and retried for three seconds. If speech itself fails — which cannot be announced by speaking — the app beeps every ten seconds and shows a warning on screen. Without internet it switches to the Windows built-in voice, and returns to the online voice automatically when the network comes back.

Full details: [docs/devlog.md](docs/devlog.md)

---

## Measured results

39 tests, recorded as they were run, on a front-facing laptop webcam in one room.
Method and full caveats: [docs/devlog.md](docs/devlog.md) (Day 9).

**Object detection is reliable for common objects and blind to items outside the
model's vocabulary.** Every failure was silence, never a wrong answer:

| Always detected (13/13) | Never detected (0/6) |
|---|---|
| bottle, phone, headphones, keyboard | cream tube, spray bottle, pen |

The second group is not in Objects365's 365 classes, so the app reads their text
instead — which for a product is usually the more useful half.

| Feature | Result | Tests |
|---|---|---|
| Text reading | 100% usable (3 exact, 3 partial) | 6 |
| Object detection | 66.7% | 24 |
| Object identification | 44.4% (4 correct, 4 no label, 1 wrong) | 9 |

**What these numbers do not show:** reading was tested 6 times of a planned 16,
and handwritten notes were never tested — handwriting is significantly harder
for OCR. The per-lighting breakdown is not reported, because too few tests ran
in each condition for it to mean anything.

## Known limitations

- **Speech needs internet for the better voice.** Without it, speech falls back to the Windows built-in voice, which sounds noticeably more robotic. Images are processed locally; only the text to be spoken is sent to the speech service.
- **Many specific products are not recognised** — cream tubes, spray bottles and paper cups are not among the model's 365 classes. The app reads their text instead, which is usually the more useful half.
- **Printed faces** on posters or photographs can be announced as "person".
- **SPACE and I currently do the same thing**, differing only in that SPACE requires reading mode.
- **Diagnostic output** prints every detection on each identification — useful while developing, noisy for a demonstration.
- **Headphones cannot be identified deliberately** while `CAMERA_FACES_USER` is true: worn items are excluded from the held-object choice, which prevents a common wrong answer at the cost of this rare one. Setting it to false removes the conflict.
- Tested on one machine, running Windows 11 with a front-facing laptop webcam.

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
    ├── src/evaluation/            test recorder and analysis scripts
    ├── evaluation/results.csv     recorded test results
    ├── models/                    pretrained model files
    ├── docs/                      development log, GPU setup, images
    └── requirements.txt

The files in `archive/` were superseded by `src/assistant/main.py`. They are kept because the development log describes how they were built.

---

## Project status

- **Day 0** — Research and planning completed.
- **Day 1** — Python environment set up; GPU-accelerated PyTorch verified on an RTX 5060.
- **Day 2** — First object detection module, tested on a live webcam feed.
- **Day 3** — Real-time FPS measurement (~30 FPS) and a structured confidence-threshold experiment, documenting a genuine failure case.
- **Day 4** — OCR implemented; Tesseract failed on real-world objects and was replaced with EasyOCR after direct comparison.
- **Day 5** — Text-to-speech for detection and OCR; change detection so objects are announced once rather than every frame.
- **Day 5 (Part A)** — Upgraded to YOLO26; replaced pyttsx3 with edge-tts after five failures; length-aware OCR filtering; portability fixes.
- **Day 6** — Detection and OCR integrated into one application with mode switching; OCR reading order fixed.
- **Day 6 (Part A)** — Describe, identify and repeat commands; upgraded to an Objects365-pretrained model (365 classes).
- **Day 7** — Diagnosed a Windows Smart App Control block, migrated to stable PyTorch 2.11 (CUDA 12.8) — correcting the Day 1 nightly-build choice — and tidied the repository.
- **Day 8** — Made the application robust: text reading moved to a background thread so the video no longer freezes; speech given priority, batching and cooldowns after a five-second backlog; camera, model and speech failures explained aloud or by beep; and an offline voice fallback for when the internet drops.
- **Day 9** — Systematic evaluation: 39 recorded tests across detection, identification and reading, with the results and their limits documented.

Full day-by-day record: [docs/devlog.md](docs/devlog.md)

---

## Models and attribution

- **Object detection:** [Ultralytics YOLO26](https://docs.ultralytics.com/), pretrained on Objects365
- **Text reading:** [EasyOCR](https://github.com/JaidedAI/EasyOCR)
- **Speech:** [edge-tts](https://pypi.org/project/edge-tts/), using Microsoft Edge's online voices, with [pyttsx3](https://pypi.org/project/pyttsx3/) as an offline fallback

---

## Research foundation

The project began with a structured review of 2024–2026 academic literature and existing open-source and commercial assistive systems, to identify genuine gaps rather than assume novelty. This project does not claim novelty; its aim is a working system that is honestly evaluated and fully understood.

## Roadmap

- The remaining reading tests, including handwriting
- A demo video
- Put diagnostic output behind a debug switch, and resolve the SPACE / I overlap
- Later: voice commands, LLM scene descriptions, Hindi support