# Guide (VisionAssist)

A software-first AI accessibility assistant for blind and visually impaired users. Point a camera at the world and hear what is there: objects around you, text on labels and signs, and - in development - which currency note you are holding.

Built as a first major end-to-end machine learning project, with an emphasis on honest evaluation: documented failures, measured results, and the reasoning behind every decision.

---

## What it does

| Feature | Status |
|---|---|
| Real-time object detection, announced aloud | Working |
| Text reading (OCR), announced in reading order | Working |
| Currency recognition (Indian notes) | Model trained and evaluated; not yet in the app |
| Spoken scene summary on demand | Working |
| Repeat last announcement | Working |

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

### Currency classifier (YOLO26n-cls, fine-tuned)

| Test | Accuracy |
|---|---|
| Validation, same source as training | 99.3% |
| **Independent external dataset** | **85.1%** |

The gap is almost entirely one note. External recall per class: Rs.100 98.5%, Rs.200 98.2%, Rs.10 95.7%, Rs.500 95.7%, Rs.50 91.8%, **Rs.20 46.4%**.

Inspecting all 292 misclassified Rs.20 images showed 287 were the **new 2019 green design**, which the training data did not contain. Excluding Rs.20, external accuracy is about 95.5%.

An earlier run reported 99.4%, but a check found **84% train/validation leakage** caused by augmented copies of the same photo appearing on both sides. The split was rebuilt by original photo and the model retrained.

Full details: [experiments/experiment_log.md](experiments/experiment_log.md)

### Other experiments

- **OCR:** Tesseract failed on curved, metallic and stylised text; EasyOCR read all the same test objects correctly.
- **Detection:** a YOLO26 model pretrained on Objects365 (365 classes) detects everyday objects such as glasses, pillows and bracelets that COCO's 80 classes cannot. An Open Images V7 model was tested and rejected for producing redundant labels.

---

## Known limitations

- **The new Rs.20 note is not recognised** by the currency model, and is often misread - sometimes as Rs.500. Not yet safe for real use with Rs.20 notes.
- **Speech requires an internet connection** (Microsoft Edge voices via `edge-tts`). Images are processed locally; the text to be spoken is sent to the speech service.
- **OCR pauses the video** for 1-2 seconds per reading.
- **Objects at the edge of the frame** can be announced more than once as detection confidence fluctuates.
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
    │   ├── speech/tts.py          text-to-speech module
    │   ├── currency/              currency classifier: data preparation,
    │   │                          leakage check, training, external evaluation
    │   ├── detection/detect.py    deprecated - standalone detection (Days 2-5)
    │   └── ocr/                   deprecated - standalone OCR (Days 4-5)
    ├── models/                    pretrained model files
    ├── experiments/               training runs and experiment log
    ├── docs/                      development log, GPU setup, images
    └── requirements.txt

`src/detection/detect.py`, `src/ocr/ocr_easy.py` and `src/ocr/ocr.py` are superseded by `src/assistant/main.py` and kept for reference only, since the development log describes how they were built.

---

## Project status

- **Day 0** - Research and planning completed.
- **Day 1** - Python environment set up; GPU-accelerated PyTorch verified on an RTX 5060.
- **Day 2** - First object detection module, tested on a live webcam feed.
- **Day 3** - Real-time FPS measurement (~30 FPS) and a structured confidence-threshold experiment, documenting a genuine failure case.
- **Day 4** - OCR implemented; Tesseract failed on real-world objects, replaced with EasyOCR after direct comparison.
- **Day 5** - Text-to-speech for detection and OCR; change detection so objects are announced once rather than every frame.
- **Day 5 (Part A)** - Upgraded to YOLO26; replaced pyttsx3 with edge-tts after five failures; length-aware OCR filtering; portability fixes.
- **Day 6** - Detection and OCR integrated into one application with mode switching; OCR reading order fixed.
- **Day 6 (Part A)** - Describe, identify and repeat commands; upgraded to an Objects365-pretrained model (365 classes).
- **Day 7** - Trained a currency classifier; found and fixed 84% data leakage; evaluated on an independent dataset (85.1%) and traced the gap to a missing note design; migrated to stable PyTorch after a Windows security block.

Full day-by-day record: [docs/devlog.md](docs/devlog.md)

---

## Datasets and attribution

- **Training:** [CONFIRM: dataset name, link and licence]. Rs.2000 excluded, as it has been withdrawn from circulation.
- **External evaluation:** [CONFIRM: dataset name, link and licence]. Used for testing only, never for training.
- **Pretrained models:** [Ultralytics YOLO26](https://docs.ultralytics.com/) - detection model pretrained on Objects365; classification model pretrained on ImageNet.

---

## Research foundation

The project began with a structured review of 2024-2026 academic literature and existing open-source and commercial assistive systems, to identify genuine gaps rather than assume novelty. This project does not claim novelty; its aim is a working system that is honestly evaluated and fully understood.

## Roadmap

- Integrate currency recognition into the app, with a "not sure" response when confidence is low
- Retrain with new Rs.20 notes
- Non-blocking OCR and error handling
- Speech input, and offline text-to-speech
- Hindi support