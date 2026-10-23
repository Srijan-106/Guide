## Demo

[Watch the demo](https://github.com/Srijan-106/Guide/releases/download/v1.0/Demo.mp4)

<h1 align="center">Guide</h1>

A software-first AI accessibility assistant for blind and visually impaired users. Point a camera at the world and hear what is there: the objects around you, the text and answers to your own questions about what the camera sees, in English or Hindi.

Built as a first major end-to-end machine learning project, with an emphasis on honest evaluation: documented failures, measured results, and the reasoning behind every decision.

Everything runs on the laptop except the natural voice. **Camera images never leave the machine**; only the text to be spoken is sent to the online voice service, and without internet the Windows voice is used instead.

---

## What it does

| Feature | Status |
|---|---|
| Real-time object detection (365 object types), announced aloud | Working |
| Text reading (OCR), in reading order, with careful spelling correction | Working |
| Spoken summary of everything in view, on demand | Working |
| Identify a held-up object and read the text on it | Working |
| Voice commands (push-to-talk), in English and Hindi | Working |
| Open questions about what the camera sees ("what am I wearing?"), answered by a local vision model | Working |
| Hindi mode: Hindi voice, Hindi commands, and answers translated into Hindi | Working |
| Live on-screen captions of everything said, for sighted helpers | Working |
| Repeat the last thing said | Working |

The application runs in two modes:

- **Ambient mode** (default): announces objects as they appear. Passive awareness: *what is around me?*
- **Reading mode**: object announcements are silenced; SPACE reads text aloud. Deliberate inquiry: *what does this say?*

Modes exist because testing showed object announcements interrupting text readings, which was confusing in practice.

Hindi mode is **Hinglish** by design: sentences in Hindi, object names in English ("मुझे दिख रहा है: bottle, chair"), which is how most Hindi speakers say them.

## Controls

| Key | Action |
|---|---|
| M | Switch between Ambient and Reading mode |
| SPACE | Read the text in view (Reading mode): up to 20 words, no object name |
| D | Describe everything currently visible |
| I | "What is this?": name the held-up object and read a few words on it |
| R | Repeat the last thing said |
| V | Speak: press to start, press again to stop. A command, or any question about what the camera sees |
| L | Switch language: English / Hindi |
| Q | Quit |

Voice commands do the same as the keys: "read this", "what is this", "describe", "repeat", "reading mode", "ambient mode", "Hindi", "stop", and their Hindi equivalents ("पढ़ो", "यह क्या है", "आसपास क्या दिख रहा है", "फिर से बोलो" …). Anything that is not a command is treated as a question for the vision model.

The video window must be focused (clicked) for keys to register.

---

## Installation

One line in **Windows PowerShell**:

    git clone https://github.com/Srijan-106/Guide.git; cd Guide; python setup.py

`setup.py` checks your computer first (Python, GPU, driver, GPU memory, disk space, Ollama), creates the project's own virtual environment, installs everything, downloads the models, and ends with a table of which features are ready. Optional parts can be skipped with `--no-hindi` or `--no-vision`, and `--cpu` installs without GPU support.

**Before you start:** Python 3.12 or newer and Git. For visual questions, also [Ollama](https://ollama.com).

| Requirement | Measured |
|---|---|
| Operating system | Windows 10 or 11 (tested on 11) |
| GPU, core features | NVIDIA, 4 GB (detection, OCR and speech recognition used about 3.3 GB) |
| GPU, with visual questions | NVIDIA, 8 GB (7.7 of 8 GB in use with everything loaded) |
| Disk | About 6 GB, or 12 GB with everything |
| Internet | For setup, and for the natural voice |

Without an NVIDIA GPU everything still runs on the CPU, much more slowly.

Full setup guide: [docs/setup.md](docs/setup.md) · Troubleshooting (RTX 50-series, Windows Smart App Control): [docs/gpu-setup.md](docs/gpu-setup.md)

## Running

From the project folder:

    .venv\Scripts\python src\assistant\main.py

---

## Measured results

Two separate evaluations, **117 tests in total**. In both, the test list was fixed before any test was run, each result was recorded as it happened, and every row was afterwards checked against the application's own log. Method and every correction: [docs/devlog.md](docs/devlog.md) (Days 9, 12, 13 and 15).

### 1. The feature evaluation — 86 tests

A list of ten objects, eight texts, fourteen voice commands and twelve questions, fixed on Day 9 and Day 12. Three of the ten objects were deliberately chosen as known weak cases.

| Feature | Success | Notes |
|---|---|---|
| Object detection | 17/30 = 56.7% | bottle, phone, headphones, keyboard 12/12; cream tube, spray bottle 0/6 |
| Object identification | 4/10 = 40% | 4 correct, 5 named nothing and read the text instead, 1 wrong |
| Text reading (exact or partial) | 12/16 = 75% | normal light 7/8, dim 5/8; only 3 exact |
| Voice commands, English | 7/7 | transcribed in 0.4–1.1 s |
| Voice commands, Hindi | 5/7 | 2 misheard; transcribed in 0.8–2.5 s |
| Visual questions, English | 8/8 | |
| Visual questions, Hindi | 2/4 | 1 partly correct, 1 wrong (the question was misheard) |
| Hindi translation | 3/4 | "yellow case" became "yellow bag" |
| **Trap questions** (asking about things that are not there) | **4/4 passed, 0 made up** | no cup, no dog, no watch: each answered "there is none" |

The last row matters most: a confident wrong answer cannot be checked by someone who cannot see. On the chosen object list, every detection failure was **silence, never a wrong name** — the safer failure. The app reads the text on such items instead, which for a product is usually the more useful half.

### 2. The room sweep — 31 objects

The first evaluation asks *"of these ten chosen objects, how many does it name?"*. This one asks a different question: *"of the objects actually in a room, how many does it name?"* Every object visible in one room was written down **before the application was started**, then shown in that order.

| | Result |
|---|---|
| Named correctly | **21/31 = 67.7%** |
| Said nothing | 7 |
| Gave the wrong name | 3 |

**Not named:** wired earbuds, key, speaker, door, clothes, a miniature bike, a phone charger.

**Named wrongly,** and all three are near misses rather than nonsense: a cat doll → "teddy bear", a calculator → "cell phone", a spray bottle → "bottle" (true, but not specific enough to identify it).

**Three objects were announced but not heard.** A hanger, a cap and a power socket were recorded as misses, and the log showed the application had named all three: the announcements arrived merged into a sentence with other objects and went past unnoticed. The rows were corrected, with the log evidence written into each, and the problem is recorded as a limitation.

### What these numbers do not show

One attempt per test, one speaker, one room, one front-facing laptop webcam, one lighting condition for the sweep. 100% on 7 or 8 tests is promising, not proof. Handwriting is largely not read. The 86 results describe the application **before** the fixes made on Day 13; each of those fixes was checked separately afterwards.

## Evaluation highlights

**OCR engine.** Tesseract failed on curved, metallic and stylised text. EasyOCR read the same objects correctly, for example "BOLDFIT" on a metallic bottle at 0.99 confidence, where Tesseract returned garbage.

**Detection model.** Three models were compared on everyday objects:

| Model | Result |
|---|---|
| YOLO26, COCO (80 classes) | Pen detected as "toothbrush"; no glasses, pillows or slippers. (With Objects365 the pen **is** detected — see the room sweep) |
| YOLOv8, Open Images (600 classes) | Rejected: announced "Man", "Person" and "Human face" for one person |
| **YOLO26, Objects365 (365 classes)** | **Chosen:** glasses (0.91), bracelet (0.76), pillow, slippers; one label per object |

**Vision model.** Five local vision-language models were tried. Three gave confident wrong answers (wrong shirt colour, text invented on a bottle), and the feature was first dropped. Qwen3-VL answered accurately; its 8B version was too large to share an 8 GB GPU with the rest of the app (one answer took 95 s), and the **4B Instruct** version answers in a few seconds.

**Hindi answers.** Asked to answer in Hindi directly, the vision model invented objects, so it answers in English and the answer is translated. The vision model's own translation changed facts ("on the right wrist" became "on the knee of the right hand"), and Google's free endpoint refused every request. **NLLB-200** translated all four test sentences correctly on the CPU, with a glossary that keeps object names intact and a small step that fixes two of its grammar slips.

**Spelling correction.** A spell checker fixed OCR slips (PERFORMANGE → performance) but also damaged names (Glycolic → glycol, Hazrat → hazard). Every helpful correction swapped letters that look alike; every damaging one did not. Corrections are now accepted only for look-alike letters: 22 of 22 logged cases handled correctly.

**Failure behaviour.** A missing model or busy camera is reported in plain language at startup. A camera lost mid-session is announced and retried. If the internet drops, the very first failed announcement is spoken again with the Windows voice, and speech stays offline until the network returns. If speech fails entirely, which cannot be announced by speaking, the app beeps every ten seconds. Optional features (voice commands, visual questions, Hindi translation) switch themselves off if unavailable, and the rest keeps working.

## Performance

| Measurement | Value |
|---|---|
| Object detection (YOLO inference) | about 4–5 ms per frame on the GPU |
| Video with detection running | 34–35 FPS (measured on Day 6, before the current display) |
| GPU memory | about 3.3 GB without visual questions; 7.7 of 8 GB with everything loaded |
| Visual question, app running | 3–5 s to the start of the answer |
| Hindi translation (NLLB, CPU) | 1.2–1.9 s per sentence |

## Known limitations

- **Speech needs internet for the natural voice.** Without it, the Windows voice is used, which sounds noticeably more robotic.
- **Some objects are never named, and a few are named wrongly.** Cream tubes are not among the model's 365 classes. A spray bottle is named only "bottle". In the room sweep, a key, a speaker, a door, clothes, a phone charger and wired earbuds were not named at all, and a calculator was called a "cell phone". Where an object has text on it, the app reads that instead, which for a product is usually the more useful half.
- **Handwriting is largely not read.** Neat handwriting only in good light.
- **A misheard question gets a confident answer to a different question.**
- **An announcement can be spoken and still missed.** Objects appearing together are merged into one sentence, and a name can go past unnoticed. Three objects in the room sweep were recorded as misses until the log showed they had been announced.
- **Hindi speech recognition is weaker than English.**
- **Visual questions need an 8 GB GPU**, and the model can state details it is unsure of, so answers are spoken as "It looks like …".
- **Identification picks the smallest box covering the centre**, so a box drawn inside a held object can win.
- **Objects at the edge of the frame flicker and are re-announced.** In one 25-minute session the log shows a bracelet announced 18 times, a nightstand 14, a hat 13 and a shelf 12, despite a 30-second cooldown per object.
- **Printed faces** on posters or photographs can be announced as "person".
- Tested on one machine, running Windows 11 with a front-facing laptop webcam.

---

## Documentation

| Document | What it holds |
|---|---|
| [docs/devlog.md](docs/devlog.md) | The day-by-day record: what was built, every problem, its fix and the reasoning behind it |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | How the system is built: components, threads, data flows, the models, and every decision with its reason |
| [docs/PRD.md](docs/PRD.md) | The problem, who it is for, what is built and what is not, and the constraints |
| [docs/DESIGN_SYSTEM.md](docs/DESIGN_SYSTEM.md) | How the application speaks — the honesty rules come first — and how the screen is drawn |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Where new code belongs, what must not break, when to stop and ask |
| [docs/setup.md](docs/setup.md) | Installing it, and the measured minimum requirements |
| [docs/gpu-setup.md](docs/gpu-setup.md) | Troubleshooting: RTX 50-series, Windows Smart App Control |

---

## Project structure

    Guide/
    ├── README.md                    this file
    ├── CONTRIBUTING.md              rules for changing the project
    ├── setup.py                     one-command installer
    ├── src/
    │   ├── assistant/main.py        the application: camera, modes, keys, commands
    │   ├── speech/tts.py            speech queue, online voice, offline fallback
    │   ├── voice/listener.py        microphone, Whisper, voice commands
    │   ├── vision/asker.py          visual questions through Ollama
    │   ├── language/
    │   │   ├── phrases.py           every spoken sentence, English and Hindi
    │   │   ├── translator.py        NLLB translation, glossary, grammar polishing
    │   │   └── spelling.py          look-alike-letter spelling correction
    │   ├── display/
    │   │   ├── hud.py               captions, status, key strip, detection boxes
    │   │   └── shaping.py           Hindi text shaping (HarfBuzz)
    │   ├── bidi/                    stand-in for python-bidi (see its docstring)
    │   └── evaluation/              test recorders, analysis, correction and check scripts
    ├── evaluation/                  recorded test results, and the fixed test lists
    ├── models/                      the detection model
    ├── archive/                     earlier standalone scripts, kept for reference
    ├── docs/                        devlog, architecture, requirements,
    │                                design system, setup guides, images
    └── requirements.txt

The files in `archive/` were superseded by the application. They are kept because the development log describes how they were built.

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
- **Day 7** — Diagnosed a Windows Smart App Control block, migrated to stable PyTorch 2.11 (CUDA 12.8), correcting the Day 1 nightly-build choice, and tidied the repository.
- **Day 8** — Made the application robust: background text reading, prioritised and batched speech, failures explained aloud or by beep, and an offline voice fallback.
- **Day 9** — Systematic evaluation method, recording and analysis scripts; the first 39 of 56 planned tests recorded.
- **Day 10** — Voice commands with local Whisper (push-to-talk); separate jobs for SPACE and I; a debug switch; an Indian English voice; a stand-in for a blocked package.
- **Day 11** — Visual questions with a local vision model after comparing five; Hindi mode with local NLLB translation and a glossary; a new display with live English and Hindi captions.
- **Day 12** — Checked GPU memory and speed; finished the Day 9 tests, with every result checked against the app's log; wrote the test plan for the new features before testing.
- **Day 13** — Tested voice, vision and Hindi features (86 tests in total, 4/4 trap questions passed); rebuilt the Hindi glossary; fixed the slow first answer, lost speech on network failure, dropped announcements, Hindi command recognition and the spell checker.
- **Day 14** — Repeat replays the last real message; Hindi model warm-up; one-line setup for other machines; the codebase read against its records; repository history cleaned (234 MB → 45.5 MB).
- **Day 15** — A second evaluation: 31 objects listed before the application was started, then shown in order — 67.7% named. Documentation: architecture, product requirements, design system and contributing guide.

Full day-by-day record: [docs/devlog.md](docs/devlog.md)

---

## Models and attribution

| Part | Model or library | Licence |
|---|---|---|
| Object detection | [Ultralytics YOLO26](https://docs.ultralytics.com/), pretrained on Objects365 | AGPL-3.0 |
| Text reading | [EasyOCR](https://github.com/JaidedAI/EasyOCR) | Apache-2.0 |
| Speech recognition | [OpenAI Whisper](https://github.com/openai/whisper) ("small"), run locally | MIT |
| Visual questions | [Qwen3-VL 4B Instruct](https://ollama.com/library/qwen3-vl), run locally with [Ollama](https://ollama.com) | see the model page |
| Hindi translation | [NLLB-200 distilled 600M](https://huggingface.co/facebook/nllb-200-distilled-600M) (Meta), run locally | **CC-BY-NC 4.0, non-commercial use only** |
| Speech | [edge-tts](https://pypi.org/project/edge-tts/) (Microsoft Edge's online voices), with [pyttsx3](https://pypi.org/project/pyttsx3/) as the offline fallback | see each project |
| Hindi text on screen | [HarfBuzz](https://github.com/harfbuzz/uharfbuzz) and [FreeType](https://pypi.org/project/freetype-py/) | see each project |

This project's own code is released under the MIT licence (see [LICENSE](LICENSE)). The libraries and models it uses keep their own licences, which apply when you use or redistribute them, in particular Ultralytics YOLO (AGPL-3.0) and NLLB-200 (non-commercial only).

---

## Research foundation

The project began with a structured review of 2024–2026 academic literature and existing open-source and commercial assistive systems, to identify genuine gaps rather than assume novelty. This project does not claim novelty; its aim is a working system that is honestly evaluated and fully understood.
