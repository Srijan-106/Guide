<h2 align="center">🎬 DEMO</h2>

https://github.com/user-attachments/assets/e09bf79f-5e12-42a1-af50-f5497edc1bff

<hr>

<h1 align="center">Guide</h1>

<p align="center"><i>A camera assistant for blind and visually impaired people</i></p>

**Guide helps blind and visually impaired people understand the world around them.**

The idea behind Guide is a pair of AI-powered smart glasses that tell a blind or visually impaired person what is around them, hands-free. Building the glasses themselves needs specialised hardware, so this project focuses on the part that is software: the system that sees, reads, listens and speaks, built and tested on a laptop and webcam.

Point the camera at your room and hear what is there. Guide names the objects around you, reads the text in front of you, and answers your own spoken questions, such as *"What am I wearing?"*, *"Where is my bag?"* or *"What is around me?"*, in English or Hindi.

Your camera pictures stay on your own computer. Only the words to be spoken go online, and if there is no internet, Guide keeps working with a built-in offline voice.

I built Guide on my own as a complete project, from the idea and research to the working program, the testing and the documentation. I tested it with 117 tests. When asked about something that was not in the room, it correctly said there was nothing there, 4 out of 4 times, instead of making up an answer.

## Features

- **Object awareness.** Recognises 365 kinds of everyday objects and announces them as they come into view.
- **Text reading.** Reads labels, pages and signs aloud in reading order, with common OCR misreadings corrected.
- **Scene summary.** Describes everything currently in view, on request.
- **Identify what you hold.** Names an object held up to the camera and reads the text on it.
- **Visual questions.** Answers open questions about the camera view using a local vision model, with left and right given from the user's own point of view.
- **Voice control.** Push-to-talk commands and questions in English and Hindi.
- **Hindi mode.** Hindi voice, Hindi commands and answers translated into Hindi, with object names kept in English as most Hindi speakers say them.
- **Live captions.** Everything spoken also appears on screen, so a helper can follow along.

Guide has two modes. **Ambient** (default) announces objects as they appear. **Reading** stays quiet about objects and reads text aloud when SPACE is pressed.

## Controls

| Key | Action |
|---|---|
| **M** | Switch between Ambient and Reading mode |
| **SPACE** | Read the text in view (Reading mode), up to 20 words |
| **D** | Describe everything currently in view |
| **I** | Name the object being held up and read a few words on it |
| **R** | Repeat the last announcement |
| **V** | Speak: press to start, press again to stop. Say a command or ask a question |
| **L** | Switch between English and Hindi |
| **Q** | Quit |

Voice commands do the same as the keys: "read this", "what is this", "describe", "repeat", "reading mode", "ambient mode", "Hindi", "stop", and their Hindi equivalents ("पढ़ो", "यह क्या है", "आसपास क्या दिख रहा है", "फिर से बोलो"). Anything else is treated as a question about the camera view. Click the video window first so the keys register.

## Installation

One line in **Windows PowerShell**:

    git clone https://github.com/Srijan-106/Guide.git; cd Guide; python setup.py

`setup.py` checks your computer (Python, GPU, driver, GPU memory, disk space, Ollama), creates a virtual environment, installs the dependencies, downloads the models and finishes with a table showing which features are ready. Use `--no-hindi` or `--no-vision` to skip optional parts, and `--cpu` to install without GPU support.

**Before you start:** Python 3.12 or newer and Git. For visual questions, also [Ollama](https://ollama.com).

| Requirement | Details |
|---|---|
| Operating system | Windows 10 or 11 (tested on 11) |
| GPU, core features | NVIDIA, 4 GB |
| GPU, with visual questions | NVIDIA, 8 GB |
| Disk | About 6 GB, or 12 GB with everything |
| Internet | For setup, and for the natural voice |

Without an NVIDIA GPU everything runs on the CPU, more slowly. Full guide: [docs/setup.md](docs/setup.md). Troubleshooting (RTX 50-series, Windows Smart App Control): [docs/gpu-setup.md](docs/gpu-setup.md).

**Run** from the project folder:

    .venv\Scripts\python src\assistant\main.py

## How it works

A camera frame goes to the detector and the text reader. Voice commands are recognised locally and either run an action or become a question for the vision model. Every answer passes through one speech queue, which prioritises requests, merges announcements and falls back to the offline voice if the network drops. In Hindi mode, answers are translated before they are spoken.

| Part | Technology | Licence |
|---|---|---|
| Object detection | [Ultralytics YOLO26](https://docs.ultralytics.com/), pretrained on Objects365 | AGPL-3.0 |
| Text reading | [EasyOCR](https://github.com/JaidedAI/EasyOCR) | Apache-2.0 |
| Speech recognition | [OpenAI Whisper](https://github.com/openai/whisper) ("small"), run locally | MIT |
| Visual questions | [Qwen3-VL 4B Instruct](https://ollama.com/library/qwen3-vl), run locally with [Ollama](https://ollama.com) | see the model page |
| Left and right of hands | [MediaPipe](https://github.com/google-ai-edge/mediapipe) hand landmarker | Apache-2.0 |
| Hindi translation | [NLLB-200 distilled 600M](https://huggingface.co/facebook/nllb-200-distilled-600M) (Meta), run locally | **CC-BY-NC 4.0, non-commercial use only** |
| Speech | [edge-tts](https://pypi.org/project/edge-tts/) online voices, with [pyttsx3](https://pypi.org/project/pyttsx3/) offline | see each project |
| Hindi text on screen | [HarfBuzz](https://github.com/harfbuzz/uharfbuzz) and [FreeType](https://pypi.org/project/freetype-py/) | see each project |

Each choice was made by comparing alternatives on real objects. The comparisons are in [docs/devlog.md](docs/devlog.md): EasyOCR over Tesseract, Objects365 over COCO and Open Images, Qwen3-VL over four other vision models, and NLLB-200 over the vision model's own translation.

## Evaluation

117 tests in two evaluations. The test lists were fixed before testing, and every result was checked against the application's log. Method and details: [docs/devlog.md](docs/devlog.md).

**Feature evaluation (86 tests)**

| Feature | Result |
|---|---|
| Object detection | 17/30 (56.7%) |
| Object identification | 4/10 |
| Text reading | 12/16 (75%) |
| Voice commands, English | 7/7 |
| Voice commands, Hindi | 5/7 |
| Visual questions, English | 8/8 |
| Visual questions, Hindi | 2/4 |
| Hindi translation | 3/4 |
| Trap questions (things that are not there) | 4/4, nothing invented |

Where detection missed, Guide stayed silent instead of naming the wrong object, and it reads the text on the item instead.

**Room sweep (31 objects).** Every object in one room was listed before the application was started, then shown in order. Guide named 21 correctly (67.7%), said nothing for 7, and named 3 wrongly, all near misses such as a calculator called a "cell phone".

Results come from one tester, one room and one front-facing laptop webcam.

## Performance

| Measurement | Value |
|---|---|
| Object detection | About 4–5 ms per frame on the GPU |
| Live video with everything running | About 20–25 FPS |
| GPU memory | About 3.3 GB for core features; 7.7 of 8 GB with visual questions |
| Visual question | 3–5 s to the start of the answer |
| Hindi translation (CPU) | 1.2–1.9 s per sentence |

## Current limitations

- The natural voice needs internet. Offline, the Windows voice is used.
- Detection covers 365 object classes. Some everyday items, such as keys, chargers and cream tubes, are not recognised. Where an item has text on it, Guide reads that instead.
- Handwriting is largely not read.
- Hindi speech recognition is less accurate than English, and Hindi answers are translated from English.
- Visual questions need an NVIDIA GPU with 8 GB of memory, and their answers are spoken as "It looks like …".
- Developed and tested on Windows 11 with a laptop webcam.

## Documentation

| Document | Contents |
|---|---|
| [docs/devlog.md](docs/devlog.md) | Day-by-day development record, problems, fixes and reasoning |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Components, threads, data flows, models and design decisions |
| [docs/PRD.md](docs/PRD.md) | The problem, the users, scope and constraints |
| [docs/DESIGN_SYSTEM.md](docs/DESIGN_SYSTEM.md) | How the application speaks, and how the screen is drawn |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Where new code belongs and what must not break |
| [docs/setup.md](docs/setup.md) | Installation and minimum requirements |
| [docs/gpu-setup.md](docs/gpu-setup.md) | Troubleshooting for RTX 50-series and Windows Smart App Control |

## Project structure

    Guide/
    ├── README.md                    this file
    ├── CONTRIBUTING.md              rules for changing the project
    ├── setup.py                     one-command installer
    ├── src/
    │   ├── assistant/main.py        the application: camera, modes, keys, commands
    │   ├── speech/tts.py            speech queue, online voice, offline fallback
    │   ├── voice/listener.py        microphone, Whisper, voice commands
    │   ├── vision/
    │   │   ├── asker.py             visual questions through Ollama
    │   │   └── hands.py             left or right hand, from MediaPipe
    │   ├── language/
    │   │   ├── phrases.py           every spoken sentence, English and Hindi
    │   │   ├── translator.py        NLLB translation, glossary, grammar polishing
    │   │   └── spelling.py          look-alike-letter spelling correction
    │   ├── display/
    │   │   ├── hud.py               captions, status, key strip, detection boxes
    │   │   └── shaping.py           Hindi text shaping (HarfBuzz)
    │   ├── bidi/                    stand-in for python-bidi (see its docstring)
    │   └── evaluation/              test recorders, analysis and check scripts
    ├── evaluation/                  recorded test results and the fixed test lists
    ├── models/                      the detection model
    ├── archive/                     earlier standalone scripts, kept for reference
    ├── docs/                        devlog, architecture, requirements, guides, images
    └── requirements.txt

## License

The project's own code is released under the MIT licence (see [LICENSE](LICENSE)). The libraries and models it uses keep their own licences, which apply when you use or redistribute them, in particular Ultralytics YOLO (AGPL-3.0) and NLLB-200 (non-commercial use only).
