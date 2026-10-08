# Setup

## One line

In **Windows PowerShell**:

    git clone https://github.com/Srijan-106/Guide.git; cd Guide; python setup.py

In the older **Command Prompt**, use `&&` instead of `;`.

That is the whole install. `setup.py` checks your computer, creates the project's
own virtual environment (`.venv`), installs everything into it, downloads the
models, and finishes with a table of which features are ready.

**Before you start**, two programs must already be installed:

- **Python 3.12 or newer** — from [python.org](https://www.python.org/downloads/).
  During installation, tick **"Add python.exe to PATH"**.
- **Git** — from [git-scm.com](https://git-scm.com/downloads).

Optional, for visual questions ("what am I wearing?"):

- **Ollama** — from [ollama.com](https://ollama.com). Install it before running
  setup and setup downloads the vision model for you; install it later and simply
  run `python setup.py` again.

## Minimum requirements

Measured on the development laptop (Windows 11, RTX 5060 Laptop GPU with 8 GB,
Python 3.13), 6–7 October 2026.

| | Needed | Why |
|---|---|---|
| Operating system | Windows 10 or 11 | Tested on 11. The offline voice, the warning beep and the Hindi screen font are Windows features |
| Python | 3.12 or newer | numpy 2.5.2, pinned in `requirements.txt`, needs 3.12 |
| GPU, core features | NVIDIA, 4 GB | Detection, text reading and speech recognition used about 3.3 GB together |
| GPU, with visual questions | NVIDIA, 8 GB | 7.7 of 8 GB were in use with everything loaded |
| Disk | about 6 GB, or 12 GB with everything | Installed packages 5.3 GB; models below |
| Internet | for setup, and for the natural voice | Without it the app uses the Windows voice |
| Webcam | required | |
| Microphone | for voice commands only | Everything also works from the keyboard |

**No NVIDIA GPU?** Everything still runs on the CPU, much more slowly — object
detection takes 130–165 ms per frame instead of 4–5 ms. Install with
`python setup.py --cpu`.

## What setup downloads

| Model | Feature | Size | Optional |
|---|---|---|---|
| YOLO26, Objects365 | Object detection | 48 MB | No — included in the repository |
| EasyOCR | Text reading | about 100 MB | No |
| Whisper "small" | Voice commands | about 460 MB | Works without it, keyboard only |
| NLLB-200 (600M) | Hindi answers to visual questions | about 2.4 GB | Yes: `--no-hindi` |
| qwen3-vl:4b-instruct, via Ollama | Visual questions | about 3.3 GB | Yes: `--no-vision` |

Each optional model is downloaded on its own: if one fails, the rest still
install, and the app runs without that feature.

The models are stored in your home folder, not in the project — Whisper in
`.cache\whisper`, NLLB in `.cache\huggingface`, EasyOCR in `.EasyOCR`, and the
vision model in `.ollama`. Setup checks both drives for space if your project is
on a different drive.

## Options

| Command | What it does |
|---|---|
| `python setup.py` | Check, then install everything |
| `python setup.py --check` | Check only — installs nothing |
| `python setup.py --cpu` | Install the CPU build, even if a GPU is present |
| `python setup.py --no-hindi` | Skip the Hindi translation model |
| `python setup.py --no-vision` | Skip the visual-question model |

Options can be combined: `python setup.py --cpu --no-vision`.

Running setup again is safe: anything already installed or downloaded is reused.

## What it checks

| Check | If it fails |
|---|---|
| Python 3.12 or newer | Stops — the pinned packages will not install on older versions |
| NVIDIA GPU present | Stops and suggests `--cpu` |
| Driver new enough for CUDA 12.8 | Stops and says which version is needed |
| GPU generation supported | Stops and suggests `--cpu` |
| GPU memory | Warns if below 4 GB, or below 8 GB with visual questions |
| Free disk space | Stops if there is clearly not enough |
| Ollama installed | Continues without visual questions, and says how to add them |
| OpenCV can open windows | `easyocr` installs a windowless OpenCV (`opencv-python-headless`) and `python-bidi`; setup removes both and restores the full OpenCV, then checks the video window can open |

## Starting the app

From the project folder:

    .venv\Scripts\python src\assistant\main.py

Keys: **M** mode · **SPACE** read · **D** describe · **I** identify ·
**R** repeat · **V** speak · **L** Hindi / English · **Q** quit

---

## If something goes wrong

**[docs/gpu-setup.md](gpu-setup.md)** is the detailed troubleshooting guide. It
covers:

- `CUDA capability sm_120 is not compatible` — RTX 50-series cards
- `An Application Control policy has blocked this file` — Windows Smart App Control
- Checking whether the GPU is actually being used
- The exact configuration this project was tested on

---

## Tested configurations

Verified on Windows 11 with an NVIDIA RTX 5060 Laptop GPU (8 GB, driver 592.19),
Python 3.13. The checks for machines without a GPU, with older drivers, older
GPUs or less GPU memory were tested by simulating those machines, not on real
ones. If you hit a problem on one of them, the details in `gpu-setup.md` should
help.