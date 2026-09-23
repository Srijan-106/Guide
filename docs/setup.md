# Setup

One command installs everything:

    python setup.py

It checks your system first — Python version, virtual environment, GPU, driver —
then installs. If your machine cannot run the GPU build, it says exactly why and
stops, rather than installing something that will not work.

## Full steps, from nothing

    git clone https://github.com/Srijan-106/Guide.git
    cd Guide
    python -m venv .venv
    .venv\Scripts\activate
    python setup.py

On macOS or Linux, the fourth line is `source .venv/bin/activate`.

## Options

| Command | What it does |
|---|---|
| `python setup.py` | Check, then install |
| `python setup.py --check` | Check only — installs nothing |
| `python setup.py --cpu` | Install the CPU build, even if a GPU is present |

## What it checks

| Check | If it fails |
|---|---|
| Python 3.10 or newer | Stops — the project needs a newer Python |
| Running inside a virtual environment | Warns, and shows how to create one |
| NVIDIA GPU present | Falls back to the CPU build, with a note |
| Driver new enough for CUDA 12.8 | Reports a compatibility issue and stops |
| GPU generation supported | Reports a compatibility issue and suggests `--cpu` |

## Running the application

    python src\assistant\main.py

## A GPU is not required

The project runs on the CPU, roughly 30x slower: 130–165 ms per frame, versus
4–5 ms on a GPU. Everything works; the video is just less smooth.

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

Verified on Windows 11 with an NVIDIA RTX 5060 Laptop GPU (driver 592.19,
Python 3.13). Machines without a GPU, with older drivers, or with older GPUs are
handled by the checks but have not yet been verified — if you hit a problem on
one of those, the details in `gpu-setup.md` should help.