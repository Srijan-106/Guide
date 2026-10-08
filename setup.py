"""
One-command setup for VisionAssist.

From a fresh clone, the whole install is one line (Windows PowerShell):

    git clone https://github.com/Srijan-106/Guide.git; cd Guide; python setup.py

What it does, in order:

  1. CHECKS the system - Python, GPU, driver, GPU memory, free disk space,
     Ollama. If the machine cannot run the project, it says exactly why,
     instead of installing something that will not work.
  2. Creates the project's virtual environment (.venv) if needed, and
     continues inside it - so nothing is installed into the system Python.
  3. INSTALLS PyTorch (the GPU build when possible), then requirements.txt.
  4. DOWNLOADS the models now, so the first run of the app does not stall:
       text reading         EasyOCR                 about 100 MB
       speech recognition   Whisper "small"         about 460 MB
       Hindi translation    NLLB-200 (600M)         about 2.4 GB   optional
       visual questions     qwen3-vl via Ollama     about 3.3 GB   optional
  5. REPORTS which features are ready, and how to start the app.

Each optional feature is downloaded on its own: if one fails, the rest
still install, and the app runs without it - exactly as the app itself
treats optional features.

Options:
    python setup.py --check       check only - installs nothing
    python setup.py --cpu         install the CPU build, even with a GPU
    python setup.py --no-hindi    skip the Hindi translation model
    python setup.py --no-vision   skip the visual-question model

MINIMUM REQUIREMENTS - measured on the development laptop (Windows 11,
RTX 5060 Laptop GPU 8 GB, Python 3.13), 6-7 October 2026:

    Windows 10 or 11        tested on 11. The offline voice, the beep and
                            the Hindi screen font are Windows features.
    Python 3.12 or newer    numpy 2.5.2 in requirements.txt needs 3.12
    NVIDIA GPU, 4 GB        detection, text reading and speech
                            recognition use about 3.3 GB together
    NVIDIA GPU, 8 GB        to add visual questions: 7.7 of 8 GB were in
                            use with everything loaded
    Disk                    about 6 GB for the core, 12 GB with everything
    Internet                for setup, and for the natural voice - without
                            it the app uses the Windows voice
    Webcam; microphone      the microphone only for voice commands

Without an NVIDIA GPU everything still runs on the CPU, much more slowly
(detection 130-165 ms per frame instead of 4-5 ms).
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------

# PyTorch's own package index for the CUDA 12.8 build
CUDA_INDEX = "https://download.pytorch.org/whl/cu128"

# numpy 2.5.2, pinned in requirements.txt, needs Python 3.12 (checked on
# PyPI, 7 October 2026). The earlier minimum of 3.10 would have let the
# install start, then fail halfway through requirements.txt.
MIN_PYTHON = (3, 12)

# NVIDIA driver versions that can run CUDA 12.x builds
MIN_DRIVER_WINDOWS = 527.41
MIN_DRIVER_LINUX = 525.60

# GPU memory, in GB, from measurements on an 8 GB card: detection, text
# reading and Whisper together used about 3.3 GB; with the vision model
# loaded too, 7.7 of 8 GB were in use.
GPU_GB_CORE = 4
GPU_GB_VISION = 8

# Disk space, in GB. The installed packages (.venv) measured 5.3 GB and go
# in the project folder; the models are stored in the user's home folder,
# which may be on a different drive.
DISK_GB_PACKAGES = 5.5
DISK_GB_CORE_MODELS = 0.6          # EasyOCR + Whisper small
DISK_GB_HINDI = 2.5                # NLLB-200
DISK_GB_VISION = 3.5               # qwen3-vl:4b-instruct

# The detection model ships with the repository. Must match
# DETECTION_MODEL in src/assistant/main.py.
DETECTION_MODEL = Path("models/yolo26m-objv1-150.pt")

# Packages that easyocr pulls in as dependencies, which this project must
# NOT have. Removed straight after installing requirements.txt:
#   opencv-python-headless - OpenCV without window support. It installs
#       into the same cv2 folder as the full opencv-python and, installed
#       second, replaced its files: on 8 October, after running setup,
#       OpenCV reported "GUI: NONE" and the app's video window could not
#       open.
#   python-bidi - blocked by Windows Smart App Control on the development
#       machine (Day 10). The app uses its own stand-in in src/bidi.
UNWANTED_PACKAGES = ["opencv-python-headless", "python-bidi"]

VENV_DIR = Path(".venv")


# ---------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------

def heading(text):
    print(f"\n{text}\n" + "-" * len(text))


def run(command):
    """Run a command, showing its output live. Returns True if it worked."""
    print(f"\n  > {' '.join(str(part) for part in command)}\n")
    return subprocess.run(command).returncode == 0


def warn_if_blocked(text):
    """Explain Windows Smart App Control blocks, which look like crashes."""
    if "Application Control policy" in text:
        print("  Windows Smart App Control has blocked a file that was just")
        print("  installed. See docs/gpu-setup.md for how to fix this.")


def run_python(code, label, capture=False):
    """Run Python code in this environment, as a separate process.

    A separate process, so a crash or a Windows block while loading one
    model cannot stop the rest of setup.

    capture=False shows the output live - downloads print progress bars,
    and without them a 2.4 GB download looks like a frozen screen.
    capture=True keeps the output, for when setup needs to read it.

    Returns (worked, captured output or "").
    """
    print(f"\n  {label}...", flush=True)
    if not capture:
        worked = subprocess.run([sys.executable, "-c", code]).returncode == 0
        if not worked:
            print(f"  FAILED: {label}. See the messages above. If they "
                  f"mention 'Application Control policy', see "
                  f"docs/gpu-setup.md.")
        return worked, ""

    result = subprocess.run([sys.executable, "-c", code],
                            capture_output=True, text=True)
    if result.returncode != 0:
        # The last line of the error says what went wrong
        error = result.stderr.strip().splitlines()
        print(f"  FAILED: {error[-1] if error else 'unknown error'}")
        warn_if_blocked(result.stderr)
        return False, ""
    return True, result.stdout.strip()


def free_gb(path):
    """Free space, in GB, on the drive holding path."""
    return shutil.disk_usage(path).free / 1024 ** 3


def venv_python():
    """The Python inside the project's virtual environment."""
    if sys.platform == "win32":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def in_project_venv():
    """True when running inside THIS project's .venv - not just any venv."""
    try:
        return Path(sys.prefix).resolve() == VENV_DIR.resolve()
    except OSError:
        return False


# ---------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------

def check_python():
    version = sys.version_info
    ok = version >= MIN_PYTHON
    print(f"  Python             {version.major}.{version.minor}.{version.micro}"
          f"   {'OK' if ok else 'TOO OLD'}")
    if not ok:
        print(f"                     Needs {MIN_PYTHON[0]}.{MIN_PYTHON[1]} or "
              f"newer - the pinned packages will not install on older ones.")
        print("                     Download it from python.org.")
    return ok


def check_gpu():
    """Return (can_use_gpu, gpu_memory_gb, reason), read from nvidia-smi."""
    if shutil.which("nvidia-smi") is None:
        print("  NVIDIA GPU         NOT FOUND")
        return False, 0, ("No NVIDIA driver detected. The project can run on "
                          "the CPU instead - much slower, but it works: "
                          "python setup.py --cpu")

    try:
        output = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=name,driver_version,compute_cap,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20).stdout.strip()
        # One line per GPU; the first is the one PyTorch uses by default
        first = output.splitlines()[0]
        name, driver, compute, memory = [part.strip() for part in first.split(",")]
        memory_gb = float(memory) / 1024
    except Exception as error:
        print(f"  NVIDIA GPU         could not be read ({error})")
        return False, 0, "nvidia-smi did not report usable GPU details."

    print(f"  GPU                {name}")
    print(f"  GPU memory         {memory_gb:.1f} GB")
    print(f"  Driver             {driver}")
    print(f"  Compute capability {compute}")

    minimum = MIN_DRIVER_WINDOWS if sys.platform == "win32" else MIN_DRIVER_LINUX
    try:
        if float(".".join(driver.split(".")[:2])) < minimum:
            return False, memory_gb, (
                f"Driver {driver} is older than {minimum}, which CUDA 12.8 "
                f"needs. Update it from nvidia.com/drivers.")
    except ValueError:
        pass                                   # unreadable version: do not block

    try:
        if float(compute) < 7.0:
            return False, memory_gb, (
                f"This GPU's compute capability ({compute}) is older than "
                f"the CUDA 12.8 builds support. The CPU build will work: "
                f"python setup.py --cpu")
    except ValueError:
        pass

    return True, memory_gb, "GPU looks compatible."


def check_gpu_memory(memory_gb, want_vision):
    """Warn, without stopping, when GPU memory is below what was measured."""
    if memory_gb == 0:
        return
    if memory_gb < GPU_GB_CORE - 0.5:
        print(f"\n  NOTE: {memory_gb:.1f} GB of GPU memory is less than the "
              f"{GPU_GB_CORE} GB the core features used in testing. Expect")
        print("  slowdowns or out-of-memory errors.")
    elif want_vision and memory_gb < GPU_GB_VISION - 0.5:
        print(f"\n  NOTE: visual questions used most of an {GPU_GB_VISION} GB "
              f"GPU in testing. With {memory_gb:.1f} GB, Ollama will run part")
        print("  of the vision model on the CPU, so answers will be slow.")
        print("  To skip it:  python setup.py --no-vision")


def check_disk(want_hindi, want_vision):
    """Return False if there is clearly not enough space to finish."""
    models_gb = (DISK_GB_CORE_MODELS + (DISK_GB_HINDI if want_hindi else 0)
                 + (DISK_GB_VISION if want_vision else 0))
    project_free = free_gb(".")
    home_free = free_gb(Path.home())
    same_drive = Path(".").resolve().anchor == Path.home().resolve().anchor

    ok = True
    if same_drive:
        needed = DISK_GB_PACKAGES + models_gb
        print(f"  Free disk space    {project_free:.0f} GB   (needs about "
              f"{needed:.0f} GB)")
        ok = project_free >= needed
    else:
        # Packages go in the project folder; models in the home folder
        print(f"  Free disk space    {project_free:.0f} GB on the project "
              f"drive (needs about {DISK_GB_PACKAGES:.0f} GB)")
        print(f"                     {home_free:.0f} GB on the home drive, "
              f"where models are stored (needs about {models_gb:.0f} GB)")
        ok = project_free >= DISK_GB_PACKAGES and home_free >= models_gb
    if not ok:
        print("                     NOT ENOUGH - free some space, or skip "
              "the optional models (--no-hindi, --no-vision).")
    return ok


def check_ollama(want_vision):
    """Ollama is a separate program, needed only for visual questions."""
    if not want_vision:
        print("  Ollama             skipped (--no-vision)")
        return False
    if shutil.which("ollama"):
        print("  Ollama             found")
        return True
    print("  Ollama             NOT FOUND - visual questions will be off.")
    print("                     To add them: install Ollama from ollama.com,")
    print("                     then run  python setup.py  again.")
    return False


# ---------------------------------------------------------------------
# Install
# ---------------------------------------------------------------------

def install_packages(use_gpu):
    pip = [sys.executable, "-m", "pip", "install"]

    heading("Installing PyTorch" + (" (GPU build)" if use_gpu else " (CPU build)"))
    command = pip + ["torch", "torchvision"]
    if use_gpu:
        command += ["--index-url", CUDA_INDEX]
    if not run(command):
        print("\n  PyTorch installation failed. See the messages above.")
        return False

    # Before requirements.txt: openai-whisper and transformers depend on
    # torch, and if it were missing pip would fetch the CPU-only build
    heading("Installing the remaining packages")
    return run(pip + ["-r", "requirements.txt"])


def read_text_file(path):
    """Read a text file whatever its encoding.

    A file saved by Windows PowerShell 5.1 ("pip freeze > requirements.txt",
    or Out-File) is UTF-16, not UTF-8. pip reads both; reading it as UTF-8
    crashed setup on 8 October. The first bytes (the "BOM") say which.
    """
    raw = Path(path).read_bytes()
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return raw.decode("utf-16")
    return raw.decode("utf-8-sig", errors="replace")


def pinned_version(package):
    """The version requirements.txt pins for a package, or None."""
    try:
        text = read_text_file("requirements.txt")
    except OSError:
        return None
    for line in text.splitlines():
        line = line.split("#")[0].strip()          # ignore comments
        if line.lower().startswith(package.lower() + "=="):
            return line.split("==", 1)[1].strip()
    return None


def remove_unwanted_packages():
    """Undo what easyocr's dependencies installed - see UNWANTED_PACKAGES."""
    heading("Removing packages the project must not have")
    pip = [sys.executable, "-m", "pip"]

    # Everything needed to put OpenCV back is worked out BEFORE anything is
    # removed. On 8 October the version was looked up after the removal,
    # the lookup crashed, and OpenCV was left missing entirely.
    version = pinned_version("opencv-python")
    package = f"opencv-python=={version}" if version else "opencv-python"
    reinstall = pip + ["install", "--force-reinstall", "--no-deps", package]

    # Not installed is fine: pip just says it is skipping them
    run(pip + ["uninstall", "-y"] + UNWANTED_PACKAGES)

    # Removing the headless build also deletes files in the cv2 folder it
    # shares with the full build, so the full build is put back over it
    if run(reinstall):
        return True

    print("  Putting the full OpenCV back failed. To repair it by hand:")
    print(f"    python -m pip {' '.join(reinstall[3:])}")
    return False


def verify_installation():
    heading("Checking the installation")
    code = ("import torch;"
            "print('  PyTorch', torch.__version__);"
            "print('  CUDA available:', torch.cuda.is_available());"
            "print('  Device:', torch.cuda.get_device_name(0)"
            " if torch.cuda.is_available() else 'CPU')")
    result = subprocess.run([sys.executable, "-c", code],
                            capture_output=True, text=True)
    print(result.stdout, end="")
    if result.returncode != 0:
        print(result.stderr)
        warn_if_blocked(result.stderr)
        return False

    # The app shows its video in a window, which the headless OpenCV
    # cannot open - so window support is checked, not assumed
    code = ("import cv2;"
            "gui = [line.split(':', 1)[1].strip()"
            " for line in cv2.getBuildInformation().splitlines()"
            " if line.strip().startswith('GUI:')];"
            "print(gui[0] if gui else 'unknown')")
    result = subprocess.run([sys.executable, "-c", code],
                            capture_output=True, text=True)
    gui = result.stdout.strip()
    if result.returncode != 0 or gui == "NONE":
        print("  OpenCV window support: MISSING")
        print("  The video window cannot open. Run  python setup.py  again;")
        print("  if this stays, see docs/gpu-setup.md.")
        return False
    print(f"  OpenCV window support: {gui}")
    return True


# ---------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------
# The model names are read from the app's own code, so there is one
# source of truth: changing a model there changes what setup downloads.

def download_core_models():
    """Text reading and speech recognition. Returns (reading, voice)."""
    heading("Downloading models")

    reading_ok, _ = run_python(
        "import sys; sys.path.insert(0, 'src');"
        "import easyocr;"                   # needs the bidi stub in src/
        "easyocr.Reader(['en'], gpu=False, verbose=False)",
        "Text reading (EasyOCR, about 100 MB)")

    voice_ok, _ = run_python(
        "import sys; sys.path.insert(0, 'src');"
        "import whisper;"
        "from voice.listener import MODEL_SIZE;"
        "whisper.load_model(MODEL_SIZE, device='cpu')",
        "Speech recognition (Whisper, about 460 MB)")

    return reading_ok, voice_ok


def download_hindi_model():
    ok, _ = run_python(
        "import sys; sys.path.insert(0, 'src');"
        "from language.translator import MODEL;"
        "from transformers import AutoTokenizer, AutoModelForSeq2SeqLM;"
        "AutoTokenizer.from_pretrained(MODEL);"
        "AutoModelForSeq2SeqLM.from_pretrained(MODEL)",
        "Hindi translation (NLLB-200, about 2.4 GB - this takes a while)")
    return ok


def download_vision_model():
    found, model = run_python(
        "import sys; sys.path.insert(0, 'src');"
        "from vision.asker import MODEL; print(MODEL)",
        "Visual questions: reading the model name", capture=True)
    if not found or not model:
        return False
    model = model.splitlines()[-1].strip()   # the name is the last line printed

    print(f"  Visual questions: downloading {model} through Ollama "
          f"(about 3.3 GB)")
    if run(["ollama", "pull", model]):
        return True

    print("  The download did not work. Ollama must be running: start the")
    print("  Ollama app, then run  python setup.py  again.")
    return False


# ---------------------------------------------------------------------

def report(status):
    heading("Ready")
    for feature, ready in status:
        print(f"  {feature:<36} {'ready' if ready else 'NOT available'}")

    if sys.platform == "win32":
        start = r".venv\Scripts\python src\assistant\main.py"
    else:
        start = ".venv/bin/python src/assistant/main.py"
    print("\n  Start the app from this folder with:\n")
    print(f"    {start}")
    print("\n  Keys: M mode, SPACE read, D describe, I identify, R repeat,")
    print("        V speak, L Hindi/English, Q quit")


def main():
    parser = argparse.ArgumentParser(description="Set up VisionAssist.")
    parser.add_argument("--check", action="store_true",
                        help="check the system without installing anything")
    parser.add_argument("--cpu", action="store_true",
                        help="install the CPU build even if a GPU is present")
    parser.add_argument("--no-hindi", action="store_true",
                        help="skip the Hindi translation model (about 2.4 GB)")
    parser.add_argument("--no-vision", action="store_true",
                        help="skip the visual-question model (about 3.3 GB)")
    arguments = parser.parse_args()
    want_hindi = not arguments.no_hindi
    want_vision = not arguments.no_vision

    # Always work from the project folder, wherever setup was started from
    os.chdir(Path(__file__).resolve().parent)

    # --- Use the project's own virtual environment ---------------------
    # Python is checked first: a virtual environment inherits the Python
    # it was made with, so creating one from a too-old Python is wasted.
    if not arguments.check and not in_project_venv():
        heading("Python")
        if not check_python():
            return 1
        if not venv_python().exists():
            print(f"\n  Creating the virtual environment in {VENV_DIR}/ ...")
            if not run([sys.executable, "-m", "venv", str(VENV_DIR)]):
                print("  Could not create it. See the messages above.")
                return 1
        print(f"\n  Continuing inside {VENV_DIR}/ ...")
        return subprocess.run([str(venv_python()), __file__,
                               *sys.argv[1:]]).returncode

    # --- Check ---------------------------------------------------------
    heading("System check")
    python_ok = check_python()
    print(f"  Virtual env        {'yes (.venv)' if in_project_venv() else 'no'}")
    gpu_ok, gpu_memory, reason = check_gpu()
    disk_ok = check_disk(want_hindi, want_vision)
    ollama_found = check_ollama(want_vision)
    check_gpu_memory(gpu_memory, want_vision and ollama_found)

    print()
    if not python_ok:
        print("  Cannot continue: Python is too old.")
        return 1

    use_gpu = gpu_ok and not arguments.cpu
    if arguments.cpu:
        print("  Installing the CPU build, as requested.")
    elif gpu_ok:
        print("  " + reason)
    else:
        print("  COMPATIBILITY ISSUE")
        print("  " + reason)
        print("\n  To install the CPU build instead:  python setup.py --cpu")
        return 1

    if not disk_ok:
        print("  Cannot continue: not enough free disk space (see above).")
        return 1

    if not DETECTION_MODEL.exists():
        print(f"  Cannot continue: the detection model {DETECTION_MODEL} is "
              f"missing. It ships with the repository - clone it again.")
        return 1

    if arguments.check:
        print("\n  Check only - nothing was installed.")
        return 0

    # --- Install -------------------------------------------------------
    if (not install_packages(use_gpu) or not remove_unwanted_packages()
            or not verify_installation()):
        return 1

    reading_ok, voice_ok = download_core_models()
    hindi_ok = download_hindi_model() if want_hindi else False
    vision_ok = download_vision_model() if ollama_found else False

    report([
        ("Object detection", True),
        ("Text reading", reading_ok),
        ("Voice commands (V key)", voice_ok),
        ("Hindi answers" + ("" if want_hindi else " (skipped)"), hindi_ok),
        ("Visual questions" + ("" if want_vision else " (skipped)"), vision_ok),
    ])
    return 0 if reading_ok else 1


if __name__ == "__main__":
    sys.exit(main())