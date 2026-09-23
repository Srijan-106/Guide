"""
One-command setup for VisionAssist.

    python setup.py

Checks the system first, then installs. If the machine cannot run the GPU
build, it says exactly why rather than installing something that will not
work.

    python setup.py --check    check only, install nothing
    python setup.py --cpu      install the CPU build deliberately

Note: this checks the common cases (Python version, NVIDIA driver, GPU
generation). It will be extended as more setups are tested.
"""

import argparse
import shutil
import subprocess
import sys

# PyTorch's own package index for the CUDA 12.8 build
CUDA_INDEX = "https://download.pytorch.org/whl/cu128"

# Minimum Python version the project is tested on
MIN_PYTHON = (3, 10)

# NVIDIA driver versions that can run CUDA 12.x builds
MIN_DRIVER_WINDOWS = 527.41
MIN_DRIVER_LINUX = 525.60


def heading(text):
    print(f"\n{text}\n" + "-" * len(text))


def run(command):
    """Run a command, showing its output live. Returns True if it worked."""
    print(f"\n  > {' '.join(command)}\n")
    return subprocess.run(command).returncode == 0


# ---------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------

def check_python():
    version = sys.version_info
    ok = version >= MIN_PYTHON
    print(f"  Python            {version.major}.{version.minor}.{version.micro}"
          f"   {'OK' if ok else 'TOO OLD'}")
    if not ok:
        print(f"                    Needs {MIN_PYTHON[0]}.{MIN_PYTHON[1]} or newer.")
    return ok


def check_virtual_environment():
    # In a virtual environment, sys.prefix differs from the base install
    inside = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    print(f"  Virtual env       {'yes' if inside else 'NO'}")
    if not inside:
        print("                    Recommended, so this project's packages stay")
        print("                    separate from the rest of your system:")
        print("                      python -m venv .venv")
        print("                      .venv\\Scripts\\activate      (Windows)")
        print("                      source .venv/bin/activate   (macOS/Linux)")
    return inside


def check_gpu():
    """Return (can_use_gpu, reason). Reads the GPU details from nvidia-smi."""
    if shutil.which("nvidia-smi") is None:
        print("  NVIDIA GPU        NOT FOUND")
        return False, ("No NVIDIA driver detected. The project will run on the "
                       "CPU instead - roughly 30x slower, but it does work.")

    try:
        output = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=name,driver_version,compute_cap",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20).stdout.strip()
        name, driver, compute = [part.strip() for part in output.split(",")]
    except Exception as error:
        print(f"  NVIDIA GPU        could not be read ({error})")
        return False, "nvidia-smi did not report usable GPU details."

    print(f"  GPU               {name}")
    print(f"  Driver            {driver}")
    print(f"  Compute capability {compute}")

    minimum = MIN_DRIVER_WINDOWS if sys.platform == "win32" else MIN_DRIVER_LINUX
    try:
        if float(".".join(driver.split(".")[:2])) < minimum:
            return False, (f"Driver {driver} is older than {minimum}, which CUDA "
                           f"12.8 needs. Update it from nvidia.com/drivers.")
    except ValueError:
        pass                                   # unreadable version: do not block

    try:
        if float(compute) < 7.0:
            return False, (f"This GPU's compute capability ({compute}) is older "
                           "than the CUDA 12.8 builds support. The CPU build "
                           "will work: python setup.py --cpu")
    except ValueError:
        pass

    return True, "GPU looks compatible."


# ---------------------------------------------------------------------
# Install and verify
# ---------------------------------------------------------------------

def install(use_gpu):
    pip = [sys.executable, "-m", "pip", "install"]

    heading("Installing PyTorch" + (" (GPU build)" if use_gpu else " (CPU build)"))
    command = pip + ["torch", "torchvision"]
    if use_gpu:
        command += ["--index-url", CUDA_INDEX]
    if not run(command):
        print("\n  PyTorch installation failed. See the messages above.")
        return False

    heading("Installing the remaining packages")
    return run(pip + ["-r", "requirements.txt"])


def verify():
    heading("Verifying")
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
        if "Application Control policy" in result.stderr:
            print("  Windows Smart App Control has blocked a file that was just")
            print("  installed. See docs/gpu-setup.md for how to fix this.")
        return False
    return True


# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Set up VisionAssist.")
    parser.add_argument("--check", action="store_true",
                        help="check the system without installing anything")
    parser.add_argument("--cpu", action="store_true",
                        help="install the CPU build even if a GPU is present")
    arguments = parser.parse_args()

    heading("System check")
    python_ok = check_python()
    check_virtual_environment()
    gpu_ok, reason = check_gpu()

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

    if arguments.check:
        print("\n  Check only - nothing was installed.")
        return 0

    if not install(use_gpu):
        return 1

    if not verify():
        return 1

    heading("Done")
    print("  Run the application with:  python src/assistant/main.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())