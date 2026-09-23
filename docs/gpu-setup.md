# GPU Setup Notes

VisionAssist uses YOLO (via Ultralytics) and EasyOCR, both of which run significantly faster on an NVIDIA GPU. Neither strictly requires one - both fall back to CPU - but inference is roughly 30x slower without.

---

> **Start here instead:** [docs/setup.md](setup.md) — one command that checks your
> system and installs everything. Come back to this page if something goes wrong.


## Installing PyTorch with GPU support

PyTorch is not pinned in `requirements.txt`, because the correct build depends on your GPU. Install it separately:

    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

This installs the **stable** release built for CUDA 12.8, which supports current NVIDIA GPUs including the RTX 50-series.

- Official install selector: [PyTorch - Get Started](https://pytorch.org/get-started/locally/)
- The CUDA 12.8 package index used above: [download.pytorch.org/whl/cu128](https://download.pytorch.org/whl/cu128)

Verify:

    python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"

Expected: a version ending in `+cu128` with no `.dev` in it, and `CUDA: True`.



---

## NVIDIA RTX 50-series (Blackwell) GPUs

RTX 50-series cards use CUDA compute capability `sm_120`. **Stable PyTorch has supported this since version 2.7**, via the CUDA 12.8 (`cu128`) builds. No nightly or pre-release build is needed. ([source](https://docs.salad.com/container-engine/tutorials/machine-learning/pytorch-rtx5090))

The trap is installing a build for an older CUDA version. This project initially installed a CUDA 12.4 (`cu124`) build, which produced:

    NVIDIA GeForce RTX 5060 Laptop GPU with CUDA capability sm_120 is not
    compatible with the current PyTorch installation.

That error is resolved by installing the `cu128` build shown above - not by switching to a nightly build.

**Correction note:** earlier versions of this project used a PyTorch nightly build in the mistaken belief that stable releases lacked Blackwell support. That was wrong, and the nightly build later caused a Windows security block (see below). The project now uses stable `2.11.0+cu128`.

---

## Windows: Smart App Control

Windows 11's Smart App Control can block compiled Python extension files from loading. The error looks like:

    ImportError: DLL load failed while importing _C: An Application Control
    policy has blocked this file.

Smart App Control blocks are **not** shown in Windows Security's Protection history. To confirm one, open Event Viewer and check:

    Applications and Services Logs > Microsoft > Windows > CodeIntegrity > Operational

**What happened in this project:** Smart App Control moved automatically from Evaluation mode (which blocks nothing) to On (which enforces), without notification. Previously working code then failed to start.

**What was blocked, and the fix:**

| Blocked file | Fix that worked |
|---|---|
| PyTorch nightly (`torch\_C.cp313-win_amd64.pyd`) | Switch to stable `2.11.0+cu128` |
| scipy 1.18.1 (`_ufuncs_cxx`) | Install scipy 1.17 |

Smart App Control judges files partly on reputation, so widely-used releases are more likely to pass than rare or brand-new ones. It has no exception list - individual files cannot be allowed.

If blocks keep recurring, turning Smart App Control off is the usual resolution for Python development. On Windows 11 build 26100.8116 / 26200.8116 or later this is reversible; on earlier builds it is permanent without resetting Windows. Check your build with `winver` before deciding. ([source](https://www.computerworld.com/article/4043925/windows-11-smart-app-control-explained.html))

---

## Tested configuration

| Component | Version |
|---|---|
| GPU | NVIDIA GeForce RTX 5060 Laptop GPU (8 GB) |
| Compute capability | sm_120 (Blackwell) |
| Driver | 592.19 |
| PyTorch | 2.11.0+cu128 (stable) |
| torchvision | 0.26.0+cu128 |
| scipy | 1.17 |
| Python | 3.13 |
| OS | Windows 11, build 26200.9445, Smart App Control On |

## Measured performance

| Metric | Value |
|---|---|
| YOLO inference | ~4-5 ms per frame |
| End-to-end detection loop | 30-50 FPS |
| GPU memory used by the application | ~300 MB |
| GPU utilisation during detection | ~9% |

CPU-only inference measured 130-165 ms per frame - roughly 30x slower.

---

## Checking GPU usage

While the application runs, in a separate terminal:

    nvidia-smi

Look for `python.exe` with type `C` (compute) in the process list. If it does not appear, the application is running on CPU.

---

## References

- [PyTorch - Get Started (official install instructions)](https://pytorch.org/get-started/locally/)
- [PyTorch CUDA 12.8 package index](https://download.pytorch.org/whl/cu128)
- [Running PyTorch on RTX 50-series GPUs - SaladCloud](https://docs.salad.com/container-engine/tutorials/machine-learning/pytorch-rtx5090)
- [Windows 11 Smart App Control explained - Computerworld](https://www.computerworld.com/article/4043925/windows-11-smart-app-control-explained.html)