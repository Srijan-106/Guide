# GPU Setup Notes

VisionAssist uses YOLO (via Ultralytics) and EasyOCR, both of which run significantly faster on an NVIDIA GPU. Neither strictly requires one — both fall back to CPU — but inference is roughly 20-30x slower without.

---

## Standard setup (most users)

No special steps are needed. Installing the project requirements pulls in a compatible PyTorch automatically:

   ```bash
pip install -r requirements.txt
```

Verify GPU access with:

  ```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```
Expected output on a working GPU setup:

    CUDA available: True
    Device: NVIDIA GeForce RTX ....

If it reports `CPU`, the project still runs — just more slowly.

---

## NVIDIA RTX 50-series (Blackwell) GPUs — special case

RTX 50-series cards use CUDA compute capability `sm_120`. As of this writing, the **stable PyTorch release does not support it**. Installing normally produces:

    NVIDIA GeForce RTX 5060 Laptop GPU with CUDA capability sm_120 is not
    compatible with the current PyTorch installation.
    The current PyTorch install supports CUDA capabilities sm_50 sm_60 sm_61
    sm_70 sm_75 sm_80 sm_86 sm_90.

Note that `torch.cuda.is_available()` may still report `True` in this state — but any attempt to actually use the GPU will fail or silently fall back to CPU.

### Fix: install the CUDA 12.8 nightly build

    pip uninstall torch torchvision -y
    pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128

**Why a nightly build:** support for new GPU architectures lands in PyTorch's development branch before it reaches an official stable release. "Nightly" here means recently built from the main codebase — it is still official PyTorch from official servers, not a third-party build. This is currently the only way to use Blackwell GPUs with PyTorch.

### Verify the fix

    python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

The `sm_120 is not compatible` warning should no longer appear.

---

## Tested configuration

This project was developed and tested on:

| Component | Version |
|---|---|
| GPU | NVIDIA GeForce RTX 5060 Laptop GPU (8 GB) |
| Compute capability | sm_120 (Blackwell) |
| Driver | 592.19 |
| CUDA | 12.8 |
| PyTorch | 2.12.0.dev20260408+cu128 |
| torchvision | 0.27.0.dev20260407+cu128 |
| Python | 3.13 |
| OS | Windows 11 |

### Measured performance on this configuration

| Metric | Value |
|---|---|
| YOLO inference | ~4-5 ms per frame |
| End-to-end detection loop | 30-50 FPS |
| GPU memory used by the application | ~300 MB |
| GPU utilisation during detection | ~9% |

For comparison, CPU-only inference measured 130-165 ms per frame — roughly 30x slower.

---

## Checking your own GPU usage

While the application is running, in a separate terminal:

    nvidia-smi

Look for `python.exe` in the process list, and note the `Memory-Usage` and `GPU-Util` columns. If `python.exe` does not appear, the application is running on CPU.