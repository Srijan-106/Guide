# Development Log

This file tracks daily progress on the Guide  project.
Each entry corresponds to a handwritten diary entry, expanded here in digital form.

---

## Day 0 — [5th sept]

**Goal:** Project foundation setup — no functional code written yet.

**What was done:**

- Installed Git and configured it for Windows
- Cloned the repository locally
- Created project folder structure (`src/detection`, `src/ocr`, `src/currency`,
  `src/speech`, `src/assistant`, `models/`, `experiments/`, `docs/`)
- Created `.gitignore`
- Wrote initial `README.md`
- Created this development log

**What was learned:**
- How Git and GitHub relate to each other (Git is the tool, GitHub is the hosting service)
- What a virtual environment, `.gitignore`, and `src` folder convention are for
- Conducted a research review of the assistive-technology landscape (2024-2026 papers,
  GitHub projects, commercial products) before writing any code

**Next goal:** Day 1 — Python virtual environment and dependency setup.


---

## Day 1 — [6th sept]

**Goal:** Set up an isolated Python virtual environment and verify GPU access.

**What was done:**
- Created a virtual environment (`.venv`) specific to this project
- Installed Ultralytics (YOLO) and OpenCV
- Discovered a fresh install defaults to CPU-only PyTorch
- Reinstalled PyTorch using the nightly build with CUDA 12.8 support,
  required specifically because the RTX 5060 Laptop GPU's architecture
  (Blackwell, compute capability sm_120) is not yet supported by the
  stable PyTorch release
- Verified GPU detection: confirmed working (RTX 5060 Laptop GPU recognized)
- Generated `requirements.txt` to freeze the exact working environment
- Committed and pushed changes to GitHub

**What was learned:**
- The difference between a fresh pip install defaulting to CPU-only builds
  versus explicitly requesting GPU-enabled builds
- Why very new GPU hardware sometimes requires nightly/pre-release software
  builds until official stable support catches up
- What `pip freeze` and `requirements.txt` are for, and why they matter
  for reproducibility

**Problems encountered:**
- Virtual environment creation was interrupted once (`KeyboardInterrupt`),
  requiring deletion and a clean recreation
- Needed to verify whether the installed PyTorch was CPU-only or the
  correct CUDA build before proceeding

**Next goal:** Day 2 — first real object detection code using YOLOv8n.


---

## Day 2 — [6 sept]

**Goal:** Write the first real object detection code and test it on a live webcam feed.

**What was done:**
- Created `src/detection/detect.py`
- First version tested detection on a static sample image (bus.jpg),
  confirming YOLOv8n correctly identified a bus and 3 people with
  confidence scores, running on GPU (4.3ms inference time)
- Rewrote the code to use a live webcam feed instead of a static image,
  using an infinite loop (`while True`) combined with a `break` condition
  triggered by pressing 'q'
- Used Ultralytics' built-in `.plot()` method to automatically draw
  bounding boxes and labels onto each frame
- Organized project files: moved `yolov8n.pt` into `models/`, removed
  the temporary test image
- Committed and pushed the first real working application code

**What was learned:**
- The precise difference between confidence (a model's self-reported
  certainty about one prediction) and accuracy (a measured correctness
  rate against verified ground-truth data across many examples)
- The difference between speed metrics (preprocess/inference/postprocess
  timing) and correctness metrics — these are independent of each other
- Why an infinite loop (`while True`) is necessary for continuous video
  processing, and how `break` provides the only exit condition
- How `cv2.VideoCapture`, `camera.read()`, and `cv2.imshow()` work together
  to build a live video display loop

**Problems encountered:**
- First run failed due to a network interruption during the model
  download (KeyboardInterrupt) — resolved by simply retrying without
  interrupting.

**Next goal:** Day 3 — Measure real-time FPS, test confidence thresholds, document at least one detection failure case.