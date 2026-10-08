# Development Log

This file tracks daily progress on the Guide (VisionAssist) project.
Each entry corresponds to a handwritten diary entry, expanded here in digital form.

Where a later day showed an earlier entry to be wrong, the original text is
kept and a dated correction note is added, rather than rewriting history.

---

## Day 0 — [5 sept]

**Goal:** Project foundation setup — no functional code written yet.

**What was done:**

- Installed Git and configured it for Windows
- Cloned the repository locally
- Created project folder structure (`src/detection`, `src/ocr`,
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

## Day 1 — [6 sept]

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

> **Correction (added Day 7):** the statement above is wrong. Stable PyTorch
> has supported Blackwell (sm_120) GPUs since version 2.7, through its CUDA
> 12.8 (`cu128`) builds. The failure seen on this day came from installing a
> **CUDA 12.4** stable build; the correct fix was the **stable CUDA 12.8**
> build, not a nightly one. The nightly build later caused a Windows security
> block, and the project moved to stable `2.11.0+cu128` on Day 7.

**What was learned:**
- The difference between a fresh pip install defaulting to CPU-only builds
  versus explicitly requesting GPU-enabled builds
- Why very new GPU hardware sometimes requires nightly/pre-release software
  builds until official stable support catches up *(see correction above —
  this was not actually the case here)*
- What `pip freeze` and `requirements.txt` are for, and why they matter
  for reproducibility

**Problems encountered:**
- Virtual environment creation was interrupted once (`KeyboardInterrupt`),
  requiring deletion and a clean recreation
- Needed to verify whether the installed PyTorch was CPU-only or the
  correct CUDA build before proceeding

> **Correction (Day 7):** the nightly build was never necessary. Stable PyTorch
> has supported `sm_120` since version 2.7, through its CUDA 12.8 (`cu128`)
> builds. The original error came from installing a CUDA 12.4 build. The project
> moved to stable `2.11.0+cu128` on Day 7.

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


---

## Day 3 — [7 sept]

**Goal:** Measure real-time FPS, test confidence threshold tradeoffs, document a genuine detection failure case.

**What was done:**
- Added live FPS measurement to `detect.py` using Python's `time` module,
  calculating frame-to-frame processing speed rather than relying only on
  the model's internal inference timing
- Confirmed real-world end-to-end performance: approximately 30 FPS at
  default settings, running fully on GPU
- Tested two confidence thresholds (0.2 and 0.7) on identical surroundings
  to observe the false-positive vs. missed-detection tradeoff directly
- Identified and analyzed a real detection failure case from Day 2's testing

**Experiment Log — Confidence Threshold Comparison**

| Setting | conf = 0.2 | conf = 0.7 |
|---|---|---|
| FPS observed | 27 | 41 |
| Detections shown | 4 (2 correct, 2 false positives) | 1 (correct) |
| False positives | "person 0.27" (empty wall area), "person 0.24" (pile of clothes) | None observed |
| Missed detections | None | Backpack/bed region not labeled at all |
| Conclusion | Lower threshold catches more but introduces false positives; higher threshold is cleaner but risks missing ambiguous objects | |

**Failure Case Analysis**

- **What happened:** A backpack sitting on a bed was labeled "bed" (confidence
  0.46-0.57) instead of being identified as a backpack, or correctly separating
  the two objects.
- **Likely cause:** The bounding box region was likely dominated visually by
  the bedding/blanket pattern, causing the model to classify based on the
  larger surrounding context rather than the smaller foreground object. This
  may also reflect a general known limitation of single-pass detectors like
  YOLO when handling overlapping or partially-occluded objects.
- **Why this matters for the project:** For a blind user relying on this
  system, a misidentified object like this could give inaccurate information
  about their surroundings. This is a genuine limitation to be transparent
  about in the project's documentation, not hidden.

**What was learned:**
- FPS measured end-to-end (capture + detection + drawing + display) differs
  meaningfully from raw model inference time alone, and is the more honest
  real-world performance metric
- Confidence threshold is a genuine tradeoff, not a "more accurate = better"
  setting — the right value depends on whether false positives or missed
  detections are more costly for the specific use case
- Real evaluation requires deliberately looking for failure cases, not just
  showcasing successful detections

**Next goal:** Day 4 — Begin OCR (Optical Character Recognition) for text reading.


---

## Day 4 — [8 sept]

**Goal:** Implement OCR (text reading from images) and achieve reliable
real-world performance.

**What was done:**
- Installed Tesseract OCR engine (standalone C++ program, separate from Python)
  and `pytesseract` as the Python bridge
- Resolved a `TESSDATA_PREFIX` configuration issue where Tesseract's
  compiled-in default path did not match the actual Windows installation
- Built and tested OCR on a static screenshot (successful), then on live
  webcam captures triggered by the space bar
- Found Tesseract failed badly on real-world objects; investigated the
  failure pattern through controlled testing
- Switched to EasyOCR (neural-network-based OCR trained on real-world
  photos rather than scanned documents) and re-tested identical objects
- Resolved an OpenCV conflict caused by EasyOCR installing
  `opencv-python-headless` alongside the standard `opencv-python`

**Experiment Log — Tesseract vs EasyOCR (identical test objects)**

| Test object | Conditions | Tesseract result | EasyOCR result |
|---|---|---|---|
| Saltwater bottle | Curved, high contrast, good lighting | "SALTWATER breeze" — correct | Correct |
| BOLDFIT bottle | Curved, metallic, low contrast | "a c (a) je]" — garbage | "BOLDFIT" (conf 0.99) — correct |
| Algorithms book | Angled, large stylized title font | Empty result — no text returned at all | "ALGORITHMS" (1.00), "INTRODUCTION TO" (0.75), "THIRD EDITION" (0.81), "CLIFFORD STEIN" (0.96) — correct |

**Visual evidence:**

Tesseract failing on the BOLDFIT bottle (metallic, curved, low contrast):

![Tesseract failure on BOLDFIT bottle](images/day4_tesseract_boldfit_fail.png)

EasyOCR reading the same bottle correctly:

![EasyOCR success on BOLDFIT bottle](images/day4_easyocr_boldfit_success.png)

Tesseract failing entirely on the angled book cover (returned no text at all):

![Tesseract failure on book cover](images/day4_tesseract_book_fail.png)

EasyOCR reading the same book cover correctly:

![EasyOCR success on book cover](images/day4_easyocr_book_success.png)


**Key finding:** All three objects that completely failed with Tesseract were
read correctly by EasyOCR. The root cause was not curvature alone — the
Saltwater bottle is also curved but worked with both engines — but rather
Tesseract's fundamental design assumption of flat, scanned, document-style
text. EasyOCR, trained on real-world photographs of text, handles curved
surfaces, low contrast, glare, and stylized fonts substantially better.

**Additional observation:** EasyOCR returns per-result confidence scores.
Incorrect/garbage readings consistently carry very low confidence (0.00–0.05),
while correct readings carry high confidence (0.75–1.00), meaning noise can
be filtered by threshold — the same principle applied to object detection
confidence on Day 3. Confidence values are being kept visible during
development for diagnostic purposes rather than filtered out prematurely.

**Problems encountered and resolved:**

| Problem | Cause | Fix |
|---|---|---|
| `TesseractError: Failed loading language 'eng'` | Tesseract's compiled-in default `tessdata` path pointed to a Linux build path that doesn't exist on Windows | Explicitly set `os.environ["TESSDATA_PREFIX"]` to the real installation path |
| `cv2.error: The function is not implemented` on `cv2.imshow()` | EasyOCR installed `opencv-python-headless` (built without GUI support) alongside the existing `opencv-python`, causing a conflict | Uninstalled the headless variant, then force-reinstalled `opencv-python` |
| `ModuleNotFoundError: No module named 'cv2'` | Uninstalling the headless package also removed shared underlying files | `pip install --force-reinstall opencv-python` |

**What was learned:**
- Tesseract is a standalone external program; `pytesseract` is only a bridge,
  not the engine itself — architecturally different from YOLO, which runs
  natively inside Python via PyTorch
- Classical OCR (Tesseract) and neural-network OCR (EasyOCR) have
  fundamentally different robustness characteristics on real-world input
- Diagnosing a failure correctly matters: the initial hypothesis (curvature
  causes failure) was disproven by a counter-example (a curved bottle that
  worked), leading to a more accurate conclusion about contrast, glare,
  and font style being the controlling factors for Tesseract
- Package dependency conflicts can silently break previously-working
  functionality — the same OpenCV that worked fine on Days 2–3 broke purely
  because a new package pulled in a conflicting variant

**Next goal:** Day 5 — Text-to-speech: make OCR and object detection results
speak aloud rather than only printing to terminal, making both features
genuinely usable by the target user for the first time.


---

## Day 5 — [9 sept]

**Goal:** Add text-to-speech output so that OCR and object detection results
are spoken aloud rather than only printed to a terminal — making the project
genuinely usable by its target user (blind and visually impaired people) for
the first time.

**Why this matters:** Every feature built before today printed results to a
terminal window. For a blind user, terminal output is useless. Today is the
first day the project produces output its actual audience can perceive.

**What was done:**
- Installed `pyttsx3`, an offline text-to-speech library that drives SAPI5,
  the speech engine built into Windows
- Built a dedicated, swappable speech module (`src/speech/tts.py`) so that
  changing the TTS engine later requires modifying one file rather than
  every file that produces output
- Integrated speech into OCR: readings above 0.5 confidence are spoken;
  lower-confidence garbage is still printed for diagnostics but not spoken
- Added explicit audible feedback when no text is found ("No readable text
  found"), since silence is ambiguous for a blind user
- Integrated speech into object detection using a change-detection design
  with flicker tolerance
- Diagnosed and resolved four separate speech failures (documented below)

**Design decision — why offline TTS:** `pyttsx3` was chosen over cloud
services (ElevenLabs, Google TTS, Azure) despite lower voice quality,
because it requires no internet connection, no API keys, and no per-use
cost. A blind user should not need connectivity to know what is in front
of them, and images of their private surroundings should not be sent to a
remote server. This aligns with the offline/privacy-first gap identified
during Day 0 research. The speech module is deliberately isolated so a
higher-quality engine (e.g. Piper, which is also offline and supports
Hindi) can replace it later without touching any other file.

**Design decision — change detection for object announcements:** Object
detection runs continuously at ~30 FPS. Announcing every detection every
frame would be unusable. Instead, the system tracks which objects are
currently believed present and announces only newly-appeared objects.
An object must be undetected for 60 consecutive frames (~2 seconds)
before being forgotten and becoming eligible for re-announcement. This
threshold was chosen deliberately: detection flicker typically lasts 1-3
frames, while a genuine removal-and-return takes 1-3 seconds, so the
threshold sits comfortably in the gap between them.

**Errors encountered and resolved:**

**Error 1 — Only the first announcement ever spoke**
- *Symptom:* First `speak()` call worked; all subsequent calls printed
  normally but produced no audio, with no error raised.
- *Diagnosis:* Reproduced in isolation outside the project with a two-word
  test, confirming it was a library behaviour rather than an integration bug.
- *Cause:* After `runAndWait()` completes, pyttsx3's internal event loop
  terminates. The engine still accepts `say()` calls without error, but no
  active loop remains to process them — a silent failure.
- *Fix attempted:* Create a fresh engine per utterance.

**Error 2 — `RuntimeError: run loop already started`**
- *Symptom:* Crashes in worker threads when multiple objects appeared close
  together.
- *Cause:* Spawning one thread per announcement meant multiple pyttsx3
  engines ran concurrently. SAPI5 cannot handle simultaneous speech engines.
- *Fix:* Replaced per-announcement threads with a single permanent worker
  thread consuming from a `queue.Queue`, guaranteeing announcements are
  processed strictly one at a time.

**Error 3 — Threaded speech silently failed regardless of engine strategy**
- *Symptom:* With the queue design, no crashes occurred and diagnostic
  logging confirmed each announcement completed successfully — yet only the
  first produced audio.
- *Diagnosis:* Added `[worker] speaking:` / `[worker] finished:` logging to
  confirm the worker was completing, not hanging. Separately verified via an
  isolated four-word test that repeated create/speak/destroy cycles work
  correctly *on the main thread*.
- *Conclusion:* pyttsx3/SAPI5 does not work reliably from a background
  thread on Windows, most likely due to COM thread-affinity requirements.
  Both persistent-engine and fresh-engine strategies failed identically
  inside a thread while both worked on the main thread.
- *Fix:* Abandoned threading entirely. Switched to pyttsx3's non-blocking
  mode: `engine.startLoop(False)` starts the loop without blocking, and
  `engine.iterate()` processes a small slice of pending speech work each
  time it is called. Calling `iterate()` from within the video loop lets
  speech progress incrementally without ever freezing the loop.

**Error 4 — Speech lagged behind detections and queued up**
- *Symptom:* Announcements were audibly delayed, and a backlog built up when
  several objects appeared at once.
- *Cause:* `iterate()` processes only a small slice of work per call. Calling
  it once per frame was insufficient to keep pace.
- *Fix:* Called `iterate()` five times per frame instead of once. Five was
  chosen as a balance — enough to clear the queue promptly, not so many that
  the video loop slows down.

**Known limitation — repeated announcements for unstable detections:**
Some objects (notably "bed" in testing) are announced repeatedly because
YOLO's confidence in them genuinely fluctuates across frames due to angle,
partial occlusion, and lighting. Increasing `FRAMES_BEFORE_GONE` from 20 to
60 reduced this substantially but did not eliminate it. This is a detection
stability issue rather than a speech or tracking-logic issue. Possible
future mitigations: raising the detection confidence threshold, increasing
the forget threshold further, or adding a per-object cooldown period.

**Known limitation — OCR reading order:** EasyOCR returns text regions in
its own internal processing order, not natural human reading order. When a
book cover is read aloud, fragments are announced in an arbitrary sequence
rather than title-first as a sighted reader would perceive. Raw OCR provides
text but not visual hierarchy or reading order. Possible future approaches:
sorting results by bounding-box position (top-to-bottom, left-to-right),
sorting by text size, or using a language model to organise fragments
meaningfully.

**What was learned:**
- Isolating a bug outside the main program is often the fastest route to a
  correct diagnosis — the two-word and four-word tests each resolved
  ambiguity immediately.
- Silent failures are harder than crashes: Errors 1 and 3 raised no
  exceptions and required deliberate diagnostic logging to locate.
- Solving one problem can create another: threading fixed the video-freezing
  issue but introduced concurrency crashes, which in turn revealed a
  platform-level threading limitation.
- Accessibility design differs from ordinary UI design — silence is
  ambiguous for a blind user, so explicit negative feedback ("no text found")
  is necessary, not optional.

**Next goal:** Day 6 — Integration: combine object detection and OCR into a
single application, so both features work together rather than as separate
scripts.


---

## Day 5 — Part A — [18 sept]

**Goal:** Resolve the outstanding speech failures from Day 5, upgrade the
detection model, and fix portability issues that would prevent the project
running on another machine.

### 1. Detection model upgrade — YOLOv8 to YOLO26

The project had been using YOLOv8 (2023). YOLO26, released January 2026,
is the current Ultralytics release — three generations newer. It uses
two-stage Objects365 pretraining followed by COCO fine-tuning, and reports
40.9-57.5 mAP on COCO at 1.7-11.8 ms latency.

Switched to `models/yolo26n.pt`. The API is identical, so this was a
one-line change.

### 2. Experiment — does a larger vocabulary fix missed objects?

A recurring problem: a pen is detected as "toothbrush", a protein shaker
as "remote". The hypothesis was that COCO's 80 classes are too narrow.

**Test 1 — YOLOv8s-oiv7 (Open Images V7, 600 classes)**

| Observation | Result |
|---|---|
| Pen detection | Not detected at all |
| Label quality | Redundant overlapping labels - "Man", "Person" and "Human face" all announced for one individual |
| Announcement noise | Substantially worse; constant repetition |
| FPS | 22-51 (variable, often slower than COCO) |

**Test 2 — YOLO26n (COCO, newer architecture)**

| Observation | Result |
|---|---|
| Pen detection | Still "toothbrush", at higher confidence (0.75) |
| False positives | "tie", "refrigerator", "baseball bat" announced with none present |
| FPS | ~32 |

**Conclusion:** neither a broader vocabulary nor a newer architecture
solves missed or misidentified objects. Open Images contains a "Pen"
class and still failed to detect one. The problem is detection
difficulty for small, thin, hand-held objects, not vocabulary size.

**Decision:** retained YOLO26n as the base model. The only remaining fix
for domain-specific objects is fine-tuning on custom labelled data.

### 3. OCR spacebar unresponsiveness — three separate causes

**Cause 1:** `cv2.waitKey(5)` had been introduced earlier to reduce CPU
load. This narrowed the window during which the program listens for a
keypress to 5 milliseconds per loop, causing most presses to be missed
entirely. Reverted to `waitKey(1)`.

**Cause 2:** window focus. `cv2.waitKey()` only receives keystrokes when
the OpenCV window is the active window. Presses were going to Chrome and
VS Code instead. Confirmed by adding a diagnostic print of the raw key
code - no key events appeared until the video window was clicked.

**Cause 3:** occasional empty OCR results were not a bug but blurry
frames, confirmed by saving the captured frame to disk for inspection.

### 4. pyttsx3 replaced with edge-tts

**Fifth pyttsx3 failure:** after the first announcement, the program hung
completely - unresponsive even to the quit key. Confirmed by disabling
all speech calls and running nine consecutive OCR captures without any
hang, isolating pyttsx3 as the cause.

**Decision to switch engines.** Five distinct failures on one library
(engine reuse, thread crashes, COM thread-affinity, processing backlog,
and now a hang) indicated the library was the wrong tool rather than
repeatedly misused.

**Options evaluated:**

| Engine | Internet | Quality | Notes |
|---|---|---|---|
| Piper | Not required | Good | Offline, but requires manual binary and voice model installation |
| Coqui XTTS | Not required | Very good | Heavy; company shut down 2024, community-maintained |
| edge-tts | Required | Very good | Free, no API key, one-line install, native Hindi support |
| gTTS | Required | Good | Simpler but lower quality |
| Google Cloud / ElevenLabs | Required | Very good | Paid, require API keys |

**Chosen: edge-tts.** Reasoning: the project is intended to be cloned and
run by others, and a single `pip install` line is substantially easier to
distribute than manual binary installation. Voice quality is materially
better, which matters for a tool whose output is listened to constantly.
Hindi support is included natively, addressing a previously deferred
requirement.

**Trade-off accepted and documented:** speech now requires an internet
connection. Offline TTS is noted as a possible future addition.

**New architecture:** `edge_tts` generates an MP3 file; `pygame` plays it.
A single background worker thread processes a queue, playing one
announcement to completion before starting the next. This avoids the
entire class of problem encountered with pyttsx3 - there is no event loop
to manage, no COM objects, and no thread-affinity constraints.

### 5. OCR output filtering improvements

**Problem identified:** a flat 0.5 confidence threshold systematically
discarded the most useful readings. On a skincare product, "Glycolic Acid
Cream 6% w/w" scored 0.35 and was filtered out, while the less
informative fragment "Glyco 6" scored 0.80 and was spoken.

**Cause:** EasyOCR's confidence reflects certainty about the entire
string. Longer text has more characters that could be misread, so scores
fall - meaning a flat threshold penalises exactly the most informative
results.

**Fix:** length-aware thresholds.

| Text length | Confidence required |
|---|---|
| 12+ characters | 0.25 |
| 5-11 characters | 0.40 |
| Under 5 characters | 0.75 |

Additionally: results sorted by confidence, a minimum length of 3
characters enforced, and announcements capped at 6 items, since a
cluttered page produced over 130 fragments in one capture.

**Capture resolution raised** from 640x480 to 1280x720, which visibly
improved character accuracy.

### 6. Portability fixes

Three issues would have prevented the project running on any other machine:

| Problem | Fix |
|---|---|
| `requirements.txt` pinned a PyTorch nightly build (`torch==2.12.0.dev20260408+cu128`) not available on PyPI | Removed the pin entirely; Ultralytics installs a compatible version automatically. RTX 50-series users directed to `docs/gpu-setup.md` |
| `requirements.txt` listed ~30 transitive sub-dependencies from `pip freeze` | Reduced to 6 direct dependencies; pip resolves the rest, avoiding version conflicts |
| `src/ocr/ocr.py` contained a hardcoded Tesseract path specific to the development machine | Marked deprecated with an explanatory docstring; retained for reference only |

Created `docs/gpu-setup.md` documenting the RTX 50-series `sm_120`
incompatibility, the CUDA 12.8 nightly fix, the tested configuration, and
measured performance figures.

> **Correction (added Day 7):** `gpu-setup.md` as written here repeated the
> Day 1 error that a nightly build was required. It was rewritten on Day 7.

Verified the new `requirements.txt` by creating a clean virtual
environment outside the project directory and installing from it
successfully.

### Known limitation carried forward

EasyOCR returns text regions in its own detection order, not human
reading order. When pointed at dense paragraph text, individually correct
words are announced in a meaningless sequence. The bounding-box
coordinates needed to reconstruct reading order are already returned by
EasyOCR but are currently unused. Position-based sorting is the
identified fix, not yet implemented.

**Next goal:** Day 6 — Integration: combine object detection and OCR into
a single application.


---

## Day 6 — [20 sept]

**Goal:** Combine object detection and OCR into a single application, so both features work together rather than as separate scripts that cannot run simultaneously.

**The problem being solved:** detection and OCR each existed as a standalone program, and both opened the camera. Only one could run at a time. A user could not point at a shelf and get both "bottle detected" and the label read aloud without quitting one program and starting another. Both files also contained duplicated camera, speech and display-loop logic, meaning a bug fixed in one still existed in the other.

### 1. Single application

Created `src/assistant/main.py`, which owns the camera, owns one Speaker, runs detection every frame, and runs OCR on demand. Existing modules (`src/speech/tts.py`) are imported rather than duplicated.

Settings were grouped as named constants at the top of the file, and the main loop's logic moved into named functions (`read_text_from_frame`, `track_objects`), so the loop reads as a sequence of steps rather than a wall of logic.

A `try`/`finally` block was added around the main loop, guaranteeing the camera is released and audio shut down even if something crashes.

**One implementation detail worth noting:** OCR is run on the clean `frame`, not on `annotated_frame`. The annotated version has detection boxes and labels drawn over it, which would interfere with text recognition.

### 2. Mode switching (M key)

The first working version ran both features simultaneously. Testing showed object announcements interrupting OCR readings, which was confusing in practice.

The underlying insight: the two features serve different intentions. Detection is passive ambient awareness ("what is around me"). OCR is deliberate inquiry ("what does this say"). Ambient chatter interrupting a deliberate reading is the wrong model for how the tool is actually used.

| Mode | Behaviour |
|---|---|
| Ambient (default) | Detection announces newly-appeared objects. OCR inactive. |
| Reading | Detection announcements silenced. SPACE reads text aloud. |

Detection tracking continues to run in both modes, with only the announcements silenced. If tracking stopped during reading mode, switching back would produce a burst of re-announcements for objects that had been in view the whole time.

Mode changes are announced aloud, since the user cannot see the on-screen indicator. Pressing SPACE in ambient mode says "Switch to reading mode first" rather than failing silently — silence would be ambiguous.

### 3. OCR reading order fixed

The limitation identified on Day 5 and carried forward through Day 5 Part A was resolved.

**Problem:** EasyOCR returns text regions in its own internal detection order, not human reading order. The code then sorted results by confidence for filtering purposes, which scrambled the order further. Individually correct words were announced in a meaningless sequence.

**Before:**

    USA, Edition, Global, Silberschatzs, OPERATING, SYSTEM, CONCEPTS, Jillegal, Fon Sala n

**After:**

    Edition, Silberschatzs, OPERATING, SYSTEM, CONCEPTS

**Method:** filter by confidence first, then sort the surviving results by bounding-box position — top to bottom, then left to right. Vertical position is rounded into 30-pixel bands so that words on the same visual line group together rather than being split apart by small height differences.

```python
def reading_order(result):
    box = result[0]
    top = min(point[1] for point in box)
    left = min(point[0] for point in box)
    return (round(top / 30), left)
```

The bounding-box coordinates required for this were already returned by EasyOCR on every result but had simply been unused since Day 4.

### Measured performance

31-43 FPS with both models loaded, at 1280x720 capture resolution.

### Known limitations at end of Day 6

- OCR blocks the main loop for 1-2 seconds during a capture, freezing the video. Acceptable for a user-triggered action, but noticeable.
- Object labels remain unreliable for specific products (a shaker bottle detected as "remote", a cream tube as "tie") due to COCO's 80-class vocabulary.

**Next goal:** Day 6 Part A — usability features and a model vocabulary upgrade.

---

## Day 6 Part A — [20 sept]

**Goal:** Add usability features that a genuinely integrated application needs, and address the object-vocabulary limitation carried over from Day 6.

### 1. Repeat last announcement (R key)

**Problem:** audio is transient. A user who misses an announcement — because of background noise, or simply not listening at that moment — had no way to hear it again short of physically moving the object out of frame and back in to re-trigger detection. Sighted people re-read things constantly without thinking about it; this restores that ability.

**Implementation:** a `say()` wrapper replaced direct `speaker.speak()` calls throughout the application, storing each announcement before speaking it.

```python
def say(text):
    """Speak text and remember it, so the R key can repeat it."""
    global last_announcement
    if not text or not text.strip():
        return
    last_announcement = text
    speaker.speak(text)
```

One function, one place to change — every announcement is remembered automatically rather than requiring the memory to be updated at each call site. The R key itself calls `speaker.speak()` directly rather than `say()`, so repeating does not overwrite the memory with itself.

### 2. Describe everything visible (D key)

**Problem:** detection announces only *changes*. A user entering a room hears a burst of announcements in the first few seconds, then silence. Thirty seconds later there was no way to ask "what is here?"

**Implementation:** speaks everything currently in `currently_present` — state the tracking logic already maintained — as a single summary.

    "I see: chair, glasses, person, pillow, slippers"

This bridges passive and active use: ambient mode tells the user when things change, D tells them what *is*.

### 3. Deprecating superseded files

`src/detection/detect.py` and `src/ocr/ocr_easy.py` were superseded by `main.py`. Leaving three overlapping programs in the repository would confuse anyone reading it, and the older files contain duplicated logic that would drift out of sync.

Both were marked deprecated with docstrings explaining what replaced them and why they were kept, rather than deleted — the devlog references them heavily when documenting Days 2-5, and deleting them would leave that documentation pointing at nothing.

### 4. Combined identify (I key)

**The idea:** neither detection nor OCR alone is complete. Detection gives a generic category ("bottle"). OCR gives specific content ("Glycolic Acid Cream 6%"). Together they describe an object usefully.

**The design question this raised:** what if the two disagree? Earlier testing had detection labelling a shaker bottle "remote" and a cream tube "tie", while OCR read both correctly. Announcing "I see a tie. It says Glycolic Acid Cream" would be actively misleading to someone who cannot see the object to check.

**Decision:** where text is found, the object label is deliberately omitted. The unreliable source is only used when the reliable one produces nothing.

```python
if words:
    say("It says: " + ", ".join(words))
elif detected_now:
    say("I see " + objects[0] + ", but no readable text")
else:
    say("I cannot identify anything")
```

### 5. On-screen controls

The six control keys are now drawn below the mode indicator, so they are visible without referring to the terminal. Drawn line by line in a loop, since OpenCV has no multi-line text support.

### 6. Model upgrade — COCO to Objects365

**Context:** three separate models had failed to correctly identify everyday objects outside COCO's 80 classes, establishing vocabulary as a genuine constraint rather than an assumption.

Ultralytics publishes YOLO26 models pretrained on Objects365v1 — 365 classes across 1.74 million training images. These are the same weights the COCO models were fine-tuned from, so the training quality is proven.

**Experiment result:**

| Object | COCO (yolo26m) | Objects365 (yolo26m-objv1-150) |
|---|---|---|
| Glasses | Not a class | Detected, 0.91-0.92 |
| Bracelet | Not a class | Detected, 0.76 |
| Pillow | Not a class | Detected |
| Slippers | Not a class | Detected |
| Person | Detected | Detected, 0.93-0.96 (higher confidence) |
| FPS | 31-43 | 34-35 (no meaningful cost) |

**Critically, no redundant labels.** This was precisely what made Open Images V7 unusable on Day 5 Part A — it announced "Man", "Person" and "Human face" for a single individual. Objects365's taxonomy is better curated, producing one label per object.

**Why this succeeded where Open Images failed:** Objects365 shares YOLO26's own training lineage, rather than being a separately-trained model of different provenance and quality.

`yolov8s-oiv7.pt` was deleted, since the Day 5 Part A experiment concluded it was worse on every measure and there is no scenario in which it would be loaded again.

### Known limitations carried forward

- OCR still blocks the loop for 1-2 seconds during a capture.
- Objects at the frame edge (pillow, slippers) still flicker in and out despite `FRAMES_BEFORE_GONE = 150`. This is detection instability rather than a tracking-logic fault.
- No error handling. If the camera disconnects or OCR fails, the program crashes — and a blind user would simply hear silence with no explanation.
- The I key was built but not yet tested on a real product label.

**Next goal:** Day 7 — Environment and repository maintenance.

---

## Day 7 — [20–22 sept]

**Goal:** Resolve a Windows security block that stopped the project from
running, correct the PyTorch setup chosen on Day 1, and tidy the repository.

Nothing in the project had changed. Code that had worked the previous day
suddenly refused to start, which made this a diagnosis exercise rather than a
coding one.

### 1. Windows Smart App Control blocked PyTorch

Previously working code failed at startup:

    ImportError: DLL load failed while importing _C: An Application Control policy has blocked this file.

Windows Security's Protection history showed nothing — Smart App Control blocks
are not recorded there. The block was found in Event Viewer under
**Applications and Services Logs > Microsoft > Windows > CodeIntegrity >
Operational**: Code Integrity had blocked
`torch\_C.cp313-win_amd64.pyd`, PyTorch's core compiled file, for not meeting
the required signing level (policy `{0283ac0f-fff1-49ae-ada1-8a933130cad6}`).

**Cause:** Smart App Control starts in Evaluation mode, which blocks nothing,
and Windows later switches it to On automatically without notifying the user.
The nightly PyTorch build was a rare development snapshot with no reputation
record, so once enforcement began it was blocked.

**Before changing any security setting**, the Windows build was checked with
`winver`: 26200.9445. Since the April 2026 update (builds 26100.8116 /
26200.8116 onwards), turning Smart App Control off is reversible; on earlier
builds it is permanent without resetting Windows. In the end it was never
turned off — the fix below worked with it still on.

### 2. Correction of a Day 1 mistake: stable PyTorch works

On Day 1, a stable CUDA 12.4 (`cu124`) build failed on the RTX 5060, and a
nightly build was chosen instead. That was the wrong conclusion. **Stable
PyTorch has supported the RTX 50-series (`sm_120`) since version 2.7, through
its CUDA 12.8 (`cu128`) builds.** The Day 1 problem was the CUDA version, not
stable versus nightly.

    pip uninstall torch torchvision -y
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

This installed `torch 2.11.0+cu128` and `torchvision 0.26.0+cu128`, which
loaded with Smart App Control still on and reported `CUDA: True`. Pip reported
the wheel as already cached — the same stable build had been used successfully
on this machine before the project began.

The mistake went unnoticed for six days because the nightly build worked. A
wrong conclusion that still produces a working result can sit undetected until
something else exposes it.

### 3. A second block: scipy

The application then failed on `scipy` 1.18.1 (`_ufuncs_cxx`), required by
EasyOCR. Installing the previous release fixed it:

    pip install --no-deps "scipy==1.17.*"

`--no-deps` replaced only scipy, leaving every other package untouched. Smart
App Control was kept **on** throughout. It has no exception list, so individual
files cannot be allowed — but widely used releases passed where rare ones did
not.

> **Note (added Day 10):** Smart App Control was later turned off, after it also
> blocked `python-bidi` and Whisper's `llvmlite`. See Day 10.

### 4. Repository and documentation fixes

| Change | Why |
|---|---|
| `requirements.txt`: removed `pyttsx3`, added `edge-tts` and `pygame`, pinned `scipy>=1.17,<1.18`, added the PyTorch install command | It still listed the abandoned speech engine and was missing the two packages actually used |
| `docs/gpu-setup.md` rewritten | It repeated the Day 1 claim that nightly was required; now includes a correction note, a Smart App Control section, and clickable reference links |
| Deprecated scripts moved to `archive/` with `git mv` | `git mv` records a move rather than a delete-and-recreate, so each file's history stays connected |
| Unused models removed (`yolo26m.pt`, `yolov8n.pt`) | Superseded by the Objects365 model; both are public and re-downloadable |
| `.gitignore`: added `*.cache` and `runs/`; anchored `/weights/` to the project root | `weights/` without a leading slash matched every folder of that name, not just the stray one at the root |

An Ultralytics update (8.4.157) was offered and **deliberately declined** —
after stabilising the environment, a new release could bring new compiled files
and trigger another block.

### What was learned

- Security software can break a working environment without any change to the project
- When Windows blocks a file, Event Viewer's CodeIntegrity log shows what Protection history does not
- Check whether a security change is reversible before making it
- A wrong conclusion can hide for days when the wrong path still works
- Pin known-good versions of packages rather than always taking the newest
- A broad `.gitignore` rule can silently exclude files you meant to keep

**Next goal:** Day 8 — Make the core application robust: non-blocking OCR,
error handling with spoken explanations, and fewer repeated announcements.

---

## Day 8 — [22–28 sept]

**Goal:** Turn a working demonstration into something that behaves sensibly
when things go wrong — text reading that does not freeze the video, speech that
stays responsive, errors explained aloud instead of silence, and speech that
keeps working without internet.

No new capability was added today. Every existing one was made dependable,
which for an accessibility tool matters more.

### 1. Text reading without freezing the video

Pressing SPACE ran EasyOCR on the main loop, so for one to two seconds nothing
else happened — no frames, no detection, frozen video.

OCR now runs on a background thread. Three details matter:

| Decision | Reason |
|---|---|
| `frame.copy()` | The loop replaces `frame` every iteration; the thread needs its own snapshot of the moment the key was pressed |
| An `ocr_busy` flag, and a spoken "Still reading, please wait" | A second reading would waste the GPU; silence would be ambiguous |
| `try` / `except` / `finally` inside the thread | Failures are spoken, and `finally` always clears the flag — otherwise one error would block reading for the whole session |

**Why threading was safe here but failed on Day 5:** the Day 5 failures came
from pyttsx3, whose Windows speech engine has strict thread-affinity rules.
EasyOCR is plain Python and PyTorch, with no such constraint. The same
technique is safe or unsafe depending entirely on what runs inside it.

### 2. Speech five seconds behind

**Symptom:** announcements arrived about five seconds after the key was pressed.

**Diagnosis:** at startup, ambient mode announced seven objects one at a time.
Each is a separate network round trip plus playback — one to two seconds each.
By the time M was pressed, a dozen announcements were queued ahead of it.
**Speech was not slow; the queue was long.**

Four changes:

| Change | Effect |
|---|---|
| Priority speech | Anything the user asks for (M, SPACE, D, I, R) jumps the queue and stops what is playing |
| Batched announcements | Seven objects become one sentence — one round trip instead of seven |
| Expiring ambient announcements | Dropped if not spoken within six seconds: they describe a scene that has changed |
| Per-object cooldown (30 s) | Ended the repeated "hat" and "cabinet/shelf" announcements from flickering detections |

Ambient announcements are also not queued at all while speech is failing — each
would attempt a network call that fails, producing errors and no sound.

### 3. Identifying the object being held up

The I key now announces the object and its text together — *"I see canned. It
says: PEPSI."* This reverses a Day 6 decision to omit object labels, made when
labels came from COCO and were often wrong. With Objects365 they are good
enough to be worth saying. They are kept as **two separate sentences**, so a
wrong label cannot corrupt the text.

Choosing *which* object took four attempts:

| Attempt | Rule | What went wrong |
|---|---|---|
| 1 | Largest box near the centre | Chose `cabinet/shelf` — furniture is large |
| 2 | Smallest box covering the centre | Chose the user's `glasses`, dead centre |
| 3 | Ignore worn items; prefer recent objects | Worked, but a weak label was chosen when nothing better existed |
| 4 | Add a confidence floor and confidence tie-breaks | Current behaviour |

**Bug — the wrong object chosen when two boxes tied:**

    canned   conf 0.85   distance 0.20
    bottle   conf 0.46   distance 0.20
    PROMINENT: bottle

Sorting compared `(is_background, distance, label, ...)`. When distance tied,
Python moved on to the label — and "bottle" sorts before "canned"
alphabetically. Confidence was never considered. Fixed by putting
`-confidence` in the sort key before the label, and rounding distance and area
so near-equal values genuinely tie.

**An honest finding:** a cream tube, a spray bottle and a paper cup produced no
detections at all, even at 0.30 confidence. They are not among Objects365's
classes. The app falls back to reading their text, which for a product is
arguably the more useful half.

### 4. Errors explained instead of silence

| Failure | Behaviour now |
|---|---|
| Missing model, camera busy, no audio | Printed in plain language naming what failed and what to check, then exit — speech may be the broken thing, and whoever is installing is reading the terminal |
| Camera lost mid-session | "Camera problem", retried for three seconds, then "Camera disconnected. Stopping." Recovery is also announced |
| One bad frame | Detection wrapped in `try`; the frame is skipped rather than ending the run |
| Ctrl+C | Caught, so it exits cleanly instead of printing a traceback |

**When speech itself fails**, it cannot be announced by speaking. The app
generates a **beep** every ten seconds — no network, no speech engine needed —
and shows SPEECH NOT WORKING on screen. The speech module counts consecutive
failures and translates the error, so the terminal reads
`SPEECH FAILING: no internet connection` rather than a socket error.

### 5. Speech without internet

A beep says something is wrong but nothing more. Better: when edge-tts cannot
reach the network, fall back to the **Windows built-in voice** via pyttsx3.

**Deliberately reintroducing a library that failed five times.** pyttsx3 was
abandoned on Day 5 Part A after five failures ending in a complete hang. What
changed:

| Day 5 | Day 8 |
|---|---|
| Driven from the main thread with a fragile event loop | Runs inside the speech worker thread, which owns it for the whole utterance |
| A hang froze the entire application | A hang would affect speech only — and the beep still signals failure |
| Used for every announcement | Used only when the network is down |
| One engine reused (fell silent after the first utterance) | A fresh engine per utterance — Day 5's own fix, applied deliberately |

**The library was never the whole problem. The architecture around it was.**

After two consecutive online failures it switches over, retries the failed
announcement, shows OFFLINE VOICE on screen, and stops beeping. Every sixty
seconds it tries the online voice again; on success the app says "Back online".

**Bug — the online retry fired instantly, twice over.** First attempt:
`last_online_retry` started at `0.0`, and since `time.time()` returns a number
in the billions, "time since the last retry" looked like decades. Second
attempt: setting it once at startup — but if the app had already run for more
than sixty seconds before the internet dropped, it still fired immediately.
**Correct fix:** start the countdown at the moment offline mode begins.

### Testing

| Test | Result |
|---|---|
| Video keeps running during a reading | Passed |
| Second reading refused | "Still reading, please wait" |
| Announcements batched | One sentence, not seven |
| Speech failure signalled | Beep every ten seconds, on-screen warning |
| Offline voice | Switched after two failures; Windows voice audible; no hang |
| Recovery | "Back online" spoken |
| Clean exit (Q and Ctrl+C) | No mixer errors, no traceback |

### Known limitations carried forward

- Many specific products (cream tubes, spray bottles, paper cups) are not in Objects365's classes and are not detected; their text is read instead
- Printed faces on posters or photographs can be announced as "person"
- The offline voice sounds noticeably more robotic
- SPACE and I now do the same thing, differing only in that SPACE requires reading mode
- Diagnostic output prints every detection on each identification — useful while developing, noisy for a demonstration
- Tested on one machine, with a front-facing laptop webcam

### What was learned

- The same technique can be safe or unsafe depending on what runs inside it
- A library that failed repeatedly can be reintroduced safely if the architecture around it changes — and it is worth saying why, rather than quietly reversing an earlier decision
- Slow-feeling software is often queued rather than slow
- Sorting on the wrong key produces plausible-looking wrong answers
- A variable meaning "when did this last happen" must be set when it happens; zero means "long ago" and fires immediately
- Some failures cannot be reported through the channel that failed
- Saying nothing is sometimes the correct output

**Next goal:** Day 9 — systematic evaluation of the finished application
against a fixed set of objects and texts, with real numbers rather than
impressions.

---

## Day 9 — [28–29 sept]

**Goal:** Replace "I tried it and it seemed to work" with measured results — a
fixed set of objects and texts, tested under stated conditions, recorded as the
tests happen and analysed afterwards.

### 1. Method

Three features were scored separately, because they fail in different ways:

| Feature | Question | Verdicts |
|---|---|---|
| Object detection (ambient) | Does it announce what is actually there? | correct / wrong / missed |
| Identification (I key) | Does it name the held-up object correctly? | correct / hedged right / wrong / no label |
| Text reading (SPACE) | Does it read the text correctly? | exact / partial / wrong / nothing |

Rules set **before** testing began:

- The object and text lists were fixed in advance. Swapping in easier items after seeing results is how an evaluation stops meaning anything.
- Ten objects deliberately included known weak cases — a cream tube, a spray bottle and a pen — all of which had failed in earlier informal testing.
- Results were recorded at the moment of each test, not reconstructed afterwards.
- A synonym counts as correct: "canned" for a drinks can is a right answer.

Two new scripts: `src/evaluation/record_results.py` presents each test in a
fixed order and appends a row to a CSV after every answer, so a session can be
stopped and resumed; `src/evaluation/analyse_results.py` reports per-feature,
per-item and per-condition results.

**Bug:** the recorder wrote the CSV header only when the file did not exist.
The file had been created earlier but left empty, so the header was skipped and
analysis failed with `KeyError: 'kind'`. Fixed by treating an empty file as new:

    new_file = not OUTPUT.exists() or OUTPUT.stat().st_size == 0

### 2. Results (39 tests)

> **Update (Days 12–13):** the remaining Day 9 tests were run on 6–7 October,
> after several recorded results were checked against the app's log and
> corrected. The final numbers for all 56 Day 9 tests are in Day 13.

| Feature | Result | Tests |
|---|---|---|
| Text reading | 100% (3 exact, 3 partial) | 6 |
| Object detection | 66.7% | 24 |
| Object identification | 44.4% | 9 |

**The clearest finding is not the percentage but the split:**

| Always detected (13/13) | Never detected (0/6) |
|---|---|
| bottle, phone, headphones, keyboard | cream tube, spray bottle, pen |

Every failure was recorded as **missed**, never as **wrong**. The model does not
misidentify these items — it does not see them at all. They are not among
Objects365's 365 classes. Only `cup` was mixed, at 2 of 3.

"66.7% accurate" invites the question "66.7% of what?". "Reliable for common
objects, blind to items outside its vocabulary" answers it.

Identification: 4 correct, 4 "no label", 1 wrong. **"No label" is not a wrong
answer** — the app deliberately says nothing about an object it cannot name
confidently, and reads the text instead, which for a product is usually the more
useful half.

### 3. A design conflict found by testing

Headphones were detected correctly in every ambient test (3/3), but
identification named them **wrongly**.

The cause is a Day 8 rule: while the camera faces the user, worn items — person,
glasses, hat, headphones — are excluded from being chosen as the held-up object,
because a laptop webcam always has them in shot. When headphones are
deliberately held up, that same rule excludes them and something else is chosen.

This is a genuine trade-off, not a bug: the rule prevents a common wrong answer
and causes a rare one. Setting `CAMERA_FACES_USER = False` — correct for an
outward-facing camera, which is the real use case — removes the conflict.

### 4. What these numbers do not show

- **Reading is under-tested:** 6 of a planned 16. 100% on six tests is promising, not proven.
- **Handwritten notes were never tested** — none were available. Handwriting is significantly harder for OCR, so this is a real gap.
- **The chair was not tested** — none available in view of the camera.
- **The lighting breakdown is meaningless.** Dim light shows 100%, but on five tests, and those happened to involve items that always work. It reflects which items were tested in each condition, not the effect of lighting.
- **One machine, one room, one camera.**

Reporting these limits matters as much as reporting the results. A number
without its conditions is not evidence.

### What was learned

- An overall percentage can hide the real finding: 66.7% concealed a clean split between items that always work and items that never do
- The kind of failure matters more than the rate — every failure here was silence, not a wrong answer, which is far safer for a blind user
- Fixing the test set before testing is what keeps the result honest
- Recording results as they happen avoids remembering favourably
- Testing finds design conflicts, not just bugs
- A test that was not run must be reported as not run

**Next goal:** Day 10 — a demo video, the remaining reading tests including
handwriting, and a final documentation pass.

---

## Day 10 — [29 sept]

**Goal:** Begin the features the roadmap listed as "later", starting with voice
commands. Also clear two problems carried over from Day 8: diagnostic output
too noisy for a demonstration, and SPACE and I doing the same thing.

> **Change of plan:** Day 9 set the next goal as a demo video, the remaining
> reading tests and a final documentation pass. These were postponed: the
> remaining tests were run on Day 12, and the demo video moved to the very end.
> The reason for the change was not recorded.

### 1. Voice commands (V key)

Two small scripts came first, both now in `archive/voice/`:

| Script | Why |
|---|---|
| `test_mic.py` | Measures the microphone's signal level. Whisper invents plausible text from near-silence ("Thank you", "You know what?"), so a quiet microphone looks like poor recognition |
| `test_listener.py` | Records three seconds, transcribes them, and shows which command matched: recognition and matching tested on their own |

| Decision | Reason |
|---|---|
| OpenAI Whisper, running locally on the GPU | Offline and free; understands English and Hindi; no audio leaves the machine |
| Push-to-talk: V starts recording, V again stops it | OpenCV cannot detect a key being *released*, so V toggles. It also removes the need to detect when speech has ended |
| A beep when recording starts | Instant, unlike a spoken prompt |
| Matching by keyword, not exact phrase | "can you read this please" contains "read", so it matches |
| Specific phrases checked first | "reading mode" must be checked before "read" |
| The command is run by the main loop | Transcription runs on a background thread, but commands use the camera frame and change the mode, which belong to the loop. The thread only sets `pending_command`; the loop runs it on the next frame |

**Whisper `small`, not `base`.** `base` recognised the five test commands but
struggled with an Indian English accent unless the speaker talked loudly and
slowly. `small` is about three times larger, much better with accents, and
transcribed commands in 0.4–0.6 s. Every transcription prints its time, so the
cost is measured rather than guessed. *(The code records this choice; the day
of the switch is not recorded.)*

One difference from the keys: saying "read" switches to reading mode by itself,
while SPACE in ambient mode still says "Switch to reading mode first".

### 2. SPACE and I now do different jobs

Day 8 recorded that SPACE and I did the same thing, except that SPACE needed
reading mode. They now answer different questions:

| Key | Question | What runs | Words spoken |
|---|---|---|---|
| SPACE | "Read this to me": documents, signs, letters | Text reading only, no detection | up to 20 |
| I | "What is this?" | Detection, then text reading | up to 6, after the object name |

### 3. Diagnostic output behind a switch

`DEBUG` at the top of `main.py` controls the printing of every detection (with
the reason it was kept or skipped) and every OCR word (with its confidence). It
is set to `False` before recording a demonstration. One line always prints,
`PROMINENT: …`, because it shows what an identification was based on.

In the held-object choice, each `continue` sits **outside** its `if DEBUG:`
block. Indented inside, the object would only be skipped while debugging, so
turning `DEBUG` off would silently change which object is chosen.

### 4. An Indian English voice

Speech switched to `en-IN-NeerjaNeural`, for clearer pronunciation of Indian
names and product labels.

### 5. Windows Smart App Control, again

| Blocked | Fix |
|---|---|
| `python-bidi` 0.6, a Rust rewrite shipping a compiled binary. EasyOCR imports it at startup (`from bidi import get_display`), so the whole app could not start | A documented local stand-in in `src/bidi/` |
| `llvmlite`'s DLL, installed with Whisper (through `numba`) | Smart App Control turned off for the rest of development, reversible on this build (26200.9445), as checked on Day 7 |

**Why the stand-in is safe:** `bidi` reorders right-to-left text (Arabic,
Hebrew, Urdu) mixed with left-to-right text. The app reads English only, which
is left-to-right, so the correct output is the text unchanged, and that is what
the stand-in returns. `main.py` puts `src` first on the import path before
importing EasyOCR, so the stand-in is found first. Its docstring says to delete
it and install the real package if a right-to-left language is ever added.

### Known limitations carried forward

- Voice commands are English only (Hindi followed on Day 11)
- Whisper can still invent phrases from near-silence (addressed on Day 11)
- Smart App Control is off on the development machine

### What was learned

- A package the app never calls can still stop it starting: EasyOCR imports `python-bidi` at startup
- A stand-in is safe only where its behaviour is provably the same, and it must say when that stops being true
- Check the input before blaming the model: a quiet microphone makes recognition look bad
- A background thread should hand its result to the main loop, not change shared state itself
- Two keys doing the same job means their purposes were never separated

**Next goal:** Day 11 — open questions about what the camera sees, answered by
a vision model running locally, and a Hindi mode.

---

## Day 11 — [2–5 oct]

**Goal:** Let the user ask anything about what the camera sees ("what am I
wearing?"), answered by a vision-language model on the laptop; make the whole
application work in Hindi; and give the video window a display the people
around the user can follow.

Four days of testing and building, committed together on 5 October.

### 1. Visual questions: finding a model that tells the truth

Anything said after V that matches no command now goes to a vision-language
model, running locally through Ollama, with the current camera frame. The
answer is spoken as "It looks like …".

| Model | Result |
|---|---|
| qwen2.5vl:3b | Failed outright with a token-repeat error |
| llava 7B | Wrong shirt colour, missed a held phone, and answered about an image it was never given |
| moondream 1.8B | Invented text on a bottle that was not there, and answered about a file that did not exist |

A confident wrong answer is worse than no answer for someone who cannot check
it, so after these three the feature was **dropped**. The decision was then
**reversed**:

| Model | Result |
|---|---|
| qwen3-vl:8b | Accurate: clothing colour, bracelets on the correct wrist, a bottle present; correctly said there was no cow; spotted a dog on a phone screen. 5–8 s alone, but about 6 GB: beside detection, OCR and Whisper on the 8 GB GPU, part of it ran on the CPU and one answer took 95 s |
| qwen3-vl:4b | Accurate, about 3 GB, 2–5 s with the whole application running |

**The limit was GPU memory, not the model.** The plain `qwen3-vl:4b` tag is the
*Thinking* variant: its reasoning used up the length limit, so answers came back
empty (often in Hindi), and it kept the GPU busy long enough to make the video
stutter. Fixed by using `qwen3-vl:4b-instruct`, the same model without the
reasoning step.

### 2. Making visual questions dependable

| Problem | Fix |
|---|---|
| Five questions in a row refused as "busy": one slow answer kept the model busy, and nothing showed why | A 30 s limit per question; problems printed as well as spoken |
| A one-word fragment ("और…", "um") produced a long rambling answer that blocked real questions | Fewer than two words: "Please ask a full question" |
| With the model missing, the app started normally and failed on every question | At startup, check the model itself is installed; if not, report it with the `ollama pull` command and carry on without the feature |
| The model called the user "he" or "the person", and asked about the wall it also described clothes | The instructions: address the user as "you", describe only what is visible, say so when unsure, answer only what was asked |
| Problems reported as English sentences | Short codes (`busy`, `capture`, `empty`, `no_response`), so they can be spoken in either language |

### 3. Voice commands, made reliable

| Problem | Fix |
|---|---|
| The first word was lost: "colour of my bracelet" arrived as "of my bracelet", because the microphone was opened on each key press | The microphone is opened once and left running; audio is kept only while recording |
| "Subscribe" transcribed twice when nothing was said (Whisper learned subtitle phrases from video) | `no_speech_threshold`, a list of known invented phrases, near-silent clips skipped |
| A clipping microphone made Whisper invent phrases | Each recording normalised to 70% of full volume |
| "What am I wearing" transcribed five times over (beam search looping) | Repeated copies of one phrase collapsed to one |

Whisper is also given the app's commands and example questions in advance
(`initial_prompt`), and uses beam search (5 candidates).

### 4. Spelling correction for text reading

A protein box, "Beast Life Performance Protein 1kg", was read as "BEAST,
PERFORMANGE, PROTEINa, Ikg":

| Problem | Fix |
|---|---|
| "Ikg": OCR reads 1 as I or l before a unit | Becomes "1 kg" |
| "PERFORMANGE": a misread letter | A word below 0.90 confidence and not in the dictionary is corrected to the nearest real word (pyspellchecker) |
| "BEAST": all capitals can make the voice spell it out | Words of more than three capitals spoken as "Beast" |
| "LIFE" (0.58–0.64) cut by the 0.75 bar for short fragments | Four-letter words get a 0.50 bar; four-letter junk ("Hikg", "CRuY", "Fepe") scored below 0.20 |

Confident readings are trusted as they are: BOLDFIT, read at 0.92, is a brand
name, not a typo.

### 5. Hindi mode

The L key, or saying "Hindi" / "इंग्लिश", switches everything spoken: voice,
fixed sentences, voice-command recognition and answers to visual questions.

| Decision | Reason |
|---|---|
| Hinglish: Hindi sentences, English object names ("मुझे दिख रहा है: bottle, chair") | How most Hindi speakers say them; avoids translating 365 object names |
| Every spoken sentence in `src/language/phrases.py`, in both languages | A language is added by translating one file; a missing sentence falls back to English |
| Voices `en-IN-NeerjaExpressiveNeural` and `hi-IN-SwaraNeural` | Chosen after listening to every Indian voice edge-tts offers |
| Feminine first person in Hindi ("पढ़ रही हूँ") | The Hindi voice is female |
| Whisper told which language to expect | Left to detect it, Whisper often wrote Hindi speech in Urdu script, which no Hindi keyword could match |
| The nukta dot ignored in matching | Whisper writes both पढ़ो and पढो |
| Text reading stays English only | Hindi OCR was left out of scope |

### 6. Hindi answers: answer in English, then translate

Asked to answer in Hindi directly, the vision model invented objects ("a cooler
near the window", "traffic number boards"); its English answers to the same
questions were correct. So it always answers in English, and the answer is then
translated.

| Translator (tested 5 October) | Result |
|---|---|
| The vision model itself | Changed facts: "a red bracelet on the right wrist" became "on the knee of the right hand" |
| Google Translate's free endpoint | Refused every request: 4 of 4 returned "too many requests" |
| **NLLB-200 distilled 600M** (Meta) | Correct on all 4 test sentences, 1.2–1.9 s each **on the CPU**. Chosen |

NLLB runs on the CPU, so it does not compete for GPU memory, and loads only
when Hindi is first used. Licence: **CC-BY-NC 4.0, non-commercial only**.

On its own NLLB turned "a red bracelet" into "a red wrist". The fix is a
**glossary**: known object names are swapped for their spoken Hindi form inside
the English sentence before translation, so NLLB sees them already in Hindi
script and keeps them (a terminology constraint). An example Hindi sentence in
the vision model's instructions was copied into every answer, so no examples
are given. If translation fails, the English answer is spoken.

### 7. A new display

The window is for the people around the user, who cannot hear what the
application is thinking. Its main job is large live captions of everything
said, in English or Hindi.

| Problem | Fix |
|---|---|
| Hindi captions came out as question marks: OpenCV's fonts cannot draw Devanagari | All text drawn with Pillow, and cached |
| ब्रेसलेट drawn as ब्‌रेसलेट: Devanagari needs shaping, and Pillow's shaping engine (Raqm) was missing | Text shaped with HarfBuzz and drawn with FreeType (`src/display/shaping.py`), both pip-installable |

Typeface: Nirmala UI, the only built-in Windows face that draws English and
Hindi as one family. Captions fade out after about 7 s; status pills (Listening,
Thinking, Reading text, Offline voice, Speech not working) appear only while
active; the key strip shows only available features; detection boxes are thin
sky blue, and the identified object is marigold for 3 s.

### Known limitations carried forward

- Visual questions use most of the 8 GB GPU
- The vision model sometimes states details it is unsure of; hence "It looks like …"
- Hindi speech recognition is weaker than English
- Text reading is English only
- NLLB's licence is non-commercial

### What was learned

- For a user who cannot check the answer, a confident wrong answer is the worst failure
- A "no" can be revisited: the idea failed with three models and worked with a better one
- The limit can be hardware, not the model
- Asking a model to look *and* answer in Hindi made it worse at both; one step at a time, then translate, worked
- Examples in a prompt get copied into the answer
- Indian scripts need text shaping; a font alone is not enough

**Next goal:** Day 12 — check the new features on the real machine, finish the
remaining Day 9 tests, and write the test plan for the new features before
running any of them.

---

## Day 12 — [6 oct]

**Goal:** Check the Day 11 features on the real machine, fix the small things
found, finish the evaluation planned on Day 9, and write the test plan for the
new features before running them. Features were frozen from the end of Day 11.

### 1. Checks

| Check | Result |
|---|---|
| Hindi captions on screen | Correct (Nirmala UI + HarfBuzz), confirmed by eye |
| Vision answer time, Hindi, app running | 3–5 s |
| Where the vision model runs (`ollama ps`) | 3.5 GB, 100% on the GPU |
| GPU memory, everything loaded | 7,734 of 8,151 MiB |
| GPU memory, app closed, model unloaded | 887 MiB |
| By subtraction | vision model about 3.5 GB; the app (YOLO, EasyOCR, Whisper) about 3.3 GB |

Decision: no memory changes during the feature freeze. Visual questions need an
8 GB GPU, and that is documented.

### 2. Small fixes

- "curly hair" was translated as "गुलदस्ता बाल" (bouquet hair): glossary entry "curly" → "घुंघराले"
- `requirements.txt`: every Day 10–11 package pinned to the tested version and grouped into core and optional; `pip install --dry-run` confirmed no mismatch
- The unused Thinking model removed from Ollama

### 3. Finishing the Day 9 evaluation

55 of the 56 Day 9 tests recorded (the last one followed on Day 13). Detection
17/30, identification 4/10, reading 11/15. Neat handwriting was read in normal
light only; normal handwriting never. The lighting differences in detection are
not a real effect: about ten tests per condition, each with different items.

### 4. When the records disagreed with the logs

Every result was checked against the application's own log. Several were wrong
because of **how the tests were run**, not the app:

| Problem | Evidence in the log | Action |
|---|---|---|
| The chair never left the frame | No "FORGOTTEN: chair" before the test | Removed, redone with the camera covered first |
| Three lighting conditions saved within 7 s | The light could not have changed | Removed, redone |
| Reading done with I instead of SPACE | No switch to reading mode; 6 readings for 8 tests | Removed, redone |
| "Detected" when only the label was read ("Glyco 6") | Detection means the name is spoken | Corrected to "missed" |
| Handwriting marked "wrong" | All words below 0.15; nothing spoken | Corrected to "nothing" |
| Speech lost to an internet drop | Title read, then a speech error | Removed, redone on Day 13 |

All changes are made by `src/evaluation/correct_results_2026_10_06.py`, with the
reason written into each row, and the original kept as
`evaluation/results_before_2026-10-06_corrections.csv`. **A test is removed only
when the log shows it did not test what it claims, never because the result was
bad.**

### 5. Findings recorded, not yet fixed

1. Startup announcements are dropped: the first burst overflowed the queue in all three logged startups
2. Flickering background objects were announced 5–7 times each in about 6 minutes
3. The spell checker damages correct words: Glycolic → glycol, Hazrat → hazard, gmail.com → malcolm, Rivest → rivets
4. Identification picks the smallest box over the centre: "storage box" (6%) beat "book" (19%)
5. The README is wrong about the pen: "pen/pencil" *is* an Objects365 class, just rarely detected at 0.65
6. The first network failure loses an announcement
7. A box on screen does not mean it was spoken

### 6. A test plan for the new features, written before testing

30 tests added to `record_results.py` before any was run: 7 voice commands in
English and 7 in Hindi (natural phrasings, not copied from the keyword lists),
8 English and 4 Hindi visual questions (including 4 **trap questions** about
things that are not there), and 4 Hindi translations. A visual answer counts
only if fully correct, because a half-right answer can mislead someone who
cannot check it.

### What was learned

- Check the test, not just the result: most problems today were in how tests were run
- The application's own log is the referee
- Corrections belong in code, with a reason, so they can be questioned
- Write the test plan before seeing any result

**Next goal:** Day 13 — run the 30 new tests and the last Day 9 test, then fix
what testing found.

---

## Day 13 — [7 oct]

**Goal:** Run the new tests exactly as planned, check every row against the
log, and only then fix what testing found, checking each fix on its own.

### 1. Testing the new features

**Rule for repeated attempts:** the first attempt where the test was set up
correctly counts. A misheard question that gets a confident wrong answer is
what a blind user would experience, so it is not retried away.

Corrections, in `src/evaluation/correct_results_2026_10_07.py`:

- Two Hindi commands were misheard ("आस पास का दिख रहा है", "मेरा ब्रेसलेट करो"); they were sent to the vision model as questions
- The Hindi shirt-colour question was misheard and answered about the wall; correct only on the 3rd try, so it counts as wrong
- The dog trap question was never asked ("is there another man in my room?" was asked instead, and answered correctly); it was asked again
- Translation verdicts had been scored on the app's reaction to the recorder's label "translation of: …" being spoken aloud; they were rescored on meaning alone

### 2. Final results — all 86 tests

| Feature | Success | Notes |
|---|---|---|
| Detection | 17/30 = 56.7% | bottle, phone, headphones, keyboard 12/12; cream tube, spray bottle 0/6 |
| Identification | 4/10 = 40% | 5 no label, 1 wrong (headphones, worn-item rule) |
| Reading (exact + partial) | 12/16 = 75% | normal 7/8, dim 5/8; exact only 3 |
| Voice commands, English | 7/7 | Whisper 0.4–1.1 s |
| Voice commands, Hindi | 5/7 | 2 misheard; Whisper 0.8–2.5 s |
| Visual questions, English | 8/8 | |
| Visual questions, Hindi | 2/4 | 1 partly correct, 1 wrong (misheard question) |
| Hindi translation | 3/4 | "yellow case" → "पीला बैग" (yellow bag) |
| **Trap questions** | **4/4 passed, 0 made up** | cup, dog, watch (English); dog (Hindi) |

One attempt per test, one speaker, one room, one laptop webcam: 100% on 7 or 8
tests is not proof of reliability.

### 3. Fixes, each checked separately

The 86 results describe the app **before** these fixes.

| # | Problem | Fix | Check |
|---|---|---|---|
| G | A proposed glossary list had 6 words listed twice (a Python dictionary silently keeps the last), and words with no single Hindi form ("old man" = बूढ़ा आदमी, "old bag" = पुराना बैग; light, orange, back, right). "case" became "बैग"; "turtleneck" was garbled | Glossary rebuilt as checked pairs: a repeated or upper-case word stops startup. Written rules: nouns only, no colours, verbs, directions or context-dependent words. 22 → 109 words. A narrow polishing step fixes NLLB's own slips (काला शर्ट → काली शर्ट, तुम → आप) | `check_glossary.py`: both errors fixed, nothing broke; 9/11 context sentences (known slips: "एक संतरे", "दो तकिया"); polishing 8/8 |
| 1 | First visual question after starting: "timed out". Ollama unloads the model after ~5 idle minutes; reloading exceeded 30 s | First request 120 s, then 30 s; model kept loaded 20 minutes (`keep_alive`) | First question answered in 6.2 s |
| 2 | A reading lost when the internet blinked (offline voice only after the 2nd failure); with Wi-Fi off, speech hung silently | First failure rescued by the Windows voice; offline after 2 in a row; 5 s limit on the online voice. A bug found while testing this (one shared failure counter, reset by every rescue, so offline mode never started) fixed with two counters | 6 simulated scenarios; live Wi-Fi-off test passed |
| 3 | Startup announcements expired in the queue | Ambient announcements waiting back to back merged into one sentence, repeats removed | Real startup burst: 6 announcements → 2 sentences, none dropped |
| 4 | Whisper copied "मेरा ब्रेसलेट" from the example in its Hindi prompt; "आस पास" written as two words | Prompt holds the command phrases and one example with no object; "आस पास" joined; keyword "आसपास का" | Re-test 6/6 (was 0/2) |
| 5 | The spell checker damaged names: Glycolic → glycol, Hazrat → hazard, gmail.com → malcolm | Accept a correction only if same length and every changed letter is a look-alike pair (c/e, o/e, t/f, l/t, b/p, q/o, l/e, c/g, i/l, i/j, u/v, n/h). New `src/language/spelling.py` | `check_spelling.py` 22/22: 9 helpful kept, 7 damaging blocked |

Not fixed, by decision: the smallest-box rule for identification, because
changing it would invalidate the 10 identification tests.

### 4. Tidying the repository

Voice-sample MP3s and an unused model removed; three one-off test scripts moved
to `archive/`; `*.mp3` added to `.gitignore`.

### Known limitations carried forward

- Normal handwriting is not read
- A misheard question gets a confident answer to a different question
- Hindi speech recognition is weaker than English
- Background objects at the frame edge flicker and are re-announced

### What was learned

- Count the first correctly set-up attempt; retrying until it works measures persistence, not the system
- Data that matters should be checked as it is loaded: a dictionary silently drops repeated keys
- A word is not always one word: "old" needs its noun to be translated
- A fix needs its own test, and a test can be wrong too
- A dead network hangs rather than failing, so a time limit is part of error handling
- Prompts are copied: whatever is in the hint can come back as the answer
- Look for the pattern in failures: every bad spelling correction broke the same rule

**Next goal:** Day 14 — make "repeat" replay what the user asked for, remove the
slow first Hindi answer, and a one-line setup for other machines.

---

## Day 14 — [8 oct]

**Goal:** Fix the last two problems testing exposed, make the project
installable with one command, read the whole codebase against its records, and
clean the repository before it is made public.

### 1. Repeat says the last real message

On Day 13, "फिर से बोलो" straight after a description repeated "hat", an ambient
announcement made in between. `say()` now remembers `last_message`, the last
thing said that was **not** ambient chatter, and R and "repeat" replay it. If
only ambient announcements have been said, the last of those is repeated.

### 2. Warming up the Hindi model

The first translation after NLLB loads took 9.5 and 10.6 s, against about 1 s
afterwards. While loading in the background, the model now translates one short
sentence before it is handed over, so a real request waits for the warm-up
instead of running alongside it. A failed warm-up is only printed.

### 3. One-line setup

`setup.py` (first added on 23 September) was rewritten:

    git clone https://github.com/Srijan-106/Guide.git; cd Guide; python setup.py

It checks the system (Python, GPU, driver, GPU memory, disk space on both drives,
Ollama), creates `.venv` and re-runs inside it, installs PyTorch then
`requirements.txt`, repairs OpenCV, verifies the install, downloads the models
(names read from the app's own code), and prints a table of ready features.
Options: `--check`, `--cpu`, `--no-hindi`, `--no-vision`. `docs/setup.md` gives
the measured requirements: Python 3.12+, an NVIDIA GPU with 4 GB for the core
features or 8 GB with visual questions, about 6 GB of disk (12 GB with
everything).

| Problem | Fix |
|---|---|
| The minimum was set to Python 3.10, but numpy 2.5.2 (pinned) needs 3.12 and scipy 1.17 needs 3.11, so an install on 3.10 would fail halfway | Minimum 3.12, checked before the virtual environment is created |
| After setup, OpenCV reported `GUI: NONE` and the window could not open: easyocr pulls in `opencv-python-headless`, which overwrote the full OpenCV's files, and reinstalls `python-bidi` (the Day 4 conflict, returning on every fresh install) | Setup removes both, force-reinstalls the pinned `opencv-python` without touching anything else, and checks the window support |
| Setup crashed with `UnicodeDecodeError`: `requirements.txt` had been saved by PowerShell 5.1 as UTF-16. The crash came after the headless package was removed, leaving no OpenCV at all | Text read with its encoding detected; the OpenCV version looked up **before** anything is removed; a repair command printed on failure; `requirements.txt` converted to UTF-8 |

Final run: all checks passed, PyTorch 2.11.0+cu128 with CUDA on the RTX 5060,
OpenCV window support WIN32UI, models downloaded, app started normally.

### 4. Reading the codebase against its records

Before the final documentation, every file was read and compared with this log,
the README and the results. Nothing was changed; findings were recorded:

- This log stopped at Day 9; the README and `docs/gpu-setup.md` are out of date
- The commit introducing Hindi mode mentions a transcription lock, but no version of `listener.py` ever had one: two transcriptions can run at once if V is used again while one is running
- The Whisper model size is set in two places (`main.py`, `listener.py`)
- The vision model's "loaded" flag is never reset, so after 20 idle minutes a question could time out again (from reading the code; not tested)
- Keywords match inside other words, e.g. "bread" contains "read" (from reading the code; not tested)
- Ollama must already be running when the app starts

### 5. Cleaning the repository's history

Older commits still contained files from an earlier experiment (including photos
whose source and licence were never recorded) and six model files no longer
used. Deleting a file only removes it from later commits, so the history was
rewritten: backed up first (`git bundle`), rewritten in a separate fresh copy
(`git filter-repo`), and checked before publishing. The current files are
identical, all 56 commits are kept, and none of the removed files appears in any
commit. The repository shrank from **234 MB to 45.5 MB**.

### Still open

- The README, and the architecture and design documents
- Decide on the code findings above
- Set `DEBUG = False`, then record the demo video

### What was learned

- The minimum Python version comes from the pinned packages: check each one
- A dependency can undo the environment on every install, so setup must repair after it and check rather than assume
- Work out how to undo a step before taking it
- PowerShell 5.1 writes UTF-16 text files
- Read the code against its own records before documenting it
- Back up, and prove the files are identical, before rewriting history

**Next goal:** Day 15 — the README, then the architecture and design documents,
and finally the demo video.