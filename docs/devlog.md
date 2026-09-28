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