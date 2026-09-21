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


> **Correction (Day 7):** the nightly build was never necessary. Stable PyTorch has supported `sm_120` since version 2.7 via its CUDA 12.8 (`cu128`) builds. The original error came from installing a CUDA 12.4 build. The project moved to stable `2.11.0+cu128` on Day 7.


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

## Day 3 — [ 7 -sept]

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

Two issues would have prevented the project running on any other machine:

| Problem | Fix |
|---|---|
| `requirements.txt` pinned a PyTorch nightly build (`torch==2.12.0.dev20260408+cu128`) not available on PyPI | Removed the pin entirely; Ultralytics installs a compatible version automatically. RTX 50-series users directed to `docs/gpu-setup.md` |
| `requirements.txt` listed ~30 transitive sub-dependencies from `pip freeze` | Reduced to 6 direct dependencies; pip resolves the rest, avoiding version conflicts |
| `src/ocr/ocr.py` contained a hardcoded Tesseract path specific to the development machine | Marked deprecated with an explanatory docstring; retained for reference only |

Created `docs/gpu-setup.md` documenting the RTX 50-series `sm_120` incompatibility, the CUDA 12.8 nightly fix (later found unnecessary - see Day 7), the tested configuration, and measured performance figures.

Verified the new `requirements.txt` by creating a clean virtual environment outside the project and installing from it successfully.

### Known limitation carried forward

EasyOCR returns text regions in its own detection order, not human reading order. When pointed at dense paragraph text, individually correct words are announced in a meaningless sequence. Position-based sorting is the identified fix, not yet implemented.

**Next goal:** Day 6 — Integration: combine object detection and OCR into a single application.

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

**Next goal:** Day 7 — Currency recognition: train and evaluate a classifier for Indian banknotes.


---

## Day 7 — [20–21 sept]

**Goal:** Build currency recognition (Stage 8), and move from using pretrained models to training one.

**Why this matters:** every model used until now was pretrained by someone else. This is the first model trained, evaluated and analysed within this project.

### 1. Dataset and approach

The dataset contains folders of images per denomination (`Rs.10` ... `Rs.2000`) with no bounding-box labels - a **classification** dataset, not a detection one. YOLO detection cannot train on it directly.

**Decision: classification rather than detection.** It works with the data as-is, and matches the real use case: a user holds a note up to the camera and asks what it is, rather than scanning a room for money.

Rs.2000 was excluded, since it has been withdrawn from circulation. Six classes remain.

### 2. First training run (v1)

The first attempt (`yolo26s-cls`, batch 32) ran out of GPU memory. Ultralytics retried at batch 16 automatically, which also failed.

Switched to `yolo26n-cls` at batch 8:

| Setting | Value |
|---|---|
| Model | yolo26n-cls (ImageNet pretrained) |
| Epochs | 30 |
| Batch | 8 |
| Training time | 1.95 hours |
| **Top-1 accuracy** | **99.4%** |

### 3. Data leakage discovered

99.4% looked suspiciously high. Filenames such as `10_original_103.jpg_<id>.jpg` showed the dataset contains **several augmented copies of each original photo**. The split script had shuffled individual files, so copies of the same photo could land in both training and validation.

A leakage check was written to group files by original photo.

**The first version of the check was wrong.** It split filenames on `_jpg_`, but the real separator is `.jpg_`. Every file therefore looked unique and it reported 0% leakage. This was spotted because the number of "originals" exactly equalled the number of files - implausible for an augmented dataset.

After fixing the separator:

| | Originals |
|---|---|
| In training | 3,350 |
| In validation | 1,462 |
| **In both** | **1,230 (84.1%)** |

**Fix:** rebuilt the split grouped by original photo, so every copy of a photo lands on one side only. Verified: 0 shared originals, with originals (2,863 train / 719 val) now correctly fewer than files (7,173 / 1,772).

### 4. Retraining on the clean split (v2)

| Setting | Value |
|---|---|
| Batch | 32 |
| Epochs | 28 of 30 (early stopping; best at epoch 18) |
| Training time | 1.75 hours |
| **Top-1 accuracy** | **99.3%** |
| Inference | 0.6 ms per image |

**The accuracy barely changed.** The leakage was real but was not inflating the score much - Indian notes are strongly colour-coded, so recognising an unseen photo is nearly as easy as a seen one. The fix still mattered: 99.3% is now a number that can be defended.

**Timing lesson:** batch 32 was expected to cut training time substantially. It did not (1.95 h to 1.75 h). With the GPU at around 20% utilisation, the bottleneck was not the GPU but most likely CPU-side decoding and resizing of large JPEG files. Pre-resizing images once would likely speed up future training considerably.

### 5. Confusion matrix analysis (v2 validation)

13 errors out of 1,772:

| Class | Recall | Mistaken for |
|---|---|---|
| Rs.10 | 100% | - |
| Rs.20 | 100% | - |
| Rs.500 | 99.3% | Rs.200 (2) |
| Rs.200 | 99.1% | Rs.10, Rs.100, Rs.500 |
| Rs.50 | 98.9% | Rs.10, Rs.500 (2) |
| Rs.100 | 98.7% | Rs.500 (3), Rs.10, Rs.50 |

**Safety finding:** every error that overestimated a note's value was predicted as Rs.500. For a blind user this is the most costly mistake - being handed Rs.100 in change and told it is Rs.500. Overall accuracy hides this; the confusion matrix reveals it.

### 6. Evaluation on an independent dataset

Validation images came from the same source as training - same camera, same photographers, similar backgrounds - so 99.3% only describes performance on that source.

A second, independent dataset was used for testing only: 2,577 phone photos (captured October 2022, per filename timestamps), with deliberately varied backgrounds including notes photographed in front of screens showing neon patterns.

| Class | Recall |
|---|---|
| Rs.100 | 98.5% |
| Rs.200 | 98.2% |
| Rs.10 | 95.7% |
| Rs.500 | 95.7% |
| Rs.50 | 91.8% |
| **Rs.20** | **46.4%** |
| **Overall** | **85.1%** |

Excluding Rs.20, accuracy was about 95.5% - the model generalises reasonably well, except for one note.

266 errors overestimated a note's value, 134 of them predicted as Rs.500. 226 of the 266 came from Rs.20 alone.

### 7. Root cause of the Rs.20 failure

**Hypothesis:** India introduced a new green-yellow Rs.20 design in 2019, and the training data contained only the older red design.

**Verification:** all 292 misclassified Rs.20 images were copied into one folder and inspected. **287 were the new design; 5 were the old one.** Hypothesis confirmed.

This is a **training-data coverage gap**, not a model problem. More epochs, a larger model or tuning would not help - no model can recognise something it was never shown. It also explains why the errors scattered across several classes: a six-class classifier must always choose one of its six classes, and cannot say "I don't recognise this."

**Decision:** retraining with new Rs.20 images is deferred. Two approaches were considered: photographing new Rs.20 notes personally, or using part of the external dataset's Rs.20 images with a blocked split by capture time (a random split would leak burst photos taken seconds apart). The limitation is documented instead.

### 8. Environment incident: Windows Smart App Control

Mid-session, previously working code failed to start:

    ImportError: DLL load failed while importing _C: An Application Control policy has blocked this file.

Windows Security's Protection history showed nothing. The block was found in Event Viewer (**CodeIntegrity > Operational**): Smart App Control had blocked `torch\_C.cp313-win_amd64.pyd`, the core compiled PyTorch file.

**Cause:** Smart App Control moves automatically from Evaluation mode (blocks nothing) to On (enforces), without notification. The nightly PyTorch build - a rare development snapshot - had no reputation with Windows, so was blocked.

**Correction of a Day 1 mistake:** the nightly build was never needed. Stable PyTorch has supported `sm_120` since version 2.7 via CUDA 12.8 builds. On Day 1 a CUDA 12.4 build was installed, failed, and nightly was chosen instead of trying the CUDA 12.8 stable build. The project moved to stable `2.11.0+cu128`, which loaded successfully.

A second block followed on scipy 1.18.1 (`_ufuncs_cxx`), resolved by installing scipy 1.17. Smart App Control was kept **on** throughout.

### 9. Repository hygiene

- `data/` added to `.gitignore`, keeping around 9,000 training images out of GitHub
- A `weights/` rule intended to exclude a stray duplicate folder also matched `experiments/*/weights/`, silently excluding the trained models from the first commit. Noticed because `best.pt` was missing from the committed file list. Fixed by anchoring the rule to the project root: `/weights/`
- Stray downloaded files removed; the crashed run's empty folder deleted
- `requirements.txt` corrected: it still listed `pyttsx3` (no longer used) and was missing `edge-tts` and `pygame`
- `docs/gpu-setup.md` corrected, with clickable reference links

### What was learned

- A result that looks too good deserves checking before celebrating
- A verification tool can itself be wrong - the first leakage check gave a false 0%, caught only because its output was implausible
- In-distribution accuracy can overstate real performance: 99.3% on familiar data, 85.1% on independent data
- Overall accuracy can hide the errors that matter most; the confusion matrix exposed the Rs.500 overestimation pattern
- Some failures are data problems, not model problems, and no amount of training fixes missing data
- Security software can break a working environment without any change to the project
- A broad `.gitignore` rule can silently exclude important files

### Known limitations carried forward

- The new Rs.20 design is not recognised, and is sometimes confidently misread as Rs.500
- The currency model is trained but not yet integrated into the application
- The classifier cannot express uncertainty; it always names one of six notes

**Next goal:** Day 8 — Integrate the currency classifier into `main.py`, with a "not sure" response when confidence is low, so users are protected from confident misreadings before retraining.