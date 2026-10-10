# Contributing to VisionAssist

Rules for changing this project. They exist because most of them were learned
the hard way: each one has a day in [docs/devlog.md](docs/devlog.md) behind it.

**Read first:** [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) explains how the
system works. This file says what you may do to it.

---

## Who this application is for

A blind or visually impaired user who **cannot check anything they are told**.
That single fact decides most of the rules below. A confident wrong answer is
worse than no answer, and a failure the user cannot hear might as well not have
been detected.

## When the documents disagree

The **code** is always right. After that:

1. `docs/ARCHITECTURE.md` — how it is built
2. `docs/PRD.md` — what it is for
3. `docs/DESIGN_SYSTEM.md` — how it speaks and looks
4. `README.md` — the summary

If two of them contradict each other, say so and fix the wrong one. Do not
quietly follow whichever is convenient.

---

## Where new code belongs

| Adding | Goes in |
|---|---|
| A sentence the app speaks | `src/language/phrases.py`, **in both English and Hindi**; speak it with `say(t("your_key"))` |
| A voice command | A keyword list in `COMMANDS` in `src/voice/listener.py` (both languages, more specific phrases first), a branch in `run_command()` in `main.py`, and the words in `VOCABULARY_PROMPTS` |
| A word the Hindi translator gets wrong | `_GLOSSARY_PAIRS` in `src/language/translator.py`, following the rules written above that list; then run `check_glossary.py` |
| Anything drawn on the video window | `src/display/hud.py`, using the existing colour tokens |
| A new setting | The settings block at the top of `main.py`, with a comment saying what it controls and why that value |

## Do

- **Speak through `say()`.** It remembers what was said so R can repeat it, and
  it decides priority. The single exception is `repeat_last()`, which calls the
  speaker directly so that repeating does not overwrite the memory.
- **Mark anything the user asked for as `priority=True`.** Only ambient
  announcements expire.
- **Give background threads their own copy of the frame.** The main loop
  replaces `frame` constantly.
- **Let the main loop change the mode, the language and the current frame.**
  A background thread hands a command over through `pending_command` instead.
- **Clear a busy flag in `finally`.** One unhandled error must not block a
  feature for the rest of the session.
- **Let an optional feature fail quietly on its own.** Voice commands, visual
  questions, Hindi translation and spelling correction all switch themselves
  off rather than stopping the application.
- **Say something when nothing happened.** "No readable text found" is an
  answer; silence is not.

## Don't

- **Don't state an uncertain label as a fact.** Below `MIN_LABEL_CONFIDENCE`
  the app says nothing about the object; between that and `CONFIDENT_LABEL` it
  hedges. Keep both.
- **Don't send images anywhere.** Detection, text reading, speech recognition
  and visual answers all run on the machine. Only the text to be spoken goes
  to the online voice service.
- **Don't put wording in the middle of the code.** It belongs in `phrases.py`,
  or one language silently loses it.
- **Don't add a second way to do something that already exists.** SPACE and I
  did the same thing for two days, and it confused the design until they were
  given separate jobs.
- **Don't "improve" working code that nobody asked about.**

## Stop and ask before

- **Changing a model, a threshold or a confidence value.** The 86 recorded
  results describe the current values. Changing one makes them describe
  software that no longer exists.
- **Changing a voice.** The Hindi sentences use feminine verb forms
  ("पढ़ रही हूँ") that match `hi-IN-SwaraNeural`. A male voice means rewriting
  every one of them.
- **Adding any cloud service.**
- **Changing how the user is spoken to:** the hedging, the dropped
  announcements, the priority order, the failure beep.
- **Removing the `src/bidi` stand-in.** EasyOCR imports it at startup. It is
  safe only while the app reads English; a right-to-left language means
  installing the real `python-bidi` instead.

## How to check your work

There are **no automated tests**. The evaluation is manual, and these scripts
are the checks:

| After changing | Run |
|---|---|
| The glossary or `polish_hindi()` | `python src\evaluation\check_glossary.py` — every line, not just the PASS count |
| The spelling rules | `python src\evaluation\check_spelling.py` — all 22 cases must pass |
| Anything else | Run the app and exercise the feature **by key and by voice**, in English and in Hindi |

For a change to a model or a threshold, re-record the affected tests and
re-analyse them. There are two evaluations, each with its own pair of scripts
and its own results file:

| Evaluation | Record | Analyse | Results |
|---|---|---|---|
| The 86-test feature evaluation | `record_results.py` | `analyse_results.py` | `evaluation/results.csv` |
| The 31-object room sweep | `record_room_sweep.py` | `analyse_room_sweep.py` | `evaluation/room_sweep.csv` |

**Recording results:** fix the test list before testing. Record each verdict as
it happens. Afterwards, check every row against the application's own terminal
log. Correct a wrong verdict in a dated script in `src/evaluation/`, with the
reason written into the row, and never by hand. Remove a test only when the log
shows it did not test what it claims — never because the result was
disappointing.

## Git

- **No model files**, except the detection model already in `models/`, and no
  `.venv`, caches or generated output.
- **No audio files.** The application writes its speech to the system's
  temporary folder.
- **Commit messages** say what changed and why, in one line.
- **Don't force-push** `main` unless you are deliberately rewriting history and
  have made a backup first (`git bundle create ... --all`).

## Before a demonstration

Set `DEBUG = False` in `src/assistant/main.py`. With it on, every detection and
every OCR word is printed, which is useful while developing and noisy on
screen.