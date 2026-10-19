# Guide — Design System

**The interface is speech.** The user cannot see the screen, so everything the
application knows it must say, and how it says it is the whole design. The
video window is secondary: it exists for the sighted people nearby.

This document covers the voice first, then the display, then the rules for
changing either. Everything here was taken from the code, not invented:
sentences from `src/language/phrases.py`, behaviour from `src/speech/tts.py`
and `src/assistant/main.py`, and the display from `src/display/hud.py`.

---

## Part 1 — The voice

### 1.1 Honesty before completeness

The user cannot check what they are told. A confident wrong answer is
therefore the worst possible output, worse than silence, and three rules
follow from that.

**Say nothing rather than guess.** Below 0.40 confidence, no object name is
spoken at all; the text on the object is read instead. In the evaluation this
produced five "no label" results out of ten identifications: not failures, but
the design working.

**Hedge what is uncertain.** Between 0.40 and 0.50, the name is spoken as a
possibility, not a fact.

| Confidence | English | Hindi |
|---|---|---|
| 0.50 or more | "I see bottle." | "यह bottle है।" |
| 0.40–0.50 | "This might be bottle." | "यह शायद bottle है।" |
| Below 0.40 | nothing about the object; the text is read | — |

**Mark an opinion as an opinion.** Every answer from the vision model is
wrapped: "It looks like …" / "ऐसा लगता है: …". The model sometimes states
details it is not sure of, and the wording says so without refusing to answer.

### 1.2 Never leave silence unexplained

Silence is ambiguous: the user cannot tell "nothing is there" from "it is
broken". So every outcome is spoken, including the empty ones:

| Situation | What is said |
|---|---|
| No text found | "No readable text found" |
| Nothing recognised | "I do not see anything I recognise" |
| Reading already in progress | "Still reading, please wait" |
| Nothing was heard | "I did not hear anything" |
| The question was too short | "Please ask a full question" |
| A feature is unavailable | "Visual questions are not available" / "Voice commands are not available" |
| Speech itself has failed | **a beep every 10 seconds** — it needs no network and no speech engine |

### 1.3 What gets said first

Speech is a queue of one voice, so order is a design decision.

| Rule | Behaviour |
|---|---|
| **The user comes first** | Anything the user asked for (a key or a command) jumps the queue and stops whatever is playing |
| **Stale news is dropped** | An ambient announcement not spoken within 6 seconds is dropped: it describes a scene that has changed |
| **Say it once, together** | Objects appearing at the same moment become one sentence — "chair, hat, pillow" — not four announcements |
| **Do not repeat yourself** | The same object is not announced again within 30 seconds, even if it leaves and returns |
| **Do not talk over a reading** | In reading mode, ambient announcements are silenced entirely |

### 1.4 Hindi

Hindi mode is **Hinglish**: Hindi sentences, English object names. "मुझे दिख
रहा है: bottle, chair". This is how most Hindi speakers name these objects
aloud, and it avoids translating 365 object names into words no one uses.

| Rule | Reason |
|---|---|
| The assistant speaks of itself in the **feminine** ("पढ़ रही हूँ", "देखती हूँ") | The Hindi voice, `hi-IN-SwaraNeural`, is female. These forms must change if the voice does |
| The user is always addressed as **आप**, never तुम | Respectful address. The translator slips into तुम, and a step after translation corrects it |
| Hindi text reading is **not** offered | Out of scope; text reading is English only |

### 1.5 The voices

| Language | Voice | Chosen because |
|---|---|---|
| English | `en-IN-NeerjaExpressiveNeural` | Indian English: clearer on Indian names and product labels. Picked by listening to every Indian voice the service offers |
| Hindi | `hi-IN-SwaraNeural` | Picked the same way |
| Fallback | The Windows built-in voice | Needs no internet. Noticeably more robotic, which is accepted: understandable beats nothing |

### 1.6 Writing a new sentence

Every spoken sentence lives in `src/language/phrases.py`, in both languages.
Rules for adding one:

1. **Short.** It is heard, not read; it cannot be skimmed.
2. **Plain.** No jargon, no English words in the English sentences that a
   listener would have to spell out.
3. **Says what happened, not what failed internally.** "Sorry, the vision model
   did not respond", not an error code.
4. **Both languages, every time.** A missing Hindi sentence falls back to
   English rather than crashing, but that is a safety net, not a plan.
5. **Keep the feminine forms** in Hindi, to match the voice.

---

## Part 2 — The display

### 2.1 Who it is for

Not the user, who cannot see it. It is for the people around them: a helper, a
tester, someone watching a demonstration. Their difficulty is the opposite of
the user's — they can see everything and hear nothing of what the application
is thinking. **So the display's main job is to make the speech visible.**
Everything else is deliberately quiet.

### 2.2 Colours

| Token | Value | Used for |
|---|---|---|
| Slate | `#18202B` | Panel backgrounds, at 72–82% opacity so the video stays visible |
| Paper | `#F6F1E7` | Captions and primary text |
| Mist | `#B0BAC7` | Secondary text: key labels, frame rate |
| Sky | `#7CC6E8` | Detection boxes and the ambient-mode label — context, not focus |
| Marigold | `#F2A93B` | The identified object, and reading mode — the accent, used sparingly |
| Coral | `#E8615A` | Listening, and speech failure |
| Lavender | `#A99BEA` | Thinking (a visual question in progress) |
| Saffron | `#E9C46A` | The offline voice is in use |

### 2.3 Type

**Nirmala UI**, Windows' typeface for Indian scripts and the only built-in one
that draws Latin and Devanagari as a single family. Hindi also needs
**shaping** — letters that merge (ब्र, श्म) and vowel signs drawn before the
letter they follow — which is done with HarfBuzz when Pillow's own shaping
engine is missing. Without shaping, Hindi renders as broken letters.

| Element | Size |
|---|---|
| Caption | 30 px |
| Title | 22 px bold |
| Status pill | 17 px |
| Key strip, box labels | 15–16 px |

### 2.4 Elements

| Element | Design |
|---|---|
| **Caption** | The one prominent thing: centred, up to three lines, wrapped and ellipsised, fading out 7 seconds after it was spoken |
| **Status pills** | Appear only while something is happening: Listening (with a pulsing dot), Thinking, Reading text, Offline voice, Speech not working |
| **Key strip** | Along the bottom; shows only keys that are actually available, and L is labelled with the language it switches **to** |
| **Detection boxes** | Thin sky-blue outlines with a small name tag. Confidence numbers appear only while `DEBUG` is on |
| **The identified object** | A thicker marigold box labelled "Identified: …", for 3 seconds |
| **Top strip** | Title, mode pill, language pill, and the frame rate on the right |

### 2.5 Wording on screen

Sentence case, in plain words: "Looking around", "Reading text", "Speech not
working" — not "MODE: AMBIENT" or "TTS_ERROR". The screen is read by people
who are not developers.

---

## Part 3 — Rules for changing any of this

**Stop and ask before:**

- Changing a voice. The Hindi sentences use feminine verb forms that match the
  current voice; a male voice means rewriting them.
- Changing a confidence threshold. The thresholds decide when the application
  speaks, hedges or stays silent, and they are tied to the recorded evaluation.
- Making an uncertain answer sound certain, or removing a hedge.
- Making a failure silent.
- Adding a second way to say the same thing.

**Always:**

- Put new sentences in `phrases.py`, in both languages.
- Keep a failure audible, even when speech is the thing that failed.
- Use the colour tokens rather than new colours; the palette is small on
  purpose.
- Re-run the relevant scripts in `src/evaluation/` after changing wording that
  the tests check.

**Never:**

- State a label the application is not confident about as a fact.
- Let an ambient announcement interrupt something the user asked for.
- Add text to the screen that is not also available by ear, since the user
  cannot see it.
