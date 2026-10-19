# Guide — Product Requirements

What this project is for, what it does today, and what it cannot do. Kept
short: the reasoning is in [devlog.md](devlog.md), the build is in
[ARCHITECTURE.md](ARCHITECTURE.md), and installing and running it is in the
[README](../README.md).

---

## The problem

A blind or visually impaired person cannot see what is in front of them, and
most of the everyday information around them is printed: a product label, a
sign, a page, the name of the thing someone just handed them. Asking another
person is always possible, but it costs independence, and nobody is always
there.

Existing help falls into two groups. Phone apps that send the camera image to a
company's servers work well but need internet, a subscription, or both, and the
pictures of someone's home leave their house. Dedicated hardware is accurate
and expensive. Neither answers an open question in the user's own language.

## Who it is for

**The main user** is a blind or visually impaired person with a laptop or a
webcam, who speaks English or Hindi. They control everything by keyboard or
voice, and everything the application knows, it says aloud.

**A second group matters too:** the sighted people nearby — a family member, a
helper, a teacher watching a demonstration. They can see the screen but cannot
hear what the application is thinking, which is why the video window shows
large captions of everything spoken.

This is a student project, built to be understood and defended rather than
shipped. It is not a medical device and not a mobility aid: it does not warn
about obstacles, steps or traffic, and it must not be relied on for safety.

## What it does today

| Feature | Status | Note |
|---|---|---|
| Announces objects as they come into view | Built | 365 object types; reliable for common objects |
| Reads text aloud (SPACE) | Built | Printed text; in reading order |
| Names a held-up object and reads the text on it (I) | Built | Says nothing about the object when unsure, rather than guessing |
| Describes everything in view (D) | Built | |
| Repeats the last thing said (R) | Built | |
| Voice commands (V) | Built | English and Hindi |
| Answers open questions about what the camera sees (V) | Built | A vision model running on the machine |
| Hindi mode (L) | Built | Hindi voice, commands and answers |
| On-screen captions, status and key strip | Built | For the sighted people nearby |
| Keeps working when things fail | Built | Offline voice, beeps, spoken explanations |
| One-command installation | Built | `python setup.py` |
| Reading handwriting | Partly built | Neat handwriting in good light only |
| Reading Hindi text | Not built | Text reading is English only |
| A wake word, instead of pressing V | Planned | The microphone is already open all the time |
| Obstacle or hazard warnings | Not planned | Out of scope, and unsafe to attempt at this accuracy |
| Running on a phone | Not planned | |

## Measured results

**117 tests across two evaluations.** In both, the list was fixed before any
test was run, and every row was afterwards checked against the application's
own log. Full numbers and caveats: [README](../README.md#measured-results) and
[devlog.md](devlog.md) (Days 9, 12, 13 and 15).

**The feature evaluation — 86 tests,** on a list of ten objects that
deliberately included three known weak cases:

| Feature | Result |
|---|---|
| Object detection | 17/30 = 56.7%, and every failure was silence, not a wrong name |
| Identification | 4/10 = 40% |
| Text reading (exact or partial) | 12/16 = 75% |
| Voice commands | English 7/7, Hindi 5/7 |
| Visual questions | English 8/8, Hindi 2/4 |
| Hindi translation | 3/4 |
| **Questions about things that are not there** | **4/4 answered "there is none", nothing invented** |

The last row is the one that matters most for this user: a confident wrong
answer cannot be checked by someone who cannot see.

**The room sweep — 31 objects,** asking a different question: of the objects
actually in a room, listed before the application was started, how many does it
name?

| | Result |
|---|---|
| Named correctly | **21/31 = 67.7%** |
| Said nothing | 7 |
| Gave the wrong name | 3 |

Two measurements, not a before and after. Neither replaces the other.

## Constraints

| Constraint | Why |
|---|---|
| **Windows 10 or 11** | The offline voice, the failure beep and the Hindi screen font are Windows features |
| **An NVIDIA GPU with 8 GB** for visual questions (4 GB without) | Measured: 7.7 of 8 GB in use with everything loaded |
| **Internet for the natural voice** | Without it, speech falls back to the Windows voice |
| **Non-commercial use only** | The Hindi translation model (NLLB-200) is licensed CC-BY-NC 4.0 |
| **Python 3.12 or newer** | Required by the pinned package versions |
| **Camera images never leave the machine** | A deliberate rule: detection, reading, recognition and visual answers all run locally |
| **One person, one machine, one room** | Everything measured so far was measured there |

## Current status

Working end to end, evaluated, documented, and installable on another Windows
machine with one command. Features were frozen on 6 October; since then the
work has been testing, fixing what testing found, and documentation.

**Remaining before it is finished:** a demo video, and turning the diagnostic
output off (`DEBUG = False`) before recording it.

**Known weaknesses, written up rather than hidden:** handwriting, objects
outside the model's vocabulary, Hindi speech recognition, and a misheard
question being answered confidently. The full list is in the
[README](../README.md#known-limitations) and
[ARCHITECTURE.md](ARCHITECTURE.md#11-known-limitations-and-open-issues).