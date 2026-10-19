"""
Text-to-speech module for Guide.

Primary engine: edge-tts (Microsoft Edge's online voices), played through
pygame. This replaced pyttsx3, which failed five separate times on Windows
due to SAPI5/COM event-loop and threading issues - see docs/devlog.md
(Days 5 and 5A).

Fallback engine: pyttsx3, using the Windows built-in voice, for when there
is no internet. It is used only as a fallback, and only from inside this
module's worker thread - which is what makes it safe here. The Day 5
failures came from driving it across threads on the main loop; here the
worker thread owns it for the duration of each utterance, and a fresh
engine is created each time (the Day 5 fix for it falling silent after the
first announcement).

The very FIRST online failure falls back to the offline voice for that
same announcement, so nothing is lost to one dropped network packet. On
7 October a book cover was read correctly and then never spoken, because
the internet blinked at that moment and the old code only fell back after
a second failure. A dead network also counts as a failure after
ONLINE_TIMEOUT_SECONDS, instead of hanging.

Long text is spoken in pieces: see MAX_PIECE_CHARACTERS. If a piece fails
partway through, the offline voice says only the part not yet heard.

Queue behaviour:
  priority   - an announcement the user asked for jumps ahead of queued
               ambient chatter and stops whatever is playing
  expires_in - an ambient announcement is dropped if it cannot be spoken
               within that many seconds, since a late announcement
               describes a scene that has already changed
  merging    - ambient announcements waiting back to back are spoken as
               one sentence ("chair, hat, pillow"), with repeats removed.
               At startup the camera finds objects a frame at a time, so
               4-5 separate announcements queued up; spoken one by one, at
               about 2 s each, the later ones passed their expiry and were
               dropped - in all three startups logged on 6-7 October.
"""

import asyncio
import os
import re
import tempfile
import threading
import time
from collections import deque

import edge_tts
import pygame
import pyttsx3

# The longest the online voice may take to produce an announcement before
# it counts as a failure and the offline voice says it instead. Without a
# limit, a dead network does not fail - it hangs: on 7 October, with Wi-Fi
# switched off, no error appeared at all and every announcement waited
# behind the stuck one, in silence. Short announcements normally arrive in
# about a second; on a slow connection, 5 s may occasionally trip early,
# which costs one announcement in the less natural Windows voice.
ONLINE_TIMEOUT_SECONDS = 5

# The longest piece of text sent to the online voice in one request.
# The app waits for the COMPLETE audio of a request before playing it, and
# ONLINE_TIMEOUT_SECONDS applies to each request - so a long text could
# never be spoken online. Short announcements and two-sentence answers (up
# to about 230 characters) arrived within the limit. On 20 October a
# reading from a phone screen, about 550 characters, did not: it timed out
# and was read by the Windows voice instead. Longer text is therefore split
# at sentence and phrase breaks and spoken piece by piece. Each piece
# arrives well within the limit, and the first words are heard sooner. The
# cost is a short pause, while the next piece is fetched, between pieces.
MAX_PIECE_CHARACTERS = 200

# Where a long text may be split, best first: after the end of a sentence
# (including the Hindi full stop ।), after a comma, colon or semicolon -
# which is also how the lines of a reading are joined - and at a space.
_BREAKS = (r"(?<=[.!?।])\s+", r"(?<=[,;:])\s+", r"\s+")


def split_for_speech(text, limit=MAX_PIECE_CHARACTERS):
    """Split text into pieces of at most `limit` characters, at natural breaks.

    Text within the limit comes back unchanged, as one piece - so every
    short announcement is sent exactly as before. Longer text is broken at
    the best break available (see _BREAKS), then the parts are packed back
    together, in order, into as few pieces as fit. A word is only cut in
    the middle if it is longer than the limit on its own.
    """
    if len(text) <= limit:
        return [text]

    def units(part, level):
        if len(part) <= limit:
            return [part]
        if level == len(_BREAKS):
            return [part[i:i + limit] for i in range(0, len(part), limit)]
        found = []
        for sub in re.split(_BREAKS[level], part):
            if sub:
                found.extend(units(sub, level + 1))
        return found

    pieces = []
    for unit in units(" ".join(text.split()), 0):
        if pieces and len(pieces[-1]) + 1 + len(unit) <= limit:
            pieces[-1] += " " + unit
        else:
            pieces.append(unit)
    return pieces


class OnlineVoiceError(Exception):
    """The online voice failed while speaking a text.

    `unspoken` is the part the user has not heard yet, so the offline voice
    can say just that, instead of repeating a long reading from the start.
    The message is the original error's, so the terminal shows the same
    line as before and failure_reason() still recognises network errors.
    """

    def __init__(self, error, unspoken):
        super().__init__(str(error))
        self.unspoken = unspoken


class Speaker:
    def __init__(self, voice="en-IN-NeerjaNeural"):
        self.voice = voice

        # Pending announcements, each (text, expiry_time_or_None).
        # A deque can be appended at either end, which is how priority
        # announcements jump the queue.
        self._queue = deque()

        # Condition combines a lock with "wait until something changes",
        # so the worker sleeps instead of spinning while idle
        self._condition = threading.Condition()

        self._stopping = threading.Event()      # program is shutting down
        self._interrupt = threading.Event()     # cut short what is playing

        # Counts announcements that NEITHER voice could speak, so the app
        # can tell the user that speech itself has stopped working - which
        # cannot be announced by speaking
        self.consecutive_failures = 0
        self.last_error = ""

        # Counts online failures in a row. One is a blip: only the
        # announcement that failed uses the offline voice. Two means the
        # network is gone: speech stays offline until the main loop calls
        # try_online_again(). Kept separate from the count above, which is
        # reset whenever the user hears something - if the two were one
        # counter, a rescued announcement would reset it and speech would
        # never switch to offline, waiting for the dead network every time.
        self._online_failures = 0

        # True once speech has switched to the offline Windows voice
        self.offline = False

        pygame.mixer.init()

        self.worker = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker.start()

    # -----------------------------------------------------------------
    # Public interface
    # -----------------------------------------------------------------

    def speak(self, text, priority=False, expires_in=None):
        """Queue text to be spoken. Returns immediately - never blocks."""
        if not text or not text.strip() or self._stopping.is_set():
            return

        expiry = time.time() + expires_in if expires_in else None

        with self._condition:
            if priority:
                # Drop queued ambient chatter, stop what is playing, and
                # put this announcement at the front
                self._queue = deque(
                    item for item in self._queue if item[1] is None
                )
                self._queue.appendleft((text, expiry))
                self._interrupt.set()
            else:
                self._queue.append((text, expiry))
            self._condition.notify()

    def is_working(self):
        """False once two announcements in a row could not be spoken at all.

        Counts failures of BOTH voices: an online failure that the offline
        voice rescued does not count, because the user heard it. (The old
        version returned True whenever speech was offline, which hid the
        case of the offline voice failing too.)
        """
        return self.consecutive_failures < 2

    def is_offline(self):
        """True while using the Windows built-in voice."""
        return self.offline

    def try_online_again(self):
        """Attempt the online voice again - call when the network returns."""
        if self.offline:
            print("  [speech] trying the online voice again")
            self.offline = False
            self.consecutive_failures = 0
            self._online_failures = 0

    def failure_reason(self):
        """A short explanation of why speech is failing, for the terminal."""
        error = self.last_error.lower()
        if any(word in error for word in ("resolve", "connect", "network",
                                          "timeout", "ssl", "socket")):
            return "no internet connection"
        return self.last_error or "unknown error"

    def update(self):
        """Kept for compatibility with older code. Does nothing."""
        pass

    def shutdown(self):
        """Stop speaking and close the audio system cleanly."""
        self._stopping.set()
        self._interrupt.set()

        with self._condition:
            self._queue.clear()
            self._condition.notify_all()

        # Give the worker a moment to notice and finish
        self.worker.join(timeout=3.0)

        try:
            pygame.mixer.music.stop()
            pygame.mixer.quit()
        except Exception:
            pass

    # -----------------------------------------------------------------
    # Background worker
    # -----------------------------------------------------------------

    def _worker_loop(self):
        while not self._stopping.is_set():
            with self._condition:
                while not self._queue and not self._stopping.is_set():
                    self._condition.wait(timeout=0.2)
                if self._stopping.is_set():
                    return
                text, expiry = self._queue.popleft()

                # An ambient announcement takes the others waiting right
                # behind it, so they are spoken as one sentence
                if expiry is not None:
                    text, expiry = self._merge_waiting_ambient(text, expiry)
                    if text is None:
                        continue          # every part had expired

            # Skip announcements that waited too long to still be useful
            if expiry is not None and time.time() > expiry:
                print(f"  [speech] dropped (too late): {text}")
                continue

            self._interrupt.clear()

            try:
                if self.offline:
                    self._speak_offline(text)
                else:
                    self._speak_online(text)
                    self._online_failures = 0
                self.consecutive_failures = 0

            except Exception as error:
                if self._stopping.is_set():
                    continue

                self.last_error = str(error)
                print(f"  [speech error] {error}")

                spoken = False
                if not self.offline:
                    # The online voice failed. Say it with the offline
                    # voice rather than lose it - on the FIRST failure,
                    # since the user may have asked for this announcement.
                    self._online_failures += 1
                    # Only the part not yet heard: a long reading that
                    # failed halfway is not repeated from the start
                    spoken = self._rescue_with_offline_voice(
                        getattr(error, "unspoken", text))

                    # Twice in a row is a dead network rather than a blip:
                    # stay offline, so later announcements are not delayed
                    # by waiting for the network every time
                    if self._online_failures >= 2:
                        print("  [speech] switching to the offline voice")
                        self.offline = True

                if spoken:
                    self.consecutive_failures = 0     # the user heard it
                else:
                    self.consecutive_failures += 1

    def _merge_waiting_ambient(self, text, expiry):
        """Join an ambient announcement with the ambient ones queued
        directly behind it. Called with the queue's lock held.

        Ambient announcements are object lists ("chair, glasses"), so
        they are split on ", " to remove repeats - "cabinet/shelf" queued
        twice is said once. Parts that have already expired are dropped
        as before. Only announcements directly behind are taken, so the
        order of everything else in the queue is unchanged.

        Returns (merged text, latest expiry), or (None, None) if every
        part had expired.
        """
        now = time.time()
        labels = []
        latest_expiry = expiry
        fresh_parts = 0                   # announcements actually kept

        def take(part, part_expiry):
            nonlocal fresh_parts
            if now > part_expiry:
                print(f"  [speech] dropped (too late): {part}")
                return
            fresh_parts += 1
            for label in part.split(", "):
                if label and label not in labels:
                    labels.append(label)

        take(text, expiry)
        while self._queue and self._queue[0][1] is not None:
            part, part_expiry = self._queue.popleft()
            take(part, part_expiry)
            latest_expiry = max(latest_expiry, part_expiry)

        if not labels:
            return None, None

        merged = ", ".join(labels)
        if fresh_parts > 1:
            print(f"  [speech] merged {fresh_parts} announcements: {merged}")
        return merged, latest_expiry

    def _rescue_with_offline_voice(self, text):
        """Speak text with the Windows voice after an online failure.

        Returns True if it was spoken.
        """
        print("  [speech] online voice failed - saying it with the "
              "offline voice")
        try:
            self._speak_offline(text)
            return True
        except Exception as offline_error:
            print(f"  [speech error] the offline voice also failed: "
                  f"{offline_error}")
            return False

    # -----------------------------------------------------------------
    # The two engines
    # -----------------------------------------------------------------

    def _speak_online(self, text):
        """Speak text with the online voice, one piece at a time.

        Short text is a single piece, spoken exactly as before. If a piece
        fails, the error carries the part not yet spoken (see
        OnlineVoiceError). If the user interrupts, the rest is dropped,
        just as a single long announcement was cut off before.
        """
        pieces = split_for_speech(text)
        for number, piece in enumerate(pieces):
            if self._interrupt.is_set() or self._stopping.is_set():
                return
            try:
                self._speak_piece_online(piece)
            except Exception as error:
                raise OnlineVoiceError(error,
                                       " ".join(pieces[number:])) from error

    def _speak_piece_online(self, text):
        """Generate an audio file with edge-tts, then play it."""
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
            temp_path = temp_file.name

        try:
            # edge-tts is asynchronous, so it runs in its own event loop -
            # with a time limit, because a dead network hangs rather than
            # failing (see ONLINE_TIMEOUT_SECONDS)
            try:
                asyncio.run(asyncio.wait_for(self._generate(text, temp_path),
                                             ONLINE_TIMEOUT_SECONDS))
            except asyncio.TimeoutError:
                # The words "network" and "timeout" let failure_reason()
                # report this as "no internet connection"
                raise ConnectionError(
                    f"no reply from the online voice within "
                    f"{ONLINE_TIMEOUT_SECONDS} s (network timeout)")

            if self._stopping.is_set():
                return

            pygame.mixer.music.load(temp_path)
            pygame.mixer.music.play()

            # Wait for playback, but stop early if interrupted or quitting
            while pygame.mixer.music.get_busy():
                if self._interrupt.is_set() or self._stopping.is_set():
                    pygame.mixer.music.stop()
                    break
                pygame.time.Clock().tick(20)

            pygame.mixer.music.unload()

        finally:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception:
                    pass

    async def _generate(self, text, output_path):
        """Ask Microsoft's service to synthesise the text into a file."""
        communicate = edge_tts.Communicate(text, self.voice)
        await communicate.save(output_path)

    def _speak_offline(self, text):
        """Speak using the Windows built-in voice, with no internet.

        A fresh engine is created for each utterance: reusing one caused
        pyttsx3 to fall silent after the first announcement (Day 5,
        Error 1). This runs on the worker thread, which then owns the
        engine for the whole utterance - matching the COM thread-affinity
        rule behind Day 5's silent failures.
        """
        engine = pyttsx3.init()
        try:
            engine.setProperty("rate", 170)
            engine.say(text)
            engine.runAndWait()
        finally:
            try:
                engine.stop()
            except Exception:
                pass