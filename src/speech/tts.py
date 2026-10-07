"""
Text-to-speech module for VisionAssist.

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

Queue behaviour:
  priority   - an announcement the user asked for jumps ahead of queued
               ambient chatter and stops whatever is playing
  expires_in - an ambient announcement is dropped if it cannot be spoken
               within that many seconds, since a late announcement
               describes a scene that has already changed
"""

import asyncio
import os
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
                    spoken = self._rescue_with_offline_voice(text)

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