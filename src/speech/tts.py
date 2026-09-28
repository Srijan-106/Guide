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


class Speaker:
    def __init__(self, voice="en-US-AriaNeural"):
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

        # Counts consecutive failures, so the app can tell the user that
        # speech itself has stopped working - which cannot be announced
        # by speaking
        self.consecutive_failures = 0
        self.last_error = ""

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
        """False only when speech has failed with no working fallback."""
        return self.offline or self.consecutive_failures < 2

    def is_offline(self):
        """True while using the Windows built-in voice."""
        return self.offline

    def try_online_again(self):
        """Attempt the online voice again - call when the network returns."""
        if self.offline:
            print("  [speech] trying the online voice again")
            self.offline = False
            self.consecutive_failures = 0

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
                self.consecutive_failures = 0

            except Exception as error:
                if self._stopping.is_set():
                    continue

                self.consecutive_failures += 1
                self.last_error = str(error)
                print(f"  [speech error] {error}")

                # After two online failures, switch to the offline voice
                # and retry this announcement rather than losing it
                if not self.offline and self.consecutive_failures >= 2:
                    print("  [speech] switching to offline voice")
                    self.offline = True
                    try:
                        self._speak_offline(text)
                        self.consecutive_failures = 0
                    except Exception as offline_error:
                        print(f"  [speech error] offline voice also failed: "
                              f"{offline_error}")

    # -----------------------------------------------------------------
    # The two engines
    # -----------------------------------------------------------------

    def _speak_online(self, text):
        """Generate an audio file with edge-tts, then play it."""
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
            temp_path = temp_file.name

        try:
            # edge-tts is asynchronous, so it runs in its own event loop
            asyncio.run(self._generate(text, temp_path))

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