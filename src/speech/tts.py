"""
Text-to-speech module for VisionAssist.

Uses edge-tts (Microsoft Edge's online voices) to generate speech, and
pygame to play it. This replaced pyttsx3, which failed repeatedly on
Windows due to SAPI5/COM event-loop and threading issues - see
docs/devlog.md (Days 5 and 5A).

Speech is queued and spoken one item at a time, by a single background
worker thread. Two features keep the queue from falling behind:

  priority   - an announcement the user explicitly asked for jumps ahead
               of queued ambient chatter and stops whatever is playing
  expires_in - an ambient announcement is dropped if it cannot be spoken
               within that many seconds, since a late announcement
               describes a scene that has already changed

NOTE: this requires an internet connection.
"""

import asyncio
import os
import tempfile
import threading
import time
from collections import deque

import edge_tts
import pygame


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

        # Set when the program is shutting down
        self._stopping = threading.Event()

        # Set when the current playback should be cut short
        self._interrupt = threading.Event()

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
        self.worker.join(timeout=2.0)

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
                self._speak_now(text)
            except Exception as error:
                if not self._stopping.is_set():
                    print(f"  [speech error] {error}")

    def _speak_now(self, text):
        """Generate an audio file for the text, then play it."""
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