"""
Text-to-speech module for VisionAssist.

Uses edge-tts (Microsoft Edge's online TTS service) to generate speech,
and pygame to play it. This replaced pyttsx3, which failed repeatedly on
Windows due to SAPI5/COM event-loop and threading issues - see
docs/devlog.md (Day 5 and Day 6) for the full history.

NOTE: This requires an internet connection.
"""

import asyncio
import os
import tempfile
import threading
import queue

import edge_tts
import pygame


class Speaker:
    def __init__(self, voice="en-US-AriaNeural"):
        self.voice = voice
        self.queue = queue.Queue()

        # Start pygame's audio system once
        pygame.mixer.init()

        # One permanent background thread processes announcements in order
        self.worker = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker.start()

    def speak(self, text):
        """Queue text to be spoken. Returns immediately - never blocks."""
        if not text or not text.strip():
            return
        self.queue.put(text)

    def update(self):
        """Kept for compatibility with existing code. Does nothing now,
        since playback happens on the worker thread."""
        pass

    def shutdown(self):
        """Stop audio playback cleanly when the program exits."""
        try:
            pygame.mixer.quit()
        except Exception:
            pass

    def _worker_loop(self):
        """Runs forever on a background thread, speaking one item at a time."""
        while True:
            text = self.queue.get()
            try:
                self._speak_now(text)
            except Exception as error:
                print(f"  [speech error] {error}")

    def _speak_now(self, text):
        """Generate an audio file for the text, then play it to completion."""
        # Create a temporary file for the audio
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
            temp_path = temp_file.name

        try:
            # edge-tts is asynchronous, so we run it inside its own event loop
            asyncio.run(self._generate(text, temp_path))

            # Play the generated file
            pygame.mixer.music.load(temp_path)
            pygame.mixer.music.play()

            # Wait for playback to finish before taking the next item,
            # so announcements never overlap
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)

            # Release the file so it can be deleted
            pygame.mixer.music.unload()

        finally:
            # Always clean up the temporary file
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception:
                    pass

    async def _generate(self, text, output_path):
        """Ask Microsoft's service to synthesise the text into an audio file."""
        communicate = edge_tts.Communicate(text, self.voice)
        await communicate.save(output_path)