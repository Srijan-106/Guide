"""Quick check that voice recognition works, before wiring it into the app.

Records for three seconds, transcribes, and shows which command matched.
"""

import sys
import time

sys.path.insert(0, "src")
from voice.listener import VoiceListener, match_command

listener = VoiceListener()

while True:
    input("\nPress Enter, then speak for 3 seconds (Ctrl+C to stop)...")

    listener.start_recording()
    print("  listening...")
    time.sleep(3)
    audio = listener.stop_recording()

    if audio is None:
        print("  nothing recorded")
        continue

    print("  transcribing...")
    text = listener.transcribe(audio)
    print(f"  heard:   {text!r}")
    print(f"  command: {match_command(text)}")