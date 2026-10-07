"""Check the microphone is actually picking up speech.

Whisper invents plausible-sounding text when it hears near-silence -
"Thank you", "You know what?" - so a quiet microphone looks like poor
recognition. This measures the actual signal level.
"""

import time

import numpy as np
import sounddevice as sd

SAMPLE_RATE = 16000

print("Input devices:")
for index, device in enumerate(sd.query_devices()):
    if device["max_input_channels"] > 0:
        default = " (default)" if index == sd.default.device[0] else ""
        print(f"  {index}: {device['name']}{default}")

print("\nSpeak normally for 5 seconds...\n")

recording = sd.rec(int(5 * SAMPLE_RATE), samplerate=SAMPLE_RATE,
                   channels=1, dtype="float32")

for second in range(5):
    time.sleep(1)
    # Peak level so far, as a rough loudness indicator
    peak = float(np.abs(recording[:int((second + 1) * SAMPLE_RATE)]).max())
    bars = "#" * int(peak * 50)
    print(f"  {second + 1}s   peak {peak:.3f}  {bars}")

sd.wait()

peak = float(np.abs(recording).max())
average = float(np.abs(recording).mean())

print(f"\nPeak level:    {peak:.3f}")
print(f"Average level: {average:.4f}")

if peak < 0.05:
    print("\nVery quiet - Whisper will mostly hear silence and invent text.")
    print("Raise the microphone level in Windows sound settings, or speak closer.")
elif peak < 0.15:
    print("\nQuiet. Recognition will be unreliable.")
elif peak > 0.95:
    print("\nClipping - too loud, which also harms recognition.")
else:
    print("\nLevel looks healthy.")