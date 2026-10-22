"""
Does MediaPipe tell the user's right hand from their left - and which way
round?

The vision model cannot: asked "which hand am I raising?" it said "your
right hand" every time, whichever hand was raised (22 October). MediaPipe's
hand model labels each hand it finds as "Left" or "Right", but it assumes
a MIRRORED picture, like a phone's selfie view. The laptop webcam's
picture is not mirrored, so its labels may come out the wrong way round.

This script measures it before anything is built on it. Raise your right
hand, then your left; for each, it reads 15 frames twice - the picture as
the camera gives it, and the same picture mirrored - and counts what
MediaPipe called the hand each time. Whichever way gets both hands right
is the one the application will use.

Needs, in the project folder:
    models/hand_landmarker.task      MediaPipe's hand model (about 8 MB)

    python src\\evaluation\\check_hands.py
"""

import os
import sys
import time

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision

MODEL_PATH = os.path.join("models", "hand_landmarker.task")
FRAMES_PER_HAND = 15


def load_landmarker():
    if not os.path.exists(MODEL_PATH):
        sys.exit(f"Missing {MODEL_PATH} - download it first (see the setup steps).")
    options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=2,
    )
    return vision.HandLandmarker.create_from_options(options)


def labels_in(landmarker, frame_bgr):
    """MediaPipe's label for every hand in one frame, e.g. ['Left']."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect(image)
    return [hand[0].category_name for hand in result.handedness if hand]


def measure(landmarker, camera, hand):
    """Count MediaPipe's labels for one raised hand, both ways round."""
    counts = {"as it is": {"Left": 0, "Right": 0, "none": 0},
              "mirrored": {"Left": 0, "Right": 0, "none": 0}}
    for _ in range(FRAMES_PER_HAND):
        ok, frame = camera.read()
        if not ok:
            continue
        for way, picture in (("as it is", frame),
                             ("mirrored", cv2.flip(frame, 1))):
            found = labels_in(landmarker, picture)
            if len(found) == 1:
                counts[way][found[0]] += 1
            else:
                counts[way]["none"] += 1     # no hand, or two hands seen
        time.sleep(0.1)
    return counts


def decide(results):
    """Which way round labels BOTH hands correctly, by majority of frames.

    results = {"Right": counts, "Left": counts} - counts as from measure().
    Returns "as it is", "mirrored", or None if neither is clearly right.
    """
    for way in ("as it is", "mirrored"):
        right_ok = results["Right"][way]["Right"] > results["Right"][way]["Left"]
        left_ok = results["Left"][way]["Left"] > results["Left"][way]["Right"]
        enough = (results["Right"][way]["Right"] >= FRAMES_PER_HAND // 2 and
                  results["Left"][way]["Left"] >= FRAMES_PER_HAND // 2)
        if right_ok and left_ok and enough:
            return way
    return None


def main():
    landmarker = load_landmarker()
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        sys.exit("The camera could not be opened - close the application first.")
    for _ in range(10):              # let the camera adjust its exposure
        camera.read()

    results = {}
    try:
        for hand in ("Right", "Left"):
            input(f"\nRaise your {hand.upper()} hand, open, facing the camera - "
                  f"then press Enter and hold it still...")
            results[hand] = measure(landmarker, camera, hand)
            print(f"  done")
    finally:
        camera.release()
        landmarker.close()

    print("\n" + "=" * 64)
    print("What MediaPipe called each hand, out of "
          f"{FRAMES_PER_HAND} frames")
    print("=" * 64)
    print(f"{'you raised':12}{'picture':12}{'said Right':>12}{'said Left':>12}{'no hand':>10}")
    for hand in ("Right", "Left"):
        for way in ("as it is", "mirrored"):
            c = results[hand][way]
            print(f"{hand:12}{way:12}{c['Right']:>12}{c['Left']:>12}{c['none']:>10}")

    way = decide(results)
    print()
    if way:
        print(f"RESULT: both hands are labelled correctly with the picture {way.upper()}.")
    else:
        print("RESULT: neither way labelled both hands correctly. Paste this table "
              "before anything is built on it.")


if __name__ == "__main__":
    main()