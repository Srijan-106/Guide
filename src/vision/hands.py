"""
Which hand - the user's right or left - using MediaPipe's hand model.

The vision model cannot tell: asked "which hand am I raising?" it said
"your right hand" seven times out of seven, whichever hand was raised
(22 October). MediaPipe's hand landmarker finds each hand in the picture
and labels it Left or Right, from the shape of the hand itself.

MEASURED, NOT ASSUMED (src/evaluation/check_hands.py, 22 October): with
the laptop webcam's picture exactly as the camera gives it, MediaPipe
labelled the user's right hand "Right" in 15 of 15 frames, and the left
hand "Left" in 15 of 15. Mirroring the picture first reversed every
label. Its documentation suggests a mirrored picture is expected; the
measurement says otherwise for this camera, so the picture is used as it
is. A different camera should be checked again with the same script.

Optional: if MediaPipe or its model file is missing, available is False
and the application simply gives no hand side - which is what it did
before this module existed.

Licence: MediaPipe and its hand model are Apache 2.0.
"""

import os

MODEL_PATH = os.path.join("models", "hand_landmarker.task")

# Where setup.py downloads it from (about 8 MB)
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/"
             "hand_landmarker/hand_landmarker/float16/latest/"
             "hand_landmarker.task")

# A hand whose label MediaPipe is less sure of than this is not used
MIN_LABEL_SCORE = 0.7


class HandSides:
    def __init__(self, model_path=MODEL_PATH):
        self.available = False
        self._landmarker = None
        try:
            if not os.path.exists(model_path):
                print(f"  [hands] {model_path} not found - hand sides off")
                return
            # Imported here, so the application starts without MediaPipe
            import mediapipe as mp
            from mediapipe.tasks.python import BaseOptions, vision
            options = vision.HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=model_path),
                running_mode=vision.RunningMode.IMAGE,
                num_hands=2,
            )
            self._mp = mp
            self._landmarker = vision.HandLandmarker.create_from_options(options)
            self.available = True
            print("  [hands] hand sides from MediaPipe")
        except Exception as error:
            print(f"  [hands] unavailable, hand sides off: {error}")

    def sides(self, frame_bgr):
        """The user's own side of every hand in the picture, e.g. ["right"].

        Hands MediaPipe is unsure of are left out. Never raises: any
        problem gives an empty list, which means "no side known".
        """
        if not self.available:
            return []
        try:
            import cv2
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB,
                                   data=rgb)
            result = self._landmarker.detect(image)
            found = []
            for hand in result.handedness:
                if hand and hand[0].score >= MIN_LABEL_SCORE:
                    found.append(hand[0].category_name.lower())
            return found
        except Exception as error:
            print(f"  [hands] could not check the hands: {error}")
            return []