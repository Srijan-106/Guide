"""
On-screen display for Guide.

WHO THIS IS FOR: not the blind user, who cannot see it, but the sighted
people around them - a helper, a tester, someone watching a demo. Their
difficulty is that they cannot hear what the application is thinking. So
the display's main job is to make the speech visible: large live
captions of everything the application says, in English or Hindi.
Everything else is deliberately quiet.

DESIGN
  Captions      the one prominent element - large, centred, fading out
                after the speech is over
  Type          Nirmala UI throughout: Windows' typeface for Indian
                scripts, and the only built-in one that draws Latin and
                Devanagari as a single family
  Panels        translucent slate, so the video stays visible
  Boxes         thin, soft sky blue - detections are context, not focus
  Accent        marigold, used only for the object being identified
  Status        small pills that appear only while something is active
  Wording       sentence case ("Looking around"), not "MODE: AMBIENT"

WHY PILLOW: OpenCV's built-in fonts cannot draw Devanagari at all - Hindi
captions came out as question marks. Text is drawn with Pillow instead,
and every rendered piece of text is cached, so unchanged text costs
almost nothing on later frames.

SHAPING: Devanagari also needs "shaping" - conjuncts such as ब्र and श्म
are single shapes, and some vowel signs are drawn before the letter they
follow. Pillow shapes only through its Raqm engine, which was missing on
the development machine, so ब्रेसलेट rendered as ब्‌रेसलेट. When Raqm is
missing, text is shaped with HarfBuzz instead - see display/shaping.py.
"""

import math
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, features

from display import shaping
from display.shaping import ShapedFont

# ---------------------------------------------------------------------
# Design tokens (RGB)
# ---------------------------------------------------------------------

SLATE = (24, 32, 43)          # panels
PAPER = (246, 241, 231)       # caption and primary text
MIST = (176, 186, 199)        # secondary text
SKY = (124, 198, 232)         # detection boxes
MARIGOLD = (242, 169, 59)     # the identified object; reading mode
CORAL = (232, 97, 90)         # listening; failures
LAVENDER = (169, 155, 234)    # thinking
SAFFRON = (233, 196, 106)     # offline voice

PANEL_OPACITY = 0.72

TOP_STRIP_HEIGHT = 52
KEY_STRIP_HEIGHT = 46
MARGIN = 18

# How long a caption stays on screen after it was spoken, and how long it
# takes to fade out at the end
CAPTION_SECONDS = 7.0
CAPTION_FADE = 1.2

# Status pills, in display order: (label, colour)
STATUSES = {
    "listening":   ("Listening", CORAL),
    "thinking":    ("Thinking", LAVENDER),
    "reading":     ("Reading text", MARIGOLD),
    "offline":     ("Offline voice", SAFFRON),
    "speech_down": ("Speech not working", CORAL),
}

MODE_LABELS = {
    "ambient": ("Looking around", SKY),
    "reading": ("Reading", MARIGOLD),
}

LANGUAGE_LABELS = {"en": "English", "hi": "हिंदी"}

# Windows 11 stores every Nirmala style in one collection file; older
# Windows has separate files. A collection holds several faces, so the
# ones wanted are picked by name rather than by position.
NIRMALA_COLLECTION = "C:/Windows/Fonts/Nirmala.ttc"
NIRMALA_FAMILY = "Nirmala UI"

# Separate-file candidates, best first, used when there is no collection.
# The later entries let the display run elsewhere, with less suitable type.
FONT_FILES = {
    "regular": [
        "C:/Windows/Fonts/Nirmala.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ],
    "bold": [
        "C:/Windows/Fonts/NirmalaB.ttf",
        "C:/Windows/Fonts/segoeuib.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansDevanagari-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ],
}


def _overlap(a, b):
    """Intersection over union of two [x1, y1, x2, y2] boxes."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = ((a[2] - a[0]) * (a[3] - a[1]) +
             (b[2] - b[0]) * (b[3] - b[1]) - inter)
    return inter / union if union > 0 else 0.0


def bgr(rgb):
    """OpenCV draws in BGR order; the tokens above are RGB."""
    return rgb[2], rgb[1], rgb[0]


class Hud:
    def __init__(self, font_path=None, bold_font_path=None):
        """font_path / bold_font_path override the typeface search -
        useful on systems without Nirmala UI."""

        # Devanagari needs "shaping" - see the note at the top of this
        # file. Pillow shapes text only with its Raqm engine; without
        # Raqm, HarfBuzz is used instead, through display/shaping.py.
        self.shaping = features.check("raqm")
        self.use_harfbuzz = not self.shaping and shaping.AVAILABLE
        if self.use_harfbuzz:
            print("  [display] shaping Hindi text with HarfBuzz")
        elif not self.shaping:
            print("  [display] no text shaping available - Hindi captions "
                  "will show broken conjuncts. Install it with:  "
                  "pip install uharfbuzz freetype-py")

        regular, bold = self._choose_fonts(font_path, bold_font_path)

        # Reported, because Hindi captions only render if the regular
        # face covers Devanagari - Nirmala UI does; Segoe UI does not
        print(f"  [display] typeface: {self._describe(regular)}, "
              f"bold: {self._describe(bold)}")

        self.fonts = {
            "title":   self._load(bold, 22),
            "pill":    self._load(regular, 17),
            "small":   self._load(regular, 16),
            "key":     self._load(bold, 15),
            "caption": self._load(regular, 30),
            "label":   self._load(bold, 15),
        }

        self._sprites = {}          # cache of rendered text and panels
        self._strips = {}           # cache of solid colour strips

    # -----------------------------------------------------------------
    # Fonts and caching
    # -----------------------------------------------------------------

    @staticmethod
    def _find(paths):
        import os
        for path in paths:
            if os.path.exists(path):
                return path
        return None

    @staticmethod
    def _collection_faces(path):
        """{(family, style): index} for every face in a .ttc collection."""
        faces = {}
        for index in range(32):
            try:
                name = ImageFont.truetype(path, 12, index=index).getname()
            except OSError:
                break                       # past the last face
            faces[name] = index
        return faces

    def _choose_fonts(self, font_path, bold_font_path):
        """Pick the regular and bold faces as (path, index) pairs."""
        import os

        if font_path:
            return (font_path, 0), (bold_font_path or font_path, 0)

        regular = bold = None

        if os.path.exists(NIRMALA_COLLECTION):
            faces = self._collection_faces(NIRMALA_COLLECTION)
            index = faces.get((NIRMALA_FAMILY, "Regular"))
            if index is None:               # any Nirmala UI face will do
                index = next((i for (family, _), i in faces.items()
                              if family == NIRMALA_FAMILY), None)
            if index is not None:
                regular = (NIRMALA_COLLECTION, index)

            index = faces.get((NIRMALA_FAMILY, "Bold"))
            if index is not None:
                bold = (NIRMALA_COLLECTION, index)

        if regular is None:
            path = self._find(FONT_FILES["regular"])
            regular = (path, 0) if path else None

        # Bold text - the title, keys and box labels - is always Latin, so
        # a Latin-only bold face is an acceptable fallback here
        if bold is None:
            path = self._find(FONT_FILES["bold"])
            bold = (path, 0) if path else regular

        return regular, bold

    @staticmethod
    def _describe(face):
        import os
        if face is None:
            return "built-in fallback"
        path, index = face
        name = os.path.basename(path)
        return f"{name} #{index}" if name.lower().endswith(".ttc") else name

    def _load(self, face, size):
        if face is None:
            return ImageFont.load_default(size=size)
        path, index = face
        if self.use_harfbuzz:
            return ShapedFont(path, size, index)
        engine = (ImageFont.Layout.RAQM if self.shaping
                  else ImageFont.Layout.BASIC)
        return ImageFont.truetype(path, size, index=index,
                                  layout_engine=engine)

    def _cached(self, key, build):
        """Return a cached sprite, building it on first use."""
        sprite = self._sprites.get(key)
        if sprite is None:
            if len(self._sprites) > 400:      # bound the memory used
                self._sprites.clear()
            sprite = build()
            self._sprites[key] = sprite
        return sprite

    def _put_text(self, image, xy, text, font_key, colour):
        """Draw text onto a Pillow image, shaped if the font needs it.

        Pillow fonts draw directly; HarfBuzz-shaped fonts render to an
        image of their own, which is then laid onto this one.
        """
        font = self.fonts[font_key]
        x, y = int(round(xy[0])), int(round(xy[1]))
        if isinstance(font, ShapedFont):
            rendered, pad = font.render(text, colour)
            image.paste(rendered, (x - pad, y - pad), rendered)
        else:
            ImageDraw.Draw(image).text((x, y), text, font=font,
                                       fill=colour + (255,))

    @staticmethod
    def _to_bgra(image):
        """Pillow RGBA image -> numpy BGRA array for blending."""
        array = np.asarray(image, dtype=np.uint8)
        return array[..., [2, 1, 0, 3]].copy()

    # -----------------------------------------------------------------
    # Drawing primitives
    # -----------------------------------------------------------------

    @staticmethod
    def _blit(frame, sprite, x, y, fade=1.0):
        """Alpha-blend a BGRA sprite onto the frame, clipped to its edges."""
        frame_h, frame_w = frame.shape[:2]
        h, w = sprite.shape[:2]
        x0, y0 = max(x, 0), max(y, 0)
        x1, y1 = min(x + w, frame_w), min(y + h, frame_h)
        if x0 >= x1 or y0 >= y1:
            return

        part = sprite[y0 - y:y1 - y, x0 - x:x1 - x]
        alpha = part[..., 3:4].astype(np.float32) * (fade / 255.0)
        region = frame[y0:y1, x0:x1].astype(np.float32)
        blended = part[..., :3].astype(np.float32) * alpha + region * (1 - alpha)
        frame[y0:y1, x0:x1] = blended.astype(np.uint8)

    def _strip(self, frame, top, height, opacity=PANEL_OPACITY):
        """A full-width translucent band."""
        width = frame.shape[1]
        key = (width, height)
        solid = self._strips.get(key)
        if solid is None:
            solid = np.full((height, width, 3), bgr(SLATE), dtype=np.uint8)
            self._strips[key] = solid
        region = frame[top:top + height]
        cv2.addWeighted(solid[:region.shape[0]], opacity, region,
                        1 - opacity, 0, dst=region)

    def _text(self, text, font_key, colour):
        """A sprite of text alone, with transparent background."""
        def build():
            font = self.fonts[font_key]
            ascent, descent = font.getmetrics()
            width = int(math.ceil(font.getlength(text))) + 4
            image = Image.new("RGBA", (max(width, 1), ascent + descent + 6),
                              (0, 0, 0, 0))
            self._put_text(image, (2, 2), text, font_key, colour)
            return self._to_bgra(image)
        return self._cached(("text", text, font_key, colour), build)

    def _pill(self, text, colour, font_key="pill", dot=False):
        """A rounded label: outline in the given colour, optional dot."""
        def build():
            font = self.fonts[font_key]
            ascent, descent = font.getmetrics()
            text_w = int(math.ceil(font.getlength(text)))
            pad_x = 12
            dot_space = 16 if dot else 0
            w = text_w + pad_x * 2 + dot_space
            h = ascent + descent + 10
            image = Image.new("RGBA", (w, h), (0, 0, 0, 0))
            draw = ImageDraw.Draw(image)
            draw.rounded_rectangle((0, 0, w - 1, h - 1), radius=h // 2,
                                   fill=SLATE + (int(255 * 0.82),),
                                   outline=colour + (255,), width=2)
            self._put_text(image, (pad_x + dot_space, 4), text, font_key,
                           PAPER)
            return self._to_bgra(image)
        return self._cached(("pill", text, colour, font_key, dot), build)

    # -----------------------------------------------------------------
    # Detection boxes
    # -----------------------------------------------------------------

    def draw_detections(self, frame, result, show_confidence=False,
                        chosen_box=None):
        """Thin boxes with a small name label - calmer than the default.

        chosen_box - the object picked by "what is this?". It is drawn in
        marigold with an "Identified" label naming it, in place of its
        ordinary label rather than on top of it.
        """
        names = result.names
        boxes = result.boxes.xyxy.tolist()
        classes = result.boxes.cls.tolist()
        confidences = result.boxes.conf.tolist()

        # Which detection, if any, is the identified object? The chosen
        # box comes from a separate detection pass, so match by overlap.
        chosen_index = None
        if chosen_box:
            best = 0.0
            for index, box in enumerate(boxes):
                overlap = _overlap(box, chosen_box)
                if overlap > best:
                    best, chosen_index = overlap, index
            if best < 0.5:
                chosen_index = None

        for index, (box, class_id, conf) in enumerate(zip(boxes, classes,
                                                           confidences)):
            if index == chosen_index:
                continue                      # drawn below, in marigold
            x1, y1, x2, y2 = (int(v) for v in box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), bgr(SKY), 2, cv2.LINE_AA)

            label = names[int(class_id)]
            if show_confidence:
                label = f"{label}  {conf:.2f}"
            self._place_label(frame, self._label(label, SKY), x1, y1)

        if chosen_box:
            x1, y1, x2, y2 = (int(v) for v in chosen_box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), bgr(MARIGOLD), 3,
                          cv2.LINE_AA)
            text = "Identified"
            if chosen_index is not None:
                text = f"Identified: {names[int(classes[chosen_index])]}"
            self._place_label(frame, self._label(text, MARIGOLD), x1, y1)

    def _place_label(self, frame, sprite, x1, y1):
        """Put a label just above a box, or just inside it near the top."""
        label_y = y1 - sprite.shape[0] - 2
        if label_y < TOP_STRIP_HEIGHT:            # keep clear of the strip
            label_y = y1 + 2
        self._blit(frame, sprite, x1, label_y)

    def _label(self, text, colour):
        """A small filled tag above a box."""
        def build():
            font = self.fonts["label"]
            ascent, descent = font.getmetrics()
            w = int(math.ceil(font.getlength(text))) + 14
            h = ascent + descent + 6
            image = Image.new("RGBA", (w, h), (0, 0, 0, 0))
            draw = ImageDraw.Draw(image)
            draw.rounded_rectangle((0, 0, w - 1, h - 1), radius=5,
                                   fill=colour + (235,))
            self._put_text(image, (7, 2), text, "label", SLATE)
            return self._to_bgra(image)
        return self._cached(("label", text, colour), build)

    # -----------------------------------------------------------------
    # The overlay
    # -----------------------------------------------------------------

    def draw(self, frame, mode, language, fps, statuses, caption,
             caption_age, keys):
        """Draw everything except detections.

        statuses    - keys of STATUSES currently active
        caption     - the last thing spoken
        caption_age - seconds since it was spoken
        keys        - [(key, action), ...] for the key strip
        """
        height, width = frame.shape[:2]

        self._draw_top_strip(frame, mode, language, fps)
        self._draw_statuses(frame, statuses)
        self._draw_caption(frame, caption, caption_age)
        self._draw_key_strip(frame, keys)

    def _draw_top_strip(self, frame, mode, language, fps):
        width = frame.shape[1]
        self._strip(frame, 0, TOP_STRIP_HEIGHT)

        title = self._text("Guide", "title", PAPER)
        self._blit(frame, title, MARGIN, (TOP_STRIP_HEIGHT - title.shape[0]) // 2)

        x = MARGIN + title.shape[1] + 20
        mode_text, mode_colour = MODE_LABELS.get(mode, (mode, SKY))
        for pill in (self._pill(mode_text, mode_colour),
                     self._pill(LANGUAGE_LABELS.get(language, language), MIST)):
            self._blit(frame, pill, x, (TOP_STRIP_HEIGHT - pill.shape[0]) // 2)
            x += pill.shape[1] + 10

        fps_sprite = self._text(f"{int(fps)} fps", "small", MIST)
        self._blit(frame, fps_sprite, width - MARGIN - fps_sprite.shape[1],
                   (TOP_STRIP_HEIGHT - fps_sprite.shape[0]) // 2)

    def _draw_statuses(self, frame, statuses):
        y = TOP_STRIP_HEIGHT + 14
        for key in STATUSES:
            if key not in statuses:
                continue
            text, colour = STATUSES[key]
            pill = self._pill(text, colour, dot=True)
            self._blit(frame, pill, MARGIN, y)

            # The dot is drawn live, not cached, so that it can pulse
            # while listening - motion that answers the user pressing V
            radius = 5
            if key == "listening":
                radius = int(5 + 2 * (1 + math.sin(time.time() * 6)) / 2)
            centre = (MARGIN + 14, y + pill.shape[0] // 2)
            cv2.circle(frame, centre, radius, bgr(colour), -1, cv2.LINE_AA)

            y += pill.shape[0] + 8

    def _draw_caption(self, frame, caption, age):
        if not caption or age > CAPTION_SECONDS:
            return

        height, width = frame.shape[:2]
        max_width = min(1040, width - MARGIN * 4)
        sprite = self._caption_sprite(caption, max_width)

        # Fade out over the last moments, rather than vanishing abruptly
        fade = 1.0
        remaining = CAPTION_SECONDS - age
        if remaining < CAPTION_FADE:
            fade = max(0.0, remaining / CAPTION_FADE)

        x = (width - sprite.shape[1]) // 2
        y = height - KEY_STRIP_HEIGHT - 16 - sprite.shape[0]
        self._blit(frame, sprite, x, y, fade)

    def _caption_sprite(self, caption, max_width):
        """The caption panel: text wrapped to at most three lines."""
        def build():
            font = self.fonts["caption"]
            pad_x, pad_y = 26, 14
            lines = self._wrap(caption, font, max_width - pad_x * 2, 3)

            ascent, descent = font.getmetrics()
            line_h = int((ascent + descent) * 1.18)
            text_w = max(int(math.ceil(font.getlength(line))) for line in lines)

            w = text_w + pad_x * 2
            h = line_h * len(lines) + pad_y * 2
            image = Image.new("RGBA", (w, h), (0, 0, 0, 0))
            draw = ImageDraw.Draw(image)
            draw.rounded_rectangle((0, 0, w - 1, h - 1), radius=14,
                                   fill=SLATE + (int(255 * 0.80),))
            for index, line in enumerate(lines):
                line_w = font.getlength(line)
                self._put_text(image, ((w - line_w) / 2, pad_y + index * line_h),
                               line, "caption", PAPER)
            return self._to_bgra(image)
        return self._cached(("caption", caption, max_width), build)

    @staticmethod
    def _wrap(text, font, max_width, max_lines):
        """Split text into lines that fit, ending with an ellipsis if cut."""
        words = text.split()
        lines, current = [], ""
        for word in words:
            trial = f"{current} {word}".strip()
            if font.getlength(trial) <= max_width:
                current = trial
            else:
                if current:
                    lines.append(current)
                current = word
        if current:
            lines.append(current)

        if len(lines) > max_lines:
            lines = lines[:max_lines]
            last = lines[-1]
            while last and font.getlength(last + " …") > max_width:
                last = last.rsplit(" ", 1)[0] if " " in last else last[:-1]
            lines[-1] = last + " …"
        return lines or [""]

    def _draw_key_strip(self, frame, keys):
        height, width = frame.shape[:2]
        top = height - KEY_STRIP_HEIGHT
        self._strip(frame, top, KEY_STRIP_HEIGHT)

        x = MARGIN
        for key, action in keys:
            cap = self._key_cap(key)
            label = self._text(action, "small", MIST)
            needed = cap.shape[1] + 8 + label.shape[1]
            if x + needed > width - MARGIN:
                break
            self._blit(frame, cap, x, top + (KEY_STRIP_HEIGHT - cap.shape[0]) // 2)
            x += cap.shape[1] + 8
            self._blit(frame, label, x, top + (KEY_STRIP_HEIGHT - label.shape[0]) // 2)
            x += label.shape[1] + 22

    def _key_cap(self, key):
        """A key drawn as a small keycap."""
        def build():
            font = self.fonts["key"]
            ascent, descent = font.getmetrics()
            w = max(int(math.ceil(font.getlength(key))) + 16, 28)
            h = ascent + descent + 8
            image = Image.new("RGBA", (w, h), (0, 0, 0, 0))
            draw = ImageDraw.Draw(image)
            draw.rounded_rectangle((0, 0, w - 1, h - 1), radius=6,
                                   fill=(52, 64, 80, 255),
                                   outline=(84, 98, 116, 255), width=1)
            self._put_text(image, ((w - font.getlength(key)) / 2, 3), key,
                           "key", PAPER)
            return self._to_bgra(image)
        return self._cached(("key", key), build)