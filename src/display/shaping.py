"""
Shaped text rendering, for scripts such as Devanagari.

Devanagari is not drawn letter by letter. Some vowel signs are drawn
before the consonant they follow in the text (the ि in दाहिनी), and some
letter pairs merge into a single shape (श्म in चश्मे, ब्र in ब्रेसलेट).
Arranging glyphs this way is called shaping.

Pillow shapes text only through its Raqm engine, which on Windows needs
an extra library that is not installed with it - so on the development
machine Hindi captions rendered with broken conjuncts and misplaced
vowel signs. This module shapes text with HarfBuzz, the engine web
browsers use, and draws the shaped glyphs with FreeType. Both install
with pip on every platform, so anyone running the project gets correct
Hindi without extra setup.
"""

import math

import numpy as np
from PIL import Image

try:
    import freetype
    import uharfbuzz as hb
    AVAILABLE = True
except Exception:
    AVAILABLE = False


class ShapedFont:
    """One typeface at one size, able to measure and draw shaped text."""

    def __init__(self, path, size, index=0):
        self.size = size
        with open(path, "rb") as file:
            data = file.read()

        # HarfBuzz decides WHICH glyphs to draw and WHERE
        self._hb_font = hb.Font(hb.Face(hb.Blob(data), index))
        self._hb_font.scale = (size * 64, size * 64)

        # FreeType draws each glyph
        self._ft = freetype.Face(path, index=index)
        self._ft.set_char_size(size * 64)

        metrics = self._ft.size
        self.ascent = metrics.ascender / 64
        self.descent = -metrics.descender / 64

    def _shape(self, text):
        buffer = hb.Buffer()
        buffer.add_str(text)
        buffer.guess_segment_properties()     # works out script and direction
        hb.shape(self._hb_font, buffer, {})
        return buffer.glyph_infos, buffer.glyph_positions

    def getlength(self, text):
        """Width of the text in pixels, after shaping."""
        _, positions = self._shape(text)
        return sum(position.x_advance for position in positions) / 64

    def getmetrics(self):
        return int(math.ceil(self.ascent)), int(math.ceil(self.descent))

    def render(self, text, colour):
        """The text as an RGBA Pillow image, transparent background."""
        infos, positions = self._shape(text)

        ascent, descent = self.getmetrics()
        pad = int(self.size * 0.35)           # room for marks above and below
        width = int(math.ceil(self.getlength(text))) + pad * 2
        height = ascent + descent + pad * 2
        alpha = np.zeros((height, width), dtype=np.float32)

        pen_x = pad
        baseline = pad + ascent
        for info, position in zip(infos, positions):
            self._ft.load_glyph(info.codepoint, freetype.FT_LOAD_RENDER)
            glyph = self._ft.glyph
            bitmap = glyph.bitmap
            if bitmap.width and bitmap.rows:
                glyph_alpha = np.array(bitmap.buffer, dtype=np.float32).reshape(
                    bitmap.rows, bitmap.pitch)[:, :bitmap.width] / 255.0
                x = int(round(pen_x + position.x_offset / 64 + glyph.bitmap_left))
                y = int(round(baseline - position.y_offset / 64 - glyph.bitmap_top))
                self._paint(alpha, glyph_alpha, x, y)
            pen_x += position.x_advance / 64

        image = np.zeros((height, width, 4), dtype=np.uint8)
        image[..., :3] = colour
        image[..., 3] = (alpha * 255).clip(0, 255).astype(np.uint8)
        return Image.fromarray(image, "RGBA"), pad

    @staticmethod
    def _paint(canvas, glyph, x, y):
        """Combine a glyph into the canvas, clipped to its edges."""
        h, w = glyph.shape
        x0, y0 = max(x, 0), max(y, 0)
        x1, y1 = min(x + w, canvas.shape[1]), min(y + h, canvas.shape[0])
        if x0 < x1 and y0 < y1:
            region = canvas[y0:y1, x0:x1]
            np.maximum(region, glyph[y0 - y:y1 - y, x0 - x:x1 - x], out=region)