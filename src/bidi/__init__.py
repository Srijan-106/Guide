"""
Minimal stand-in for the python-bidi package.

EasyOCR does `from bidi import get_display` at import time, so the name
must exist or EasyOCR cannot start. The real package was removed from
this project because its current release (0.6.x) is a Rust rewrite
shipping a compiled binary, which Windows Smart App Control blocked -
stopping the application from starting at all.

WHAT BIDI ACTUALLY DOES: it reorders bi-directional text so that
right-to-left languages (Arabic, Hebrew, Urdu, Persian) display
correctly when mixed with left-to-right text.

WHY A STUB IS SAFE HERE: this application reads English only
(easyocr.Reader(['en'])). English is left-to-right, so the correct
output of the bidi algorithm is the text unchanged - which is exactly
what this returns.

IF A RIGHT-TO-LEFT LANGUAGE IS EVER ADDED: delete this folder and
install the real package (pip install python-bidi). Leaving this stub in
place would silently produce wrongly-ordered text.
"""


def get_display(text, *args, **kwargs):
    """Return the text unchanged.

    For left-to-right text, the bidi algorithm is a no-op, so this
    matches the real implementation's behaviour for English.
    """
    return text