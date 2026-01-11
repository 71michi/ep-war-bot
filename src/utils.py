from __future__ import annotations
from typing import Iterable
import mimetypes

def guess_mime(filename: str) -> str:
    mt, _ = mimetypes.guess_type(filename)
    return mt or "image/png"

def is_image_filename(filename: str) -> bool:
    f = filename.lower()
    return f.endswith((".png", ".jpg", ".jpeg", ".webp"))
