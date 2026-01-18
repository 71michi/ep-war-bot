import io

from PIL import Image

from src.openai_parser import _ensure_png_bytes


def test_ensure_png_bytes_converts_jpeg_to_png():
    # create a tiny JPEG
    img = Image.new("RGB", (8, 8), color=(12, 34, 56))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    jpg = buf.getvalue()

    out = _ensure_png_bytes(jpg)
    # PNG signature
    assert out[:8] == b"\x89PNG\r\n\x1a\n"
