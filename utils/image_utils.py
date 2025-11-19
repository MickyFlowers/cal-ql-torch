import io

import numpy as np
from PIL import Image


def encode_png(pil_image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    return buffer.getvalue()

def decode_png(png_bytes: bytes) -> Image.Image:
    buffer = io.BytesIO(png_bytes)
    image = Image.open(buffer)
    return image

