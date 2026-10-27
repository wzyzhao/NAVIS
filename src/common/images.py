"""OpenCV image loading through Python paths, including Unicode paths on Windows."""

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray


def read_image(path: Path, flags: int = cv2.IMREAD_COLOR) -> NDArray[Any] | None:
    """Decode an image or return None for missing, empty or unreadable input.

    Python opens the file so OpenCV's Windows narrow-character filename API is
    avoided. Decoding flags and BGR channel ordering match cv2.imread.
    """
    try:
        data = path.read_bytes()
    except OSError:
        return None
    if not data:
        return None
    return cv2.imdecode(np.frombuffer(data, dtype=np.uint8), flags)
