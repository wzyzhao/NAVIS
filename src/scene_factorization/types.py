"""Shared array and record types used by the processing pipeline."""

from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

Image: TypeAlias = NDArray[np.uint8]
FloatImage: TypeAlias = NDArray[np.float32]
BoundingBox: TypeAlias = list[int]
Record: TypeAlias = dict[str, Any]
