"""Binary mask RLE kernels; optional JIT compilation never changes wire format."""

from collections.abc import Sequence
from functools import lru_cache
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

Backend = Literal["numpy", "numba"]


def _rle_loop(flat: NDArray[np.uint8]) -> NDArray[np.int64]:
    """Count alternating zero/one runs, including the initial empty zero run."""
    counts = np.empty(flat.size + 1, dtype=np.int64)
    index = 0
    value = 0
    run = 0
    for pixel in flat:
        if pixel == value:
            run += 1
        else:
            counts[index] = run
            index += 1
            value = pixel
            run = 1
    counts[index] = run
    return counts[: index + 1]


@lru_cache(maxsize=1)
def _compiled_rle() -> Any:
    """Load Numba only when explicitly requested; cache machine code on disk."""
    try:
        from numba import njit
    except ImportError as exc:
        raise RuntimeError("Numba backend requires: uv sync --extra acceleration") from exc
    return njit(cache=True, nogil=True)(_rle_loop)


def binary_run_lengths(mask: NDArray[Any], *, backend: Backend = "numpy") -> NDArray[np.int64]:
    """Return C-order runs of a 2D mask (positive pixels are foreground).

    The first count always describes zeros; an empty mask returns [0]. NumPy
    removes Python per-pixel overhead without JIT startup. Numba is opt-in for
    repeated large masks and releases the GIL, without fast-math changes.
    """
    if mask.ndim != 2:
        raise ValueError("binary mask must have exactly two dimensions")
    flat = np.ascontiguousarray(mask > 0, dtype=np.uint8).reshape(-1)
    if backend == "numba":
        return _compiled_rle()(flat)
    if backend != "numpy":
        raise ValueError(f"Unknown mask backend: {backend}")
    if not flat.size:
        return np.array([0], dtype=np.int64)
    transitions = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    counts = np.diff(np.concatenate(([0], transitions, [flat.size])))
    if flat[0]:
        counts = np.concatenate(([0], counts))
    return counts.astype(np.int64, copy=False)


def expand_binary_runs(counts: Sequence[int], height: int, width: int, *, order: str = "C") -> NDArray[np.uint8]:
    """Expand zero-first runs into a bounded uint8 array without Python pixel lists.

    Preserve the metadata decoder's compatibility behavior: short streams are
    zero padded and long streams are clipped to the declared shape. Unlike
    expanding a Python list, memory is bounded by that shape even for huge runs.
    """
    if height < 0 or width < 0:
        raise ValueError("mask dimensions must be nonnegative")
    flat = np.zeros(height * width, dtype=np.uint8)
    offset = 0
    for index, count in enumerate(counts):
        if count < 0:
            raise ValueError("run lengths must be nonnegative")
        end = min(flat.size, offset + int(count))
        if index % 2:
            flat[offset:end] = 1
        offset = end
        if offset == flat.size:
            break
    return flat.reshape((height, width), order="C" if order.upper() == "C" else "F")
