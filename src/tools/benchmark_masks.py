"""Measure RLE kernels on a synthetic ROI mask; report cold JIT separately."""

import argparse
import json
from collections.abc import Callable
from statistics import median
from time import perf_counter

import numpy as np

from common.masks import binary_run_lengths


def legacy_counts(mask: np.ndarray) -> list[int]:
    """Original per-pixel implementation retained only as a benchmark reference."""
    counts = []
    value = run = 0
    for pixel in np.asarray(mask > 0, dtype=np.uint8).reshape(-1):
        pixel = int(pixel)
        if pixel == value:
            run += 1
        else:
            counts.append(run)
            value, run = pixel, 1
    return [*counts, run]


def elapsed(function: Callable[[], object], repeats: int) -> float:
    """Return median elapsed seconds, keeping each invocation's result alive."""
    samples = []
    for _ in range(repeats):
        started = perf_counter()
        result = function()
        samples.append(perf_counter() - started)
        del result
    return median(samples)


def main() -> None:
    """Benchmark equivalent kernels without touching experiment inputs/outputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--numba", action="store_true")
    args = parser.parse_args()
    if min(args.height, args.width, args.repeats) < 1:
        parser.error("image dimensions and repeats must be positive")
    mask = np.zeros((args.height, args.width), dtype=np.uint8)
    mask[args.height // 4 : 3 * args.height // 4, args.width // 4 : 3 * args.width // 4] = 255
    expected = legacy_counts(mask)
    np.testing.assert_array_equal(binary_run_lengths(mask), expected)
    baseline = elapsed(lambda: legacy_counts(mask), args.repeats)
    numpy_time = elapsed(lambda: binary_run_lengths(mask), args.repeats)
    report = {
        "shape": list(mask.shape),
        "repeats": args.repeats,
        "legacy_ms": baseline * 1000,
        "numpy_ms": numpy_time * 1000,
        "numpy_speedup": baseline / numpy_time,
    }
    if args.numba:
        started = perf_counter()
        actual = binary_run_lengths(mask, backend="numba")
        cold_time = perf_counter() - started
        np.testing.assert_array_equal(actual, expected)
        warm_time = elapsed(lambda: binary_run_lengths(mask, backend="numba"), args.repeats)
        report.update(
            numba_first_call_ms=cold_time * 1000,
            numba_warm_ms=warm_time * 1000,
            numba_warm_speedup=baseline / warm_time,
        )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
