"""Regression tests for wire compatibility and independent-task execution."""

import base64
import importlib.util
import math
from pathlib import Path
from threading import Event

import numpy as np
import pytest

from dcvc_compression.dcvc_codec.offline_schedule_simulator import binary_mask_to_metadata_rle
from rebuild_video.quality import compute_reference_quality
from common.masks import binary_run_lengths, expand_binary_runs
from common.parallel import ordered_map


def write_png(path: Path, image: np.ndarray) -> None:
    """Write fixtures through Python's Unicode-aware filesystem API."""
    import cv2

    success, encoded = cv2.imencode(".png", image)
    assert success
    path.write_bytes(encoded.tobytes())


def legacy_counts(mask: np.ndarray) -> list[int]:
    counts, value, run = [], 0, 0
    for pixel in (mask > 0).reshape(-1):
        if pixel == value:
            run += 1
        else:
            counts.append(run)
            value, run = int(pixel), 1
    return [*counts, run]


@pytest.mark.parametrize("backend", ["numpy", "numba"])
def test_rle_matches_legacy_on_boundaries_and_noncontiguous_masks(backend: str) -> None:
    if backend == "numba" and importlib.util.find_spec("numba") is None:
        pytest.skip("optional Numba dependency is not installed")
    rng = np.random.default_rng(42)
    masks = [
        np.empty((0, 3)),
        np.zeros((4, 6)),
        np.ones((4, 6)),
        np.array([[1, 0, 1, 0]]),
        rng.integers(-2, 3, (31, 37))[:, ::2],
    ]
    for mask in masks:
        assert binary_run_lengths(mask, backend=backend).tolist() == legacy_counts(mask)


def test_rle_wire_bytes_and_initial_zero_run() -> None:
    encoded = binary_mask_to_metadata_rle(np.ones((1, 130), dtype=np.uint8))
    assert encoded == {
        "format": "rle",
        "order": "C",
        "size": [1, 130],
        "encoding": "varint-base64",
        "data": base64.b64encode(b"\x00\x82\x01").decode(),
    }


def test_metadata_mask_reconstruction_preserves_wire_order() -> None:
    from rebuild_video.video_reconstructor import VideoReconstructor

    decoder = VideoReconstructor.__new__(VideoReconstructor)
    mask = np.array([[255, 0, 255], [0, 0, 255]], dtype=np.uint8)
    wire = binary_mask_to_metadata_rle(mask)
    np.testing.assert_array_equal(decoder._decode_compact_rle(wire), mask > 0)


def test_mask_expansion_pads_clips_and_supports_fortran_order() -> None:
    np.testing.assert_array_equal(expand_binary_runs([0, 10**30], 2, 3), np.ones((2, 3)))
    np.testing.assert_array_equal(expand_binary_runs([1, 1], 2, 2), [[0, 1], [0, 0]])
    np.testing.assert_array_equal(expand_binary_runs([1, 1], 2, 2, order="F"), [[0, 0], [1, 0]])


def test_ordered_map_keeps_input_order_and_bounds_submission() -> None:
    released = Event()
    submitted = []

    def items():
        for number in range(20):
            submitted.append(number)
            yield number

    def work(number: int) -> int:
        if number == 0:
            assert released.wait(5)
        else:
            released.set()
        return number

    results = ordered_map(work, items(), workers=2)
    assert next(results) == 0
    assert len(submitted) == 4
    assert list(results) == list(range(1, 20))


def test_worker_error_is_propagated() -> None:
    def fail(number: int) -> int:
        raise RuntimeError(str(number))

    with pytest.raises(RuntimeError, match="0"):
        list(ordered_map(fail, range(3), workers=2))
    with pytest.raises(ValueError):
        list(ordered_map(abs, [], workers=0))


def test_parallel_quality_matches_serial_and_historical_metrics(tmp_path: Path) -> None:
    import cv2

    frames = tmp_path / "reconstructed_frames" / "node1"
    reference = tmp_path / "reference"
    frames.mkdir(parents=True)
    reference.mkdir()
    for index, value in enumerate([0, 10, 255], start=1):
        original = np.full((12, 16, 3), value, dtype=np.uint8)
        write_png(reference / f"{index:06d}.png", original)
        write_png(frames / f"{index:06d}.png", original)
    serial = compute_reference_quality(tmp_path, reference)
    assert compute_reference_quality(tmp_path, reference, workers=3) == serial
    metrics = serial["devices"]["node1"]
    assert metrics["avg_psnr_db"] == 99.0
    assert metrics["avg_mae"] == 0.0
    assert metrics["avg_ssim"] == 1.0
    assert metrics["worst_frame"]["frame_idx"] == 1


def test_quality_resizes_and_skips_missing_pairs(tmp_path: Path) -> None:
    import cv2

    frames = tmp_path / "reconstructed_frames" / "node1"
    reference = tmp_path / "reference"
    frames.mkdir(parents=True)
    (reference / "000001").mkdir(parents=True)
    write_png(reference / "000001" / "original.png", np.zeros((8, 8, 3), dtype=np.uint8))
    write_png(frames / "000001.png", np.full((4, 4, 3), 10, dtype=np.uint8))
    (frames / "000002.png").write_bytes(b"missing reference")
    (frames / "notes.png").write_bytes(b"not a frame")
    metrics = compute_reference_quality(tmp_path, reference, workers=2)["devices"]["node1"]
    assert metrics["compared_frames"] == 1
    assert metrics["avg_psnr_db"] == pytest.approx(10 * math.log10(255**2 / 100))
    assert metrics["avg_mae"] == 10
    assert metrics["avg_ssim"] == pytest.approx(6.5025 / 106.5025)
