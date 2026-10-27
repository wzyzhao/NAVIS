"""Reference image metrics with deterministic, bounded frame concurrency."""

from __future__ import annotations

import math
from functools import partial
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any, TypedDict

from common.parallel import ordered_map
from common.paths import PROJECT_ROOT
from rebuild_video.utils import load_json, schedule_path_from_config

if TYPE_CHECKING:
    from numpy.typing import NDArray

    ImageArray = NDArray[Any]


class FrameQuality(TypedDict):
    frame_idx: int
    psnr_db: float
    mae: float
    ssim: float


def infer_reference_root(config: dict[str, Any]) -> Path | None:
    """Infer the original frame directory from the first scheduled sequence."""
    schedule_path = schedule_path_from_config(config)
    if not schedule_path or not schedule_path.is_file():
        return None
    try:
        schedule = load_json(schedule_path)
        frames = schedule.get("frames", schedule) if isinstance(schedule, dict) else schedule
        sequence = str(frames[0].get("sequence")) if frames else ""
    except (OSError, ValueError, TypeError, KeyError, IndexError, AttributeError):
        return None
    if not sequence:
        return None
    candidate = PROJECT_ROOT / "src" / "scene_factorization" / "scene_factorization_output" / sequence
    return candidate if candidate.is_dir() else None


def _compare_frame(frame_path: Path, *, reference_root: Path) -> FrameQuality | None:
    """Compare one readable BGR pair, preserving the historical metric definitions."""
    import cv2
    import numpy as np

    from common.images import read_image

    if not frame_path.stem.isdigit():
        return None
    frame_idx = int(frame_path.stem)
    reference_path = reference_root / f"{frame_idx:06d}" / "original.png"
    if not reference_path.is_file():
        reference_path = reference_root / f"{frame_idx:06d}.png"
    if not reference_path.is_file():
        return None
    reference = read_image(reference_path)
    reconstructed = read_image(frame_path)
    if reference is None or reconstructed is None:
        return None
    if reference.shape != reconstructed.shape:
        reconstructed = cv2.resize(reconstructed, (reference.shape[1], reference.shape[0]))
    diff = reference.astype(np.float32) - reconstructed.astype(np.float32)
    mse = float(np.mean(diff * diff))
    return FrameQuality(
        frame_idx=frame_idx,
        psnr_db=99.0 if mse == 0 else 10.0 * math.log10(255.0**2 / mse),
        mae=float(np.mean(np.abs(diff))),
        ssim=_global_ssim(reference, reconstructed, np),
    )


def compute_reference_quality(root: Path, reference_root: Path | None, *, workers: int = 1) -> dict[str, Any]:
    """Aggregate PSNR, MAE and global channel SSIM in sorted frame order.

    Independent image reads and NumPy/OpenCV operations can use worker threads.
    Submission is bounded; images are released by each task. Missing/unreadable
    pairs are skipped, matching the serial implementation. Identical images use
    the historical PSNR sentinel 99 dB; SSIM is global, not windowed MS-SSIM.
    """
    if workers < 1:
        raise ValueError("workers must be at least 1")
    if reference_root is None or not reference_root.is_dir():
        return {"available": False}
    try:
        import cv2  # noqa: F401
        import numpy  # noqa: F401
    except ImportError as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    results = {}
    compare = partial(_compare_frame, reference_root=reference_root)
    for device_dir in sorted((root / "reconstructed_frames").glob("*")):
        if not device_dir.is_dir():
            continue
        compared = [
            row for row in ordered_map(compare, sorted(device_dir.glob("*.png")), workers=workers) if row is not None
        ]
        if compared:
            psnrs = [row["psnr_db"] for row in compared]
            ssims = [row["ssim"] for row in compared]
            results[device_dir.name] = {
                "compared_frames": len(compared),
                "avg_psnr_db": sum(psnrs) / len(compared),
                "min_psnr_db": min(psnrs),
                "avg_mae": sum(row["mae"] for row in compared) / len(compared),
                "avg_ssim": sum(ssims) / len(compared),
                "min_ssim": min(ssims),
                "worst_frame": min(compared, key=lambda row: row["psnr_db"]),
            }
    return {"available": bool(results), "reference_root": str(reference_root), "devices": results}


def _global_ssim(reference: ImageArray, reconstructed: ImageArray, np: ModuleType) -> float:
    """Compute mean per-channel global SSIM with population moments in float64."""
    reference = reference.astype(np.float64)
    reconstructed = reconstructed.astype(np.float64)
    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2
    scores = []
    for channel in range(reference.shape[2]):
        x = reference[:, :, channel]
        y = reconstructed[:, :, channel]
        mu_x = x.mean()
        mu_y = y.mean()
        var_x = ((x - mu_x) ** 2).mean()
        var_y = ((y - mu_y) ** 2).mean()
        cov_xy = ((x - mu_x) * (y - mu_y)).mean()
        score = ((2.0 * mu_x * mu_y + c1) * (2.0 * cov_xy + c2)) / (
            (mu_x * mu_x + mu_y * mu_y + c1) * (var_x + var_y + c2)
        )
        scores.append(float(score))
    return sum(scores) / len(scores)
