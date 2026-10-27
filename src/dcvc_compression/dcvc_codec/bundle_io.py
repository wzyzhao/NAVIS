import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from common.images import read_image

DEFAULT_ROI_COUNT = 12


@dataclass
class BundleROI:
    roi_idx: int
    bbox: list[int]
    roi_bbox: list[int]
    track_id: int
    class_id: int
    category: str
    motion_score: float
    uncertainty: float
    confidence: float | None
    continuity_score: float | None
    background_motion_score: float | None
    mask_pixels: int | None
    mask_coverage: float | None
    edge_mask_coverage: float | None
    masked_black_ratio: float | None
    image: np.ndarray
    mask: np.ndarray


@dataclass
class FrameBundle:
    sequence: str
    frame_id: int
    motion_score: float
    background_change_score: float
    uncertainty: float
    original_frame: np.ndarray
    background: np.ndarray
    foreground: np.ndarray
    mask: np.ndarray
    selected_rois: list[BundleROI]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def imread_required(path: Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    """Read an image or raise with its path when OpenCV cannot decode it."""
    image = read_image(path, flags)
    if image is None:
        raise FileNotFoundError(path)
    return image


def build_roi_from_metadata(
    item: dict,
    original_frame: np.ndarray,
    background_frame: np.ndarray,
    mask: np.ndarray,
    rois_dir: Path | None = None,
    rois_index: dict[int, dict] | None = None,
) -> BundleROI | None:
    x1, y1, x2, y2 = [int(v) for v in item["roi_bbox"]]
    if x2 <= x1 or y2 <= y1:
        return None

    track_id = int(item["track_id"])
    if rois_dir is not None and rois_index is not None and track_id in rois_index:
        roi_item = rois_index[track_id]
        image = imread_required(rois_dir / roi_item["image"])
        roi_mask = imread_required(rois_dir / roi_item["mask"], cv2.IMREAD_GRAYSCALE)
        if roi_mask.ndim == 3 and roi_mask.shape[2] == 1:
            roi_mask = roi_mask[:, :, 0]
    else:
        image = original_frame[y1:y2, x1:x2].copy()
        roi_mask = np.full((y2 - y1, x2 - x1), 255, dtype=np.uint8)

    if image.size == 0 or roi_mask.size == 0:
        return None

    mask_coverage = _optional_float(item.get("mask_coverage"))
    edge_mask_coverage = _optional_float(item.get("edge_mask_coverage"))
    motion_score = _required_float(item, "motion_score")
    uncertainty = _required_float(item, "uncertainty")
    confidence = _optional_float(item.get("confidence"))
    continuity_score = _optional_float(item.get("continuity_score"))
    background_motion_score = compute_background_motion_score(
        original_frame=original_frame,
        background_frame=background_frame,
        roi_bbox=[x1, y1, x2, y2],
        roi_mask=roi_mask,
    )
    return BundleROI(
        roi_idx=int(item.get("roi_idx", item.get("index", 0)) or 0),
        bbox=[int(v) for v in item["bbox"]],
        roi_bbox=[x1, y1, x2, y2],
        track_id=track_id,
        class_id=int(item["class_id"]),
        category=str(item.get("category", "other")),
        motion_score=max(0.0, min(1.0, float(motion_score))),
        uncertainty=max(0.0, min(1.0, float(uncertainty))),
        confidence=confidence,
        continuity_score=continuity_score,
        background_motion_score=background_motion_score,
        mask_pixels=_optional_int(item.get("mask_pixels")),
        mask_coverage=mask_coverage,
        edge_mask_coverage=edge_mask_coverage,
        masked_black_ratio=_optional_float(item.get("masked_black_ratio")),
        image=image,
        mask=roi_mask,
    )


def _optional_int(value) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


def _optional_float(value) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _required_float(mapping: dict, key: str) -> float:
    if mapping.get(key) in (None, ""):
        raise ValueError(f"scene metadata is missing required field {key!r}")
    return float(mapping[key])


def compute_background_motion_score(
    original_frame: np.ndarray,
    background_frame: np.ndarray,
    roi_bbox: list[int],
    roi_mask: np.ndarray | None = None,
    diff_threshold: int = 24,
) -> float:
    """Blend changed-pixel ratio and mean BGR-to-gray difference within an ROI."""
    x1, y1, x2, y2 = roi_bbox
    original_patch = original_frame[y1:y2, x1:x2]
    background_patch = background_frame[y1:y2, x1:x2]
    if original_patch.size == 0 or background_patch.size == 0:
        return 0.0
    diff = cv2.absdiff(original_patch, background_patch)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    if roi_mask is not None and roi_mask.shape[:2] == diff_gray.shape[:2]:
        pixels = diff_gray[roi_mask > 0]
    else:
        pixels = diff_gray.reshape(-1)
    if pixels.size == 0:
        return 0.0
    changed_ratio = float(np.mean(pixels >= int(diff_threshold)))
    mean_delta = float(np.mean(pixels) / 255.0)
    return max(0.0, min(1.0, 0.75 * changed_ratio + 0.25 * mean_delta))


def load_bundle_frame(frame_dir: Path) -> FrameBundle:
    """Load one scene bundle and its ROI images in metadata order."""
    meta = json.loads((frame_dir / "metadata.json").read_text(encoding="utf-8"))
    original_frame = imread_required(frame_dir / "original.png")
    background = imread_required(frame_dir / "background.png")
    source_mask = imread_required(frame_dir / "mask.png", cv2.IMREAD_GRAYSCALE)
    if source_mask.ndim == 3 and source_mask.shape[2] == 1:
        source_mask = source_mask[:, :, 0]

    selected_rois = []
    rois_dir = frame_dir / "rois"
    exported_rois = list(meta.get("selected_rois", []))
    roi_items = exported_rois[:DEFAULT_ROI_COUNT]
    rois_index = {int(item["track_id"]): item for item in exported_rois} if rois_dir.exists() else None
    for roi_idx, item in enumerate(roi_items):
        item = {**item, "roi_idx": int(item.get("roi_idx", roi_idx) or roi_idx)}
        roi = build_roi_from_metadata(
            item,
            original_frame,
            background,
            source_mask,
            rois_dir=rois_dir if rois_dir.exists() else None,
            rois_index=rois_index,
        )
        if roi is not None:
            selected_rois.append(roi)

    motion_score = _required_float(meta, "motion_score")
    background_change_score = _required_float(meta, "background_change_score")
    uncertainty = _required_float(meta, "uncertainty")

    return FrameBundle(
        sequence=str(meta["sequence"]),
        frame_id=int(meta["frame_id"]),
        motion_score=max(0.0, min(1.0, float(motion_score))),
        background_change_score=max(0.0, min(1.0, float(background_change_score))),
        uncertainty=max(0.0, min(1.0, float(uncertainty))),
        original_frame=original_frame,
        background=background,
        foreground=np.empty((0, 0, 3), dtype=original_frame.dtype),
        mask=source_mask,
        selected_rois=selected_rois,
    )


def iter_bundle_frames(bundle_root: Path) -> Iterator[FrameBundle]:
    """Yield source frames in sorted sequence/frame order for temporal codecs."""
    for sequence_dir in sorted(path for path in bundle_root.iterdir() if path.is_dir()):
        for frame_dir in sorted(path for path in sequence_dir.iterdir() if path.is_dir()):
            if not (frame_dir / "metadata.json").exists():
                continue
            yield load_bundle_frame(frame_dir)
