import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from common.images import read_image

DEFAULT_ROI_COUNT = 12


@dataclass
class BundleROI:
    bbox: list[int]
    roi_bbox: list[int]
    track_id: int
    class_id: int
    priority_score: float
    image: np.ndarray
    mask: np.ndarray


@dataclass
class FrameBundle:
    sequence: str
    frame_id: int
    original_frame: np.ndarray
    background: np.ndarray
    mask: np.ndarray
    metadata: list[dict]
    selected_rois: list[BundleROI]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def imread_required(path: Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    """Read a required BGR image or mask and report undecodable paths."""
    image = read_image(path, flags)
    if image is None:
        raise FileNotFoundError(path)
    return image


def build_roi_from_metadata(
    item: dict,
    original_frame: np.ndarray,
    mask: np.ndarray,
    rois_dir: Path | None = None,
    rois_index: dict[int, dict] | None = None,
) -> BundleROI | None:
    x1, y1, x2, y2 = [int(v) for v in item["roi_bbox"]]
    if x2 <= x1 or y2 <= y1:
        return None

    image = None
    roi_mask = None
    track_id = int(item["track_id"])
    if rois_dir is not None and rois_index is not None and track_id in rois_index:
        roi_item = rois_index[track_id]
        image = imread_required(rois_dir / roi_item["image"])
        roi_mask = imread_required(rois_dir / roi_item["mask"], cv2.IMREAD_GRAYSCALE)
    else:
        image = original_frame[y1:y2, x1:x2].copy()
        roi_mask = np.full((y2 - y1, x2 - x1), 255, dtype=np.uint8)

    if image.size == 0 or roi_mask.size == 0:
        return None

    return BundleROI(
        bbox=[int(v) for v in item["bbox"]],
        roi_bbox=[x1, y1, x2, y2],
        track_id=track_id,
        class_id=int(item.get("class_id", item.get("class", -1))),
        priority_score=float(item.get("priority_score", 0.0)),
        image=image,
        mask=roi_mask,
    )


def load_bundle_frame(frame_dir: Path, roi_count: int = DEFAULT_ROI_COUNT) -> FrameBundle:
    """Load one baseline frame, limiting selected ROI records to roi_count."""
    meta = json.loads((frame_dir / "metadata.json").read_text(encoding="utf-8"))
    original_frame = imread_required(frame_dir / "original.png")
    background = imread_required(frame_dir / "background.png")
    mask = imread_required(frame_dir / "mask.png", cv2.IMREAD_GRAYSCALE)

    selected_rois = []
    rois_dir = frame_dir / "rois"
    exported_rois = list(meta.get("selected_rois", []))
    roi_limit = max(0, int(roi_count or 0))
    roi_items = list(meta.get("selected_metadata", []))[:roi_limit]
    if not roi_items:
        roi_items = exported_rois[:roi_limit]
    rois_index = {int(item["track_id"]): item for item in exported_rois} if rois_dir.exists() else None
    for item in roi_items:
        roi = build_roi_from_metadata(
            item, original_frame, mask, rois_dir=rois_dir if rois_dir.exists() else None, rois_index=rois_index
        )
        if roi is not None:
            selected_rois.append(roi)

    return FrameBundle(
        sequence=str(meta["sequence"]),
        frame_id=int(meta["frame_id"]),
        original_frame=original_frame,
        background=background,
        mask=mask,
        metadata=list(meta.get("metadata", meta.get("selected_metadata", []))),
        selected_rois=selected_rois,
    )


def iter_bundle_frames(
    bundle_root: Path,
    sequence_filter: str | Sequence[str] | None = None,
    roi_count: int = DEFAULT_ROI_COUNT,
) -> Iterator[FrameBundle]:
    """Yield selected sequences and frames in stable filesystem-name order."""
    allowed_sequences = _normalize_sequence_filter(sequence_filter)
    for sequence_dir in sorted(path for path in bundle_root.iterdir() if path.is_dir()):
        if allowed_sequences and sequence_dir.name not in allowed_sequences:
            continue
        for frame_dir in sorted(path for path in sequence_dir.iterdir() if path.is_dir()):
            if (frame_dir / "metadata.json").exists():
                yield load_bundle_frame(frame_dir, roi_count=roi_count)


def _normalize_sequence_filter(sequence_filter: str | Sequence[str] | None) -> set[str]:
    if sequence_filter in (None, "", []):
        return set()
    if isinstance(sequence_filter, str):
        return {value.strip() for value in sequence_filter.split(",") if value.strip()}
    return {str(value).strip() for value in sequence_filter if str(value).strip()}
