import json
import shutil
from pathlib import Path

import cv2
import numpy as np

from .config import (
    EXPORT_ROI_MASK_DILATE_ITERATIONS,
)
from .types import BoundingBox, Image, Record


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def clip_bbox(bbox: BoundingBox, width: int, height: int) -> BoundingBox:
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1 = max(0, min(x1, width))
    y1 = max(0, min(y1, height))
    x2 = max(0, min(x2, width))
    y2 = max(0, min(y2, height))
    return [x1, y1, x2, y2]


def refine_roi_mask(mask: Image) -> Image:
    refined = mask.copy()
    if EXPORT_ROI_MASK_DILATE_ITERATIONS > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        refined = cv2.morphologyEx(refined, cv2.MORPH_CLOSE, kernel, iterations=1)
        refined = cv2.dilate(refined, kernel, iterations=EXPORT_ROI_MASK_DILATE_ITERATIONS)
    return refined


def fill_mask_holes(mask: Image) -> Image:
    if mask.size == 0:
        return mask
    flood = mask.copy()
    padded = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
    cv2.floodFill(flood, padded, (0, 0), 255)
    holes = cv2.bitwise_not(flood)
    return cv2.bitwise_or(mask, holes)


def keep_largest_components(mask: Image, max_components: int = 2) -> Image:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask

    component_ids = np.argsort(stats[1:, cv2.CC_STAT_AREA])[-max_components:] + 1
    keep = np.zeros(num_labels, dtype=bool)
    keep[component_ids] = True
    return keep[labels].astype(np.uint8) * 255


def build_instance_roi_mask(instance_mask: Image | None, roi_bbox: BoundingBox) -> Image | None:
    x1, y1, x2, y2 = roi_bbox
    if instance_mask is None:
        return None

    normalized = instance_mask
    if normalized.ndim == 3 and normalized.shape[2] == 1:
        normalized = normalized[:, :, 0]
    patch_mask = normalized[y1:y2, x1:x2].copy()
    if patch_mask.size == 0:
        return None

    _, enhanced = cv2.threshold(patch_mask, 127, 255, cv2.THRESH_BINARY)
    enhanced = refine_roi_mask(enhanced)
    enhanced = fill_mask_holes(enhanced)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    enhanced = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel, iterations=1)
    enhanced = keep_largest_components(enhanced, max_components=2)
    return enhanced


def build_roi_patch(
    frame: Image,
    roi_bbox: BoundingBox,
    instance_mask: Image | None = None,
) -> tuple[Image | None, Image | None, Image | None]:
    x1, y1, x2, y2 = roi_bbox
    patch = frame[y1:y2, x1:x2].copy()
    if patch.size == 0:
        return None, None, None

    patch_mask = build_instance_roi_mask(instance_mask, roi_bbox)
    if patch_mask is None or patch_mask.size == 0 or np.count_nonzero(patch_mask) < 20:
        return None, None, None

    masked_patch = np.zeros_like(patch)
    masked_patch[patch_mask > 0] = patch[patch_mask > 0]
    return patch, masked_patch, patch_mask


def roi_quality_metrics(
    raw_patch: Image | None, masked_patch: Image | None, patch_mask: Image | None
) -> Record:
    if raw_patch is None or masked_patch is None or patch_mask is None or patch_mask.size == 0:
        return {}
    binary = patch_mask > 0
    mask_pixels = int(np.count_nonzero(binary))
    total_pixels = int(patch_mask.size)
    if mask_pixels <= 0 or total_pixels <= 0:
        return {
            "mask_pixels": mask_pixels,
            "mask_coverage": 0.0,
            "edge_mask_coverage": 0.0,
            "masked_black_ratio": 1.0,
        }
    edge = np.zeros_like(binary, dtype=bool)
    edge[0, :] = True
    edge[-1, :] = True
    edge[:, 0] = True
    edge[:, -1] = True
    edge_mask_pixels = int(np.count_nonzero(binary & edge))
    masked_pixels = masked_patch[binary]
    dark = (masked_pixels[:, 0] < 8) & (masked_pixels[:, 1] < 8) & (masked_pixels[:, 2] < 8)
    return {
        "mask_pixels": mask_pixels,
        "mask_coverage": float(mask_pixels / total_pixels),
        "edge_mask_coverage": float(edge_mask_pixels / max(1, int(np.count_nonzero(edge)))),
        "masked_black_ratio": float(np.count_nonzero(dark) / max(1, mask_pixels)),
    }


def export_factorized_frame(
    output_root: str | Path,
    sample: Record,
    frame: Image,
    sf_out: Record,
    max_foregrounds: int = 10,
) -> Path:
    """Write reconstruction images, ROI patches, and compact frame metadata."""
    frame_dir = ensure_dir(Path(output_root) / sample["sequence"] / f"{sample['frame_id']:06d}")
    rois_dir = frame_dir / "rois"
    if rois_dir.exists():
        shutil.rmtree(rois_dir)
    ensure_dir(rois_dir)
    height, width = frame.shape[:2]

    raw_metadata = sf_out["metadata"]
    selected_raw_all = [item for item in raw_metadata if item.get("selected", False)]
    selected_raw = selected_raw_all[:max_foregrounds]
    selected_rois = []

    for index, item in enumerate(selected_raw):
        roi_bbox = clip_bbox(item["roi_bbox"], width, height)
        raw_patch, patch, patch_mask = build_roi_patch(
            frame,
            roi_bbox,
            instance_mask=item.get("instance_mask"),
        )
        if patch is None or patch_mask is None:
            continue
        quality = roi_quality_metrics(raw_patch, patch, patch_mask)

        image_name = f"roi_{index:02d}_track_{int(item['track_id']):04d}.png"
        mask_name = f"roi_{index:02d}_track_{int(item['track_id']):04d}_mask.png"
        cv2.imwrite(str(rois_dir / image_name), patch)
        cv2.imwrite(str(rois_dir / mask_name), patch_mask)

        roi_record = {
            "bbox": [int(v) for v in item["bbox"]],
            "roi_bbox": roi_bbox,
            "track_id": int(item["track_id"]),
            "class_id": int(item["class"]),
            "category": item.get("category", "other"),
            "motion_score": float(item.get("motion_score", 0.0)),
            "uncertainty": float(item.get("uncertainty", 1.0 - item.get("confidence", 0.0))),
            "confidence": float(item.get("confidence", 0.0)),
            "continuity_score": float(item.get("continuity_score", 0.0)),
            "mask_pixels": quality.get("mask_pixels"),
            "mask_coverage": quality.get("mask_coverage"),
            "edge_mask_coverage": quality.get("edge_mask_coverage"),
            "masked_black_ratio": quality.get("masked_black_ratio"),
            "image": image_name,
            "mask": mask_name,
        }
        selected_rois.append(roi_record)

    cv2.imwrite(str(frame_dir / "original.png"), frame)
    cv2.imwrite(str(frame_dir / "background.png"), sf_out["background"])
    cv2.imwrite(str(frame_dir / "mask.png"), sf_out["mask"])

    frame_metadata = {
        "sequence": sample["sequence"],
        "frame_id": int(sample["frame_id"]),
        "shape": list(frame.shape),
        "motion_score": frame_motion_score(sf_out, selected_rois),
        "background_change_score": float(sf_out.get("background_change_score", 0.0)),
        "uncertainty": frame_uncertainty(selected_rois),
        "selected_count": len(selected_rois),
        "max_foregrounds": int(max_foregrounds),
        "selected_rois": selected_rois,
    }
    with open(frame_dir / "metadata.json", "w", encoding="utf-8") as file:
        json.dump(frame_metadata, file, ensure_ascii=True, indent=2)

    return frame_dir


def frame_motion_score(sf_out: Record, selected_rois: list[Record]) -> float:
    motion_mask = sf_out.get("motion_mask")
    if motion_mask is not None and motion_mask.size > 0:
        return float(np.mean(motion_mask > 0))
    if not selected_rois:
        return 0.0
    return float(max(float(item.get("motion_score", 0.0) or 0.0) for item in selected_rois))


def frame_uncertainty(selected_rois: list[Record]) -> float:
    if not selected_rois:
        return 0.0
    return float(np.mean([float(item.get("uncertainty", 0.0) or 0.0) for item in selected_rois]))
