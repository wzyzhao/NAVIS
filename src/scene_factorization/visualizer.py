from pathlib import Path

import cv2
import numpy as np

from .config import COCO_CLASSES
from .types import Image, Record

WINDOW_ORIGINAL = "Scene Factorization - Original"
WINDOW_BACKGROUND = "Scene Factorization - Background"
WINDOW_FOREGROUND = "Scene Factorization - Foreground"
WINDOW_MASK = "Scene Factorization - Object Mask"

DEFAULT_PREVIEW_FPS = 30.0


def build_foreground(frame: Image, mask: Image) -> Image:
    foreground = np.zeros_like(frame)
    foreground[mask > 0] = frame[mask > 0]
    return foreground


def colorize_mask(mask: Image) -> Image:
    if mask.ndim == 2:
        normalized = mask
        if mask.dtype != np.uint8:
            normalized = cv2.normalize(mask, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        if np.max(normalized) <= 1:
            normalized = normalized * 255
        return cv2.cvtColor(normalized, cv2.COLOR_GRAY2BGR)
    return mask.copy()


def _class_name(class_id: int) -> str:
    if 0 <= class_id < len(COCO_CLASSES):
        return COCO_CLASSES[class_id]
    return f"class_{class_id}"


def _add_status_bar(image: Image, text: str) -> Image:
    panel = image.copy()
    cv2.rectangle(panel, (0, 0), (panel.shape[1], 30), (20, 20, 20), -1)
    cv2.putText(
        panel,
        text,
        (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (240, 240, 240),
        1,
        cv2.LINE_AA,
    )
    return panel


def annotate_frame(frame: Image, metadata: list[Record]) -> Image:
    annotated = frame.copy()
    for item in metadata:
        x1, y1, x2, y2 = item["bbox"]
        rx1, ry1, rx2, ry2 = item["roi_bbox"]
        color = (0, 0, 255) if item.get("selected", False) else (0, 255, 0)

        cv2.rectangle(annotated, (rx1, ry1), (rx2, ry2), color, 1)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        class_name = _class_name(item["class"])
        label = f"{class_name} id={item['track_id']} m={item['motion_score']:.2f}"
        cv2.putText(
            annotated,
            label,
            (x1, max(y1 - 8, 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )

    return annotated


def build_preview_views(
    frame: Image, sf_out: Record, sample: Record | None = None
) -> dict[str, Image]:
    annotated = annotate_frame(frame, sf_out["metadata"])
    foreground = build_foreground(frame, sf_out["mask"])
    background = sf_out["background"]
    mask = colorize_mask(sf_out["mask"])

    status_parts = []
    if sample is not None:
        status_parts.append(f"{sample['sequence']} #{sample['frame_id']:06d}")
    selected_count = sum(1 for item in sf_out["metadata"] if item.get("selected", False))
    status_parts.append(f"selected={selected_count}")
    status_parts.append("q/esc quit")
    status_text = " | ".join(status_parts)

    return {
        WINDOW_ORIGINAL: _add_status_bar(annotated, f"Original | {status_text}"),
        WINDOW_BACKGROUND: _add_status_bar(background, f"Background | {status_text}"),
        WINDOW_FOREGROUND: _add_status_bar(foreground, f"Foreground | {status_text}"),
        WINDOW_MASK: _add_status_bar(mask, f"Object Mask | {status_text}"),
    }


def build_preview_grid(frame: Image, sf_out: Record, sample: Record | None = None) -> Image:
    views = build_preview_views(frame, sf_out, sample=sample)
    ordered = [
        views[WINDOW_ORIGINAL],
        views[WINDOW_BACKGROUND],
        views[WINDOW_FOREGROUND],
        views[WINDOW_MASK],
    ]
    height, width = ordered[0].shape[:2]
    tile_size = (width // 2, height // 2)
    tiles = [cv2.resize(image, tile_size, interpolation=cv2.INTER_AREA) for image in ordered]
    top = np.hstack((tiles[0], tiles[1]))
    bottom = np.hstack((tiles[2], tiles[3]))
    return np.vstack((top, bottom))


def show_preview(
    frame: Image, sf_out: Record, sample: Record | None = None, fps: float = DEFAULT_PREVIEW_FPS
) -> bool:
    """Show the four layer views and return whether processing should continue."""
    views = build_preview_views(frame, sf_out, sample=sample)
    for window_name, image in views.items():
        cv2.imshow(window_name, image)
    wait_ms = max(1, int(round(1000.0 / fps))) if fps and fps > 0 else 1
    key = cv2.waitKey(wait_ms) & 0xFF
    return key not in (27, ord("q"), ord("Q"))


def close_preview() -> None:
    for window_name in (WINDOW_ORIGINAL, WINDOW_BACKGROUND, WINDOW_FOREGROUND, WINDOW_MASK):
        try:
            cv2.destroyWindow(window_name)
        except cv2.error:
            pass


class VisualizationVideoWriter:
    """Write a side by side diagnostic video for each sequence."""

    def __init__(self, output_path: str | Path, fps: float = DEFAULT_PREVIEW_FPS) -> None:
        self.output_path = str(output_path)
        self.fps = float(fps) if fps and fps > 0 else DEFAULT_PREVIEW_FPS
        self.writer = None

    def write(self, frame: Image, sf_out: Record, sample: Record | None = None) -> None:
        image = build_preview_grid(frame, sf_out, sample=sample)
        if self.writer is None:
            height, width = image.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.writer = cv2.VideoWriter(self.output_path, fourcc, self.fps, (width, height))
            if not self.writer.isOpened():
                raise RuntimeError(f"Unable to open visualization video: {self.output_path}")
        self.writer.write(image)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.release()
            self.writer = None
