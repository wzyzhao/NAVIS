import cv2
import numpy as np

from .config import (
    BACKGROUND_CONFIDENCE_GROWTH,
    BACKGROUND_DIFF_THRESHOLD,
    BACKGROUND_FAST_LEARNING_RATE,
    BACKGROUND_LEARNING_RATE,
    BACKGROUND_MIN_COMPONENT_AREA,
    BACKGROUND_PERSISTENCE_ALPHA,
    BACKGROUND_PERSISTENCE_THRESHOLD,
)
from .types import BoundingBox, Image


class BackgroundModel:
    """Maintain a conservative static background and temporal motion mask."""

    def __init__(
        self,
        diff_threshold: int = BACKGROUND_DIFF_THRESHOLD,
        learning_rate: float = BACKGROUND_LEARNING_RATE,
        fast_learning_rate: float = BACKGROUND_FAST_LEARNING_RATE,
        confidence_growth: float = BACKGROUND_CONFIDENCE_GROWTH,
        min_component_area: int = BACKGROUND_MIN_COMPONENT_AREA,
        persistence_alpha: float = BACKGROUND_PERSISTENCE_ALPHA,
        persistence_threshold: float = BACKGROUND_PERSISTENCE_THRESHOLD,
        protected_mask_dilate_iterations: int = 2,
    ) -> None:
        self.diff_threshold = diff_threshold
        self.learning_rate = learning_rate
        self.fast_learning_rate = fast_learning_rate
        self.confidence_growth = confidence_growth
        self.min_component_area = min_component_area
        self.persistence_alpha = persistence_alpha
        self.persistence_threshold = persistence_threshold
        self.protected_mask_dilate_iterations = protected_mask_dilate_iterations
        self.reset()

    def reset(self, initial_background: Image | None = None) -> None:
        """Initialize or clear the background cache for a sequence."""
        if initial_background is None:
            self.background = None
            self.confidence = None
            self.persistence = None
            return

        self.background = initial_background.astype(np.float32)
        self.confidence = np.full(initial_background.shape[:2], 0.75, dtype=np.float32)
        self.persistence = np.zeros(initial_background.shape[:2], dtype=np.float32)

    def apply(
        self,
        frame: Image,
        protected_bboxes: list[BoundingBox] | None = None,
        protected_masks: list[Image] | None = None,
    ) -> tuple[Image, Image]:
        """Update unprotected pixels and return background and motion mask."""
        protected_mask = self._build_protected_mask(
            frame.shape[:2],
            bboxes=protected_bboxes,
            instance_masks=protected_masks,
        )
        protected = (
            protected_mask > 0
            if protected_mask is not None
            else np.zeros(frame.shape[:2], dtype=bool)
        )

        if self.background is None:
            initial = self._initial_background(frame, protected_mask)
            self.background = initial.astype(np.float32)
            self.confidence = (~protected).astype(np.float32)
            self.persistence = np.zeros(frame.shape[:2], dtype=np.float32)
            return initial, np.zeros(frame.shape[:2], dtype=np.uint8)

        diff_gray = self._background_difference(frame)
        raw_motion = np.where(diff_gray >= self.diff_threshold, 255, 0).astype(np.uint8)
        motion_mask = self._apply_persistence(self._refine_mask(raw_motion))

        self._update_background(frame, diff_gray, protected)
        background = self._render_background(frame, protected)
        return background, motion_mask

    def _initial_background(self, frame: Image, protected_mask: Image | None) -> Image:
        if protected_mask is None:
            return frame.copy()
        return cv2.inpaint(frame, protected_mask, 3, cv2.INPAINT_TELEA)

    def _update_background(self, frame: Image, diff_gray: Image, protected: np.ndarray) -> None:
        frame_float = frame.astype(np.float32)
        low_confidence = self.confidence < 0.35
        stable = diff_gray < self.diff_threshold
        update_pixels = (~protected) & (stable | low_confidence)
        if not np.any(update_pixels):
            return

        fast_pixels = update_pixels & low_confidence
        normal_pixels = update_pixels & (~fast_pixels)

        if np.any(normal_pixels):
            self.background[normal_pixels] = (1.0 - self.learning_rate) * self.background[
                normal_pixels
            ] + self.learning_rate * frame_float[normal_pixels]
        if np.any(fast_pixels):
            self.background[fast_pixels] = (1.0 - self.fast_learning_rate) * self.background[
                fast_pixels
            ] + self.fast_learning_rate * frame_float[fast_pixels]

        self.confidence[update_pixels] = np.clip(
            self.confidence[update_pixels] + self.confidence_growth,
            0.0,
            1.0,
        )

    def _render_background(self, frame: Image, protected: np.ndarray) -> Image:
        background = np.clip(self.background, 0, 255).astype(np.uint8)
        low_confidence = self.confidence < 0.2
        holes = (low_confidence & protected).astype(np.uint8) * 255
        if np.count_nonzero(holes) == 0:
            return background
        return cv2.inpaint(background, holes, 3, cv2.INPAINT_TELEA)

    def _background_difference(self, frame: Image) -> Image:
        background = np.clip(self.background, 0, 255).astype(np.uint8)
        diff = cv2.absdiff(frame, background)
        return cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    def _build_protected_mask(
        self,
        size: tuple[int, int],
        bboxes: list[BoundingBox] | None = None,
        instance_masks: list[Image] | None = None,
        padding_ratio: float = 0.08,
    ) -> Image | None:
        if not bboxes and not instance_masks:
            return None

        height, width = size
        mask = np.zeros((height, width), dtype=np.uint8)

        if instance_masks:
            for instance_mask in instance_masks:
                if instance_mask is None:
                    continue
                normalized = instance_mask
                if normalized.shape[:2] != (height, width):
                    normalized = cv2.resize(
                        normalized, (width, height), interpolation=cv2.INTER_NEAREST
                    )
                mask[normalized > 0] = 255

        if bboxes:
            for bbox in bboxes:
                x1, y1, x2, y2 = [int(v) for v in bbox]
                box_w = max(x2 - x1, 1)
                box_h = max(y2 - y1, 1)
                pad_x = max(int(round(box_w * padding_ratio)), 2)
                pad_y = max(int(round(box_h * padding_ratio)), 2)

                x1 = max(x1 - pad_x, 0)
                y1 = max(y1 - pad_y, 0)
                x2 = min(x2 + pad_x, width)
                y2 = min(y2 + pad_y, height)
                if x2 > x1 and y2 > y1:
                    mask[y1:y2, x1:x2] = 255

        if np.count_nonzero(mask) == 0:
            return None

        if self.protected_mask_dilate_iterations > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.dilate(mask, kernel, iterations=self.protected_mask_dilate_iterations)

        return mask

    def _refine_mask(self, raw_mask: Image) -> Image:
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        mask = cv2.medianBlur(raw_mask, 5)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_large, iterations=2)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        keep = np.zeros(num_labels, dtype=bool)
        for label_id in range(1, num_labels):
            x, y, w, h, area = stats[label_id]
            extent = area / max(w * h, 1)
            if area < self.min_component_area:
                continue
            if area < self.min_component_area * 3 and extent < 0.12:
                continue
            keep[label_id] = True

        clean = keep[labels].astype(np.uint8) * 255

        clean = cv2.dilate(clean, kernel_small, iterations=1)
        return clean

    def _apply_persistence(self, mask: Image) -> Image:
        binary = (mask > 0).astype(np.float32)
        self.persistence = (
            1.0 - self.persistence_alpha
        ) * self.persistence + self.persistence_alpha * binary

        persistent = self.persistence >= self.persistence_threshold
        large_components = self._keep_large_components(mask, self.min_component_area * 4)
        return np.where(persistent | (large_components > 0), 255, 0).astype(np.uint8)

    @staticmethod
    def _keep_large_components(mask: Image, min_area: int) -> Image:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        keep = np.zeros(num_labels, dtype=bool)
        for label_id in range(1, num_labels):
            area = stats[label_id, cv2.CC_STAT_AREA]
            if area >= min_area:
                keep[label_id] = True
        return keep[labels].astype(np.uint8) * 255
