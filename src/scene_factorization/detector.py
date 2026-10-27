from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO

from .config import (
    DETECTOR_ALLOWED_CLASSES,
    DETECTOR_CLASS_CONF,
    DETECTOR_CONF,
    DETECTOR_DEVICE,
    DETECTOR_MIN_BOX_AREA,
    DETECTOR_MODEL_PATH,
    ROI_PERSON_ATTACHED_CLASSES,
)
from .types import BoundingBox, Image, Record


class Detector:
    """Run YOLO instance segmentation and normalize detections."""

    def __init__(
        self,
        model_path: str = DETECTOR_MODEL_PATH,
        device: str = DETECTOR_DEVICE,
        conf: float = DETECTOR_CONF,
        class_conf: dict[int, float] | None = DETECTOR_CLASS_CONF,
        allowed_classes: set[int] | None = DETECTOR_ALLOWED_CLASSES,
        min_box_area: int = DETECTOR_MIN_BOX_AREA,
    ) -> None:
        self.model = YOLO(model_path)
        self.device = device
        self.conf = conf
        self.class_conf = dict(class_conf or {})
        self.allowed_classes = set(allowed_classes) if allowed_classes else None
        self.min_box_area = min_box_area

    def detect(self, frame: Image) -> list[Record]:
        """Return filtered detections with full frame masks where available."""
        inference_conf = (
            min([self.conf, *self.class_conf.values()]) if self.class_conf else self.conf
        )
        results = self.model(frame, verbose=False, conf=inference_conf, device=self.device)[0]
        masks = self._extract_masks(results, frame.shape[:2])

        detections = []
        for index, box in enumerate(results.boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            area = max(0, x2 - x1) * max(0, y2 - y1)

            if self.allowed_classes is not None and cls not in self.allowed_classes:
                continue
            if conf < self.class_conf.get(cls, self.conf):
                continue
            if area < self.min_box_area:
                continue

            detections.append(
                {
                    "bbox": [x1, y1, x2, y2],
                    "score": conf,
                    "class": cls,
                    "instance_mask": masks[index]
                    if masks is not None and index < len(masks)
                    else None,
                }
            )

        return self._suppress_attached_duplicates(detections)

    @staticmethod
    def _extract_masks(results: Any, frame_size: tuple[int, int]) -> list[Image] | None:
        if results.masks is None:
            return None

        height, width = frame_size
        masks = []

        if results.masks.xy:
            for polygon in results.masks.xy:
                mask = np.zeros((height, width), dtype=np.uint8)
                points = np.round(polygon).astype(np.int32)
                if points.size:
                    points[:, 0] = np.clip(points[:, 0], 0, width - 1)
                    points[:, 1] = np.clip(points[:, 1], 0, height - 1)
                    cv2.fillPoly(mask, [points], 255)
                masks.append(mask)
            return masks

        if results.masks.data is None:
            return None

        for mask_tensor in results.masks.data:
            mask = mask_tensor.detach().cpu().numpy()
            if mask.shape[:2] != (height, width):
                mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
            masks.append((mask >= 0.5).astype(np.uint8) * 255)
        return masks

    def _suppress_attached_duplicates(self, detections: list[Record]) -> list[Record]:
        kept = []
        attached = sorted(
            [det for det in detections if det["class"] in ROI_PERSON_ATTACHED_CLASSES],
            key=lambda det: det["score"],
            reverse=True,
        )
        others = [det for det in detections if det["class"] not in ROI_PERSON_ATTACHED_CLASSES]

        for detection in attached:
            duplicate = False
            for other in kept:
                iou = self._bbox_iou(detection["bbox"], other["bbox"])
                containment = self._bbox_intersection_over_smaller(detection["bbox"], other["bbox"])
                if iou >= 0.55 or containment >= 0.85:
                    duplicate = True
                    break
            if not duplicate:
                kept.append(detection)

        return others + kept

    @staticmethod
    def _bbox_iou(a: BoundingBox, b: BoundingBox) -> float:
        x_a = max(a[0], b[0])
        y_a = max(a[1], b[1])
        x_b = min(a[2], b[2])
        y_b = min(a[3], b[3])
        inter = max(0, x_b - x_a) * max(0, y_b - y_a)
        if inter <= 0:
            return 0.0

        area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
        area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
        return inter / max(area_a + area_b - inter, 1e-6)

    @staticmethod
    def _bbox_intersection_over_smaller(a: BoundingBox, b: BoundingBox) -> float:
        x_a = max(a[0], b[0])
        y_a = max(a[1], b[1])
        x_b = min(a[2], b[2])
        y_b = min(a[3], b[3])
        inter = max(0, x_b - x_a) * max(0, y_b - y_a)
        area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
        area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
        return inter / max(min(area_a, area_b), 1e-6)
