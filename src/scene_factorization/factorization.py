import math

import cv2
import numpy as np

from .background import BackgroundModel
from .config import (
    BACKGROUND_DIFF_THRESHOLD,
    MAX_FOREGROUNDS,
    ROI_CONTEXT_SCALE,
    ROI_MIN_EDGE,
    ROI_MIN_MASK_COVERAGE,
    ROI_MIN_MOTION_SCORE,
    ROI_NMS_THRESHOLD,
    ROI_PERSON_ATTACHED_CLASSES,
    ROI_VEHICLE_CLASSES,
    ROI_VEHICLE_MOTION_THRESHOLD,
    ROI_VEHICLE_PIXEL_MOTION_THRESHOLD,
)
from .detector import Detector
from .tracker import Track, Tracker
from .types import BoundingBox, Image, Record


def bbox_iou(a: BoundingBox, b: BoundingBox) -> float:
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


def bbox_intersection_ratio(inner: BoundingBox, outer: BoundingBox) -> float:
    x_a = max(inner[0], outer[0])
    y_a = max(inner[1], outer[1])
    x_b = min(inner[2], outer[2])
    y_b = min(inner[3], outer[3])
    inter = max(0, x_b - x_a) * max(0, y_b - y_a)
    area = max(0, inner[2] - inner[0]) * max(0, inner[3] - inner[1])
    return inter / max(area, 1e-6)


def bbox_center_inside(inner: BoundingBox, outer: BoundingBox) -> bool:
    cx = (inner[0] + inner[2]) / 2.0
    cy = (inner[1] + inner[3]) / 2.0
    return outer[0] <= cx <= outer[2] and outer[1] <= cy <= outer[3]


def bbox_center_distance(a: BoundingBox, b: BoundingBox) -> float:
    ax = (a[0] + a[2]) / 2.0
    ay = (a[1] + a[3]) / 2.0
    bx = (b[0] + b[2]) / 2.0
    by = (b[1] + b[3]) / 2.0
    return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)


class SceneFactorization:
    """Combine detection, tracking, and background estimates into frame layers."""

    def __init__(
        self,
        bg_model: BackgroundModel,
        detector: Detector,
        tracker: Tracker,
        max_foregrounds: int = MAX_FOREGROUNDS,
        roi_context_scale: float = ROI_CONTEXT_SCALE,
        roi_min_edge: int = ROI_MIN_EDGE,
        roi_min_mask_coverage: float = ROI_MIN_MASK_COVERAGE,
        roi_nms_threshold: float = ROI_NMS_THRESHOLD,
        roi_min_motion_score: float = ROI_MIN_MOTION_SCORE,
        roi_vehicle_motion_threshold: float = ROI_VEHICLE_MOTION_THRESHOLD,
        roi_vehicle_pixel_motion_threshold: float = ROI_VEHICLE_PIXEL_MOTION_THRESHOLD,
        roi_pixel_motion_diff_threshold: int = BACKGROUND_DIFF_THRESHOLD,
        roi_person_attached_classes: set[int] | None = None,
        roi_vehicle_classes: set[int] | None = None,
    ) -> None:
        self.bg_model = bg_model
        self.detector = detector
        self.tracker = tracker
        self.max_foregrounds = max_foregrounds
        self.roi_context_scale = roi_context_scale
        self.roi_min_edge = roi_min_edge
        self.roi_min_mask_coverage = roi_min_mask_coverage
        self.roi_nms_threshold = roi_nms_threshold
        self.roi_min_motion_score = roi_min_motion_score
        self.roi_vehicle_motion_threshold = roi_vehicle_motion_threshold
        self.roi_vehicle_pixel_motion_threshold = roi_vehicle_pixel_motion_threshold
        self.roi_pixel_motion_diff_threshold = roi_pixel_motion_diff_threshold
        self.roi_person_attached_classes = set(
            ROI_PERSON_ATTACHED_CLASSES
            if roi_person_attached_classes is None
            else roi_person_attached_classes
        )
        self.roi_vehicle_classes = set(
            ROI_VEHICLE_CLASSES if roi_vehicle_classes is None else roi_vehicle_classes
        )
        self.previous_selected_ids = set()
        self.previous_background = None
        self.previous_frame_gray = None

    def reset(self, initial_background: Image | None = None) -> None:
        """Clear temporal state at a sequence boundary."""
        self.bg_model.reset(initial_background=initial_background)
        self.tracker.reset()
        self.previous_selected_ids = set()
        self.previous_background = None
        self.previous_frame_gray = None

    def process(self, frame: Image) -> Record:
        """Factorize one BGR frame while preserving sequence state."""
        detections = self.detector.detect(frame)
        protected_bboxes = [
            detection["bbox"] for detection in detections if detection.get("instance_mask") is None
        ]
        protected_bboxes.extend(self.tracker.active_bboxes())
        protected_masks = [
            detection.get("instance_mask")
            for detection in detections
            if detection.get("instance_mask") is not None
        ]
        fill_background, motion_mask = self.bg_model.apply(
            frame,
            protected_bboxes=protected_bboxes,
            protected_masks=protected_masks,
        )
        tracks = self.tracker.update(frame, detections)

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        pixel_diff = (
            cv2.absdiff(frame_gray, self.previous_frame_gray)
            if self.previous_frame_gray is not None
            else None
        )
        metadata = self._build_metadata(frame, motion_mask, tracks, pixel_diff)
        self.previous_selected_ids = {item["track_id"] for item in metadata if item["selected"]}
        object_mask = self._build_object_mask(frame.shape[:2], metadata)
        background = self._compose_background(frame, fill_background, object_mask)
        background_change_score = self._background_change_score(background)
        self.previous_background = background.copy()
        self.previous_frame_gray = frame_gray

        return {
            "background": background,
            "background_change_score": background_change_score,
            "metadata": metadata,
            "mask": object_mask,
            "motion_mask": motion_mask,
        }

    def _build_metadata(
        self, frame: Image, mask: Image, tracks: list[Track], pixel_diff: Image | None
    ) -> list[Record]:
        height, width = frame.shape[:2]
        metadata = []

        for track in tracks:
            bbox = self._clip_bbox(track.to_tlbr(), width, height)
            if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                continue

            roi_bbox = self._expand_bbox(bbox, width, height)
            mask_coverage = self._mask_coverage(track.instance_mask, roi_bbox)
            background_motion_score = self._motion_score(mask, bbox)
            bbox_motion_score = float(np.clip(track.motion_score, 0.0, 1.0))
            pixel_motion_score = self._pixel_motion_score(pixel_diff, track.instance_mask, bbox)
            motion_score = self._effective_motion_score(
                track.cls, bbox_motion_score, pixel_motion_score
            )
            stability_score = min(track.hits / max(track.n_init + 3, 1), 1.0)
            confidence_score = float(np.clip(track.score, 0.0, 1.0))
            category = self._roi_category(track.cls)
            motion_threshold = self._motion_threshold(track.cls)
            continuity_score = 1.0 if track.id in self.previous_selected_ids else stability_score
            motion_eligible = motion_score >= motion_threshold
            uncertainty = 0.70 * (1.0 - confidence_score) + 0.30 * (1.0 - stability_score)

            metadata.append(
                {
                    "bbox": bbox,
                    "roi_bbox": roi_bbox,
                    "instance_mask": track.instance_mask,
                    "track_id": track.id,
                    "motion_score": motion_score,
                    "bbox_motion_score": bbox_motion_score,
                    "pixel_motion_score": pixel_motion_score,
                    "background_motion_score": background_motion_score,
                    "mask_coverage": mask_coverage,
                    "uncertainty": uncertainty,
                    "class": track.cls,
                    "confidence": confidence_score,
                    "continuity_score": continuity_score,
                    "motion_eligible": motion_eligible,
                    "motion_threshold": motion_threshold,
                    "category": category,
                    "selected": False,
                }
            )

        self._annotate_attached_context(metadata)

        self._select_rois(metadata)

        return metadata

    def _annotate_attached_context(self, metadata: list[Record]) -> None:
        people = [item for item in metadata if item["class"] == 0]
        for item in metadata:
            item["attached_to_person"] = False
            item["attached_person_overlap"] = 0.0
            item["attached_person_track_id"] = None
            item["strong_attached_to_person"] = False
            if item["class"] not in self.roi_person_attached_classes or not people:
                continue

            ranked_people = sorted(
                people,
                key=lambda person: (
                    bbox_intersection_ratio(item["bbox"], person["bbox"]),
                    -bbox_center_distance(item["bbox"], person["bbox"]),
                    person["confidence"],
                ),
                reverse=True,
            )
            person = ranked_people[0]
            overlap = bbox_intersection_ratio(item["bbox"], person["bbox"])
            center_inside = any(
                bbox_center_inside(item["bbox"], person["bbox"]) for person in people
            )
            item["attached_person_overlap"] = float(overlap)
            item["attached_person_track_id"] = person["track_id"]
            item["attached_to_person"] = overlap >= 0.15 or center_inside
            item["strong_attached_to_person"] = overlap >= 0.45 and person["confidence"] >= 0.35
            if item["attached_to_person"]:
                item["motion_score"] = float(
                    max(item["motion_score"], 0.85 * person["motion_score"])
                )
            item["uncertainty"] = float(np.clip(item["uncertainty"], 0.0, 1.0))
            item["motion_eligible"] = self._is_motion_eligible(item)

        metadata.sort(
            key=lambda item: (
                item["motion_score"],
                item["pixel_motion_score"],
                item["confidence"],
            ),
            reverse=True,
        )

    def _select_rois(self, metadata: list[Record]) -> None:
        selected = []
        attached_person_counts = {}
        person_by_id = {item["track_id"]: item for item in metadata if item["class"] == 0}

        def conflicts_with_selected(item: Record) -> bool:
            for other in selected:
                if self._allow_roi_overlap(item, other):
                    continue
                if bbox_iou(item["roi_bbox"], other["roi_bbox"]) > self.roi_nms_threshold:
                    return True
            return False

        def can_select(item: Record) -> bool:
            if len(selected) >= self.max_foregrounds:
                return False
            if item["selected"]:
                return False
            if item.get("instance_mask") is None:
                return False
            if item.get("mask_coverage", 0.0) < self.roi_min_mask_coverage:
                return False
            if not self._is_motion_eligible(item):
                return False
            if item["class"] in self.roi_person_attached_classes:
                person_id = item.get("attached_person_track_id")
                if not item.get("attached_to_person", False):
                    return False
                person = person_by_id.get(person_id)
                person_selected = person is not None and person.get("selected", False)
                if not person_selected:
                    return False
                if person_id is not None and attached_person_counts.get(person_id, 0) >= 1:
                    return False
            return not conflicts_with_selected(item)

        def add_item(item: Record) -> None:
            item["selected"] = True
            selected.append(item)
            if item["class"] in self.roi_person_attached_classes:
                person_id = item.get("attached_person_track_id")
                if person_id is not None:
                    attached_person_counts[person_id] = attached_person_counts.get(person_id, 0) + 1

        for item in metadata:
            if can_select(item):
                add_item(item)
            if len(selected) >= self.max_foregrounds:
                break

    def _is_motion_eligible(self, item: Record) -> bool:
        threshold = item["motion_threshold"]
        if item["class"] == 0:
            return item["motion_score"] >= threshold
        if item["class"] in self.roi_person_attached_classes:
            return item.get("attached_to_person", False) and item["motion_score"] >= threshold
        if item["class"] in self.roi_vehicle_classes:
            return (
                item["motion_score"] >= threshold
                and item.get("pixel_motion_score", 0.0) >= self.roi_vehicle_pixel_motion_threshold
            )
        return item["motion_score"] >= threshold

    def _allow_roi_overlap(self, item: Record, other: Record) -> bool:
        classes = {item["class"], other["class"]}
        return 0 in classes and bool(classes & self.roi_person_attached_classes)

    def _roi_category(self, class_id: int) -> str:
        if class_id in self.roi_person_attached_classes:
            return "attached"
        if class_id == 0:
            return "person"
        if class_id in self.roi_vehicle_classes:
            return "vehicle"
        return "other"

    def _motion_threshold(self, class_id: int) -> float:
        if class_id == 0:
            return self.roi_min_motion_score
        if class_id in self.roi_person_attached_classes:
            return self.roi_min_motion_score
        if class_id in self.roi_vehicle_classes:
            return max(self.roi_vehicle_motion_threshold, 0.40)
        return self.roi_min_motion_score

    def _effective_motion_score(
        self, class_id: int, bbox_motion_score: float, pixel_motion_score: float
    ) -> float:
        if class_id in self.roi_vehicle_classes:
            return float(min(bbox_motion_score, pixel_motion_score))
        if class_id in self.roi_person_attached_classes:
            return float(max(0.65 * bbox_motion_score, pixel_motion_score))
        if class_id == 0:
            return float(max(bbox_motion_score, 0.75 * pixel_motion_score))
        return float(max(0.75 * bbox_motion_score, pixel_motion_score))

    @staticmethod
    def _build_object_mask(size: tuple[int, int], metadata: list[Record]) -> Image:
        height, width = size
        mask = np.zeros((height, width), dtype=np.uint8)
        for item in metadata:
            if not item.get("selected", False):
                continue
            instance_mask = item.get("instance_mask")
            if instance_mask is None:
                continue
            normalized = instance_mask
            if normalized.ndim == 3 and normalized.shape[2] == 1:
                normalized = normalized[:, :, 0]
            if normalized.shape[:2] != (height, width):
                normalized = cv2.resize(
                    normalized, (width, height), interpolation=cv2.INTER_NEAREST
                )
            mask[normalized > 0] = 255
        return mask

    @staticmethod
    def _compose_background(frame: Image, fill_background: Image, selected_mask: Image) -> Image:
        background = frame.copy()
        background[selected_mask > 0] = fill_background[selected_mask > 0]
        return background

    def _background_change_score(self, background: Image) -> float:
        if self.previous_background is None:
            return 0.0
        diff = cv2.absdiff(background, self.previous_background)
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        changed_ratio = float(np.mean(diff_gray >= self.bg_model.diff_threshold))
        mean_delta = float(np.mean(diff_gray) / 255.0)
        return float(np.clip(0.75 * changed_ratio + 0.25 * mean_delta, 0.0, 1.0))

    @staticmethod
    def _clip_bbox(bbox: BoundingBox, width: int, height: int) -> BoundingBox:
        x1, y1, x2, y2 = bbox
        return [
            max(0, min(int(x1), width - 1)),
            max(0, min(int(y1), height - 1)),
            max(0, min(int(x2), width)),
            max(0, min(int(y2), height)),
        ]

    def _expand_bbox(self, bbox: BoundingBox, width: int, height: int) -> BoundingBox:
        x1, y1, x2, y2 = bbox
        box_w = x2 - x1
        box_h = y2 - y1
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        target_w = max(box_w * self.roi_context_scale, self.roi_min_edge)
        target_h = max(box_h * self.roi_context_scale, self.roi_min_edge)

        nx1 = int(round(cx - target_w / 2.0))
        ny1 = int(round(cy - target_h / 2.0))
        nx2 = int(round(cx + target_w / 2.0))
        ny2 = int(round(cy + target_h / 2.0))

        nx1 = max(nx1, 0)
        ny1 = max(ny1, 0)
        nx2 = min(nx2, width)
        ny2 = min(ny2, height)
        return [nx1, ny1, nx2, ny2]

    @staticmethod
    def _motion_score(mask: Image, bbox: BoundingBox) -> float:
        x1, y1, x2, y2 = bbox
        patch = mask[y1:y2, x1:x2]
        if patch.size == 0:
            return 0.0
        return float(np.mean(patch > 0))

    def _pixel_motion_score(
        self, diff: Image | None, instance_mask: Image | None, bbox: BoundingBox
    ) -> float:
        """Score changed pixels inside a tracked instance or fallback box."""
        if diff is None:
            return 0.0
        if instance_mask is not None:
            mask = instance_mask
            if mask.ndim == 3 and mask.shape[2] == 1:
                mask = mask[:, :, 0]
            if mask.shape[:2] != diff.shape[:2]:
                mask = cv2.resize(
                    mask, (diff.shape[1], diff.shape[0]), interpolation=cv2.INTER_NEAREST
                )
            pixels = diff[mask > 0]
        else:
            x1, y1, x2, y2 = bbox
            pixels = diff[y1:y2, x1:x2].reshape(-1)
        if pixels.size == 0:
            return 0.0
        changed_ratio = float(np.mean(pixels >= self.roi_pixel_motion_diff_threshold))
        mean_delta = float(np.mean(pixels) / 255.0)
        return float(np.clip(0.75 * changed_ratio + 0.25 * mean_delta, 0.0, 1.0))

    @staticmethod
    def _mask_coverage(instance_mask: Image | None, roi_bbox: BoundingBox) -> float:
        if instance_mask is None:
            return 0.0
        x1, y1, x2, y2 = roi_bbox
        patch = instance_mask[y1:y2, x1:x2]
        if patch.size == 0:
            return 0.0
        return float(np.mean(patch > 0))
