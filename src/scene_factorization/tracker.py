from collections import deque

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from .config import (
    TRACKER_APPEARANCE_WEIGHT,
    TRACKER_FEATURE_HISTORY,
    TRACKER_IOU_WEIGHT,
    TRACKER_MAX_AGE,
    TRACKER_MAX_APPEARANCE_DISTANCE,
    TRACKER_MAX_IOU_DISTANCE,
    TRACKER_MOTION_EMA_ALPHA,
    TRACKER_MOTION_GAIN,
    TRACKER_MOTION_JITTER_PIXELS,
    TRACKER_MOTION_JITTER_SCALE,
    TRACKER_N_INIT,
)
from .types import BoundingBox, Image, Record


def bbox_to_xywh(bbox: BoundingBox) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    w = max(float(x2 - x1), 1.0)
    h = max(float(y2 - y1), 1.0)
    cx = float(x1) + w / 2.0
    cy = float(y1) + h / 2.0
    return np.array([cx, cy, w, h], dtype=np.float32)


def xywh_to_bbox(state: np.ndarray) -> BoundingBox:
    cx, cy, w, h = state[:4]
    w = max(float(w), 1.0)
    h = max(float(h), 1.0)
    x1 = int(round(cx - w / 2.0))
    y1 = int(round(cy - h / 2.0))
    x2 = int(round(cx + w / 2.0))
    y2 = int(round(cy + h / 2.0))
    return [x1, y1, x2, y2]


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < 1e-6:
        return 1.0
    similarity = float(np.dot(a, b) / denom)
    return 1.0 - np.clip(similarity, -1.0, 1.0)


def iou(a: BoundingBox, b: BoundingBox) -> float:
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


class KalmanFilterCV:
    """Constant velocity filter for center, size, and their velocities."""

    def __init__(self) -> None:
        self.F = np.eye(8, dtype=np.float32)
        for i in range(4):
            self.F[i, i + 4] = 1.0

        self.H = np.zeros((4, 8), dtype=np.float32)
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.H[2, 2] = 1.0
        self.H[3, 3] = 1.0

        self.Q = np.eye(8, dtype=np.float32)
        self.Q[:4, :4] *= 1.0
        self.Q[4:, 4:] *= 0.25
        self.R = np.eye(4, dtype=np.float32) * 10.0

    def initiate(self, measurement: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mean = np.zeros(8, dtype=np.float32)
        mean[:4] = measurement
        covariance = np.eye(8, dtype=np.float32) * 20.0
        covariance[4:, 4:] *= 10.0
        return mean, covariance

    def predict(self, mean: np.ndarray, covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mean = self.F @ mean
        covariance = self.F @ covariance @ self.F.T + self.Q
        return mean, covariance

    def update(
        self, mean: np.ndarray, covariance: np.ndarray, measurement: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        innovation = measurement - (self.H @ mean)
        innovation_cov = self.H @ covariance @ self.H.T + self.R
        kalman_gain = covariance @ self.H.T @ np.linalg.inv(innovation_cov)
        mean = mean + kalman_gain @ innovation
        covariance = covariance - kalman_gain @ self.H @ covariance
        return mean, covariance


class Track:
    """Tracked detection with motion and appearance history."""

    def __init__(
        self,
        detection: Record,
        track_id: int,
        kf: KalmanFilterCV,
        n_init: int,
        max_age: int,
        feature_history: int,
    ) -> None:
        self.kf = kf
        self.mean, self.covariance = self.kf.initiate(bbox_to_xywh(detection["bbox"]))
        self.id = track_id
        self.cls = detection["class"]
        self.score = detection["score"]
        self.age = 1
        self.hits = 1
        self.time_since_update = 0
        self.n_init = n_init
        self.max_age = max_age
        self.features = deque(maxlen=feature_history)
        self.feature = None
        self.instance_mask = detection.get("instance_mask")
        self.motion_score = 0.0
        self.last_detection_bbox = list(detection["bbox"])
        self.confirmed = False
        self.deleted = False
        self.update_appearance(detection["feature"])

    def predict(self) -> None:
        self.mean, self.covariance = self.kf.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1

    def update(self, detection: Record) -> None:
        previous_bbox = self.last_detection_bbox
        frame_gap = max(int(self.time_since_update), 1)
        measurement = bbox_to_xywh(detection["bbox"])
        self.mean, self.covariance = self.kf.update(self.mean, self.covariance, measurement)
        measured_motion = self._bbox_motion_score(
            previous_bbox, detection["bbox"], frame_gap=frame_gap
        )
        alpha = max(0.0, min(float(TRACKER_MOTION_EMA_ALPHA), 1.0))
        self.motion_score = (1.0 - alpha) * self.motion_score + alpha * measured_motion
        self.last_detection_bbox = list(detection["bbox"])
        self.cls = detection["class"]
        self.score = detection["score"]
        self.instance_mask = detection.get("instance_mask")
        self.hits += 1
        self.time_since_update = 0
        self.update_appearance(detection["feature"])
        if self.hits >= self.n_init:
            self.confirmed = True

    def mark_missed(self) -> None:
        self.motion_score *= 0.8
        if (
            not self.confirmed and self.time_since_update > 0
        ) or self.time_since_update > self.max_age:
            self.deleted = True

    def update_appearance(self, feature: np.ndarray | None) -> None:
        if feature is None:
            return
        self.features.append(feature)
        stacked = np.stack(self.features, axis=0)
        aggregated = stacked.mean(axis=0)
        norm = np.linalg.norm(aggregated)
        self.feature = aggregated / norm if norm > 1e-6 else aggregated

    def to_tlbr(self) -> BoundingBox:
        return xywh_to_bbox(self.mean)

    @staticmethod
    def _bbox_motion_score(
        previous_bbox: BoundingBox, current_bbox: BoundingBox, frame_gap: int = 1
    ) -> float:
        prev_center = np.array(
            [
                (previous_bbox[0] + previous_bbox[2]) / 2.0,
                (previous_bbox[1] + previous_bbox[3]) / 2.0,
            ],
            dtype=np.float32,
        )
        curr_center = np.array(
            [
                (current_bbox[0] + current_bbox[2]) / 2.0,
                (current_bbox[1] + current_bbox[3]) / 2.0,
            ],
            dtype=np.float32,
        )
        distance = float(np.linalg.norm(curr_center - prev_center))
        prev_width = max(float(previous_bbox[2] - previous_bbox[0]), 1.0)
        prev_height = max(float(previous_bbox[3] - previous_bbox[1]), 1.0)
        curr_width = max(float(current_bbox[2] - current_bbox[0]), 1.0)
        curr_height = max(float(current_bbox[3] - current_bbox[1]), 1.0)
        width = 0.5 * (prev_width + curr_width)
        height = 0.5 * (prev_height + curr_height)
        scale = max(np.sqrt(width * height), 1.0)
        jitter_radius = max(
            float(TRACKER_MOTION_JITTER_PIXELS),
            float(TRACKER_MOTION_JITTER_SCALE) * scale,
        )
        adjusted_distance = max(0.0, distance - jitter_radius) / max(int(frame_gap), 1)
        return float(np.clip((adjusted_distance / scale) * float(TRACKER_MOTION_GAIN), 0.0, 1.0))


class Tracker:
    """Associate detections across frames using IoU and HSV appearance."""

    def __init__(
        self,
        max_age: int = TRACKER_MAX_AGE,
        n_init: int = TRACKER_N_INIT,
        max_iou_distance: float = TRACKER_MAX_IOU_DISTANCE,
        max_appearance_distance: float = TRACKER_MAX_APPEARANCE_DISTANCE,
        appearance_weight: float = TRACKER_APPEARANCE_WEIGHT,
        iou_weight: float = TRACKER_IOU_WEIGHT,
        feature_history: int = TRACKER_FEATURE_HISTORY,
    ) -> None:
        self.max_age = max_age
        self.n_init = n_init
        self.max_iou_distance = max_iou_distance
        self.max_appearance_distance = max_appearance_distance
        self.appearance_weight = appearance_weight
        self.iou_weight = iou_weight
        self.feature_history = feature_history
        self.kf = KalmanFilterCV()
        self.reset()

    def reset(self) -> None:
        self.tracks = []
        self.next_id = 0

    def active_bboxes(self) -> list[BoundingBox]:
        return [
            track.to_tlbr()
            for track in self.tracks
            if not track.deleted and 0 < track.time_since_update <= self.max_age
        ]

    def update(self, frame: Image, detections: list[Record]) -> list[Track]:
        """Advance tracks and return current confirmed and tentative tracks."""
        prepared_detections = [self._prepare_detection(frame, det) for det in detections]

        for track in self.tracks:
            track.predict()

        confirmed_indices = [i for i, track in enumerate(self.tracks) if track.confirmed]
        tentative_indices = [i for i, track in enumerate(self.tracks) if not track.confirmed]

        matches = []
        unmatched_detections = list(range(len(prepared_detections)))

        stage_matches, unmatched_confirmed, unmatched_detections = self._match(
            confirmed_indices,
            unmatched_detections,
            prepared_detections,
            use_appearance=True,
        )
        matches.extend(stage_matches)

        candidate_indices = tentative_indices + unmatched_confirmed
        stage_matches, _, unmatched_detections = self._match(
            candidate_indices,
            unmatched_detections,
            prepared_detections,
            use_appearance=False,
        )
        matches.extend(stage_matches)

        matched_track_indices = {track_idx for track_idx, _ in matches}
        unmatched_tracks = [
            idx for idx in range(len(self.tracks)) if idx not in matched_track_indices
        ]

        for track_idx, det_idx in matches:
            self.tracks[track_idx].update(prepared_detections[det_idx])

        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()

        for det_idx in unmatched_detections:
            self._start_track(prepared_detections[det_idx])

        self.tracks = [track for track in self.tracks if not track.deleted]

        confirmed_tracks = [
            track for track in self.tracks if track.confirmed and track.time_since_update <= 1
        ]
        tentative_tracks = [
            track for track in self.tracks if not track.confirmed and track.time_since_update <= 1
        ]
        return confirmed_tracks + tentative_tracks

    def _match(
        self,
        track_indices: list[int],
        detection_indices: list[int],
        detections: list[Record],
        use_appearance: bool,
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        if not track_indices or not detection_indices:
            return [], track_indices, detection_indices

        cost_matrix = np.full((len(track_indices), len(detection_indices)), 1e6, dtype=np.float32)

        for row, track_idx in enumerate(track_indices):
            track = self.tracks[track_idx]
            track_bbox = track.to_tlbr()
            for col, det_idx in enumerate(detection_indices):
                detection = detections[det_idx]
                det_bbox = detection["bbox"]
                overlap = iou(track_bbox, det_bbox)
                if overlap < 1.0 - self.max_iou_distance:
                    continue

                if track.cls != detection["class"]:
                    continue

                iou_cost = 1.0 - overlap
                if (
                    use_appearance
                    and track.feature is not None
                    and detection["feature"] is not None
                ):
                    appearance_cost = cosine_distance(track.feature, detection["feature"])
                    if appearance_cost > self.max_appearance_distance:
                        continue
                    cost = self.appearance_weight * appearance_cost + self.iou_weight * iou_cost
                else:
                    cost = iou_cost

                cost_matrix[row, col] = cost

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matches = []
        assigned_tracks = set()
        assigned_detections = set()

        for row, col in zip(row_ind, col_ind, strict=True):
            if cost_matrix[row, col] >= 1e5:
                continue
            track_idx = track_indices[row]
            det_idx = detection_indices[col]
            matches.append((track_idx, det_idx))
            assigned_tracks.add(track_idx)
            assigned_detections.add(det_idx)

        unmatched_tracks = [idx for idx in track_indices if idx not in assigned_tracks]
        unmatched_detections = [idx for idx in detection_indices if idx not in assigned_detections]
        return matches, unmatched_tracks, unmatched_detections

    def _start_track(self, detection: Record) -> None:
        track = Track(
            detection=detection,
            track_id=self.next_id,
            kf=self.kf,
            n_init=self.n_init,
            max_age=self.max_age,
            feature_history=self.feature_history,
        )
        self.tracks.append(track)
        self.next_id += 1

    @staticmethod
    def _prepare_detection(frame: Image, detection: Record) -> Record:
        bbox = [int(v) for v in detection["bbox"]]
        bbox[0] = max(bbox[0], 0)
        bbox[1] = max(bbox[1], 0)
        bbox[2] = min(bbox[2], frame.shape[1])
        bbox[3] = min(bbox[3], frame.shape[0])

        prepared = dict(detection)
        prepared["bbox"] = bbox
        prepared["feature"] = Tracker._extract_feature(frame, bbox)
        return prepared

    @staticmethod
    def _extract_feature(frame: Image, bbox: BoundingBox) -> np.ndarray | None:
        x1, y1, x2, y2 = bbox
        if x2 <= x1 or y2 <= y1:
            return None

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        margin_x = max((x2 - x1) // 10, 1)
        margin_y = max((y2 - y1) // 10, 1)
        crop = crop[margin_y : -margin_y or None, margin_x : -margin_x or None]
        if crop.size == 0:
            return None

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 4, 4], [0, 180, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten().astype(np.float32)
        norm = np.linalg.norm(hist)
        return hist / norm if norm > 1e-6 else hist
