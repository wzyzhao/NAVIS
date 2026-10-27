import json
from collections import defaultdict, deque
from collections.abc import Iterable
from pathlib import Path

from nbiot_transport.core.metadata_bundle import expand_metadata_bundle
from nbiot_transport.core.packetizer import Packet
from nbiot_transport.feedback.ack_utils import result_arrived


def packet_header(packet: Packet) -> dict:
    return dict(packet.metadata or {})


def _bbox_center(bbox: list[int]) -> tuple[float, float]:
    return ((float(bbox[0]) + float(bbox[2])) / 2.0, (float(bbox[1]) + float(bbox[3])) / 2.0)


def _shift_bbox(bbox: list[int], dx: float, dy: float) -> list[int]:
    return [int(round(bbox[0] + dx)), int(round(bbox[1] + dy)), int(round(bbox[2] + dx)), int(round(bbox[3] + dy))]


class MetadataOnlyReceiverState:
    """Receiver-side state for degraded metadata-only operation.

    This implements the PDF's section-6 strategy:
    - keep a historical background model and freshness flag;
    - keep per-track ROI cache;
    - use metadata bbox history to predict ROI placeholders when no ROI arrives;
    - emit recovery hints for stale background and stale tracks.
    """

    def __init__(
        self,
        stale_after_frames: int = 30,
        roi_cache_size: int = 5,
        track_stale_after_frames: int = 10,
        background_change_threshold: float = 0.6,
    ):
        self.stale_after_frames = max(1, int(stale_after_frames))
        self.roi_cache_size = max(1, int(roi_cache_size))
        self.track_stale_after_frames = max(1, int(track_stale_after_frames))
        self.background_change_threshold = float(background_change_threshold)

        self.background_state = {}
        self.track_cache = defaultdict(lambda: deque(maxlen=self.roi_cache_size))
        self.track_metadata = defaultdict(lambda: deque(maxlen=3))
        self.predicted_rois = []
        self.recovery_hints = []

        self.metadata_frames = defaultdict(int)
        self.roi_frames = defaultdict(int)
        self.metadata_only_frames = defaultdict(int)

    def apply_confirmed_packets(self, packets: Iterable[Packet]) -> None:
        packets_by_frame = defaultdict(list)
        for packet in packets:
            header = packet_header(packet)
            bundle_items = expand_metadata_bundle(header) if packet.layer == "metadata" else []
            if bundle_items:
                for item in bundle_items:
                    frame_idx = int(item.get("frame_idx", item.get("frame_id", 0)) or 0)
                    if frame_idx <= 0:
                        continue
                    packets_by_frame[(packet.device_id, frame_idx)].append(
                        Packet(
                            packet_id=f"{packet.packet_id}:bundle_frame:{frame_idx}",
                            device_id=packet.device_id,
                            frame_idx=frame_idx,
                            frame_id=int(item.get("frame_id", frame_idx) or frame_idx),
                            layer="metadata",
                            payload=b"",
                            priority=packet.priority,
                            metadata=item,
                        )
                    )
                continue
            packets_by_frame[(packet.device_id, packet.frame_idx)].append(packet)

        for (device_id, frame_idx), frame_packets in sorted(packets_by_frame.items()):
            headers = {packet.packet_id: packet_header(packet) for packet in frame_packets}
            metadata_packets = [packet for packet in frame_packets if packet.layer == "metadata"]
            roi_packets = [packet for packet in frame_packets if packet.layer == "roi_patch"]
            background_packets = [packet for packet in frame_packets if packet.layer == "background_keyframe_summary"]

            if metadata_packets:
                self.metadata_frames[device_id] += 1
                for packet in metadata_packets:
                    self._apply_metadata_packet(device_id, frame_idx, headers[packet.packet_id])

            if roi_packets:
                self.roi_frames[device_id] += 1
                for packet in roi_packets:
                    self._apply_roi_packet(device_id, frame_idx, packet, headers[packet.packet_id])

            if metadata_packets and not roi_packets:
                self.metadata_only_frames[device_id] += 1
                self._apply_metadata_only_frame(device_id, frame_idx)

            if background_packets:
                self.background_state[device_id] = {
                    "last_frame_idx": frame_idx,
                    "fresh": True,
                    "reason": "background_keyframe_summary_received",
                }
            else:
                self._refresh_background_state(device_id, frame_idx)

    def _apply_metadata_packet(self, device_id: str, frame_idx: int, header: dict) -> None:
        background_change_score = float(header.get("background_change_score", 0.0) or 0.0)
        if background_change_score >= self.background_change_threshold:
            self._request(
                device_id=device_id,
                frame_idx=frame_idx,
                request_type="background_keyframe_summary",
                reason="background_change_score_high",
                priority=0.8 + background_change_score,
            )

        for roi in self._iter_metadata_rois(header):
            track_id = self._track_id(roi)
            bbox = self._bbox(roi)
            if track_id is None or bbox is None:
                continue
            self.track_metadata[(device_id, track_id)].append(
                {
                    "frame_idx": frame_idx,
                    "bbox": bbox,
                    "class_id": roi.get("class_id"),
                    "motion_score": float(roi.get("motion_score", 0.0) or 0.0),
                    "uncertainty": float(roi.get("uncertainty", 0.0) or 0.0),
                }
            )

    def _apply_roi_packet(self, device_id: str, frame_idx: int, packet: Packet, header: dict) -> None:
        track_id = header.get("track_id", packet.roi_idx)
        if track_id is None:
            return
        self.track_cache[(device_id, track_id)].append(
            {
                "frame_idx": frame_idx,
                "roi_idx": packet.roi_idx,
                "rate_idx": packet.rate_idx,
                "bytes": packet.bytes,
                "bbox": header.get("bbox") or header.get("roi_bbox"),
            }
        )

    def _apply_metadata_only_frame(self, device_id: str, frame_idx: int) -> None:
        active_tracks = [
            (track_key, history)
            for track_key, history in self.track_metadata.items()
            if track_key[0] == device_id and history and int(history[-1]["frame_idx"]) == frame_idx
        ]
        for (track_device, track_id), history in active_tracks:
            current = history[-1]
            cached = self._latest_roi_cache(track_device, track_id)
            predicted_bbox = self._predict_bbox(history)
            confidence = 0.55 if cached else 0.25
            if len(history) >= 2:
                confidence += 0.2
            self.predicted_rois.append(
                {
                    "device_id": device_id,
                    "frame_idx": frame_idx,
                    "track_id": track_id,
                    "bbox": predicted_bbox,
                    "source": "roi_cache" if cached else "metadata_bbox",
                    "confidence": min(confidence, 0.95),
                    "cached_roi_frame_idx": cached.get("frame_idx") if cached else None,
                }
            )

            if cached is None or frame_idx - int(cached["frame_idx"]) > self.track_stale_after_frames:
                self._request(
                    device_id=device_id,
                    frame_idx=frame_idx,
                    request_type="roi_patch",
                    track_id=track_id,
                    bbox=predicted_bbox,
                    reason="track_roi_cache_stale",
                    priority=(
                        0.7
                        + 0.25 * float(current.get("motion_score", 0.0) or 0.0)
                        + 0.25 * float(current.get("uncertainty", 0.0) or 0.0)
                    ),
                )

    def _refresh_background_state(self, device_id: str, frame_idx: int) -> None:
        state = self.background_state.get(device_id)
        if state is None:
            self.background_state[device_id] = {
                "last_frame_idx": None,
                "fresh": False,
                "reason": "no_background_received",
            }
            self._request(
                device_id=device_id,
                frame_idx=frame_idx,
                request_type="background_keyframe_summary",
                reason="no_background_model",
                priority=0.6,
            )
            return

        last_frame = state.get("last_frame_idx")
        fresh = last_frame is not None and (frame_idx - int(last_frame)) <= self.stale_after_frames
        state["fresh"] = fresh
        state["reason"] = "fresh" if fresh else "background_stale"
        if not fresh:
            self._request(
                device_id=device_id,
                frame_idx=frame_idx,
                request_type="background_keyframe_summary",
                reason="background_stale",
                priority=0.7,
            )

    def _predict_bbox(self, history: deque) -> list[int]:
        current = history[-1]["bbox"]
        if len(history) < 2:
            return list(current)
        previous = history[-2]["bbox"]
        cx0, cy0 = _bbox_center(previous)
        cx1, cy1 = _bbox_center(current)
        return _shift_bbox(current, cx1 - cx0, cy1 - cy0)

    def _latest_roi_cache(self, device_id: str, track_id) -> dict | None:
        cache = self.track_cache.get((device_id, track_id))
        if not cache:
            return None
        return cache[-1]

    def _request(
        self,
        device_id: str,
        frame_idx: int,
        request_type: str,
        reason: str,
        priority: float,
        track_id=None,
        bbox=None,
    ) -> None:
        self.recovery_hints.append(
            {
                "device_id": device_id,
                "frame_idx": frame_idx,
                "request_type": request_type,
                "track_id": track_id,
                "bbox": bbox,
                "reason": reason,
                "priority": float(priority),
            }
        )

    @staticmethod
    def _iter_metadata_rois(header: dict) -> list[dict]:
        if isinstance(header.get("rois"), list):
            return header["rois"]
        return []

    @staticmethod
    def _track_id(roi: dict):
        return roi.get("track_id", roi.get("roi_idx"))

    @staticmethod
    def _bbox(roi: dict):
        bbox = roi.get("bbox", roi.get("roi_bbox"))
        if isinstance(bbox, list) and len(bbox) == 4:
            return [int(v) for v in bbox]
        return None

    def to_json(self) -> dict:
        hints = sorted(self.recovery_hints, key=lambda item: (-item["priority"], item["frame_idx"]))
        return {
            "policy": "metadata-only receiver uses historical background, track prediction, ROI cache, and recovery hints",
            "stale_after_frames": self.stale_after_frames,
            "roi_cache_size": self.roi_cache_size,
            "track_stale_after_frames": self.track_stale_after_frames,
            "background_change_threshold": self.background_change_threshold,
            "metadata_frames": dict(self.metadata_frames),
            "roi_frames": dict(self.roi_frames),
            "metadata_only_frames": dict(self.metadata_only_frames),
            "background_state": self.background_state,
            "predicted_roi_count": len(self.predicted_rois),
            "predicted_rois": self.predicted_rois[-100:],
            "recovery_hint_count": len(hints),
            "recovery_hints": hints[:100],
            "roi_cache_keys": [f"{device_id}:{track_id}" for device_id, track_id in self.track_cache.keys()],
        }


def write_receiver_state(
    output_root: Path,
    packets: list[Packet],
    results,
    stale_after_frames: int,
    roi_cache_size: int,
    track_stale_after_frames: int = 10,
    background_change_threshold: float = 0.6,
):
    output_root.mkdir(parents=True, exist_ok=True)
    confirmed_ids = {result.packet_id for result in results if result_arrived(result)}
    confirmed_packets = [packet for packet in packets if packet.packet_id in confirmed_ids]
    state = MetadataOnlyReceiverState(
        stale_after_frames=stale_after_frames,
        roi_cache_size=roi_cache_size,
        track_stale_after_frames=track_stale_after_frames,
        background_change_threshold=background_change_threshold,
    )
    state.apply_confirmed_packets(confirmed_packets)
    output_path = output_root / "receiver_state.json"
    output_path.write_text(json.dumps(state.to_json(), ensure_ascii=True, indent=2), encoding="utf-8")
    return output_path
