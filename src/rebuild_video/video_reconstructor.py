from __future__ import annotations

import base64
import json
import queue
import re
import shutil
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from common.encoding_layout import load_artifact_json

if TYPE_CHECKING:
    from numpy.typing import NDArray

    ImageArray = NDArray[Any]

from nbiot_transport.core.metadata_bundle import expand_metadata_bundle
from rebuild_video.side_info_codec import decode_metadata

BACKGROUND_LAYER = "background_keyframe_summary"


def _safe_payload_stem_part(value: object) -> str:
    text = str(value or "")
    return "".join(ch for ch in text if ch.isalnum()) or "x"


def _payload_file_name_from_item(device_id: object, frame: dict, item: dict, rate_idx: int, item_index: int) -> str:
    frame_idx = int(frame.get("frame_idx", 0) or 0)
    layer = str(item.get("layer") or "payload")
    roi_idx = item.get("roi_idx")
    item_rate_idx = int(item.get("rate_idx", rate_idx) or 0)
    if layer == "roi_patch" and roi_idx not in (None, ""):
        suffix = f"rate{item_rate_idx}_roi{int(roi_idx):03d}"
    elif layer == BACKGROUND_LAYER:
        suffix = f"rate{item_rate_idx}_background"
    else:
        suffix = _safe_payload_stem_part(layer)
    return f"{_safe_payload_stem_part(device_id)}_frame{frame_idx}_{suffix}.bin"


def _payload_filename_fields(name: str) -> dict | None:
    match = re.fullmatch(
        r"(?P<device>[A-Za-z0-9]+)_frame(?P<frame>\d+)_(?:(?P<suffix>metadata)|rate(?P<rate>\d+)_(?P<rated_suffix>background|roi(?P<roi>\d+)))\.bin",
        name,
    )
    if match is None:
        return None
    suffix = match.group("suffix") or match.group("rated_suffix")
    layer = "roi_patch" if suffix.startswith("roi") else (BACKGROUND_LAYER if suffix == "background" else "metadata")
    return {
        "rate_idx": int(match.group("rate") or 0),
        "frame_idx": int(match.group("frame")),
        "layer": layer,
        "roi_idx": int(match.group("roi")) if match.group("roi") is not None else None,
    }


@dataclass
class ROITrackCache:
    image: object
    mask: object | None
    bbox: list[int]
    frame_idx: int
    roi_idx: int
    track_id: object
    previous_bbox: list[int] | None = None
    previous_frame_idx: int | None = None


@dataclass
class DeviceState:
    background: object | None = None
    background_frame_idx: int | None = None
    background_frames: set[int] = field(default_factory=set)
    background_history: dict[int, object] = field(default_factory=dict)
    metadata: dict[int, dict] = field(default_factory=dict)
    rois: dict[tuple[int, int], Path] = field(default_factory=dict)
    roi_track_cache: dict[str, ROITrackCache] = field(default_factory=dict)
    frames: set[int] = field(default_factory=set)


class DecoderWorker:
    def __init__(self, command: list[str], stderr_path: Path) -> None:
        self.command = command
        self.stderr_path = stderr_path
        self.process = None
        self.stderr_file = None
        self.lock = threading.Lock()

    def start(self) -> None:
        if self.process is not None and self.process.poll() is None:
            return
        self.close()
        self.stderr_file = self.stderr_path.open("a", encoding="utf-8")
        self.process = subprocess.Popen(
            self.command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.stderr_file,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        ready = self._read_json_response("decoder worker exited during startup")
        if not ready.get("ready"):
            raise RuntimeError(ready.get("error") or "decoder worker failed to start")

    def decode(self, layer: str, device_id: str, source: Path, output: Path, frame_idx: int | None = None) -> dict:
        """Serialize access to the persistent decoder and match a reply to its request."""
        with self.lock:
            self.start()
            request = {
                "request_id": uuid.uuid4().hex,
                "layer": layer,
                "device_id": device_id,
                "input_path": str(source.resolve()),
                "output_path": str(output.resolve()),
            }
            if frame_idx is not None:
                request["frame_idx"] = int(frame_idx)
            self.process.stdin.write(json.dumps(request, separators=(",", ":")) + "\n")
            self.process.stdin.flush()
            response = self._read_json_response(
                "decoder worker exited while decoding",
                request_id=request["request_id"],
            )
            if response.get("request_id") != request["request_id"] or not response.get("ok"):
                raise RuntimeError(response.get("error") or "decoder worker protocol error")
            return response

    def _read_json_response(self, exit_error: str, request_id: str | None = None) -> dict:
        while True:
            line = self.process.stdout.readline() if self.process and self.process.stdout else ""
            if not line:
                raise RuntimeError(exit_error)
            try:
                response = json.loads(line)
            except json.JSONDecodeError:
                self._log_worker_stdout(line)
                continue
            if not isinstance(response, dict):
                self._log_worker_stdout(line)
                continue
            if request_id is not None and response.get("request_id") != request_id:
                self._log_worker_stdout(line)
                continue
            return response

    def _log_worker_stdout(self, line: str) -> None:
        with self.stderr_path.open("a", encoding="utf-8") as stream:
            stream.write(f"[decoder stdout] {line.rstrip()}\n")

    def close(self) -> None:
        process, self.process = self.process, None
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
        if self.stderr_file is not None:
            self.stderr_file.close()
            self.stderr_file = None


class VideoReconstructor:
    def __init__(self, root: Path, config: dict) -> None:
        self.root = root
        self.enabled = bool(config.get("reconstruction_enabled", False))
        self.width = int(config.get("reconstruction_width", 1920))
        self.height = int(config.get("reconstruction_height", 1080))
        self.fps = float(config.get("reconstruction_fps", 30))
        self.expected_frames = int(config.get("reconstruction_expected_frames", 0) or 0)
        self.fill_missing_frames = bool(config.get("reconstruction_fill_missing_frames", True))
        self.roi_mask_erode_pixels = max(0, int(config.get("reconstruction_roi_mask_erode_pixels", 1) or 0))
        self.roi_mask_feather_pixels = max(0, int(config.get("reconstruction_roi_mask_feather_pixels", 5) or 0))
        self.roi_black_border_suppression = bool(config.get("reconstruction_roi_black_border_suppression", True))
        self.roi_black_threshold = max(0, min(255, int(config.get("reconstruction_roi_black_threshold", 18) or 0)))
        self.roi_content_mask_dilate_pixels = max(
            0, int(config.get("reconstruction_roi_content_mask_dilate_pixels", 2) or 0)
        )
        self.roi_internal_black_suppression = bool(config.get("reconstruction_roi_internal_black_suppression", True))
        self.roi_internal_black_min_area_ratio = max(
            0.0,
            min(1.0, float(config.get("reconstruction_roi_internal_black_min_area_ratio", 0.015) or 0.0)),
        )
        self.roi_cache_ttl_frames = max(0, int(config.get("roi_cache_ttl_frames", 10) or 0))
        self.roi_proxy_fade_frames = max(1, int(config.get("roi_proxy_fade_frames", 5) or 1))
        self.roi_proxy_min_alpha = max(
            0.0,
            min(1.0, float(config.get("roi_proxy_min_alpha", 0.25) or 0.0)),
        )
        self.roi_proxy_motion_compensation = bool(config.get("roi_proxy_motion_compensation", True))
        self.roi_proxy_max_motion_scale = max(
            0.0,
            float(config.get("roi_proxy_max_motion_scale", 1.5) or 0.0),
        )
        self.roi_min_effective_alpha_ratio = max(
            0.0,
            min(1.0, float(config.get("reconstruction_roi_min_effective_alpha_ratio", 0.02) or 0.0)),
        )
        self.cv2 = None
        self.np = None
        self.states: dict[str, DeviceState] = {}
        self.lock = threading.RLock()
        self.manifest = root / "received_payload_manifest.jsonl"
        self.status_log = root / "reconstruction_status.jsonl"
        self.error_log = root / "reconstruction_errors.jsonl"
        self.frames_root = root / "reconstructed_frames"
        self.videos_root = root / "reconstructed_videos"
        self.decoded_root = root / "decoded_layers"
        self.worker = None
        self.work_queue: queue.Queue = queue.Queue()
        self.thread = None
        self.closed = False
        self.video_frame_counts: dict[str, int] = {}
        root.mkdir(parents=True, exist_ok=True)
        if self.enabled:
            import cv2
            import numpy as np

            self.cv2, self.np = cv2, np
            for path in (self.frames_root, self.videos_root, self.decoded_root):
                path.mkdir(parents=True, exist_ok=True)
            command = config.get("dcvc_decoder_command") or []
            if not command:
                raise ValueError("reconstruction_enabled requires dcvc_decoder_command")
            self.worker = DecoderWorker([str(value) for value in command], root / "dcvc_decoder_stderr.log")
            self.thread = threading.Thread(target=self._run, name="video-reconstructor", daemon=True)
            self.thread.start()

    def accept_payload(self, header: dict, path: Path) -> dict | None:
        keys = (
            "message_id_hex",
            "device_id",
            "frame_idx",
            "frame_id",
            "item_index",
            "layer",
            "roi_idx",
            "rate_idx",
            "payload_crc32",
        )
        record = {key: header.get(key) for key in keys}
        record["payload_path"] = str(path)
        self._append_json(self.manifest, record)
        return self._queue_record(record, path)

    def _queue_record(self, record: dict, path: Path) -> dict | None:
        if not self.enabled:
            return None
        self.work_queue.put((record, path))
        return {"reconstruction_status": "queued"}

    def recover_manifest(self) -> int:
        if not self.enabled or not self.manifest.is_file():
            return 0
        recovered = []
        for line in self.manifest.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            path = Path(record.get("payload_path", ""))
            if not path.is_file() and path.name:
                path = self.root / "reassembled_payloads" / path.name
            if path.is_file():
                record = self._record_with_filename_fields(record, path)
                recovered.append((record, path))
        order = {"metadata": 0, BACKGROUND_LAYER: 1, "roi_patch": 2}
        recovered.sort(
            key=lambda value: (
                int(value[0].get("frame_idx") or 0),
                order.get(str(value[0].get("layer")), 3),
                int(value[0].get("roi_idx") or 0),
            )
        )
        for record, path in recovered:
            self._queue_record(record, path)
        return len(recovered)

    @staticmethod
    def _record_with_filename_fields(record: dict, path: Path) -> dict:
        fields = _payload_filename_fields(path.name)
        if fields is None:
            return record
        updated = dict(record)
        updated["frame_idx"] = int(fields["frame_idx"])
        updated.setdefault("frame_id", int(fields["frame_idx"]))
        updated["layer"] = fields["layer"]
        updated["rate_idx"] = int(fields["rate_idx"])
        if fields.get("roi_idx") is not None:
            updated["roi_idx"] = int(fields["roi_idx"])
        return updated

    def recover_directory(self, schedule_json: Path) -> int:
        if not self.enabled or not schedule_json.is_file():
            return 0
        schedule = load_artifact_json(schedule_json)
        frames = schedule.get("frames", schedule) if isinstance(schedule, dict) else schedule
        frame_map = {int(frame["frame_idx"]): frame for frame in frames}
        recovered = []
        for path in (self.root / "reassembled_payloads").glob("*.bin"):
            fields = _payload_filename_fields(path.name)
            if fields is None:
                continue
            frame_idx = int(fields["frame_idx"])
            file_layer = fields["layer"]
            file_rate_idx = int(fields["rate_idx"])
            frame = frame_map.get(frame_idx)
            item = None
            if frame is not None:
                for candidate in frame.get("items", []):
                    if candidate.get("layer") != file_layer:
                        continue
                    if file_layer == "roi_patch" and int(candidate.get("roi_idx", -1)) != int(
                        fields.get("roi_idx", -2)
                    ):
                        continue
                    item = candidate
                    break
            header = {
                "device_id": str((item or {}).get("device_id") or (frame or {}).get("sequence") or "node"),
                "frame_idx": frame_idx,
                "frame_id": (frame or {}).get("frame_id", frame_idx),
                "item_index": int((item or {}).get("item_index", 0) or 0),
                "layer": file_layer,
                "roi_idx": fields.get("roi_idx") if fields.get("roi_idx") is not None else (item or {}).get("roi_idx"),
                "rate_idx": (item or {}).get("rate_idx", file_rate_idx),
            }
            recovered.append((header, path))
        order = {"metadata": 0, BACKGROUND_LAYER: 1, "roi_patch": 2}
        recovered.sort(
            key=lambda value: (
                int(value[0]["frame_idx"]),
                order.get(str(value[0]["layer"]), 3),
                int(value[0].get("roi_idx") or 0),
            )
        )
        for header, path in recovered:
            self.accept_payload(header, path)
        return len(recovered)

    def _run(self) -> None:
        while True:
            item = self.work_queue.get()
            try:
                if item is None:
                    return
                record, path = item
                try:
                    result = self.process(record, path)
                    self._append_json(self.status_log, {**record, **result, "completed_at": time.time()})
                except Exception as exc:
                    self._append_json(
                        self.error_log,
                        {**record, "error": f"{type(exc).__name__}: {exc}", "failed_at": time.time()},
                    )
            finally:
                self.work_queue.task_done()

    def process(self, record: dict, path: Path) -> dict:
        """Apply one payload to per-device temporal state and render newly available frames."""
        device_id = str(record.get("device_id") or "node")
        frame_idx = int(record.get("frame_idx") or 0)
        layer = str(record.get("layer"))
        with self.lock:
            state = self.states.setdefault(device_id, DeviceState())
            if layer == "metadata":
                metadata = self._read_metadata(path)
                bundle_items = expand_metadata_bundle(metadata)
                if bundle_items:
                    bundle_frames = []
                    for item in bundle_items:
                        item_frame_idx = int(item.get("frame_idx", item.get("frame_id", 0)) or 0)
                        if item_frame_idx <= 0:
                            continue
                        state.metadata[item_frame_idx] = item
                        bundle_frames.append(item_frame_idx)
                    result = {
                        "reconstruction_status": "metadata_bundle_ready",
                        "metadata_bundle_frame_count": len(bundle_frames),
                        "metadata_bundle_frames": bundle_frames,
                    }
                    rendered = []
                    for item_frame_idx in bundle_frames:
                        rendered.extend(self._render_ready(device_id, item_frame_idx, layer, state))
                    if rendered:
                        result["reconstructed_frame_paths"] = [str(value) for value in rendered]
                        video_path = self._write_video_if_complete(device_id, state)
                        if video_path is not None:
                            result["reconstructed_video_path"] = str(video_path)
                    return result
                state.metadata[frame_idx] = metadata
                result = {"reconstruction_status": "metadata_ready"}
            elif layer in ("roi_patch", BACKGROUND_LAYER):
                result = self._decode_layer(device_id, frame_idx, layer, record, path, state)
            else:
                return {"reconstruction_status": "stored"}
            rendered = self._render_ready(device_id, frame_idx, layer, state)
            if rendered:
                result["reconstructed_frame_paths"] = [str(value) for value in rendered]
                video_path = self._write_video_if_complete(device_id, state)
                if video_path is not None:
                    result["reconstructed_video_path"] = str(video_path)
            return result

    def _decode_layer(
        self, device_id: str, frame_idx: int, layer: str, record: dict, path: Path, state: DeviceState
    ) -> dict:
        roi_idx = int(record.get("roi_idx") or 0) if layer == "roi_patch" else None
        suffix = f"roi_{roi_idx:03d}" if roi_idx is not None else "background"
        output = self.decoded_root / device_id / f"frame_{frame_idx:06d}_{suffix}.png"
        if layer == BACKGROUND_LAYER:
            decoded = self.worker.decode(BACKGROUND_LAYER, device_id, path, output, frame_idx=frame_idx)
        else:
            decoded = self.worker.decode(layer, device_id, path, output, frame_idx=frame_idx)
        if decoded.get("pending"):
            return {"reconstruction_status": "decoder_pending"}
        image = self.cv2.imread(str(output))
        if image is None:
            raise RuntimeError(f"cannot read decoder output: {output}")
        if layer == BACKGROUND_LAYER:
            if self.width <= 0 or self.height <= 0:
                self.height, self.width = image.shape[:2]
                state.background = image.copy()
            else:
                state.background = self.cv2.resize(image, (self.width, self.height))
            state.background_frame_idx = frame_idx
            state.background_frames.add(frame_idx)
            state.background_history[int(frame_idx)] = state.background.copy()
            background = {
                "reference_frame_idx": None,
            }
            return {
                "reconstruction_status": "background_ready",
                "background": background,
            }
        state.rois[(frame_idx, roi_idx)] = output
        return {"reconstruction_status": "roi_ready"}

    def _render_ready(self, device_id: str, frame_idx: int, layer: str, state: DeviceState) -> list[Path]:
        if layer == BACKGROUND_LAYER:
            candidates = sorted(index for index in state.metadata if index >= frame_idx)
        else:
            candidates = [frame_idx] if frame_idx in state.metadata else []
        rendered = []
        for index in candidates:
            output = self._render_frame(device_id, index, state)
            if output is not None:
                rendered.append(output)
        return rendered

    def _render_frame(self, device_id: str, frame_idx: int, state: DeviceState) -> Path | None:
        metadata = state.metadata.get(frame_idx)
        if metadata is None:
            return None
        if state.background is not None:
            frame = state.background.copy()
            background_status = "decoded" if state.background_frame_idx == frame_idx else "reused"
            background_age_frames = (
                None if state.background_frame_idx is None else max(0, frame_idx - int(state.background_frame_idx))
            )
        else:
            return None
        records = {
            int(item.get("roi_idx", index)): item
            for index, item in enumerate(metadata.get("rois", []))
            if isinstance(item, dict)
        }
        applied = 0
        current_roi_applied = 0
        metadata_proxy_applied = 0
        roi_mask_applied = 0
        missing_roi = 0
        missing_roi_mask = 0
        for roi_idx, item in records.items():
            roi_path = state.rois.get((frame_idx, roi_idx))
            bbox = item.get("roi_bbox") or item.get("bbox")
            if not isinstance(bbox, list) or len(bbox) != 4:
                missing_roi += 1
                continue
            clipped = self._clip_bbox(bbox)
            if clipped is None:
                missing_roi += 1
                continue
            roi = self.cv2.imread(str(roi_path)) if roi_path is not None else None
            if roi is not None:
                if not self._composite_roi(frame, roi, item, clipped, state):
                    missing_roi += 1
                    missing_roi_mask += 1
                    continue
                self._cache_roi_patch(state, frame_idx, roi_idx, item, clipped, roi)
                roi_mask_applied += 1
                applied += 1
                current_roi_applied += 1
                continue

            cached = self._track_cache_for_item(state, item)
            if cached is None:
                missing_roi += 1
                continue
            age = max(0, frame_idx - int(cached.frame_idx))
            if age > self.roi_cache_ttl_frames:
                missing_roi += 1
                continue
            proxy_bbox = self._proxy_bbox_for_missing_roi(cached, clipped, age)
            proxy_item = dict(item)
            if "mask_rle" not in proxy_item and cached.mask is not None:
                proxy_item["_cached_mask"] = cached.mask
            alpha_scale = self._roi_proxy_alpha_scale(age)
            if not self._composite_roi(frame, cached.image, proxy_item, proxy_bbox, state, alpha_scale=alpha_scale):
                missing_roi += 1
                missing_roi_mask += 1
                continue
            roi_mask_applied += 1
            applied += 1
            metadata_proxy_applied += 1
        folder = self.frames_root / device_id
        folder.mkdir(parents=True, exist_ok=True)
        output = folder / f"{frame_idx:06d}.png"
        if not self.cv2.imwrite(str(output), frame):
            raise RuntimeError(f"failed to write reconstructed frame: {output}")
        state.frames.add(frame_idx)
        self._append_json(
            self.status_log,
            {
                "device_id": device_id,
                "frame_idx": frame_idx,
                "event": "frame_rendered",
                "background": background_status,
                "roi_applied": applied,
                "background_frame_idx": state.background_frame_idx,
                "background_age_frames": background_age_frames,
                "roi_current_applied": current_roi_applied,
                "roi_metadata_proxy_applied": metadata_proxy_applied,
                "roi_mask_applied": roi_mask_applied,
                "roi_missing": missing_roi,
                "roi_missing_mask": missing_roi_mask,
                "roi_expected": len(records),
                "output_path": str(output),
            },
        )
        return output

    def _write_video(self, device_id: str, state: DeviceState) -> Path:
        output = self.videos_root / f"{device_id}.mp4"
        temp = self.videos_root / f".{device_id}.{uuid.uuid4().hex}.mp4"
        writer = self.cv2.VideoWriter(
            str(temp),
            self.cv2.VideoWriter_fourcc(*"mp4v"),
            self.fps,
            (self.width, self.height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"cannot open video writer: {temp}")
        try:
            for frame_idx in sorted(state.frames):
                frame = self.cv2.imread(str(self.frames_root / device_id / f"{frame_idx:06d}.png"))
                if frame is not None:
                    writer.write(frame)
        finally:
            writer.release()
        temp.replace(output)
        return output

    def _fill_missing_frames(self, device_id: str, state: DeviceState) -> int:
        if not self.fill_missing_frames or self.expected_frames <= 0 or not state.frames:
            return 0
        folder = self.frames_root / device_id
        existing = set(state.frames)
        filled = 0
        previous = None
        first_existing = min(existing)
        first_path = folder / f"{first_existing:06d}.png"
        for frame_idx in range(1, self.expected_frames + 1):
            output = folder / f"{frame_idx:06d}.png"
            if frame_idx in existing and output.is_file():
                previous = output
                continue
            source = previous if previous is not None else first_path
            if not source.is_file():
                continue
            shutil.copyfile(source, output)
            state.frames.add(frame_idx)
            existing.add(frame_idx)
            previous = output
            filled += 1
            self._append_json(
                self.status_log,
                {
                    "device_id": device_id,
                    "frame_idx": frame_idx,
                    "event": "frame_filled",
                    "fill_source_frame_idx": self._frame_idx_from_path(source),
                    "output_path": str(output),
                },
            )
        return filled

    def _write_video_if_complete(self, device_id: str, state: DeviceState) -> Path | None:
        if self.expected_frames <= 0:
            return None
        expected = set(range(1, self.expected_frames + 1))
        if not expected.issubset(state.frames):
            return None
        if self.video_frame_counts.get(device_id) == len(state.frames):
            return None
        self.video_frame_counts[device_id] = len(state.frames)
        return self._write_video(device_id, state)

    def finalize_videos(self) -> list[Path]:
        """Fill configured frame gaps and finalize each device's ordered video output."""
        if not self.enabled:
            return []
        videos = []
        with self.lock:
            for device_id, state in self.states.items():
                if state.frames:
                    self._fill_missing_frames(device_id, state)
                    if self.video_frame_counts.get(device_id) == len(state.frames):
                        continue
                    self.video_frame_counts[device_id] = len(state.frames)
                    videos.append(self._write_video(device_id, state))
        return videos

    def _clip_bbox(self, bbox: list) -> tuple[int, int, int, int] | None:
        x1, y1, x2, y2 = (int(value) for value in bbox)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(self.width, x2), min(self.height, y2)
        return (x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None

    def _composite_roi(
        self,
        frame: ImageArray,
        roi: ImageArray,
        item: dict,
        bbox: tuple[int, int, int, int],
        state: DeviceState | None = None,
        alpha_scale: float = 1.0,
    ) -> bool:
        x1, y1, x2, y2 = bbox
        width, height = x2 - x1, y2 - y1
        resized_roi = self.cv2.resize(roi, (width, height))
        alpha = self._build_roi_alpha(item, resized_roi, (width, height), bbox, state)
        if alpha is None:
            return False
        alpha = (alpha * max(0.0, min(1.0, float(alpha_scale)))).clip(0.0, 1.0)
        if float(alpha.mean()) < self.roi_min_effective_alpha_ratio:
            return False
        alpha = alpha[:, :, None]
        target = frame[y1:y2, x1:x2].astype(self.np.float32)
        blended = alpha * resized_roi.astype(self.np.float32) + (1.0 - alpha) * target
        frame[y1:y2, x1:x2] = blended.astype(self.np.uint8)
        return True

    def _build_roi_alpha(
        self,
        item: dict,
        resized_roi: ImageArray,
        target_size: tuple[int, int],
        bbox: tuple[int, int, int, int],
        state: DeviceState | None = None,
    ) -> ImageArray | None:
        mask = self._decode_roi_mask(item, target_size, bbox, state)
        if mask is None:
            return None
        width, height = target_size
        mask = self.cv2.resize(mask, (width, height), interpolation=self.cv2.INTER_NEAREST)
        mask = (mask > 0).astype(self.np.uint8)
        if width >= 8 and height >= 8:
            kernel_size = 3 if min(width, height) < 48 else 5
            kernel = self.np.ones((kernel_size, kernel_size), dtype=self.np.uint8)
            mask = self.cv2.morphologyEx(mask, self.cv2.MORPH_CLOSE, kernel)
            alpha_mask = mask
            if self.roi_mask_erode_pixels > 0:
                erode_size = 2 * self.roi_mask_erode_pixels + 1
                erode_kernel = self.np.ones((erode_size, erode_size), dtype=self.np.uint8)
                alpha_mask = self.cv2.erode(alpha_mask, erode_kernel, iterations=1)
                if not alpha_mask.any():
                    alpha_mask = mask
            alpha = (alpha_mask * 255).astype(self.np.uint8)
            blur_size = self._odd_kernel_size(self.roi_mask_feather_pixels, min(width, height))
            alpha = self.cv2.GaussianBlur(alpha, (blur_size, blur_size), 0)
            alpha = (alpha.astype(self.np.float32) / 255.0).clip(0.0, 1.0)
        else:
            alpha = mask.astype(self.np.float32)
        alpha = self._apply_content_validity_alpha(alpha, mask, resized_roi)
        return self._suppress_border_black_alpha(alpha, mask, resized_roi)

    def _apply_content_validity_alpha(self, alpha: ImageArray, mask: ImageArray, resized_roi: ImageArray) -> ImageArray:
        """Suppress large or boundary-connected dark components in one label lookup."""
        if not self.roi_black_border_suppression or self.roi_black_threshold <= 0:
            return alpha
        if resized_roi.size == 0 or not mask.any():
            return alpha
        gray = self.cv2.cvtColor(resized_roi, self.cv2.COLOR_BGR2GRAY)
        dark = ((gray <= self.roi_black_threshold) & (mask > 0)).astype(self.np.uint8)
        if not dark.any():
            return alpha
        _, labels, stats, _ = self.cv2.connectedComponentsWithStats(dark, connectivity=8)
        mask_area = max(1, int(mask.sum()))
        minimum_large_area = max(8, int(mask_area * self.roi_internal_black_min_area_ratio))
        h, w = mask.shape
        x, y = stats[:, self.cv2.CC_STAT_LEFT], stats[:, self.cv2.CC_STAT_TOP]
        cw, ch = stats[:, self.cv2.CC_STAT_WIDTH], stats[:, self.cv2.CC_STAT_HEIGHT]
        invalid_labels = (
            (x == 0)
            | (y == 0)
            | (x + cw >= w)
            | (y + ch >= h)
            | (stats[:, self.cv2.CC_STAT_AREA] >= minimum_large_area)
        )
        invalid_labels[0] = False
        invalid = invalid_labels[labels].astype(self.np.uint8)
        if not invalid.any():
            return alpha
        if self.roi_content_mask_dilate_pixels > 0:
            size = 2 * self.roi_content_mask_dilate_pixels + 1
            kernel = self.np.ones((size, size), dtype=self.np.uint8)
            invalid = self.cv2.dilate(invalid, kernel, iterations=1)
        blur_size = self._odd_kernel_size(max(3, self.roi_mask_feather_pixels), min(h, w))
        invalid_soft = self.cv2.GaussianBlur((invalid * 255).astype(self.np.uint8), (blur_size, blur_size), 0)
        invalid_soft = (invalid_soft.astype(self.np.float32) / 255.0).clip(0.0, 1.0)
        return (alpha * (1.0 - invalid_soft)).clip(0.0, 1.0)

    def _suppress_border_black_alpha(self, alpha: ImageArray, mask: ImageArray, resized_roi: ImageArray) -> ImageArray:
        if not self.roi_black_border_suppression or self.roi_black_threshold <= 0:
            return alpha
        if resized_roi.size == 0 or not mask.any():
            return alpha
        gray = self.cv2.cvtColor(resized_roi, self.cv2.COLOR_BGR2GRAY)
        black = ((gray <= self.roi_black_threshold) & (mask > 0)).astype(self.np.uint8)
        if not black.any():
            return alpha
        component_count, labels = self.cv2.connectedComponents(black, connectivity=8)
        if component_count <= 1:
            return alpha
        border_labels = set(labels[0, :].tolist())
        border_labels.update(labels[-1, :].tolist())
        border_labels.update(labels[:, 0].tolist())
        border_labels.update(labels[:, -1].tolist())
        border_labels.discard(0)
        if not border_labels:
            return alpha
        suppressed_black = self.np.isin(labels, list(border_labels)).astype(self.np.uint8)
        if self.roi_internal_black_suppression:
            minimum_area = max(4, int(mask.sum() * self.roi_internal_black_min_area_ratio))
            for label in range(1, component_count):
                if label in border_labels:
                    continue
                component = labels == label
                if int(component.sum()) >= minimum_area:
                    suppressed_black[component] = 1
        if self.roi_content_mask_dilate_pixels > 0:
            size = 2 * self.roi_content_mask_dilate_pixels + 1
            kernel = self.np.ones((size, size), dtype=self.np.uint8)
            suppressed_black = self.cv2.dilate(suppressed_black, kernel, iterations=1)
        suppression = self.cv2.GaussianBlur((suppressed_black * 255).astype(self.np.uint8), (5, 5), 0)
        suppression = (suppression.astype(self.np.float32) / 255.0).clip(0.0, 1.0)
        suppressed = alpha.copy()
        suppressed *= 1.0 - suppression
        return suppressed.clip(0.0, 1.0)

    @staticmethod
    def _odd_kernel_size(configured: int, min_dimension: int) -> int:
        if configured <= 0:
            configured = 3 if min_dimension < 48 else 5
        configured = max(3, min(int(configured), max(3, int(min_dimension) | 1)))
        if configured % 2 == 0:
            configured += 1
        return configured

    def _decode_roi_mask(
        self,
        item: dict,
        target_size: tuple[int, int],
        bbox: tuple[int, int, int, int],
        state: DeviceState | None = None,
    ) -> ImageArray | None:
        rle = item.get("mask_rle")
        if isinstance(rle, dict):
            decoded = self._decode_compact_rle(rle)
            if decoded is not None:
                return decoded
        cached_mask = item.get("_cached_mask")
        if cached_mask is not None:
            return cached_mask
        if state is not None:
            cached = self._track_cache_for_item(state, item)
            if cached is not None and cached.mask is not None:
                return cached.mask
        width, height = target_size
        if width > 0 and height > 0:
            return self.np.ones((height, width), dtype=self.np.uint8)
        return None

    def _cache_roi_patch(
        self,
        state: DeviceState,
        frame_idx: int,
        roi_idx: int,
        item: dict,
        bbox: tuple[int, int, int, int],
        roi: ImageArray,
    ) -> None:
        track_key = self._track_key(item)
        if track_key is None:
            return
        previous = state.roi_track_cache.get(track_key)
        mask = None
        rle = item.get("mask_rle")
        if isinstance(rle, dict):
            mask = self._decode_compact_rle(rle)
        if mask is None:
            cached = state.roi_track_cache.get(track_key)
            if cached is not None:
                mask = cached.mask
        if mask is None:
            x1, y1, x2, y2 = bbox
            mask = self.np.ones((max(1, y2 - y1), max(1, x2 - x1)), dtype=self.np.uint8)
        state.roi_track_cache[track_key] = ROITrackCache(
            image=roi.copy(),
            mask=mask.copy() if hasattr(mask, "copy") else mask,
            bbox=[int(value) for value in bbox],
            frame_idx=int(frame_idx),
            roi_idx=int(roi_idx),
            track_id=item.get("track_id", roi_idx),
            previous_bbox=previous.bbox if previous is not None else None,
            previous_frame_idx=previous.frame_idx if previous is not None else None,
        )

    def _track_cache_for_item(self, state: DeviceState, item: dict) -> ROITrackCache | None:
        track_key = self._track_key(item)
        if track_key is None:
            return None
        return state.roi_track_cache.get(track_key)

    @staticmethod
    def _track_key(item: dict) -> str | None:
        track_id = item.get("track_id", item.get("roi_idx"))
        if track_id in (None, ""):
            return None
        return str(track_id)

    def _roi_proxy_alpha_scale(self, age_frames: int) -> float:
        if age_frames <= 0:
            return 1.0
        fade = 1.0 - min(1.0, float(age_frames) / float(self.roi_proxy_fade_frames))
        return max(self.roi_proxy_min_alpha, fade)

    def _proxy_bbox_for_missing_roi(
        self,
        cached: ROITrackCache,
        metadata_bbox: tuple[int, int, int, int],
        age_frames: int,
    ) -> tuple[int, int, int, int]:
        if not self.roi_proxy_motion_compensation or cached.previous_bbox is None or cached.previous_frame_idx is None:
            return metadata_bbox
        previous = cached.previous_bbox
        current = cached.bbox
        metadata_cx = (metadata_bbox[0] + metadata_bbox[2]) / 2.0
        metadata_cy = (metadata_bbox[1] + metadata_bbox[3]) / 2.0
        cached_cx = (current[0] + current[2]) / 2.0
        cached_cy = (current[1] + current[3]) / 2.0
        cached_w = max(1, int(current[2]) - int(current[0]))
        cached_h = max(1, int(current[3]) - int(current[1]))
        metadata_moved = (
            abs(metadata_cx - cached_cx) > cached_w * 0.25 or abs(metadata_cy - cached_cy) > cached_h * 0.25
        )
        if metadata_moved:
            return metadata_bbox
        dt = max(1, int(cached.frame_idx) - int(cached.previous_frame_idx))
        dx = ((current[0] + current[2]) - (previous[0] + previous[2])) / 2.0 / dt
        dy = ((current[1] + current[3]) - (previous[1] + previous[3])) / 2.0 / dt
        max_dx = cached_w * self.roi_proxy_max_motion_scale
        max_dy = cached_h * self.roi_proxy_max_motion_scale
        shift_x = float(self.np.clip(dx * age_frames, -max_dx, max_dx))
        shift_y = float(self.np.clip(dy * age_frames, -max_dy, max_dy))
        x1, y1, x2, y2 = metadata_bbox
        predicted = (
            int(round(x1 + shift_x)),
            int(round(y1 + shift_y)),
            int(round(x2 + shift_x)),
            int(round(y2 + shift_y)),
        )
        return self._clip_bbox(list(predicted)) or metadata_bbox

    def _decode_compact_rle(self, rle: dict) -> ImageArray | None:
        """Decode metadata masks with bounded storage and historical pad/clip behavior."""
        from common.masks import expand_binary_runs

        size = rle.get("size") or rle.get("shape")
        data = rle.get("data")
        if (
            rle.get("encoding") != "varint-base64"
            or not isinstance(data, str)
            or not isinstance(size, list)
            or len(size) != 2
        ):
            return None
        counts = self._decode_varint_counts(base64.b64decode(data))
        height, width = int(size[0]), int(size[1])
        order = str(rle.get("order", "F")).upper()
        return expand_binary_runs(counts, height, width, order=order)

    @staticmethod
    def _decode_varint_counts(data: bytes) -> list[int]:
        counts = []
        value = 0
        shift = 0
        for byte in data:
            value |= (byte & 0x7F) << shift
            if byte & 0x80:
                shift += 7
                continue
            counts.append(value)
            value = 0
            shift = 0
        if shift:
            raise ValueError("truncated varint RLE")
        return counts

    @staticmethod
    def _read_metadata(path: Path) -> dict:
        return decode_metadata(path.read_bytes())

    @staticmethod
    def _frame_idx_from_path(path: Path) -> int | None:
        try:
            return int(path.stem)
        except ValueError:
            return None

    @staticmethod
    def _append_json(path: Path, value: dict) -> None:
        with path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(value, ensure_ascii=True, separators=(",", ":")) + "\n")

    def close(self, wait: bool = True, write_videos: bool = True) -> None:
        if self.closed:
            return
        self.closed = True
        if self.thread is not None:
            if wait:
                self.work_queue.join()
                if write_videos:
                    self.finalize_videos()
            self.work_queue.put(None)
            if wait:
                self.work_queue.join()
                self.thread.join(timeout=10)
        if self.worker is not None:
            self.worker.close()
