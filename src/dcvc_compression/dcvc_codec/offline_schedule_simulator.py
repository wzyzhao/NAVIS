import base64
import json
import re
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from common.encoding_layout import metric_stream_directory, write_candidate_manifest
from common.masks import Backend, binary_run_lengths
from dcvc_compression.dcvc_codec.bundle_io import ensure_dir
from dcvc_compression.dcvc_codec.dcvc_pipeline import (
    find_reconstructed_yuv,
    iter_yuv420_frames,
    strip_dcvc_sequence_suffix,
)
from dcvc_compression.dcvc_codec.side_information import encode_binary_mask, encode_metadata

DEFAULT_METADATA_MAX_ROIS = 0
BACKGROUND_LAYER = "background_keyframe_summary"


def compute_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    mse = np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)
    if mse == 0:
        return 100.0
    return float(10 * np.log10(255 * 255 / mse))


def load_background_reconstructions(
    background_yuv: Path,
    width: int,
    height: int,
    expected_frames: int,
) -> list[np.ndarray]:
    frames = []
    for frame in iter_yuv420_frames(background_yuv, width, height):
        frames.append(frame)
        if len(frames) == expected_frames:
            break
    if len(frames) < expected_frames:
        raise RuntimeError(
            f"Expected {expected_frames} decoded background frames, got {len(frames)} from {background_yuv}"
        )
    return frames


def source_sequence_from_background_sequence(sequence_name: str) -> str:
    base_name = strip_dcvc_sequence_suffix(sequence_name)
    return re.sub(r"_(bg_lf\d*|fg_hf)$", "", base_name)


def source_sequence_from_layer_sequence(sequence_name: str) -> str:
    return source_sequence_from_background_sequence(sequence_name)


def required_background_key_from_sequence(sequence_name: str) -> tuple[str, int] | None:
    base_name = strip_dcvc_sequence_suffix(sequence_name)
    match = re.match(r"(?P<sequence>.+)_bg_keyframe_f(?P<frame>\d{6})$", base_name)
    if match is None:
        return None
    return match.group("sequence"), int(match.group("frame"))


def encoded_mask_bits(image: np.ndarray) -> int:
    return len(encode_binary_mask(image)) * 8


def side_metadata_bits(bundle) -> int:
    payload = build_metadata_payload(
        device_id=bundle.sequence,
        sequence=bundle.sequence,
        frame_idx=getattr(bundle, "frame_idx", bundle.frame_id),
        frame_id=bundle.frame_id,
        rois=_bundle_rois_to_json(bundle.selected_rois),
        motion_score=float(getattr(bundle, "motion_score", 0.0)),
        background_change_score=float(getattr(bundle, "background_change_score", 0.0)),
        uncertainty=float(getattr(bundle, "uncertainty", 0.0)),
        background_version=f"{bundle.sequence}:bg:{bundle.frame_id}",
    )
    return len(encode_metadata(payload)) * 8


def build_metadata_payload(
    device_id: str,
    sequence: str,
    frame_idx: int,
    frame_id: int,
    rois: list[dict],
    motion_score: float,
    background_change_score: float,
    uncertainty: float,
    background_version: str,
) -> dict:
    rois = select_metadata_rois(rois)
    primary_roi = rois[0] if rois else {}
    return {
        "device_id": device_id,
        "timestamp": float(frame_idx),
        "sequence": sequence,
        "frame_idx": int(frame_idx),
        "frame_id": int(frame_id),
        "track_id": primary_roi.get("track_id", primary_roi.get("roi_idx", -1)),
        "bbox": primary_roi.get("bbox", [0, 0, 0, 0]),
        "motion_score": float(motion_score),
        "background_change_score": float(background_change_score),
        "uncertainty": float(uncertainty),
        "background_version": background_version,
        "foreground_count": len(rois),
        "rois": rois,
    }


def _bundle_rois_to_json(rois) -> list[dict]:
    return [
        {
            "roi_idx": getattr(roi, "roi_idx", idx),
            "track_id": roi.track_id,
            "bbox": roi.bbox,
            "roi_bbox": roi.roi_bbox,
            "class_id": roi.class_id,
            "category": getattr(roi, "category", "other"),
            "motion_score": float(getattr(roi, "motion_score", 0.0)),
            "uncertainty": float(getattr(roi, "uncertainty", 0.0)),
            "mask_pixels": getattr(roi, "mask_pixels", None),
            "mask_coverage": getattr(roi, "mask_coverage", None),
            "edge_mask_coverage": getattr(roi, "edge_mask_coverage", None),
            "masked_black_ratio": getattr(roi, "masked_black_ratio", None),
        }
        for idx, roi in enumerate(rois)
    ]


def select_metadata_rois(rois: list[dict], max_rois: int = DEFAULT_METADATA_MAX_ROIS) -> list[dict]:
    if max_rois <= 0:
        return list(rois)
    return sorted(
        rois,
        key=lambda item: (
            float(item.get("motion_score", 0.0) or 0.0),
            -int(item.get("roi_idx", 0) or 0),
        ),
        reverse=True,
    )[:max_rois]


def load_single_roi_reconstruction(
    fg_stream_dir: Path,
    point: dict,
    record: dict,
    yuv_cache: dict[str, Path | None],
) -> np.ndarray:
    qp_i = point["raw_metrics"].get("qp_i", point["raw_metrics"].get("qp_p"))
    cache_key = f"{point['sequence']}|{qp_i}"
    if cache_key not in yuv_cache:
        yuv_cache[cache_key] = find_reconstructed_yuv(fg_stream_dir, point["sequence"], qp_i)
    recon_yuv = yuv_cache[cache_key]
    if recon_yuv is None:
        raise FileNotFoundError(f"Missing decoded foreground ROI YUV for {point['sequence']} q{qp_i}")
    frames = load_background_reconstructions(
        recon_yuv,
        int(record["padded_width"]),
        int(record["padded_height"]),
        1,
    )
    roi_height = int(record["height"])
    roi_width = int(record["width"])
    return frames[0][:roi_height, :roi_width]


def write_metadata_payload(path: Path, payload: dict) -> Path:
    ensure_dir(path.parent)
    path.write_bytes(encode_metadata(payload))
    return path


def write_binary_payload(path: Path, payload: bytes) -> Path:
    ensure_dir(path.parent)
    path.write_bytes(payload)
    return path


def encode_background_dcvc_sequence_payload(
    stream_path: Path,
    reconstructed_background: np.ndarray,
    frame_idx: int,
) -> tuple[bytes, np.ndarray]:
    frame_payloads = read_background_frame_payloads(stream_path, int(frame_idx))
    return frame_payloads[int(frame_idx) - 1], reconstructed_background.copy()


def encode_background_dcvc_frame_payload(
    stream_path: Path,
    reconstructed_background: np.ndarray,
) -> tuple[bytes, np.ndarray]:
    frame_payloads = read_background_frame_payloads(stream_path, 1)
    return frame_payloads[0], reconstructed_background.copy()


def relative_payload_path(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")
    except ValueError:
        return str(path.resolve())


def find_stream_bin(stream_dir: Path, sequence: str, qp) -> Path | None:
    if qp is None:
        return None
    if stream_dir.name == "encoded":
        stream_dir = metric_stream_directory(stream_dir, sequence)
    direct = stream_dir / f"{sequence}_q{qp}.bin"
    if direct.exists():
        return direct
    matches = sorted(stream_dir.rglob(f"{sequence}*q{qp}.bin"))
    return matches[0] if matches else None


def find_foreground_stream_bin(
    stream_dir: Path,
    sequence: str,
    qp,
    cache: dict[str, Path | None],
) -> Path | None:
    if qp is None:
        return None
    if stream_dir.name == "encoded":
        stream_dir = metric_stream_directory(stream_dir, sequence)
    file_name = f"{sequence}_q{qp}.bin"
    if file_name not in cache:
        direct = stream_dir / file_name
        cache[file_name] = direct if direct.exists() else find_stream_bin(stream_dir, sequence, qp)
    return cache[file_name]


def read_stream_slice(path: Path | None, offset: int, size: int, fallback_header: dict) -> bytes:
    if path is not None and path.exists() and size > 0:
        with path.open("rb") as f:
            f.seek(offset)
            payload = f.read(size)
        if len(payload) < size:
            payload += bytes(size - len(payload))
        return payload
    encoded = json.dumps(fallback_header, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    return encoded[:size] if len(encoded) >= size else encoded + bytes(size - len(encoded))


def read_background_frame_payloads(stream_path: Path, expected_frames: int) -> list[bytes]:
    data = stream_path.read_bytes()
    units = parse_dcvc_nal_units(data)
    frames: list[bytes] = []
    pending_headers = bytearray()
    for nal_type, payload in units:
        if nal_type == 0:
            pending_headers.extend(payload)
            continue
        if nal_type not in {1, 2}:
            raise ValueError(f"unsupported DCVC NAL type {nal_type} in {stream_path}")
        frame_payload = bytes(pending_headers) + payload
        pending_headers.clear()
        frames.append(frame_payload)
    if len(frames) < expected_frames:
        raise RuntimeError(
            f"Expected at least {expected_frames} background frames, got {len(frames)} from {stream_path}"
        )
    return frames[:expected_frames]


def parse_dcvc_nal_units(data: bytes) -> list[tuple[int, bytes]]:
    units = []
    offset = 0
    size = len(data)
    while offset < size:
        start = offset
        flag = data[offset]
        offset += 1
        nal_type = flag >> 4
        if nal_type == 0:
            offset = _skip_uint_adaptive(data, offset)
            offset = _skip_uint_adaptive(data, offset)
            offset += 1
        elif nal_type in {1, 2}:
            offset += 1
            stream_length, offset = _read_uint_adaptive_from_bytes(data, offset)
            offset += stream_length
        else:
            raise ValueError(f"{nal_type} is not a valid NalType at byte offset {start}")
        if offset > size:
            raise EOFError(f"truncated DCVC stream at byte offset {start}")
        units.append((nal_type, data[start:offset]))
    return units


def _skip_uint_adaptive(data: bytes, offset: int) -> int:
    _, offset = _read_uint_adaptive_from_bytes(data, offset)
    return offset


def _read_uint_adaptive_from_bytes(data: bytes, offset: int) -> tuple[int, int]:
    if offset >= len(data):
        raise EOFError("truncated adaptive integer")
    a3 = data[offset]
    offset += 1
    if (a3 >> 7) == 0:
        return a3, offset
    if offset >= len(data):
        raise EOFError("truncated adaptive integer")
    a2 = data[offset]
    offset += 1
    if (a3 >> 6) == 0x02:
        return ((a3 & 0x3F) << 8) + a2, offset
    if offset + 2 > len(data):
        raise EOFError("truncated adaptive integer")
    a1 = data[offset]
    a0 = data[offset + 1]
    offset += 2
    return ((a3 & 0x3F) << 24) + (a2 << 16) + (a1 << 8) + a0, offset


def metadata_roi_from_record(record: dict, include_mask: bool = True) -> dict:
    item = {
        "roi_idx": int(record.get("roi_idx", 0)),
        "track_id": record.get("track_id", record.get("roi_idx")),
        "bbox": record.get("bbox", record.get("roi_bbox")),
        "roi_bbox": record.get("roi_bbox", record.get("bbox")),
        "class_id": record.get("class_id"),
        "category": record.get("category", "other"),
        "motion_score": float(record.get("motion_score", 0.0) or 0.0),
        "uncertainty": float(record.get("uncertainty", 0.0) or 0.0),
    }
    for key in ("confidence", "continuity_score"):
        if record.get(key) is not None:
            item[key] = float(record.get(key) or 0.0)
    for key in ("mask_pixels", "mask_coverage", "edge_mask_coverage", "masked_black_ratio"):
        if record.get(key) is not None:
            item[key] = record.get(key)
    mask = record.get("mask")
    if include_mask and mask is not None:
        item["mask_rle"] = binary_mask_to_metadata_rle(mask)
    return item


def _metadata_mask_track_key(record: dict) -> str:
    track_id = record.get("track_id")
    if track_id is not None and str(track_id) != "":
        return f"track:{track_id}"
    return f"roi:{record.get('sequence', '')}:{record.get('roi_idx', 0)}"


def should_include_metadata_mask(
    record: dict,
    frame_idx: int,
    mask_state: dict[str, int],
    mode: str,
    refresh_interval: int,
    motion_threshold: float,
    uncertainty_threshold: float,
) -> bool:
    if mode == "none":
        return False
    if mode == "all":
        return True
    key = _metadata_mask_track_key(record)
    last_frame = mask_state.get(key)
    if last_frame is None:
        mask_state[key] = int(frame_idx)
        return True
    motion = float(record.get("motion_score", 0.0) or 0.0)
    uncertainty = float(record.get("uncertainty", 0.0) or 0.0)
    due = int(refresh_interval or 0) > 0 and int(frame_idx) - int(last_frame) >= int(refresh_interval)
    important = motion >= float(motion_threshold) or uncertainty >= float(uncertainty_threshold)
    if due or important:
        mask_state[key] = int(frame_idx)
        return True
    return False


def binary_mask_to_metadata_rle(mask: np.ndarray, *, backend: Backend = "numpy") -> dict:
    """Encode a 2D mask using the existing C-order, zero-first varint wire format."""
    counts = binary_run_lengths(mask, backend=backend)
    height, width = mask.shape
    encoded_counts = b"".join(_encode_varint(count) for count in counts)
    return {
        "format": "rle",
        "order": "C",
        "size": [int(height), int(width)],
        "encoding": "varint-base64",
        "data": base64.b64encode(encoded_counts).decode("ascii"),
    }


def _encode_varint(value: int) -> bytes:
    value = int(value)
    if value < 0:
        raise ValueError("varint cannot encode negative values")
    output = bytearray()
    while value >= 0x80:
        output.append((value & 0x7F) | 0x80)
        value >>= 7
    output.append(value)
    return bytes(output)


def export_transport_candidate_manifest(
    grouped: dict[str, list],
    bg_points: list[dict],
    required_bg_points: list[dict] | None,
    fg_points: list[dict],
    foreground_export: dict,
    bg_stream_dir: Path,
    required_bg_stream_dir: Path | None,
    fg_stream_dir: Path,
    output_root: Path,
    rate_plan: list[dict],
    required_background_interval_derived_frames: int = 0,
    required_background_interval_seconds: float = 0.0,
    background_summary_change_threshold: float = 0.0,
    background_summary_min_interval_frames: int = 300,
    metadata_roi_mask_mode: str = "track_refresh",
    metadata_roi_mask_refresh_interval: int = 15,
    metadata_roi_mask_motion_threshold: float = 0.65,
    metadata_roi_mask_uncertainty_threshold: float = 0.65,
) -> Path:
    payload_root = ensure_dir(output_root / "payloads")
    bg_points_by_sequence = defaultdict(dict)
    required_bg_points_by_frame = defaultdict(dict)
    fg_points_by_roi_sequence = defaultdict(dict)
    roi_records_by_frame = defaultdict(list)
    for point in bg_points:
        bg_points_by_sequence[source_sequence_from_layer_sequence(point["sequence"])][point["rate_idx"]] = point
    for point in required_bg_points or []:
        key = required_background_key_from_sequence(point["sequence"])
        if key is not None:
            required_bg_points_by_frame[key][point["rate_idx"]] = point
    for point in fg_points:
        fg_points_by_roi_sequence[strip_dcvc_sequence_suffix(point["sequence"])][point["rate_idx"]] = point
    for record in foreground_export["roi_records"]:
        roi_records_by_frame[(record["sequence"], record["frame_idx"])].append(record)

    frames = []
    foreground_stream_cache: dict[str, Path | None] = {}
    metadata_mask_state: dict[str, int] = {}
    metadata_roi_mask_mode = str(metadata_roi_mask_mode or "track_refresh")
    metadata_roi_mask_refresh_interval = max(0, int(metadata_roi_mask_refresh_interval or 0))
    plan = rate_plan[0]
    for base_sequence, bundles in grouped.items():
        bg_point = bg_points_by_sequence.get(base_sequence, {}).get(plan["background_rate_idx"])
        if bg_point is None:
            raise RuntimeError(f"Missing background DCVC point for {base_sequence} rate {plan['background_rate_idx']}")
        bg_qp_i = bg_point["raw_metrics"].get("qp_i", bg_point["raw_metrics"].get("qp_p"))
        bg_stream_path = find_stream_bin(bg_stream_dir, bg_point["sequence"], bg_qp_i)
        if bg_stream_path is None:
            raise FileNotFoundError(f"Missing background DCVC stream for {bg_point['sequence']} q{bg_qp_i}")
        bg_recon_yuv = find_reconstructed_yuv(bg_stream_dir, bg_point["sequence"], bg_qp_i)
        if bg_recon_yuv is None:
            raise FileNotFoundError(f"Missing decoded background YUV for {bg_point['sequence']} q{bg_qp_i}")
        bg_frames = load_background_reconstructions(
            bg_recon_yuv,
            bundles[0].original_frame.shape[1],
            bundles[0].original_frame.shape[0],
            len(bundles),
        )
        required_background_interval = int(required_background_interval_derived_frames or 0)
        background_summary_min_interval = max(0, int(background_summary_min_interval_frames or 0))
        last_optional_summary_frame_idx: int | None = None
        for frame_idx, bundle in enumerate(bundles, start=1):
            roi_records = roi_records_by_frame.get((base_sequence, frame_idx), [])
            metadata_payload = build_metadata_payload(
                device_id=base_sequence,
                sequence=base_sequence,
                frame_idx=frame_idx,
                frame_id=bundle.frame_id,
                rois=[
                    metadata_roi_from_record(
                        record,
                        include_mask=should_include_metadata_mask(
                            record,
                            frame_idx,
                            metadata_mask_state,
                            metadata_roi_mask_mode,
                            metadata_roi_mask_refresh_interval,
                            metadata_roi_mask_motion_threshold,
                            metadata_roi_mask_uncertainty_threshold,
                        ),
                    )
                    for record in roi_records
                ],
                motion_score=float(getattr(bundle, "motion_score", 0.0)),
                background_change_score=float(getattr(bundle, "background_change_score", 0.0)),
                uncertainty=float(getattr(bundle, "uncertainty", 0.0)),
                background_version=f"{base_sequence}:bg:{plan['background_rate_idx']}:f{frame_idx:06d}",
            )
            metadata_path = payload_root / base_sequence / "metadata" / f"frame_{frame_idx:06d}.bin"
            write_metadata_payload(metadata_path, metadata_payload)
            items = [
                {
                    "layer": "metadata",
                    "bytes": metadata_path.stat().st_size,
                    "frame_idx": frame_idx,
                    "priority": 1.0,
                    "motion_score": float(getattr(bundle, "motion_score", 0.0)),
                    "background_change_score": float(getattr(bundle, "background_change_score", 0.0)),
                    "uncertainty": float(getattr(bundle, "uncertainty", 0.0)),
                    "payload_path": relative_payload_path(metadata_path, output_root),
                }
            ]

            if frame_idx == 1:
                background_path = (
                    payload_root
                    / base_sequence
                    / "background"
                    / f"frame_{frame_idx:06d}_bg_keyframe_rate{int(bg_point['rate_idx'])}.bin"
                )
                background_payload, _reconstructed_background = encode_background_dcvc_sequence_payload(
                    bg_stream_path,
                    bg_frames[frame_idx - 1],
                    frame_idx,
                )
                write_binary_payload(background_path, background_payload)
                init_item = {
                    "layer": BACKGROUND_LAYER,
                    "bytes": background_path.stat().st_size,
                    "frame_idx": frame_idx,
                    "rate_idx": int(bg_point["rate_idx"]),
                    "priority": 2.0,
                    "payload_path": relative_payload_path(background_path, output_root),
                    "background": {
                        "version": f"{base_sequence}:bg:{plan['background_rate_idx']}:f{frame_idx:06d}",
                        "mode": "required",
                        "reference_frame_idx": None,
                    },
                }
                items.insert(0, init_item)
            background_change_score = float(getattr(bundle, "background_change_score", 0.0) or 0.0)
            required = bool(required_background_interval > 0 and frame_idx % required_background_interval == 0)
            change_due = (
                background_summary_change_threshold > 0
                and background_change_score >= background_summary_change_threshold
            )
            if change_due and background_summary_min_interval > 0 and last_optional_summary_frame_idx is not None:
                change_due = frame_idx - last_optional_summary_frame_idx >= background_summary_min_interval
            if frame_idx != 1 and (required or change_due):
                required_point = required_bg_points_by_frame.get((base_sequence, frame_idx), {}).get(
                    plan["background_rate_idx"]
                )
                if required_point is None:
                    raise RuntimeError(
                        f"Missing background keyframe DCVC point for {base_sequence} "
                        f"frame {frame_idx} rate {plan['background_rate_idx']}. "
                        "Use run_level=refresh_full after changing background keyframe-summary settings."
                    )
                required_qp_i = required_point["raw_metrics"].get("qp_i", required_point["raw_metrics"].get("qp_p"))
                required_stream_path = find_stream_bin(
                    required_bg_stream_dir, required_point["sequence"], required_qp_i
                )
                if required_stream_path is None:
                    raise FileNotFoundError(
                        f"Missing background keyframe DCVC stream for {required_point['sequence']} q{required_qp_i}"
                    )
                required_recon_yuv = find_reconstructed_yuv(
                    required_bg_stream_dir, required_point["sequence"], required_qp_i
                )
                if required_recon_yuv is not None:
                    required_recon_frames = load_background_reconstructions(
                        required_recon_yuv,
                        bundles[0].original_frame.shape[1],
                        bundles[0].original_frame.shape[0],
                        1,
                    )
                    reconstructed_background = required_recon_frames[0]
                else:
                    reconstructed_background = bg_frames[frame_idx - 1]
                background_payload, reconstructed_background = encode_background_dcvc_frame_payload(
                    required_stream_path,
                    reconstructed_background,
                )
                background_stats = {
                    "reference_frame_idx": None,
                    "summary": None
                    if required
                    else {
                        "trigger": "background_change",
                        "change_score": background_change_score,
                        "min_interval_frames": background_summary_min_interval,
                    },
                }
                background_path = (
                    payload_root
                    / base_sequence
                    / "background"
                    / f"frame_{frame_idx:06d}_bg_keyframe_rate{int(bg_point['rate_idx'])}.bin"
                )
                write_binary_payload(background_path, background_payload)
                summary_item = {
                    "layer": BACKGROUND_LAYER,
                    "bytes": background_path.stat().st_size,
                    "frame_idx": frame_idx,
                    "rate_idx": int(bg_point["rate_idx"]),
                    "priority": 1.5 if required else max(0.2, min(1.0, background_change_score)),
                    "payload_path": relative_payload_path(background_path, output_root),
                    "background_change_score": background_change_score,
                    "background": {
                        "version": f"{base_sequence}:bg:{plan['background_rate_idx']}:f{frame_idx:06d}",
                        "mode": "required" if required else "scheduled",
                        **background_stats,
                    },
                }
                items.append(summary_item)
                if not required:
                    last_optional_summary_frame_idx = frame_idx

            for record in roi_records:
                rate_points = fg_points_by_roi_sequence.get(record["roi_sequence"], {})
                for rate_idx, fg_point in sorted(rate_points.items()):
                    roi_qp = fg_point["raw_metrics"].get("qp_i", fg_point["raw_metrics"].get("qp_p"))
                    stream_path = find_foreground_stream_bin(
                        fg_stream_dir,
                        fg_point["sequence"],
                        roi_qp,
                        foreground_stream_cache,
                    )
                    if stream_path is None:
                        continue
                    roi_pixels = int(record["padded_width"]) * int(record["padded_height"])
                    payload_bytes = max(
                        1,
                        int((float(fg_point["raw_metrics"]["frame_bpp"][0]) * roi_pixels + 7) // 8),
                    )
                    payload = read_stream_slice(
                        stream_path,
                        0,
                        payload_bytes,
                        {"layer": "roi_patch", "sequence": base_sequence, "frame_idx": frame_idx},
                    )
                    roi_path = (
                        payload_root
                        / base_sequence
                        / "roi"
                        / f"frame_{frame_idx:06d}_roi_{int(record['roi_idx']):03d}_rate{int(rate_idx)}.bin"
                    )
                    write_binary_payload(roi_path, payload)
                    item = {
                        "layer": "roi_patch",
                        "bytes": roi_path.stat().st_size,
                        "frame_idx": frame_idx,
                        "roi_idx": int(record["roi_idx"]),
                        "rate_idx": int(rate_idx),
                        "quality_rank": int(rate_idx),
                        "candidate_group_id": f"{base_sequence}:{frame_idx}:roi:{int(record['roi_idx'])}",
                        "priority": 0.0,
                        "payload_path": relative_payload_path(roi_path, output_root),
                    }
                    item.update(metadata_roi_from_record(record, include_mask=False))
                    items.append(item)

            frames.append(
                {
                    "frame_idx": frame_idx,
                    "frame_id": bundle.frame_id,
                    "sequence": base_sequence,
                    "rate_idx": int(plan["rate_idx"]),
                    "items": items,
                }
            )

    manifest_path = output_root / "candidate_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "format": "dcvc_transport_candidates_v2",
                "selection_owner": "nbiot_transport",
                "background_policy": {
                    "layer": BACKGROUND_LAYER,
                    "required_interval_seconds": float(required_background_interval_seconds or 0.0),
                    "required_interval_derived_frames": int(required_background_interval_derived_frames or 0),
                    "change_threshold": float(background_summary_change_threshold or 0.0),
                    "summary_min_interval_frames": int(background_summary_min_interval_frames or 0),
                },
                "metadata_policy": {
                    "roi_mask_mode": metadata_roi_mask_mode,
                    "roi_mask_refresh_interval": metadata_roi_mask_refresh_interval,
                    "roi_mask_motion_threshold": float(metadata_roi_mask_motion_threshold),
                    "roi_mask_uncertainty_threshold": float(metadata_roi_mask_uncertainty_threshold),
                },
                "frames": frames,
            },
            ensure_ascii=True,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    indexed_path = write_candidate_manifest(output_root, json.loads(manifest_path.read_text(encoding="utf-8")))
    manifest_path.unlink()
    return indexed_path


def compose_layered_roi_results(
    grouped: dict[str, list],
    bg_points: list[dict],
    fg_points: list[dict],
    foreground_export: dict,
    bg_stream_dir: Path,
    fg_stream_dir: Path,
    rate_plan: list[dict],
) -> dict:
    points = []
    frame_metrics = []
    bg_points_by_sequence = defaultdict(dict)
    fg_points_by_roi_sequence = defaultdict(dict)
    roi_records_by_frame = defaultdict(list)
    foreground_yuv_cache: dict[str, Path | None] = {}

    for point in bg_points:
        bg_points_by_sequence[source_sequence_from_layer_sequence(point["sequence"])][point["rate_idx"]] = point
    for point in fg_points:
        fg_points_by_roi_sequence[strip_dcvc_sequence_suffix(point["sequence"])][point["rate_idx"]] = point
    for record in foreground_export["roi_records"]:
        roi_records_by_frame[(record["sequence"], record["frame_idx"])].append(record)

    for plan in rate_plan:
        for base_sequence, bundles in grouped.items():
            bg_point = bg_points_by_sequence.get(base_sequence, {}).get(plan["background_rate_idx"])
            if bg_point is None:
                raise RuntimeError(
                    f"Missing background DCVC point for {base_sequence} rate {plan['background_rate_idx']}"
                )

            bg_qp_i = bg_point["raw_metrics"].get("qp_i", bg_point["raw_metrics"].get("qp_p"))
            bg_recon_yuv = find_reconstructed_yuv(bg_stream_dir, bg_point["sequence"], bg_qp_i)
            if bg_recon_yuv is None:
                raise FileNotFoundError(f"Missing decoded background YUV for {bg_point['sequence']} q{bg_qp_i}")

            height = bundles[0].original_frame.shape[0]
            frame_width = bundles[0].original_frame.shape[1]

            psnr_values = []
            mask_bits_total = 0
            metadata_bits_total = 0
            background_bits_total = 0.0
            foreground_bits_total = 0.0
            scheduled_foreground_roi_count = 0
            confirmed_foreground_roi_count = 0
            confirmed_payload_bytes_total = 0
            confirmed_packet_count_total = 0
            frame_idx = 0
            background_frames = iter_yuv420_frames(bg_recon_yuv, frame_width, height)
            for frame_idx, (bundle, bg_frame) in enumerate(zip(bundles, background_frames), start=1):
                frame_pixels = bundle.original_frame.shape[0] * bundle.original_frame.shape[1]
                raw_background_bpp = bg_point["raw_metrics"]["frame_bpp"][frame_idx - 1]
                background_bytes = max(1, int((raw_background_bpp * frame_pixels + 7) // 8))
                foreground_bits = 0.0
                foreground_payload_bytes = 0
                background_bits = raw_background_bpp * frame_pixels
                roi_records = roi_records_by_frame.get((base_sequence, frame_idx), [])
                metadata_bits = side_metadata_bits(bundle)
                metadata_bytes = (metadata_bits + 7) // 8
                scheduled_roi_records = roi_records
                confirmed_roi_records = roi_records
                schedule_plan = None
                reconstructed = bg_frame.copy()
                scheduled_foreground_roi_count += len(scheduled_roi_records)
                confirmed_foreground_roi_count += len(confirmed_roi_records)

                scheduled_mask = np.zeros(bundle.mask.shape, dtype=np.uint8)
                for record in confirmed_roi_records:
                    foreground_rate_idx = int(record.get("scheduled_rate_idx", plan["foreground_rate_idx"]))
                    fg_point = fg_points_by_roi_sequence.get(record["roi_sequence"], {}).get(foreground_rate_idx)
                    if fg_point is None:
                        raise RuntimeError(
                            f"Missing foreground ROI DCVC point for {record['roi_sequence']} rate {foreground_rate_idx}"
                        )
                    decoded_roi = load_single_roi_reconstruction(
                        fg_stream_dir,
                        fg_point,
                        record,
                        foreground_yuv_cache,
                    )
                    x1, y1, x2, y2 = record["roi_bbox"]
                    mask = record["mask"]
                    if mask.shape[:2] != decoded_roi.shape[:2]:
                        mask = cv2.resize(
                            mask, (decoded_roi.shape[1], decoded_roi.shape[0]), interpolation=cv2.INTER_NEAREST
                        )
                    alpha = (mask > 0)[..., None]
                    reconstructed[y1:y2, x1:x2] = np.where(alpha, decoded_roi, reconstructed[y1:y2, x1:x2])
                    scheduled_mask[y1:y2, x1:x2] = np.where(mask > 0, 255, scheduled_mask[y1:y2, x1:x2])
                    roi_pixels = int(record["padded_width"]) * int(record["padded_height"])
                    roi_bits = float(fg_point["raw_metrics"]["frame_bpp"][0]) * roi_pixels
                    foreground_bits += roi_bits
                    foreground_payload_bytes += max(1, int((roi_bits + 7) // 8))

                foreground_bits_total += foreground_bits
                background_bits_total += background_bits
                confirmed_payload_bytes = metadata_bytes + background_bytes + foreground_payload_bytes
                confirmed_packet_count = 1 + 1 + len(confirmed_roi_records)
                confirmed_payload_bytes_total += confirmed_payload_bytes
                confirmed_packet_count_total += confirmed_packet_count
                confirmed_metadata = True
                confirmed_background_summary = True
                background_bpp = background_bits / frame_pixels
                foreground_bpp = foreground_bits / frame_pixels
                core_video_bpp = background_bpp + foreground_bpp
                mask_payload = encode_binary_mask(scheduled_mask) if confirmed_roi_records else b""
                mask_bits = len(mask_payload) * 8
                mask_bpp = mask_bits / frame_pixels
                metadata_bpp = metadata_bits / frame_pixels
                estimated_total_bpp = core_video_bpp + mask_bpp + metadata_bpp
                mask_bits_total += mask_bits
                metadata_bits_total += metadata_bits

                psnr = compute_psnr(bundle.original_frame, reconstructed)
                psnr_values.append(psnr)
                frame_metrics.append(
                    {
                        "sequence": base_sequence,
                        "background_sequence": bg_point["sequence"],
                        "foreground_mode": "roi_crop",
                        "rate_idx": plan["rate_idx"],
                        "frame_idx": frame_idx,
                        "background_rate_idx": plan["background_rate_idx"],
                        "foreground_rate_idx": plan["foreground_rate_idx"],
                        "mode": "full_quality_evaluation",
                        "link_budget_bytes": schedule_plan.budget_bytes if schedule_plan is not None else None,
                        "scheduled_payload_bytes": schedule_plan.used_bytes if schedule_plan is not None else None,
                        "confirmed_payload_bytes": confirmed_payload_bytes if schedule_plan is not None else None,
                        "confirmed_packet_count": confirmed_packet_count if schedule_plan is not None else None,
                        "confirmed_metadata": confirmed_metadata if schedule_plan is not None else None,
                        "background_summary_sent": (
                            getattr(schedule_plan, "background_summary_sent", True)
                            if schedule_plan is not None
                            else True
                        ),
                        "background_summary_confirmed": confirmed_background_summary
                        if schedule_plan is not None
                        else True,
                        "background_bpp": background_bpp,
                        "foreground_bpp": foreground_bpp,
                        "core_video_bpp": core_video_bpp,
                        "mask_bpp": mask_bpp,
                        "metadata_bpp": metadata_bpp,
                        "estimated_total_bpp": estimated_total_bpp,
                        "psnr": psnr,
                        "foreground_count_available": len(bundle.selected_rois),
                        "foreground_count_encoded": len(scheduled_roi_records),
                        "foreground_count_confirmed": len(confirmed_roi_records),
                    }
                )
            if frame_idx < len(bundles):
                raise RuntimeError(
                    f"Expected {len(bundles)} decoded background frames, got {frame_idx} from {bg_recon_yuv}"
                )

            frame_pixels = bundles[0].original_frame.shape[0] * bundles[0].original_frame.shape[1]
            foreground_bpp = foreground_bits_total / (frame_pixels * len(bundles))
            background_bpp = background_bits_total / (frame_pixels * len(bundles))
            core_video_bpp = background_bpp + foreground_bpp
            mask_bpp = mask_bits_total / (frame_pixels * len(bundles))
            metadata_bpp = metadata_bits_total / (frame_pixels * len(bundles))
            estimated_total_bpp = core_video_bpp + mask_bpp + metadata_bpp
            points.append(
                {
                    "dataset": bg_point["dataset"],
                    "sequence": base_sequence,
                    "background_sequence": bg_point["sequence"],
                    "foreground_mode": "roi_crop",
                    "rate_idx": plan["rate_idx"],
                    "background_rate_idx": plan["background_rate_idx"],
                    "foreground_rate_idx": plan["foreground_rate_idx"],
                    "avg_psnr": float(np.mean(psnr_values)),
                    "core_video_bpp": core_video_bpp,
                    "background_bpp": background_bpp,
                    "foreground_bpp": foreground_bpp,
                    "mask_bpp": mask_bpp,
                    "metadata_bpp": metadata_bpp,
                    "estimated_total_bpp": estimated_total_bpp,
                    "frame_count": len(bundles),
                    "foreground_roi_count": sum(
                        len(roi_records_by_frame.get((base_sequence, idx), [])) for idx in range(1, len(bundles) + 1)
                    ),
                    "scheduled_foreground_roi_count": scheduled_foreground_roi_count,
                    "confirmed_foreground_roi_count": confirmed_foreground_roi_count,
                    "confirmed_payload_bytes": confirmed_payload_bytes_total,
                    "confirmed_packet_count": confirmed_packet_count_total,
                    "background_qp_i": bg_point["raw_metrics"].get("qp_i"),
                    "background_qp_p": bg_point["raw_metrics"].get("qp_p"),
                }
            )

    return {
        "points": points,
        "frame_metrics": frame_metrics,
    }
