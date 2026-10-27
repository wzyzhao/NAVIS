from collections.abc import Iterable
from pathlib import Path

from common.encoding_layout import load_artifact_json
from nbiot_transport.core.packetizer import Packet, build_fallback_payload, envelope_payload, fragment_payload

BACKGROUND_LAYER = "background_keyframe_summary"


def _background_item(item: dict) -> dict:
    value = item.get("background")
    return dict(value) if isinstance(value, dict) else {}


def build_payload_packets(
    payload_bytes: int,
    packet_count: int,
    device_id: str = "standalone",
    max_packet_payload_bytes: int = 0,
) -> list[Packet]:
    if payload_bytes <= 0:
        raise ValueError("payload_bytes must be positive.")
    if packet_count <= 0:
        raise ValueError("packet_count must be positive.")

    packets = []
    for idx in range(packet_count):
        packet_id = f"{device_id}:payload:{idx + 1}"
        header = {
            "packet_id": packet_id,
            "device_id": device_id,
            "source": "standalone_payload",
            "packet_index": idx + 1,
            "packet_count": packet_count,
            "frame_idx": idx + 1,
            "frame_id": idx + 1,
            "layer": "payload",
            "item_index": idx,
        }
        raw_payload = build_fallback_payload(header, payload_bytes)
        payload_crc32 = _crc32(raw_payload)
        header["payload_crc32"] = payload_crc32
        for fragment_id, fragment_header, fragment, fragment_index, fragment_count in fragment_payload(
            packet_id,
            header,
            raw_payload,
            max_packet_payload_bytes,
        ):
            fragment_header["packet_id"] = fragment_id
            packets.append(
                Packet(
                    packet_id=fragment_id,
                    device_id=device_id,
                    frame_idx=idx + 1,
                    frame_id=idx + 1,
                    layer="payload",
                    payload=envelope_payload(fragment_header, fragment),
                    priority=1.0,
                    fragment_index=fragment_index,
                    fragment_count=fragment_count,
                    payload_crc32=payload_crc32,
                    metadata=dict(header),
                )
            )
    return packets


def iter_schedule_frames(schedule_json: Path) -> Iterable[dict]:
    data = load_artifact_json(schedule_json)
    if isinstance(data, dict) and "frames" in data:
        return data["frames"]
    if isinstance(data, list):
        return data
    raise ValueError(f"Unsupported schedule JSON format: {schedule_json}")


def _normalize_sequence_filter(sequence_filter) -> set[str]:
    if sequence_filter in (None, "", []):
        return set()
    if isinstance(sequence_filter, str):
        return {value.strip() for value in sequence_filter.split(",") if value.strip()}
    return {str(value).strip() for value in sequence_filter if str(value).strip()}


def build_schedule_packets(
    schedule_json: Path,
    max_frames: int,
    max_packets: int,
    device_id: str = "node",
    max_packet_payload_bytes: int = 0,
    metadata_packet_payload_bytes: int = 0,
    roi_packet_payload_bytes: int = 0,
    background_packet_payload_bytes: int = 0,
    sequence_filter=None,
) -> list[Packet]:
    if schedule_json is None:
        raise ValueError("schedule_json is required.")
    frames = list(iter_schedule_frames(schedule_json))
    allowed_sequences = _normalize_sequence_filter(sequence_filter)
    if allowed_sequences:
        frames = [frame for frame in frames if str(frame.get("sequence", "")) in allowed_sequences]
    else:
        manifest_sequences = {str(frame.get("sequence", "")) for frame in frames if frame.get("sequence")}
        if len(manifest_sequences) > 1:
            raise ValueError(
                "candidate manifest contains multiple sequences; set sequence_filter "
                f"to one or more explicit sequence names. Found: {sorted(manifest_sequences)}"
            )
    if max_frames > 0:
        frames = frames[:max_frames]

    packets = []
    background_summary_added = False
    for frame in frames:
        sequence = frame.get("sequence", device_id)
        rate_idx = int(frame.get("rate_idx", 0))
        frame_idx = int(frame["frame_idx"])
        frame_id = int(frame.get("frame_id", frame_idx))
        items = list(frame.get("items", []))
        if not background_summary_added:
            existing_index = next(
                (
                    index
                    for index, item in enumerate(items)
                    if item.get("layer") == BACKGROUND_LAYER and int(item.get("frame_idx", frame_idx)) == 1
                ),
                None,
            )
            if existing_index is not None:
                background_summary = dict(items.pop(existing_index))
                background = _background_item(background_summary)
                background["mode"] = "required"
                background.pop("required", None)
                background_summary["background"] = background
                background_summary["priority"] = max(2.0, float(background_summary.get("priority", 0.0) or 0.0))
                items.insert(0, background_summary)
                background_summary_added = True
            else:
                background_summary = _find_background_summary_item(schedule_json.parent, sequence)
                if background_summary is not None:
                    items.insert(0, background_summary)
                    background_summary_added = True
        for item_idx, item in enumerate(items):
            layer = str(item.get("layer"))
            background = _background_item(item)
            packet_base_id = f"{device_id}:{sequence}:{rate_idx}:{frame_idx}:{item_idx}:{layer}"
            declared_bytes = max(1, int(item.get("bytes", 1)))
            header = {
                "packet_id": packet_base_id,
                "device_id": device_id,
                "source": "schedule_json",
                "sequence": sequence,
                "rate_idx": rate_idx,
                "frame_idx": frame_idx,
                "frame_id": frame_id,
                "layer": layer,
                "roi_idx": item.get("roi_idx"),
                "track_id": item.get("track_id"),
                "bbox": item.get("bbox") or item.get("roi_bbox"),
                "class_id": item.get("class_id"),
                "category": item.get("category"),
                "motion_score": float(item.get("motion_score", 0.0) or 0.0),
                "background_change_score": float(item.get("background_change_score", 0.0) or 0.0),
                "uncertainty": float(item.get("uncertainty", 0.0) or 0.0),
                "confidence": item.get("confidence"),
                "continuity_score": item.get("continuity_score"),
                "mask_pixels": item.get("mask_pixels"),
                "mask_coverage": item.get("mask_coverage"),
                "edge_mask_coverage": item.get("edge_mask_coverage"),
                "masked_black_ratio": item.get("masked_black_ratio"),
                "background": background,
                "timestamp": item.get("timestamp"),
                "item_rate_idx": item.get("rate_idx"),
                "priority": item.get("priority", 0.0),
                "item_index": item_idx,
                "candidate_group_id": item.get("candidate_group_id"),
                "quality_rank": int(item.get("quality_rank", item.get("rate_idx", 0)) or 0),
            }
            raw_payload = _load_item_payload(item, schedule_json.parent)
            if raw_payload is None:
                raw_payload = build_fallback_payload(header, declared_bytes)
            payload_crc32 = _crc32(raw_payload)
            header["payload_crc32"] = payload_crc32
            packet_payload_limit = _packet_payload_limit_for_layer(
                layer,
                max_packet_payload_bytes=max_packet_payload_bytes,
                metadata_packet_payload_bytes=metadata_packet_payload_bytes,
                roi_packet_payload_bytes=roi_packet_payload_bytes,
                background_packet_payload_bytes=background_packet_payload_bytes,
            )
            for packet_id, fragment_header, fragment, fragment_index, fragment_count in fragment_payload(
                packet_base_id,
                header,
                raw_payload,
                packet_payload_limit,
            ):
                fragment_header["packet_id"] = packet_id
                payload = envelope_payload(fragment_header, fragment)
                packets.append(
                    Packet(
                        packet_id=packet_id,
                        device_id=device_id,
                        frame_idx=frame_idx,
                        frame_id=frame_id,
                        layer=layer,
                        payload=payload,
                        roi_idx=item.get("roi_idx"),
                        rate_idx=item.get("rate_idx"),
                        priority=float(item.get("priority", 0.0)),
                        fragment_index=fragment_index,
                        fragment_count=fragment_count,
                        payload_crc32=payload_crc32,
                        metadata=dict(header),
                    )
                )
                if max_packets > 0 and len(packets) >= max_packets:
                    return packets
    return packets


def _find_background_summary_item(base_dir: Path, sequence: str) -> dict | None:
    background_dir = base_dir / "payloads" / "background" / str(sequence)
    candidates = sorted(background_dir.glob("frame_000001_bg_keyframe_rate*.bin"))
    if not candidates:
        return None
    path = candidates[0]
    return {
        "layer": BACKGROUND_LAYER,
        "bytes": path.stat().st_size,
        "frame_idx": 1,
        "priority": 2.0,
        "payload_path": str(path.relative_to(base_dir)),
        "background": {
            "version": f"{sequence}:bg:f000001",
            "mode": "required",
            "reference_frame_idx": None,
        },
    }


def _load_item_payload(item: dict, base_dir: Path) -> bytes | None:
    path_value = item.get("payload_path")
    if path_value:
        path = Path(path_value)
        if not path.is_absolute():
            path = base_dir / path
        return path.read_bytes()
    return None


def _packet_payload_limit_for_layer(
    layer: str,
    max_packet_payload_bytes: int = 0,
    metadata_packet_payload_bytes: int = 0,
    roi_packet_payload_bytes: int = 0,
    background_packet_payload_bytes: int = 0,
) -> int:
    layer_limits = {
        "metadata": int(metadata_packet_payload_bytes or 0),
        "roi_patch": int(roi_packet_payload_bytes or 0),
        BACKGROUND_LAYER: int(background_packet_payload_bytes or 0),
    }
    return layer_limits.get(layer, 0) or int(max_packet_payload_bytes or 0)


def _crc32(payload: bytes) -> int:
    import zlib

    return zlib.crc32(payload) & 0xFFFFFFFF


def build_packets_from_config(args) -> list[Packet]:
    if args.source == "payload":
        return build_payload_packets(
            args.payload_bytes,
            args.packet_count,
            max_packet_payload_bytes=int(getattr(args, "max_packet_payload_bytes", 0) or 0),
        )
    if args.source == "candidates":
        return build_schedule_packets(
            args.candidate_manifest_json,
            args.max_frames,
            args.max_packets,
            device_id="node_1",
            max_packet_payload_bytes=int(getattr(args, "max_packet_payload_bytes", 0) or 0),
            metadata_packet_payload_bytes=int(getattr(args, "metadata_packet_payload_bytes", 0) or 0),
            roi_packet_payload_bytes=int(getattr(args, "roi_packet_payload_bytes", 0) or 0),
            background_packet_payload_bytes=int(getattr(args, "background_packet_payload_bytes", 0) or 0),
            sequence_filter=getattr(args, "sequence_filter", None),
        )

    packets = []
    for node in args.nodes:
        packets.extend(
            build_schedule_packets(
                node["schedule_json"],
                int(node.get("max_frames", args.max_frames)),
                int(node.get("max_packets", args.max_packets)),
                device_id=node["device_id"],
                max_packet_payload_bytes=int(
                    node.get("max_packet_payload_bytes", getattr(args, "max_packet_payload_bytes", 0)) or 0
                ),
                metadata_packet_payload_bytes=int(
                    node.get("metadata_packet_payload_bytes", getattr(args, "metadata_packet_payload_bytes", 0)) or 0
                ),
                roi_packet_payload_bytes=int(
                    node.get("roi_packet_payload_bytes", getattr(args, "roi_packet_payload_bytes", 0)) or 0
                ),
                background_packet_payload_bytes=int(
                    node.get("background_packet_payload_bytes", getattr(args, "background_packet_payload_bytes", 0))
                    or 0
                ),
                sequence_filter=node.get("sequence_filter", getattr(args, "sequence_filter", None)),
            )
        )
    return packets
