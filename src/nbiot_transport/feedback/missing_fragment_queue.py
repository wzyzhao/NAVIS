import json
import time
from pathlib import Path

from nbiot_transport.core.packetizer import Packet


def load_missing_fragment_queue(path: Path | None) -> list[dict]:
    if path is None or not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return []
    fragments = data.get("missing_fragments", [])
    completed_keys = _received_manifest_keys(path.parent / "received_payload_manifest.jsonl")
    return [dict(item) for item in fragments if isinstance(item, dict) and _queue_item_key(item) not in completed_keys]


def reconcile_missing_fragment_queue(path: Path | None, received_manifest_path: Path | None = None) -> dict:
    if path is None or not path.exists():
        return {"available": False, "purged_by_received_manifest_count": 0, "missing_fragments": []}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return {"available": False, "purged_by_received_manifest_count": 0, "missing_fragments": []}
    fragments = [dict(item) for item in data.get("missing_fragments", []) if isinstance(item, dict)]
    manifest_path = received_manifest_path or path.parent / "received_payload_manifest.jsonl"
    completed_keys = _received_manifest_keys(manifest_path)
    completed_metadata_frames = _received_manifest_frames(manifest_path, "metadata")
    if not completed_keys and not completed_metadata_frames:
        return {**data, "available": True, "purged_by_received_manifest_count": 0}
    active = [
        item
        for item in fragments
        if _queue_item_key(item) not in completed_keys
        and not _metadata_queue_item_completed(item, completed_metadata_frames)
    ]
    purged = len(fragments) - len(active)
    payload = {
        **data,
        "available": True,
        "reconciled_at": time.time(),
        "purged_by_received_manifest_count": purged,
        "missing_fragments": active,
    }
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    return payload


def enqueue_missing_metadata_from_delivery(
    path: Path,
    received_manifest_path: Path,
    delivery_log_path: Path,
    expected_frames: int,
    session_id: str | None = None,
) -> dict:
    if expected_frames <= 0 or not delivery_log_path.exists():
        return {"available": False, "added_count": 0, "missing_metadata_frames": []}

    received_metadata_frames = _received_manifest_frames(received_manifest_path, "metadata")
    missing_frames = sorted(set(range(1, int(expected_frames) + 1)) - received_metadata_frames)
    if not missing_frames:
        return {"available": True, "added_count": 0, "missing_metadata_frames": []}

    delivery = json.loads(delivery_log_path.read_text(encoding="utf-8"))
    messages = delivery.get("messages", []) if isinstance(delivery, dict) else []
    metadata_messages = {
        int(item.get("frame_idx")): item
        for item in messages
        if isinstance(item, dict) and item.get("layer") == "metadata" and item.get("frame_idx") is not None
    }

    previous = load_missing_fragment_queue(path)
    active = {item["packet_id"]: dict(item) for item in previous if isinstance(item, dict) and item.get("packet_id")}
    now = time.time()
    added = 0
    for frame_idx in missing_frames:
        message = metadata_messages.get(frame_idx)
        if not message or not message.get("packet_id"):
            continue
        packet_id = str(message["packet_id"])
        item = active.get(packet_id, {})
        if not item:
            added += 1
        active[packet_id] = {
            "packet_id": packet_id,
            "session_id": session_id or item.get("session_id"),
            "device_id": message.get("device_id") or item.get("device_id"),
            "frame_idx": frame_idx,
            "layer": "metadata",
            "bytes": message.get("payload_bytes_sent") or item.get("bytes"),
            "priority": max(float(item.get("priority", 0.0) or 0.0), _layer_priority("metadata")),
            "attempts": int(item.get("attempts", 0) or 0),
            "first_seen": item.get("first_seen", now),
            "last_seen": now,
            "last_error": message.get("ack_error") or "missing_metadata_manifest",
        }

    payload = {
        "updated_at": now,
        "purged_by_received_manifest_count": 0,
        "missing_fragments": sorted(
            active.values(),
            key=lambda item: (
                str(item.get("device_id")),
                int(item.get("frame_idx") or 0),
                item["packet_id"],
            ),
        ),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    return {
        **payload,
        "available": True,
        "added_count": added,
        "missing_metadata_frames": missing_frames,
    }


def update_missing_fragment_queue(
    path: Path,
    previous_items: list[dict],
    packets: list[Packet],
    results,
    session_id: str | None = None,
    received_manifest_path: Path | None = None,
) -> dict:
    packets_by_id = {packet.packet_id: packet for packet in packets}
    active = {item["packet_id"]: dict(item) for item in previous_items if item.get("packet_id")}
    completed_bases = set()
    missing_seen = set()
    now = time.time()
    completed_manifest_keys = _received_manifest_keys(
        received_manifest_path or path.parent / "received_payload_manifest.jsonl"
    )

    for result in results:
        ack = getattr(result, "application_ack", None)
        packet = packets_by_id.get(result.packet_id)
        base_id = _base_packet_id(result.packet_id)
        if isinstance(ack, dict) and ack.get("complete") is True:
            completed_bases.add(base_id)
            continue

        missing_ids = _missing_ids_from_ack(result)
        if not missing_ids and _is_retriable_send_failure(result):
            missing_ids = [result.packet_id]
        for packet_id in missing_ids:
            missing_packet = packets_by_id.get(packet_id) or packet
            item = active.get(packet_id, {})
            layer = missing_packet.layer if missing_packet is not None else item.get("layer")
            attempt_increment = 0 if packet_id in missing_seen else 1
            missing_seen.add(packet_id)
            active[packet_id] = {
                "packet_id": packet_id,
                "session_id": session_id,
                "device_id": missing_packet.device_id if missing_packet is not None else item.get("device_id"),
                "frame_idx": missing_packet.frame_idx if missing_packet is not None else item.get("frame_idx"),
                "layer": layer,
                "bytes": missing_packet.bytes if missing_packet is not None else item.get("bytes"),
                "priority": max(float(item.get("priority", 3.0) or 3.0), _layer_priority(layer)),
                "attempts": int(item.get("attempts", 0) or 0) + attempt_increment,
                "first_seen": item.get("first_seen", now),
                "last_seen": now,
                "last_error": _ack_error(result),
            }

    for base_id in completed_bases:
        for packet_id in list(active):
            if _base_packet_id(packet_id) == base_id:
                active.pop(packet_id, None)
    purged_by_manifest = 0
    if completed_manifest_keys:
        for packet_id, item in list(active.items()):
            if _queue_item_key(item) in completed_manifest_keys or _packet_id_key(packet_id) in completed_manifest_keys:
                active.pop(packet_id, None)
                purged_by_manifest += 1

    payload = {
        "updated_at": now,
        "purged_by_received_manifest_count": purged_by_manifest,
        "missing_fragments": sorted(
            active.values(),
            key=lambda item: (str(item.get("device_id")), int(item.get("frame_idx") or 0), item["packet_id"]),
        ),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    return payload


def _missing_ids_from_ack(result) -> list[str]:
    ack = getattr(result, "application_ack", None)
    if not isinstance(ack, dict) or ack.get("complete") is not False:
        return []
    ranges = ack.get("missing_fragment_ranges")
    packet_id = ack.get("packet_id") or result.packet_id
    fragment_count = ack.get("fragment_count")
    if not isinstance(ranges, list):
        if ":fragment:" in result.packet_id and ack.get("error") in {"missing_messages", "not_received"}:
            return [result.packet_id]
        return []
    base_id = _base_packet_id(packet_id)
    return [
        f"{base_id}:fragment:{index + 1}-of-{fragment_count}" if fragment_count else f"{base_id}:fragment:{index + 1}"
        for start, end in _iter_missing_ranges(ranges)
        for index in range(start, end + 1)
    ]


def _iter_missing_ranges(ranges) -> list[tuple[int, int]]:
    compact = []
    for item in ranges:
        if not isinstance(item, list) or len(item) != 2:
            continue
        start, end = item
        if not isinstance(start, int) or not isinstance(end, int):
            continue
        if end < start:
            continue
        compact.append((start, end))
    return compact


def _is_failed_fragment(result) -> bool:
    if ":fragment:" not in result.packet_id:
        return False
    ack = getattr(result, "application_ack", None)
    if isinstance(ack, dict) and ack.get("complete") is True:
        return False
    return not bool(getattr(result, "confirmed", False)) or not isinstance(ack, dict)


def _is_retriable_send_failure(result) -> bool:
    ack = getattr(result, "application_ack", None)
    if isinstance(ack, dict) and ack.get("complete") is True:
        return False
    if _is_failed_fragment(result):
        return True
    if bool(getattr(result, "confirmed", False)):
        return False
    return getattr(result, "layer", None) in {
        "metadata",
        "background_keyframe_summary",
        "roi_patch",
    }


def _ack_error(result):
    ack = getattr(result, "application_ack", None)
    if isinstance(ack, dict):
        return ack.get("error")
    return getattr(result, "error", None)


def _layer_priority(layer: str | None) -> float:
    if layer in ("metadata", "background_keyframe_summary"):
        return 10.0
    return 3.0


def _base_packet_id(packet_id: str) -> str:
    if ":fragment:" in packet_id:
        return packet_id.split(":fragment:", 1)[0]
    return packet_id


def _received_manifest_keys(path: Path | None) -> set[tuple[str, int, int, str]]:
    if path is None or not path.exists():
        return set()
    keys = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        key = _queue_item_key(item)
        if key is not None:
            keys.add(key)
    return keys


def _received_manifest_frames(path: Path | None, layer: str) -> set[int]:
    if path is None or not path.exists():
        return set()
    frames = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if item.get("layer") == layer and item.get("frame_idx") is not None:
            frames.add(int(item.get("frame_idx")))
            if item.get("frame_start") is not None and item.get("frame_end") is not None:
                try:
                    frames.update(range(int(item.get("frame_start")), int(item.get("frame_end")) + 1))
                except (TypeError, ValueError):
                    pass
    return frames


def _metadata_queue_item_completed(item: dict, completed_metadata_frames: set[int]) -> bool:
    if not completed_metadata_frames:
        return False
    if item.get("layer") != "metadata" and item.get("request_type") != "metadata":
        return False
    try:
        frame_idx = int(item.get("frame_idx"))
    except (TypeError, ValueError):
        return False
    packet_id = str(item.get("packet_id") or "")
    if ":metadata_bundle:" in packet_id:
        suffix = packet_id.rsplit(":metadata_bundle:", 1)[-1]
        if "-" in suffix:
            start, end = suffix.split("-", 1)
            try:
                return all(frame in completed_metadata_frames for frame in range(int(start), int(end) + 1))
            except ValueError:
                return frame_idx in completed_metadata_frames
    return frame_idx in completed_metadata_frames


def _queue_item_key(item: dict) -> tuple[str, int, int, str] | None:
    try:
        packet_id = item.get("packet_id")
        if item.get("frame_idx") in (None, "") and packet_id:
            return _packet_id_key(str(packet_id))
        device_id = str(item.get("device_id") or "")
        frame_idx = int(item.get("frame_idx"))
        layer = str(item.get("layer") or item.get("request_type") or "")
        item_index = item.get("item_index")
        if item_index in (None, ""):
            if packet_id:
                parsed = _packet_id_key(str(packet_id))
                if parsed is not None:
                    return parsed
            return None
        return (device_id, frame_idx, int(item_index), layer)
    except (TypeError, ValueError):
        return None


def _packet_id_key(packet_id: str) -> tuple[str, int, int, str] | None:
    base_id = _base_packet_id(packet_id)
    parts = base_id.split(":")
    if len(parts) < 4:
        return None
    try:
        return (str(parts[0]), int(parts[-3]), int(parts[-2]), str(parts[-1]))
    except (TypeError, ValueError):
        return None
