import hashlib
import json
import struct
import uuid
import zlib
from dataclasses import dataclass
from math import ceil

PROTOCOL_MAGIC = b"NBT1"
PROTOCOL_VERSION = 1
HEADER_STRUCT = struct.Struct("!4sBBHQ16sIIHhhHHIII")
HEADER_SIZE = HEADER_STRUCT.size
WINDOW_BUNDLE_MAGIC = b"NBW1"
WINDOW_BUNDLE_VERSION = 1
WINDOW_BUNDLE_ENTRY_STRUCT = struct.Struct("!QBBIIHhhHHIII")
LAYER_CODES = {
    "payload": 0,
    "metadata": 1,
    "roi_patch": 2,
    "background_keyframe_summary": 3,
    "session_start": 4,
    "round_ack": 5,
    "window_bundle": 7,
}
CODE_LAYERS = {value: key for key, value in LAYER_CODES.items()}


@dataclass
class Packet:
    packet_id: str
    frame_idx: int
    frame_id: int
    layer: str
    payload: bytes
    device_id: str = "node"
    roi_idx: int | None = None
    rate_idx: int | None = None
    priority: float = 0.0
    fragment_index: int = 0
    fragment_count: int = 1
    payload_crc32: int | None = None
    metadata: dict | None = None

    @property
    def bytes(self) -> int:
        return len(self.payload)

    @property
    def crc32(self) -> int:
        return zlib.crc32(self.payload) & 0xFFFFFFFF

    def to_json(self) -> dict:
        return {
            "packet_id": self.packet_id,
            "device_id": self.device_id,
            "frame_idx": self.frame_idx,
            "frame_id": self.frame_id,
            "layer": self.layer,
            "bytes": self.bytes,
            "roi_idx": self.roi_idx,
            "rate_idx": self.rate_idx,
            "priority": self.priority,
            "fragment_index": self.fragment_index,
            "fragment_count": self.fragment_count,
            "payload_crc32": self.payload_crc32 if self.payload_crc32 is not None else self.crc32,
            "crc32": self.crc32,
        }


def build_session_start_packet(device_id: str = "node") -> Packet:
    session_id = uuid.uuid4().hex
    packet_id = f"{device_id}:session:{session_id}:session_start"
    raw_payload = b"NBS1" + bytes.fromhex(session_id)
    header = {
        "packet_id": packet_id,
        "device_id": device_id,
        "frame_idx": 0,
        "frame_id": 0,
        "layer": "session_start",
        "item_index": 0,
        "payload_crc32": zlib.crc32(raw_payload) & 0xFFFFFFFF,
    }
    return Packet(
        packet_id=packet_id,
        device_id=device_id,
        frame_idx=0,
        frame_id=0,
        layer="session_start",
        payload=envelope_payload(header, raw_payload),
        payload_crc32=header["payload_crc32"],
        priority=100.0,
        metadata={**header, "session_id": session_id},
    )


def build_round_ack_packet(device_id: str, round_index: int, packets: list[Packet]) -> Packet:
    if not packets:
        raise ValueError("round ACK query requires at least one expected packet")
    expected_message_ids = []
    seen = set()
    for packet in packets:
        base_id = packet.packet_id.split(":fragment:", 1)[0]
        if base_id in seen:
            continue
        seen.add(base_id)
        expected_message_ids.append(f"{message_id_for_packet(base_id):016x}")
    packet_id = f"{device_id}:round:{round_index}:round_ack"
    raw_payload = json.dumps(
        {
            "ack_type": "round_query",
            "round_index": int(round_index),
            "expected_message_ids": expected_message_ids,
        },
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    header = {
        "packet_id": packet_id,
        "device_id": device_id,
        "frame_idx": 0,
        "frame_id": 0,
        "layer": "round_ack",
        "item_index": int(round_index) % 65536,
        "payload_crc32": zlib.crc32(raw_payload) & 0xFFFFFFFF,
    }
    return Packet(
        packet_id=packet_id,
        device_id=device_id,
        frame_idx=0,
        frame_id=0,
        layer="round_ack",
        payload=envelope_payload(header, raw_payload),
        payload_crc32=header["payload_crc32"],
        priority=100.0,
        metadata={**header, "round_index": int(round_index)},
    )


def build_window_bundle_packets(
    device_id: str,
    round_index: int,
    packets: list[Packet],
    max_payload_bytes: int = 0,
) -> list[Packet]:
    if not packets:
        return []
    base_packet_id = f"{device_id}:round:{round_index}:window_bundle"
    raw_payload = build_window_bundle_payload(packets)
    payload_crc32 = zlib.crc32(raw_payload) & 0xFFFFFFFF
    frame_indices = [packet.frame_idx for packet in packets if packet.frame_idx > 0]
    header = {
        "packet_id": base_packet_id,
        "device_id": device_id,
        "frame_idx": min(frame_indices) if frame_indices else 0,
        "frame_id": min(frame_indices) if frame_indices else 0,
        "layer": "window_bundle",
        "item_index": int(round_index) % 65536,
        "payload_crc32": payload_crc32,
    }
    bundled_packets = []
    for packet_id, fragment_header, fragment, fragment_index, fragment_count in fragment_payload(
        base_packet_id,
        header,
        raw_payload,
        max_payload_bytes,
    ):
        fragment_header["packet_id"] = packet_id
        bundled_packets.append(
            Packet(
                packet_id=packet_id,
                device_id=device_id,
                frame_idx=header["frame_idx"],
                frame_id=header["frame_id"],
                layer="window_bundle",
                payload=envelope_payload(fragment_header, fragment),
                priority=max((packet.priority for packet in packets), default=100.0),
                fragment_index=fragment_index,
                fragment_count=fragment_count,
                payload_crc32=payload_crc32,
                metadata={
                    **header,
                    "round_index": int(round_index),
                    "bundled_message_count": len({packet.packet_id.split(":fragment:", 1)[0] for packet in packets}),
                    "bundled_packet_count": len(packets),
                },
            )
        )
    return bundled_packets


def build_window_bundle_payload(packets: list[Packet]) -> bytes:
    entries = bytearray()
    payloads = []
    for packet in packets:
        header = parse_envelope_header(packet.payload)
        payload_start = int(header["header_size"])
        payload_end = payload_start + int(header["payload_size"])
        raw_payload = packet.payload[payload_start:payload_end]
        entries.extend(
            WINDOW_BUNDLE_ENTRY_STRUCT.pack(
                int(header["message_id"]),
                int(LAYER_CODES.get(str(header.get("layer", "payload")), 0)),
                0,
                int(header.get("frame_idx", 0) or 0),
                int(header.get("frame_id", 0) or 0),
                int(header.get("item_index", 0) or 0),
                _signed_short(header.get("roi_idx"), -1),
                _signed_short(header.get("rate_idx"), -1),
                int(header.get("fragment_index", 0) or 0),
                int(header.get("fragment_count", 1) or 1),
                len(raw_payload),
                int(header.get("payload_crc32", zlib.crc32(raw_payload) & 0xFFFFFFFF)),
                zlib.crc32(raw_payload) & 0xFFFFFFFF,
            )
        )
        payloads.append(raw_payload)
    return (
        WINDOW_BUNDLE_MAGIC
        + bytes([WINDOW_BUNDLE_VERSION])
        + len(packets).to_bytes(2, "big")
        + bytes(entries)
        + b"".join(payloads)
    )


def build_fallback_payload(metadata: dict, declared_bytes: int) -> bytes:
    encoded = json.dumps(metadata, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    if len(encoded) >= declared_bytes:
        return encoded
    return encoded + bytes(declared_bytes - len(encoded))


def envelope_payload(header: dict, payload: bytes) -> bytes:
    """Pack an NBT1 header and payload, including message and fragment CRC32."""
    packet_id = str(header["packet_id"])
    message_id = message_id_for_packet(packet_id.split(":fragment:", 1)[0])
    device_id = str(header.get("device_id", "node")).encode("ascii", errors="ignore")[:16].ljust(16, b"\0")
    prefix = HEADER_STRUCT.pack(
        PROTOCOL_MAGIC,
        PROTOCOL_VERSION,
        LAYER_CODES.get(str(header.get("layer", "payload")), 0),
        0,
        message_id,
        device_id,
        int(header.get("frame_idx", 0) or 0),
        int(header.get("frame_id", header.get("frame_idx", 0)) or 0),
        int(header.get("item_index", header.get("packet_index", 0)) or 0),
        _signed_short(header.get("roi_idx"), -1),
        _signed_short(header.get("item_rate_idx", header.get("rate_idx")), -1),
        int(header.get("fragment_index", 0) or 0),
        int(header.get("fragment_count", 1) or 1),
        len(payload),
        int(header.get("payload_crc32", zlib.crc32(payload) & 0xFFFFFFFF)),
        zlib.crc32(payload) & 0xFFFFFFFF,
    )
    return prefix + payload


def parse_envelope_header(data: bytes) -> dict:
    """Read an NBT1 header; payload completeness and CRC are checked by the receiver."""
    if len(data) < HEADER_SIZE:
        raise ValueError("incomplete binary envelope header")
    values = HEADER_STRUCT.unpack(data[:HEADER_SIZE])
    (
        magic,
        version,
        layer_code,
        flags,
        message_id,
        device_raw,
        frame_idx,
        frame_id,
        item_index,
        roi_idx,
        rate_idx,
        fragment_index,
        fragment_count,
        payload_size,
        payload_crc32,
        fragment_crc32,
    ) = values
    if magic != PROTOCOL_MAGIC or version != PROTOCOL_VERSION:
        raise ValueError("unsupported binary envelope")
    return {
        "version": version,
        "flags": flags,
        "message_id": message_id,
        "message_id_hex": f"{message_id:016x}",
        "device_id": device_raw.split(b"\0", 1)[0].decode("ascii", errors="replace"),
        "frame_idx": frame_idx,
        "frame_id": frame_id,
        "item_index": item_index,
        "layer": CODE_LAYERS.get(layer_code, "payload"),
        "roi_idx": None if roi_idx < 0 else roi_idx,
        "rate_idx": None if rate_idx < 0 else rate_idx,
        "fragment_index": fragment_index,
        "fragment_count": fragment_count,
        "payload_size": payload_size,
        "payload_crc32": payload_crc32,
        "fragment_crc32": fragment_crc32,
        "header_size": HEADER_SIZE,
    }


def message_id_for_packet(packet_id: str) -> int:
    return int.from_bytes(hashlib.blake2b(packet_id.encode("utf-8"), digest_size=8).digest(), "big")


def fragment_payload(
    packet_base_id: str, header: dict, payload: bytes, max_payload_bytes: int
) -> list[tuple[str, dict, bytes, int, int]]:
    """Split a message into ordered fragments; a nonpositive limit disables splitting."""
    if max_payload_bytes <= 0 or len(payload) <= max_payload_bytes:
        item_header = dict(header)
        item_header.update({"fragment_index": 0, "fragment_count": 1})
        return [(packet_base_id, item_header, payload, 0, 1)]
    fragment_count = int(ceil(len(payload) / max_payload_bytes))
    fragments = []
    for idx in range(fragment_count):
        fragment = payload[idx * max_payload_bytes : (idx + 1) * max_payload_bytes]
        item_header = dict(header)
        item_header.update({"fragment_index": idx, "fragment_count": fragment_count})
        fragments.append(
            (f"{packet_base_id}:fragment:{idx + 1}-of-{fragment_count}", item_header, fragment, idx, fragment_count)
        )
    return fragments


def build_packets(plan, sequence: str, rate_idx: int) -> list[Packet]:
    packets = []
    for item_idx, item in enumerate(plan.items):
        packet_id = f"{sequence}:{rate_idx}:{plan.frame_idx}:{item_idx}:{item.layer}"
        header = {
            "packet_id": packet_id,
            "sequence": sequence,
            "rate_idx": rate_idx,
            "frame_idx": plan.frame_idx,
            "frame_id": plan.frame_id,
            "layer": item.layer,
            "roi_idx": item.roi_idx,
            "item_rate_idx": item.rate_idx,
            "item_index": item_idx,
        }
        packets.append(
            Packet(
                packet_id=packet_id,
                frame_idx=plan.frame_idx,
                frame_id=plan.frame_id,
                layer=item.layer,
                payload=build_fallback_payload(header, int(item.bytes)),
                roi_idx=item.roi_idx,
                rate_idx=item.rate_idx,
                priority=float(getattr(item, "priority", 0.0)),
                metadata=header,
            )
        )
    return packets


def _signed_short(value, default: int) -> int:
    if value in (None, ""):
        return default
    return max(-32768, min(int(value), 32767))
