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


def build_fallback_payload(metadata: dict, declared_bytes: int) -> bytes:
    encoded = json.dumps(metadata, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    if len(encoded) >= declared_bytes:
        return encoded
    return encoded + bytes(declared_bytes - len(encoded))


def envelope_payload(header: dict, payload: bytes) -> bytes:
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


def fragment_payload(packet_base_id: str, header: dict, payload: bytes, max_payload_bytes: int):
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
