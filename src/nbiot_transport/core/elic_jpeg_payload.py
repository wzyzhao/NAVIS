"""NEJ1: explicit codec metadata and length-delimited blobs, never wire pickle."""

import json
import struct

MAGIC = b"NEJ1"


def encode(codec: str, data: dict) -> bytes:
    blobs = []

    def convert(value):
        if isinstance(value, bytes):
            blobs.append(value)
            return {"blob_index": len(blobs) - 1}
        if isinstance(value, (tuple, list)):
            return [convert(v) for v in value]
        if isinstance(value, dict):
            return {k: convert(v) for k, v in value.items()}
        return value

    header = {"codec": codec, "data": convert(data)}
    header["lengths"] = [len(b) for b in blobs]
    raw = json.dumps(header, separators=(",", ":")).encode()
    return MAGIC + struct.pack("!I", len(raw)) + raw + b"".join(blobs)


def decode(payload: bytes) -> tuple[str, dict]:
    if len(payload) < 8 or payload[:4] != MAGIC:
        raise ValueError("Not a NEJ1 payload")
    n = struct.unpack("!I", payload[4:8])[0]
    if n > 1024 * 1024 or 8 + n > len(payload):
        raise ValueError("Invalid NEJ1 header length")
    header = json.loads(payload[8 : 8 + n])
    if header["codec"] not in {"elic_background", "jpeg_roi"}:
        raise ValueError("Unsupported codec")
    blobs, cursor = [], 8 + n
    for length in header["lengths"]:
        if not isinstance(length, int) or length < 0 or cursor + length > len(payload):
            raise ValueError("Invalid NEJ1 blob length")
        blobs.append(payload[cursor : cursor + length])
        cursor += length
    if cursor != len(payload):
        raise ValueError("Trailing NEJ1 bytes")

    def restore(value):
        if isinstance(value, dict):
            if set(value) == {"blob_index"}:
                i = value["blob_index"]
                if not isinstance(i, int) or not 0 <= i < len(blobs):
                    raise ValueError("Bad blob index")
                return blobs[i]
            return {k: restore(v) for k, v in value.items()}
        if isinstance(value, list):
            return [restore(v) for v in value]
        return value

    return header["codec"], restore(header["data"])
