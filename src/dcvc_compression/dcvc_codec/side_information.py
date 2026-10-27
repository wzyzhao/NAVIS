from __future__ import annotations

import json
import struct
import zlib

import numpy as np

METADATA_MAGIC = b"NBM1"
MASK_MAGIC = b"NBK1"


def encode_metadata(payload: dict) -> bytes:
    """Serialize JSON metadata with the NBM1 prefix and zlib compression."""
    raw = json.dumps(payload, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    return METADATA_MAGIC + zlib.compress(raw, level=9)


def decode_metadata(data: bytes) -> dict:
    """Decode NBM1 metadata, rejecting other wire formats and non-object JSON."""
    if not data.startswith(METADATA_MAGIC):
        raise ValueError("metadata payload must use NBM1 encoding")
    data = zlib.decompress(data[len(METADATA_MAGIC) :])
    value = json.loads(data.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("metadata payload must be a JSON object")
    return value


def encode_binary_mask(mask: np.ndarray) -> bytes:
    """Encode positive mask pixels as NBK1 little-endian runs in row-major order."""
    binary = np.asarray(mask > 0, dtype=np.uint8)
    height, width = binary.shape[:2]
    flat = binary.reshape(-1)
    if flat.size == 0:
        runs = b""
    else:
        changes = np.flatnonzero(flat[1:] != flat[:-1]) + 1
        lengths = np.diff(np.concatenate(([0], changes, [flat.size]))).astype("<u4")
        runs = bytes((int(flat[0]),)) + lengths.tobytes()
    header = MASK_MAGIC + struct.pack("<II", width, height)
    return header + zlib.compress(runs, level=9)


def decode_binary_mask(data: bytes) -> np.ndarray:
    """Decode NBK1 into a uint8 mask with values 0/255 and validate run lengths."""
    if not data.startswith(MASK_MAGIC) or len(data) < 12:
        raise ValueError("invalid binary mask payload")
    width, height = struct.unpack("<II", data[4:12])
    runs = zlib.decompress(data[12:])
    if not runs:
        return np.zeros((height, width), dtype=np.uint8)
    value = runs[0]
    if value not in (0, 1) or (len(runs) - 1) % 4:
        raise ValueError("invalid binary mask runs")
    lengths = np.frombuffer(runs[1:], dtype="<u4")
    if int(lengths.sum(dtype=np.uint64)) != width * height:
        raise ValueError("binary mask run length mismatch")
    values = (np.arange(lengths.size, dtype=np.uint8) & 1) ^ value
    flat = np.repeat(values, lengths.astype(np.intp))
    return (flat.reshape(height, width) * 255).astype(np.uint8)
