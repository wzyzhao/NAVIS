from __future__ import annotations

import json
import zlib

METADATA_MAGIC = b"NBM1"


def decode_metadata(data: bytes) -> dict:
    if not data.startswith(METADATA_MAGIC):
        raise ValueError("metadata payload must use NBM1 encoding")
    data = zlib.decompress(data[len(METADATA_MAGIC) :])
    value = json.loads(data.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("metadata payload must be a JSON object")
    return value
