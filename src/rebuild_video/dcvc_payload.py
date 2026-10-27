from __future__ import annotations

from collections import Counter


def validate_dcvc_payload(data: bytes) -> dict:
    try:
        units = parse_dcvc_nal_units(data)
        return {
            "valid": True,
            "unit_count": len(units),
            "nal_types": dict(Counter(unit[0] for unit in units)),
            "first_nal_type": units[0][0] if units else None,
        }
    except Exception as exc:
        return {
            "valid": False,
            "error": f"{type(exc).__name__}: {exc}",
            "first_byte_hex": data[:1].hex(),
            "first_nal_type": data[0] >> 4 if data else None,
        }


def parse_dcvc_nal_units(data: bytes) -> list[tuple[int, bytes]]:
    units = []
    offset = 0
    while offset < len(data):
        start = offset
        flag = data[offset]
        offset += 1
        nal_type = flag >> 4
        if nal_type == 0:
            _, offset = read_uint_adaptive(data, offset)
            _, offset = read_uint_adaptive(data, offset)
            offset += 1
        elif nal_type in {1, 2}:
            offset += 1
            stream_length, offset = read_uint_adaptive(data, offset)
            offset += stream_length
        else:
            raise ValueError(f"{nal_type} is not a valid NalType at byte offset {start}")
        if offset > len(data):
            raise EOFError(f"truncated DCVC stream at byte offset {start}")
        units.append((nal_type, data[start:offset]))
    return units


def read_uint_adaptive(data: bytes, offset: int) -> tuple[int, int]:
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
