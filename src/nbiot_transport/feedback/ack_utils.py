"""Normalize receiver confirmations without treating local modem sends as delivery."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from nbiot_transport.transport.sim7080_transport import SendResult


def result_to_dict(result: object) -> dict:
    """Read a result's JSON representation, falling back to its instance fields."""
    if hasattr(result, "to_json"):
        return result.to_json()
    return dict(getattr(result, "__dict__", {}))


def normalize_application_ack(result: object) -> dict[str, Any]:
    """Normalize current and legacy ACK fields, preserving unknown values as None."""
    raw = result_to_dict(result)
    ack = raw.get("application_ack") or raw.get("app_ack") or raw.get("remote_ack")
    if isinstance(ack, dict):
        observed = ack.get("observed")
        confirmed = ack.get("confirmed")
        complete = ack.get("complete")
        packet_id = ack.get("packet_id") or ack.get("ack_packet_id")
        received_bytes = ack.get("received_bytes")
        rtt_ms = ack.get("rtt_ms") or ack.get("elapsed_ms")
        sequence = ack.get("sequence") or ack.get("ack_id")
        error = ack.get("error")
    else:
        observed = None
        confirmed = raw.get("app_confirmed")
        if confirmed is None:
            confirmed = raw.get("remote_confirmed")
        complete = raw.get("app_complete")
        packet_id = raw.get("app_ack_packet_id") or raw.get("remote_ack_packet_id")
        received_bytes = raw.get("app_received_bytes") or raw.get("remote_received_bytes")
        rtt_ms = raw.get("app_ack_rtt_ms") or raw.get("remote_ack_rtt_ms")
        sequence = raw.get("app_ack_id") or raw.get("remote_ack_id")
        error = raw.get("app_ack_error") or raw.get("remote_ack_error")

    return {
        "observed": _to_observed(observed, confirmed, rtt_ms, sequence),
        "confirmed": _to_optional_bool(confirmed),
        "complete": _to_optional_bool(complete),
        "packet_id": packet_id,
        "received_bytes": _to_optional_int(received_bytes),
        "rtt_ms": _to_optional_float(rtt_ms),
        "sequence": sequence,
        "error": error,
    }


def result_arrived(result: object) -> bool:
    """Require observed application confirmation and completeness when available."""
    ack = normalize_application_ack(result)
    if ack["observed"]:
        if ack["complete"] is not None:
            return ack["confirmed"] is True and ack["complete"] is True
        return ack["confirmed"] is True
    return False


def result_delivered_bytes(result: object) -> int:
    """Prefer receiver-reported bytes, otherwise return the result's byte count."""
    ack = normalize_application_ack(result)
    if ack["received_bytes"] is not None:
        return int(ack["received_bytes"])
    return int(getattr(result, "bytes", 0) or 0)


def missing_fragment_packet_ids(results: Sequence[SendResult]) -> list[str]:
    """Return sorted unique retry IDs from failed sends and receiver missing ranges."""
    missing = []
    complete_bases = set()
    for result in results:
        ack = getattr(result, "application_ack", None)
        if not isinstance(ack, dict) or ack.get("complete") is not True:
            continue
        packet_id = str(ack.get("packet_id") or result.packet_id)
        complete_bases.add(packet_id.split(":fragment:", 1)[0])

    for result in results:
        ack = getattr(result, "application_ack", None)
        # A failed AT+CASEND has no application ACK at all.  The transport may
        # already have reopened the socket, so this logical packet must still
        # enter the normal round retransmission set.
        if not bool(getattr(result, "confirmed", False)):
            missing.append(str(result.packet_id))
            continue
        # TCP accepted the payload locally but the entire round query received
        # no server ACK.  Treat every logical message as missing so the caller
        # can reopen the socket and safely retransmit idempotent packet IDs.
        if not isinstance(ack, dict) or ack.get("observed") is not True:
            missing.append(str(result.packet_id))
            continue
        if not isinstance(ack, dict) or ack.get("complete") is not False:
            continue
        ranges = ack.get("missing_fragment_ranges")
        packet_id = ack.get("packet_id") or result.packet_id
        if not isinstance(ranges, list):
            if ack.get("error") in {"missing_messages", "not_received", "incomplete_fragments"}:
                missing.append(str(result.packet_id))
            continue
        base_id = str(packet_id).split(":fragment:", 1)[0]
        if base_id in complete_bases:
            continue
        fragment_count = ack.get("fragment_count")
        for start, end in _iter_missing_ranges(ranges):
            for index in range(start, end + 1):
                suffix = f"{index + 1}-of-{fragment_count}" if fragment_count else str(index + 1)
                missing.append(f"{base_id}:fragment:{suffix}")
    return sorted(set(missing))


def _iter_missing_ranges(ranges: Sequence[object]) -> list[tuple[int, int]]:
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


def _to_observed(observed: object, confirmed: object, rtt_ms: object, sequence: object) -> bool:
    if observed is not None:
        return bool(observed)
    return confirmed is not None or rtt_ms is not None or sequence is not None


def _to_optional_bool(value: object) -> bool | None:
    if value is None:
        return None
    return bool(value)


def _to_optional_int(value: object) -> int | None:
    if isinstance(value, (int, float)):
        return int(value)
    return None


def _to_optional_float(value: object) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None
