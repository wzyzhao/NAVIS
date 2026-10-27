from collections.abc import Iterable

from nbiot_transport.core.packetizer import Packet, message_id_for_packet

ROUND_ACK_FIELDS = (
    "type",
    "round_index",
    "expected_message_count",
    "received_message_count",
    "missing_message_count",
    "missing_message_ids",
)


def base_packet_id(packet_id: str) -> str:
    return str(packet_id).split(":fragment:", 1)[0]


def message_id_hex_for_base(base_id: str) -> str:
    return f"{message_id_for_packet(base_id):016x}"


def message_id_hex_for_packet(packet_id: str) -> str:
    return message_id_hex_for_base(base_packet_id(packet_id))


def expected_message_ids_for_packets(packets: Iterable[Packet]) -> set[str]:
    return {message_id_hex_for_packet(packet.packet_id) for packet in packets}


def ack_matches_message_ids(ack: dict, expected_message_ids: set[str] | None) -> bool:
    expected = {str(message_id).lower() for message_id in (expected_message_ids or set())}
    if ack.get("type") == "round_ack" and ack.get("ack_format") == "summary_v1":
        return (
            ack.get("confirmed") is True
            and ack.get("complete") is True
            and ack.get("expected_message_count") == len(expected)
            and ack.get("received_message_count") == len(expected)
            and ack.get("missing_message_count") == 0
        )
    if ack.get("type") == "round_ack" and isinstance(ack.get("messages"), dict):
        ack_message_ids = {str(message_id).lower() for message_id in ack["messages"]}
        return bool(ack_message_ids.intersection(expected)) if expected else bool(ack_message_ids)
    return False


def round_ack_fields(round_ack: dict) -> dict:
    if not isinstance(round_ack, dict):
        return {}
    fields = {key: round_ack.get(key) for key in ROUND_ACK_FIELDS if key in round_ack}
    if fields.get("type") == "round_ack":
        fields["round_confirmed"] = bool(round_ack.get("confirmed"))
    return fields


def apply_round_ack_to_results(results: list, round_ack: dict) -> None:
    messages = round_ack.get("messages") if isinstance(round_ack, dict) else None
    fields = round_ack_fields(round_ack)
    if isinstance(round_ack, dict) and round_ack.get("ack_format") == "summary_v1":
        for result in results:
            result.application_ack = {
                "observed": True,
                "confirmed": True,
                "complete": True,
                "packet_id": base_packet_id(result.packet_id),
                "message_id": message_id_hex_for_packet(result.packet_id),
                "rtt_ms": round_ack.get("rtt_ms"),
                "error": None,
                **fields,
            }
        return
    if not isinstance(messages, dict):
        for result in results:
            result.application_ack = {
                "observed": bool(round_ack.get("observed")) if isinstance(round_ack, dict) else False,
                "confirmed": None,
                "complete": None,
                "packet_id": result.packet_id,
                "message_id": message_id_hex_for_packet(result.packet_id),
                "rtt_ms": round_ack.get("rtt_ms") if isinstance(round_ack, dict) else None,
                "error": round_ack.get("error") if isinstance(round_ack, dict) else "missing round ACK",
                **fields,
            }
        return

    for result in results:
        base_id = base_packet_id(result.packet_id)
        message_id = message_id_hex_for_base(base_id)
        message = messages.get(message_id) or {}
        complete = message.get("complete")
        confirmed = message.get("confirmed")
        result.application_ack = {
            "observed": True,
            "confirmed": bool(confirmed) if confirmed is not None else False,
            "complete": bool(complete) if complete is not None else False,
            "packet_id": base_id,
            "message_id": message_id,
            "fragment_count": message.get("fragment_count"),
            "missing_fragment_ranges": message.get("missing_fragment_ranges"),
            "received_bytes": message.get("received_bytes") if confirmed else 0,
            "rtt_ms": round_ack.get("rtt_ms"),
            "sequence": round_ack.get("sequence"),
            "error": message.get("error") or round_ack.get("error"),
            **fields,
        }
