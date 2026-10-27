import json
import time
import zlib
from dataclasses import dataclass
from pathlib import Path

from nbiot_transport.core.background_metadata import background_required
from nbiot_transport.core.packet_sources import build_packets_from_config
from nbiot_transport.core.packetizer import (
    HEADER_SIZE,
    Packet,
    build_session_start_packet,
    envelope_payload,
    fragment_payload,
)
from nbiot_transport.core.scheduler import schedule_with_mandatory_reservation
from nbiot_transport.feedback.missing_fragment_queue import update_missing_fragment_queue
from nbiot_transport.feedback.online_link_controller import (
    OnlineLinkController,
    smooth_link_state,
    summarize_round_results,
)
from nbiot_transport.feedback.receiver_state import write_receiver_state
from nbiot_transport.logging.console_output import print_run_summary
from nbiot_transport.logging.delivery_log import write_delivery_log
from nbiot_transport.runtime.config_loader import load_config, validate_config
from nbiot_transport.transport.sim7080_transport import SendResult, Sim7080Transport

BACKGROUND_LAYER = "background_keyframe_summary"


@dataclass
class SendRun:
    args: object
    packets: list[Packet]
    results: list[SendResult]
    diagnostics: list
    scheduler_summary: dict


def load_packets(args) -> list[Packet]:
    packets = build_packets_from_config(args)
    if not packets:
        raise RuntimeError("No packets to send.")
    return packets


def create_transport(args):
    if getattr(args, "source", "") == "multi_schedule":
        raise ValueError("Create one single-node transport per modem through the multi-node coordinator")
    return Sim7080Transport(
        serial_port=args.serial_port,
        baudrate=args.serial_baud,
        apn=args.sim_apn,
        remote_host=args.remote_host,
        remote_port=args.remote_port,
        protocol=args.sim_protocol,
        connect_id=args.sim_connect_id,
        timeout_seconds=args.sim_timeout_seconds,
        send_confirm_timeout_seconds=args.sim_send_timeout_seconds,
        network_registration_timeout_seconds=float(getattr(args, "network_registration_timeout_seconds", 120.0)),
        network_registration_poll_seconds=float(getattr(args, "network_registration_poll_seconds", 3.0)),
        application_ack_enabled=bool(getattr(args, "enable_application_ack", False)),
        application_ack_timeout_seconds=float(getattr(args, "application_ack_timeout_seconds", 5.0)),
        ack_initial_delay_seconds=float(getattr(args, "ack_initial_delay_seconds", 2.0)),
        ack_poll_interval_seconds=float(getattr(args, "ack_poll_interval_seconds", 1.0)),
        round_ack_query_attempts=int(getattr(args, "round_ack_query_attempts", 2)),
        carecv_read_bytes=int(getattr(args, "carecv_read_bytes", 1024)),
        fragment_retransmit_enabled=bool(getattr(args, "fragment_retransmit_enabled", True)),
        fragment_retransmit_attempts=int(getattr(args, "fragment_retransmit_attempts", 1)),
        roi_patch_fragment_retransmit_attempts=int(getattr(args, "roi_patch_fragment_retransmit_attempts", 2)),
        background_summary_fragment_retransmit_attempts=int(
            getattr(args, "background_summary_fragment_retransmit_attempts", 3)
        ),
        socket_reopen_on_send_failures=int(getattr(args, "socket_reopen_on_send_failures", 1)),
        socket_reopen_attempts=int(getattr(args, "socket_reopen_attempts", 1)),
        window_shared_header_enabled=bool(getattr(args, "window_shared_header_enabled", True)),
        window_bundle_packet_payload_bytes=int(getattr(args, "window_bundle_packet_payload_bytes", 1024)),
        initialize=not args.skip_sim_init,
    )


def _frame_indices(packets: list[Packet]) -> list[int]:
    return sorted({packet.frame_idx for packet in packets if packet.frame_idx > 0})


def _packets_for_frame_range(packets: list[Packet], frame_start: int, frame_end: int) -> list[Packet]:
    return [packet for packet in packets if frame_start <= packet.frame_idx <= frame_end]


def _metadata_bytes_by_frame(packets: list[Packet]) -> dict[int, int]:
    metadata_bytes: dict[int, int] = {}
    for packet in packets:
        if packet.frame_idx > 0 and packet.layer == "metadata":
            metadata_bytes[packet.frame_idx] = metadata_bytes.get(packet.frame_idx, 0) + packet.bytes
    return metadata_bytes


def _metadata_raw_payload(packet: Packet) -> bytes:
    return packet.payload[HEADER_SIZE:]


def _decode_metadata_payload(payload: bytes) -> dict:
    if not payload.startswith(b"NBM1"):
        raise ValueError("metadata payload must use NBM1 encoding")
    decoded = zlib.decompress(payload[4:])
    value = json.loads(decoded.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("metadata payload must decode to a JSON object")
    return value


def _encode_metadata_payload(payload: dict) -> bytes:
    encoded = json.dumps(payload, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    return b"NBM1" + zlib.compress(encoded, level=9)


def _compact_roi(roi: dict) -> list:
    bbox = roi.get("roi_bbox") or roi.get("bbox") or [0, 0, 0, 0]
    compact = [
        int(roi.get("roi_idx", 0) or 0),
        int(roi.get("track_id", roi.get("roi_idx", 0)) or 0),
        [int(value) for value in bbox[:4]],
        int(roi.get("class_id", -1) or -1),
        round(float(roi.get("motion_score", 0.0) or 0.0), 4),
        round(float(roi.get("uncertainty", 0.0) or 0.0), 4),
    ]
    mask_rle = roi.get("mask_rle")
    if isinstance(mask_rle, dict):
        compact.append(mask_rle)
    return compact


def _compact_metadata_frame(metadata: dict, frame_start: int) -> list:
    frame_idx = int(metadata.get("frame_idx", metadata.get("frame_id", frame_start)) or frame_start)
    return [
        frame_idx - int(frame_start),
        int(metadata.get("frame_id", frame_idx) or frame_idx),
        round(float(metadata.get("motion_score", 0.0) or 0.0), 4),
        round(float(metadata.get("background_change_score", 0.0) or 0.0), 4),
        round(float(metadata.get("uncertainty", 0.0) or 0.0), 4),
        [_compact_roi(roi) for roi in metadata.get("rois", []) if isinstance(roi, dict)],
    ]


def _bundle_window_metadata_packets(
    window_packets: list[Packet],
    frame_start: int,
    frame_end: int,
    max_payload_bytes: int = 0,
) -> list[Packet]:
    # Each compact header describes exactly one device/sequence.
    partitions = {}
    for packet in window_packets:
        key = (packet.device_id, str((packet.metadata or {}).get("sequence", "")))
        partitions.setdefault(key, []).append(packet)
    if len(partitions) > 1:
        return [
            packet
            for group in partitions.values()
            for packet in _bundle_window_metadata_packets(group, frame_start, frame_end, max_payload_bytes)
        ]
    metadata_packets = [packet for packet in window_packets if packet.layer == "metadata"]
    if len(metadata_packets) <= 1:
        return window_packets

    groups: dict[str, list[Packet]] = {}
    for packet in metadata_packets:
        base_id = packet.packet_id.split(":fragment:", 1)[0]
        groups.setdefault(base_id, []).append(packet)

    frame_payloads = []
    for group in groups.values():
        group.sort(key=lambda packet: (packet.fragment_index, packet.packet_id))
        raw_payload = b"".join(_metadata_raw_payload(packet) for packet in group)
        metadata = _decode_metadata_payload(raw_payload)
        frame_payloads.append(metadata)
    frame_payloads.sort(key=lambda item: int(item.get("frame_idx", item.get("frame_id", 0)) or 0))

    if len(frame_payloads) <= 1:
        return window_packets

    device_id = metadata_packets[0].device_id
    sequence = str(frame_payloads[0].get("sequence") or "default")
    base_packet_id = f"{device_id}:metadata_bundle:{sequence}:{frame_start}-{frame_end}"
    common = {
        "device_id": frame_payloads[0].get("device_id", device_id),
        "sequence": frame_payloads[0].get("sequence"),
        "background_version": frame_payloads[0].get("background_version"),
    }
    raw_payload = _encode_metadata_payload(
        {
            "metadata_bundle": True,
            "bundle_format": "compact_v4",
            "frame_start": int(frame_start),
            "frame_end": int(frame_end),
            "frame_count": len(frame_payloads),
            "common": common,
            "frames_compact": [_compact_metadata_frame(metadata, frame_start) for metadata in frame_payloads],
        }
    )
    payload_crc32 = zlib.crc32(raw_payload) & 0xFFFFFFFF
    header = {
        "packet_id": base_packet_id,
        "device_id": device_id,
        "frame_idx": int(frame_start),
        "frame_id": int(frame_start),
        "layer": "metadata",
        "item_index": 0,
        "payload_crc32": payload_crc32,
        "metadata_bundle": True,
        "frame_start": int(frame_start),
        "frame_end": int(frame_end),
        "frame_count": len(frame_payloads),
        "bundle_format": "compact_v4",
        "common": common,
        "frames_compact": [_compact_metadata_frame(metadata, frame_start) for metadata in frame_payloads],
    }
    max_payload_bytes = max(0, int(max_payload_bytes or 0))
    bundle_packets = []
    for packet_id, fragment_header, fragment, fragment_index, fragment_count in fragment_payload(
        base_packet_id,
        header,
        raw_payload,
        max_payload_bytes,
    ):
        fragment_header["packet_id"] = packet_id
        bundle_packets.append(
            Packet(
                packet_id=packet_id,
                device_id=device_id,
                frame_idx=int(frame_start),
                frame_id=int(frame_start),
                layer="metadata",
                payload=envelope_payload(fragment_header, fragment),
                priority=max((packet.priority for packet in metadata_packets), default=100.0),
                fragment_index=fragment_index,
                fragment_count=fragment_count,
                payload_crc32=payload_crc32,
                metadata=dict(header),
            )
        )
    return [packet for packet in window_packets if packet.layer != "metadata"] + bundle_packets


def _window_frame_cap(args) -> int:
    configured_cap = int(getattr(args, "max_scheduling_window_frames", 0) or 0)
    if configured_cap > 0:
        return configured_cap
    return int(getattr(args, "scheduling_window_frames", 30))


def next_dynamic_packet_window(
    packets: list[Packet],
    frame_indices: list[int],
    start_pos: int,
    args,
    quota_bytes: int,
) -> tuple[int, int, list[Packet], int]:
    if not frame_indices:
        return 0, 0, list(packets), 1

    frame_cap = max(1, _window_frame_cap(args))
    if not bool(getattr(args, "dynamic_window_enabled", True)):
        selected_frames = frame_indices[start_pos : start_pos + frame_cap]
        frame_start, frame_end = selected_frames[0], selected_frames[-1]
        return (
            frame_start,
            frame_end,
            _packets_for_frame_range(packets, frame_start, frame_end),
            start_pos + len(selected_frames),
        )

    metadata_bytes = _metadata_bytes_by_frame(packets)
    metadata_target = max(1, int(quota_bytes * float(getattr(args, "window_metadata_quota_ratio", 0.6))))
    selected_frames = []
    selected_metadata_bytes = 0
    pos = start_pos
    while pos < len(frame_indices) and len(selected_frames) < frame_cap:
        frame_idx = frame_indices[pos]
        next_metadata_bytes = selected_metadata_bytes + metadata_bytes.get(frame_idx, 0)
        if selected_frames and next_metadata_bytes > metadata_target:
            break
        selected_frames.append(frame_idx)
        selected_metadata_bytes = next_metadata_bytes
        pos += 1

    if not selected_frames:
        selected_frames = [frame_indices[start_pos]]
        pos = start_pos + 1

    frame_start, frame_end = selected_frames[0], selected_frames[-1]
    return frame_start, frame_end, _packets_for_frame_range(packets, frame_start, frame_end), pos


def split_forced_window_packets(packets: list[Packet]) -> tuple[list[Packet], list[Packet]]:
    forced_packets = []
    schedulable_packets = []
    for packet in packets:
        if packet.layer == BACKGROUND_LAYER and background_required(packet.metadata):
            forced_packets.append(packet)
        else:
            schedulable_packets.append(packet)
    return forced_packets, schedulable_packets


def insert_forced_packets_before_scheduled(selected: list[Packet], forced_packets: list[Packet]) -> list[Packet]:
    if not forced_packets:
        return selected
    required_background = [packet for packet in forced_packets if packet.layer == BACKGROUND_LAYER]
    required_background = sorted(
        required_background,
        key=lambda packet: (
            packet.frame_idx,
            packet.packet_id.split(":fragment:", 1)[0],
            packet.fragment_index,
            packet.packet_id,
        ),
    )
    metadata_packets = [packet for packet in selected if packet.layer == "metadata"]
    scheduled_rest = [packet for packet in selected if packet.layer != "metadata"]
    return metadata_packets + required_background + scheduled_rest


def add_forced_packets_to_summary(summary: dict, forced_packets: list[Packet]) -> dict:
    if not forced_packets:
        return summary
    updated = dict(summary)
    forced_bytes = sum(packet.bytes for packet in forced_packets)
    forced_count = len(forced_packets)
    background_count = sum(1 for packet in forced_packets if packet.layer == BACKGROUND_LAYER)
    background_bytes = sum(packet.bytes for packet in forced_packets if packet.layer == BACKGROUND_LAYER)
    updated["input_packets"] = int(updated.get("input_packets", 0) or 0) + forced_count
    updated["input_bytes"] = int(updated.get("input_bytes", 0) or 0) + forced_bytes
    updated["selected_packets"] = int(updated.get("selected_packets", 0) or 0) + forced_count
    updated["selected_bytes"] = int(updated.get("selected_bytes", 0) or 0) + forced_bytes
    updated["background_packets"] = int(updated.get("background_packets", 0) or 0) + background_count
    updated["forced_required_background_packets"] = background_count
    updated["forced_required_background_bytes"] = background_bytes
    updated["forced_packet_policy"] = "required_background_after_scheduled_metadata_outside_quota_scheduler"
    quota = int(updated.get("quota_bytes", updated.get("global_quota_bytes", 0)) or 0)
    updated["total_selection_overflow_bytes"] = max(0, updated["selected_bytes"] - quota) if quota > 0 else 0
    return updated


def split_missing_fragment_backlog(args, packets: list[Packet]) -> tuple[list[Packet], list[Packet], dict]:
    max_attempts = max(1, int(getattr(args, "missing_fragment_max_attempts", 3) or 3))
    missing_items = [
        item
        for item in (getattr(args, "missing_fragment_queue", []) or [])
        if item.get("layer") in {"metadata", BACKGROUND_LAYER, "roi_patch"} and item.get("packet_id")
    ]
    retryable_items = [item for item in missing_items if int(item.get("attempts", 0) or 0) < max_attempts]
    if not missing_items:
        return (
            [],
            packets,
            {
                "queued_count": 0,
                "queued_matched_request_count": 0,
                "retryable_count": 0,
                "retry_exhausted_count": 0,
                "matched_packets": 0,
                "by_layer": {},
                "max_attempts": max_attempts,
            },
        )

    packets_by_id = {packet.packet_id: packet for packet in packets}
    packets_by_base: dict[str, list[Packet]] = {}
    for packet in packets:
        packets_by_base.setdefault(packet.packet_id.split(":fragment:", 1)[0], []).append(packet)

    selected: dict[str, Packet] = {}
    queued_packet_ids: set[str] = set()
    requested_by_layer: dict[str, int] = {}
    queued_matched_request_count = 0
    for item in missing_items:
        layer = str(item.get("layer"))
        packet_id = str(item.get("packet_id"))
        matches = []
        if packet_id in packets_by_id:
            matches = [packets_by_id[packet_id]]
        else:
            matches = packets_by_base.get(packet_id.split(":fragment:", 1)[0], [])
        matched = False
        for packet in matches:
            if packet.layer != layer:
                continue
            queued_packet_ids.add(packet.packet_id)
            matched = True
        if matched:
            queued_matched_request_count += 1

    matched_request_count = 0
    for item in retryable_items:
        layer = str(item.get("layer"))
        requested_by_layer[layer] = requested_by_layer.get(layer, 0) + 1
        packet_id = str(item.get("packet_id"))
        matches = []
        if packet_id in packets_by_id:
            matches = [packets_by_id[packet_id]]
        else:
            matches = packets_by_base.get(packet_id.split(":fragment:", 1)[0], [])
        matched = False
        for packet in matches:
            if packet.layer != layer:
                continue
            selected[packet.packet_id] = packet
            matched = True
        if matched:
            matched_request_count += 1

    backlog = sorted(
        selected.values(),
        key=lambda packet: (
            _missing_backlog_layer_rank(packet),
            packet.frame_idx,
            packet.packet_id.split(":fragment:", 1)[0],
            packet.fragment_index,
            packet.packet_id,
        ),
    )
    matched_by_layer: dict[str, int] = {}
    for packet in backlog:
        matched_by_layer[packet.layer] = matched_by_layer.get(packet.layer, 0) + 1
    remaining = [packet for packet in packets if packet.packet_id not in queued_packet_ids]
    return (
        backlog,
        remaining,
        {
            "queued_count": len(missing_items),
            "queued_matched_request_count": queued_matched_request_count,
            "retryable_count": len(retryable_items),
            "retry_exhausted_count": len(missing_items) - len(retryable_items),
            "matched_request_count": matched_request_count,
            "matched_packets": len(backlog),
            "by_layer": matched_by_layer,
            "requested_by_layer": requested_by_layer,
            "max_attempts": max_attempts,
        },
    )


def _missing_backlog_layer_rank(packet: Packet) -> int:
    if packet.layer == "metadata":
        return 0
    if packet.layer == BACKGROUND_LAYER and background_required(packet.metadata):
        return 1
    if packet.layer == "roi_patch":
        return 2
    if packet.layer == BACKGROUND_LAYER:
        return 3
    return 4


def _packet_bases(packets: list[Packet]) -> set[str]:
    return {packet.packet_id.split(":fragment:", 1)[0] for packet in packets}


def _complete_backlog_bases(results: list[SendResult], packets: list[Packet]) -> set[str]:
    expected_bases = _packet_bases(packets)
    results_by_base: dict[str, list[SendResult]] = {}
    for result in results:
        base_id = result.packet_id.split(":fragment:", 1)[0]
        if base_id not in expected_bases:
            continue
        results_by_base.setdefault(base_id, []).append(result)
    complete = set()
    for base_id, base_results in results_by_base.items():
        observed_acks = [
            result.application_ack
            for result in base_results
            if isinstance(result.application_ack, dict) and result.application_ack.get("observed") is True
        ]
        if any(ack.get("confirmed") is True and ack.get("complete") is True for ack in observed_acks):
            complete.add(base_id)
    return complete


def _missing_fragment_backlog_record(
    packets: list[Packet], results: list[SendResult], started: float, summary: dict
) -> dict:
    complete_bases = _complete_backlog_bases(results, packets)
    expected_bases = _packet_bases(packets)
    return {
        "stage": "missing_fragment_backlog",
        "packet_count": len(packets),
        "message_count": len(expected_bases),
        "bytes": sum(packet.bytes for packet in packets),
        "elapsed_seconds": time.monotonic() - started,
        "complete": expected_bases.issubset(complete_bases),
        "completed_message_count": len(expected_bases & complete_bases),
        "incomplete_message_count": len(expected_bases - complete_bases),
        "frame_start": min((packet.frame_idx for packet in packets), default=None),
        "frame_end": max((packet.frame_idx for packet in packets), default=None),
        **summary,
    }


def _message_complete(results: list[SendResult], packets: list[Packet]) -> bool:
    if not packets:
        return True
    base_ids = {packet.packet_id.split(":fragment:", 1)[0] for packet in packets}
    for result in results:
        ack = result.application_ack
        if not isinstance(ack, dict):
            continue
        if ack.get("confirmed") is not True or ack.get("complete") is not True:
            continue
        ack_base = str(ack.get("packet_id") or result.packet_id).split(":fragment:", 1)[0]
        if ack_base in base_ids:
            return True
    return False


def _module_unconfirmed_packets(packets: list[Packet], results: list[SendResult]) -> list[Packet]:
    confirmed_packet_ids = {result.packet_id for result in results if bool(result.confirmed)}
    return [packet for packet in packets if packet.packet_id not in confirmed_packet_ids]


def _aggregate_round_summaries(input_packets: list[Packet], selected_packets: list[Packet], rounds: list[dict]) -> dict:
    return {
        "enabled": True,
        "online": True,
        "policy": "windowed online scheduling; each completed round ACK determines the next round quota",
        "input_packets": len(input_packets),
        "selected_packets": len(selected_packets),
        "input_bytes": sum(packet.bytes for packet in input_packets),
        "selected_bytes": sum(packet.bytes for packet in selected_packets),
        "dropped_packets": len(input_packets) - len(selected_packets),
        "metadata_packets": sum(packet.layer == "metadata" for packet in selected_packets),
        "roi_packets": sum(packet.layer == "roi_patch" for packet in selected_packets),
        "background_packets": sum(packet.layer == BACKGROUND_LAYER for packet in selected_packets),
        "round_count": len(rounds),
        "rounds": rounds,
        "link_state": rounds[-1]["feedback"]["smoothed"] if rounds else {},
        "dynamic_window_enabled": bool(rounds[-1].get("window", {}).get("dynamic", False)) if rounds else False,
    }


def send_packets(args, packets: list[Packet]) -> SendRun:
    if getattr(args, "source", "") == "multi_schedule":
        from nbiot_transport.runtime.concurrent_sender import send_packets as send_multi_node

        return send_multi_node(args, packets)
    session_packet = build_session_start_packet(packets[0].device_id if packets else "node")
    scheduled_packets = list(packets)
    missing_backlog_packets, scheduled_packets, missing_backlog_summary = split_missing_fragment_backlog(
        args,
        scheduled_packets,
    )
    sent_packets = [session_packet]
    results = []
    round_records = []
    startup_records = []
    controller = OnlineLinkController(args)
    node_states = {}
    transport = create_transport(args)
    try:
        if hasattr(transport, "send_packet"):
            results.append(transport.send_packet(session_packet))
        else:
            results.extend(transport.send_packets([session_packet]))
        if missing_backlog_packets:
            backlog_started = time.monotonic()
            backlog_results = transport.send_packets(missing_backlog_packets)
            sent_packets.extend(missing_backlog_packets)
            results.extend(backlog_results)
            backlog_record = _missing_fragment_backlog_record(
                missing_backlog_packets,
                backlog_results,
                backlog_started,
                missing_backlog_summary,
            )
            startup_records.append(backlog_record)
            if not backlog_record.get("complete"):
                transport.diagnostics.append(
                    {
                        "missing_fragment_backlog_incomplete": True,
                        "missing_fragment_backlog_policy": "queued missing fragments remain for a later run; ordinary scheduling continues",
                        "incomplete_message_count": backlog_record.get("incomplete_message_count"),
                    }
                )

        frame_indices = _frame_indices(scheduled_packets)
        round_index = 1
        frame_pos = 0
        while frame_pos < len(frame_indices):
            quota_bytes = controller.current_quota_bytes
            frame_start, frame_end, window_packets, next_frame_pos = next_dynamic_packet_window(
                scheduled_packets,
                frame_indices,
                frame_pos,
                args,
                quota_bytes,
            )
            if bool(getattr(args, "metadata_bundle_per_window", True)):
                window_packets = _bundle_window_metadata_packets(
                    window_packets,
                    frame_start,
                    frame_end,
                    max_payload_bytes=int(getattr(args, "metadata_packet_payload_bytes", 0) or 0),
                )
            link_state = controller.link_state
            if getattr(args, "source", "") == "multi_schedule":
                link_state["nodes"] = dict(node_states)
            selected, round_summary = schedule_with_mandatory_reservation(
                window_packets,
                args,
                quota_bytes=quota_bytes,
                link_state=link_state,
            )
            if not selected:
                raise RuntimeError(f"No packets selected for scheduling round {round_index}.")
            round_started = time.monotonic()
            round_results = transport.send_packets(selected)
            round_elapsed_seconds = time.monotonic() - round_started
            sent_packets.extend(selected)
            results.extend(round_results)
            try:
                feedback = controller.observe_round(round_results, elapsed_seconds=round_elapsed_seconds)
                if getattr(args, "source", "") == "multi_schedule":
                    device_for_packet = {packet.packet_id: packet.device_id for packet in selected}
                    observations = {}
                    for device_id in dict.fromkeys(device_for_packet.values()):
                        node_results = [r for r in round_results if device_for_packet.get(r.packet_id) == device_id]
                        service_time = getattr(transport, "last_node_elapsed_seconds", {}).get(
                            device_id, round_elapsed_seconds
                        )
                        observed = summarize_round_results(node_results, elapsed_seconds=service_time)
                        if observed["basis"] != "application_ack":
                            raise RuntimeError(f"Missing application ACK for node {device_id}")
                        node_states[device_id] = smooth_link_state(
                            node_states.get(device_id, {}), observed, controller.alpha
                        )
                        observations[device_id] = observed
                    feedback["nodes"] = observations
                    feedback["node_link_states"] = dict(node_states)
            except RuntimeError as exc:
                feedback = {
                    "ack_required": True,
                    "error": str(exc),
                    "observed": {
                        "basis": "application_ack_required",
                        "attempted_message_count": len(_packet_bases(selected)),
                        "delivered_message_count": 0,
                    },
                    "smoothed": dict(controller.smoothed),
                    "next_quota_bytes": controller.current_quota_bytes,
                }
                feedback["round_elapsed_seconds"] = round_elapsed_seconds
                round_records.append(
                    {
                        "round_index": round_index,
                        "frame_start": frame_start,
                        "frame_end": frame_end,
                        "quota_bytes": quota_bytes,
                        "link_state_before": link_state,
                        "scheduler": round_summary,
                        "feedback": feedback,
                        "window": {
                            "dynamic": bool(getattr(args, "dynamic_window_enabled", True)),
                            "metadata_quota_ratio": float(getattr(args, "window_metadata_quota_ratio", 0.6)),
                            "metadata_bytes": sum(
                                packet.bytes for packet in window_packets if packet.layer == "metadata"
                            ),
                            "frame_cap": _window_frame_cap(args),
                        },
                    }
                )
                diagnostics = list(transport.diagnostics)
                diagnostics.append(
                    {
                        "ack_required_error": str(exc),
                        "round_index": round_index,
                        "policy": "online scheduling aborted; application round ACK is mandatory",
                    }
                )
                return _finalize_send_run(
                    args,
                    session_packet,
                    missing_backlog_packets,
                    scheduled_packets,
                    sent_packets,
                    results,
                    diagnostics,
                    round_records,
                    startup_records,
                )
            feedback["round_elapsed_seconds"] = round_elapsed_seconds
            round_records.append(
                {
                    "round_index": round_index,
                    "frame_start": frame_start,
                    "frame_end": frame_end,
                    "quota_bytes": quota_bytes,
                    "link_state_before": link_state,
                    "scheduler": round_summary,
                    "feedback": feedback,
                    "window": {
                        "dynamic": bool(getattr(args, "dynamic_window_enabled", True)),
                        "metadata_quota_ratio": float(getattr(args, "window_metadata_quota_ratio", 0.6)),
                        "metadata_bytes": sum(packet.bytes for packet in window_packets if packet.layer == "metadata"),
                        "frame_cap": _window_frame_cap(args),
                    },
                }
            )
            frame_pos = next_frame_pos
            round_index += 1
        diagnostics = list(transport.diagnostics)
    finally:
        transport.close()

    return _finalize_send_run(
        args,
        session_packet,
        missing_backlog_packets,
        scheduled_packets,
        sent_packets,
        results,
        diagnostics,
        round_records,
        startup_records,
    )


def _finalize_send_run(
    args,
    session_packet: Packet,
    missing_backlog_packets: list[Packet],
    scheduled_packets: list[Packet],
    sent_packets: list[Packet],
    results: list[SendResult],
    diagnostics: list,
    round_records: list[dict],
    startup_records: list[dict],
) -> SendRun:
    startup_count = 1 + len(missing_backlog_packets)
    scheduler_summary = _aggregate_round_summaries(scheduled_packets, sent_packets[startup_count:], round_records)
    scheduler_summary["session_start_enabled"] = True
    scheduler_summary["session_id"] = session_packet.metadata["session_id"]
    scheduler_summary["startup"] = startup_records
    scheduler_summary["missing_fragment_backlog_packets"] = len(missing_backlog_packets)
    scheduler_summary["missing_fragment_backlog_bytes"] = sum(packet.bytes for packet in missing_backlog_packets)
    scheduler_summary["missing_fragment_backlog_enabled"] = bool(missing_backlog_packets)
    scheduler_summary["missing_fragment_max_attempts"] = int(getattr(args, "missing_fragment_max_attempts", 3) or 3)

    if getattr(args, "enable_receiver_state", False):
        receiver_state_path = write_receiver_state(
            args.output_root,
            sent_packets,
            results,
            stale_after_frames=int(getattr(args, "metadata_stale_after_frames", 30)),
            roi_cache_size=int(getattr(args, "roi_cache_size", 5)),
            track_stale_after_frames=int(getattr(args, "track_stale_after_frames", 10)),
            background_change_threshold=float(getattr(args, "background_change_threshold", 0.6)),
        )
        try:
            import json

            receiver_state = json.loads(receiver_state_path.read_text(encoding="utf-8"))
            scheduler_summary["receiver_state_path"] = str(receiver_state_path)
            scheduler_summary["receiver_state_recovery_hint_count"] = receiver_state.get("recovery_hint_count", 0)
            scheduler_summary["receiver_state_recovery_hints"] = receiver_state.get("recovery_hints", [])
        except Exception as exc:
            diagnostics.append({"receiver_state_error": str(exc)})

    try:
        queue = update_missing_fragment_queue(
            args.missing_fragment_queue_json,
            getattr(args, "missing_fragment_queue", []),
            sent_packets,
            results,
            session_id=session_packet.metadata["session_id"],
            received_manifest_path=args.output_root / "received_payload_manifest.jsonl",
        )
        scheduler_summary["missing_fragment_queue_path"] = str(args.missing_fragment_queue_json)
        scheduler_summary["missing_fragment_count"] = len(queue.get("missing_fragments", []))
    except Exception as exc:
        diagnostics.append({"missing_fragment_queue_error": str(exc)})

    return SendRun(
        args=args,
        packets=sent_packets,
        results=results,
        diagnostics=diagnostics,
        scheduler_summary=scheduler_summary,
    )


def run_sender(config_path: Path | None = None) -> SendRun:
    args = load_config(config_path) if config_path is not None else load_config()
    validate_config(args)
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite a completed or partial run: {args.output_root}. "
            "Choose a new runNN sender directory."
        )
    return send_packets(args, load_packets(args))


def write_run_artifacts(run: SendRun) -> Path:
    return write_delivery_log(
        run.args.output_root,
        run.args,
        run.packets,
        run.results,
        diagnostics=run.diagnostics,
        scheduler_summary=run.scheduler_summary,
    )


def run(config_path: Path | None = None) -> None:
    send_run = run_sender(config_path)
    output_path = write_run_artifacts(send_run)
    print_run_summary(send_run, output_path)
