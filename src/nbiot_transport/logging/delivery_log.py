import json
import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from nbiot_transport.core.metadata_bundle import expand_metadata_bundle
from nbiot_transport.core.packetizer import Packet
from nbiot_transport.feedback.ack_utils import normalize_application_ack
from nbiot_transport.feedback.round_ack import base_packet_id, message_id_hex_for_packet
from nbiot_transport.transport.sim7080_transport import SendResult

REQUIRED_METADATA_FIELDS = (
    "device_id",
    "timestamp",
    "track_id",
    "bbox",
    "motion_score",
    "background_change_score",
    "uncertainty",
    "background_version",
)

ACK_RATE_EXCLUDED_LAYERS = {"session_start"}


def _safe_rate(numerator: float, denominator: float) -> float | None:
    if denominator <= 0:
        return None
    return numerator / denominator


def _mean(values: Iterable[float]) -> float | None:
    values = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not values:
        return None
    return sum(values) / len(values)


def _percentile(values: Iterable[float], percentile: float) -> float | None:
    values = sorted(float(value) for value in values if value is not None and math.isfinite(float(value)))
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    rank = (len(values) - 1) * percentile
    lower = int(math.floor(rank))
    upper = int(math.ceil(rank))
    if lower == upper:
        return values[lower]
    return values[lower] + (values[upper] - values[lower]) * (rank - lower)


def _packet_by_id(packets: list[Packet]) -> dict[str, Packet]:
    return {packet.packet_id: packet for packet in packets}


def summarize_results(results: list[SendResult]) -> dict:
    confirmed = [result for result in results if result.confirmed]
    return {
        "packet_count": len(results),
        "confirmed_packet_count": len(confirmed),
        "confirmed_payload_bytes": int(sum(result.bytes for result in confirmed)),
        "confirmed_wire_bytes": int(sum(_result_wire_bytes(result) for result in confirmed)),
        "failed_packet_count": len(results) - len(confirmed),
    }


def summarize_end_to_end_ack(results: list[SendResult]) -> dict:
    all_message_records = build_message_delivery_records([], results)
    message_records = [record for record in all_message_records if record.get("layer") not in ACK_RATE_EXCLUDED_LAYERS]
    observed = [record for record in message_records if record["ack_observed"]]
    confirmed = [record for record in observed if record["ack_confirmed"] is True]
    complete = [record for record in observed if record["ack_complete"] is True]
    failed = [record for record in observed if record["ack_confirmed"] is False]
    module_confirmed = [record for record in message_records if record.get("module_confirmed") is True]
    missing = len(message_records) - len(observed)
    failed_or_missing_count = len(failed) + missing
    rtts = [record["ack_rtt_ms"] for record in observed if record["ack_rtt_ms"] is not None]
    message_count = len(message_records)
    return {
        "status": "available" if observed else "not_available",
        "meaning": (
            "remote application ACK was observed at message level"
            if observed
            else "application ACK was not observed; module-side confirmation is diagnostic only"
        ),
        "ack_observed_count": len(observed),
        "ack_confirmed_count": len(confirmed),
        "ack_complete_count": len(complete),
        "ack_failed_count": failed_or_missing_count,
        "ack_observed_failed_count": len(failed),
        "ack_missing_count": missing,
        "module_confirmed_message_count": len(module_confirmed),
        "module_delivery_rate": _safe_rate(len(module_confirmed), message_count),
        "ack_coverage_rate": _safe_rate(len(observed), message_count),
        "ack_success_rate": _safe_rate(len(confirmed), message_count),
        "ack_failure_rate": _safe_rate(failed_or_missing_count, message_count),
        "ack_observed_failure_rate": _safe_rate(len(failed), len(observed)),
        "ack_unobserved_rate": _safe_rate(missing, message_count),
        "ack_required": True,
        "ack_missing_is_delivery_failure": True,
        "message_count": message_count,
        "rtt_ms_avg": _mean(rtts),
        "rtt_ms_p95": _percentile(rtts, 0.95),
    }


def summarize_metadata_schema(packets: list[Packet]) -> dict:
    metadata_packets = [packet for packet in packets if packet.layer == "metadata"]
    coverage = {field: 0 for field in REQUIRED_METADATA_FIELDS}
    malformed_packets = []
    incomplete_packets = []
    complete_packet_count = 0
    metadata_records = []
    for packet in metadata_packets:
        try:
            payloads = _decode_packet_metadata_records(packet)
        except Exception as exc:
            malformed_packets.append({"packet_id": packet.packet_id, "error": str(exc)})
            continue
        for payload in payloads:
            metadata_records.append((packet, payload))

    for packet, payload in metadata_records:
        missing = []
        for field in REQUIRED_METADATA_FIELDS:
            if field in payload and payload[field] not in (None, ""):
                coverage[field] += 1
            else:
                missing.append(field)
        if missing:
            incomplete_packets.append({"packet_id": packet.packet_id, "missing_fields": missing})
        else:
            complete_packet_count += 1

    total = len(metadata_records)
    missing_fields = {field: total - count for field, count in coverage.items() if total > 0 and count < total}
    return {
        "required_fields": list(REQUIRED_METADATA_FIELDS),
        "metadata_packet_count": total,
        "schema_complete_packet_count": complete_packet_count,
        "field_coverage": coverage,
        "missing_fields": missing_fields,
        "malformed_packets": malformed_packets,
        "incomplete_packets": incomplete_packets,
        "status": "no_metadata_packets"
        if total == 0
        else ("complete" if not missing_fields and not malformed_packets else "incomplete"),
    }


def _decode_packet_header(packet: Packet) -> dict:
    return dict(packet.metadata or {})


def _decode_packet_metadata_records(packet: Packet) -> list[dict]:
    header = _decode_packet_header(packet)
    expanded = expand_metadata_bundle(header)
    if expanded:
        return expanded
    return [header]


def summarize_link_estimate(results: list[SendResult], scheduler_summary: dict | None = None) -> dict:
    confirmed = [result for result in results if result.confirmed]
    elapsed = [
        float(result.elapsed_ms) for result in confirmed if result.elapsed_ms is not None and result.elapsed_ms > 0
    ]
    elapsed_seconds = _scheduler_elapsed_seconds(scheduler_summary)
    confirmed_wire_bytes = sum(_result_wire_bytes(result) for result in confirmed)
    complete_ack_results = [
        result
        for result in results
        if normalize_application_ack(result)["confirmed"] is True
        and normalize_application_ack(result)["complete"] is True
    ]
    application_wire_bytes = sum(_result_wire_bytes(result) for result in complete_ack_results)
    module_goodput_bps = _safe_rate(confirmed_wire_bytes * 8.0, elapsed_seconds) if elapsed_seconds > 0 else None
    application_goodput_bps = _safe_rate(application_wire_bytes * 8.0, elapsed_seconds) if elapsed_seconds > 0 else None
    ack_summary = summarize_end_to_end_ack(results)
    estimated_loss_rate = ack_summary["ack_failure_rate"]
    return {
        "basis": "application_ack" if ack_summary["ack_observed_count"] else "application_ack_required",
        "module_confirmed_goodput_bps": module_goodput_bps,
        "application_goodput_bps": application_goodput_bps,
        "elapsed_seconds": elapsed_seconds,
        "module_elapsed_ms_avg": _mean(elapsed),
        "module_elapsed_ms_p95": _percentile(elapsed, 0.95),
        "application_rtt_ms_avg": ack_summary["rtt_ms_avg"],
        "application_rtt_ms_p95": ack_summary["rtt_ms_p95"],
        "estimated_loss_rate": estimated_loss_rate,
        "dynamic_quota_ready": bool(ack_summary["ack_observed_count"]),
    }


def summarize_payload_packetization(packets: list[Packet]) -> dict:
    total_bytes = sum(packet.bytes for packet in packets)
    by_layer: dict[str, dict[str, int]] = {}
    by_device: dict[str, dict[str, int]] = {}
    for packet in packets:
        layer_stats = by_layer.setdefault(packet.layer, {"packet_count": 0, "payload_bytes": 0})
        device_stats = by_device.setdefault(packet.device_id, {"packet_count": 0, "payload_bytes": 0})
        layer_stats["packet_count"] += 1
        layer_stats["payload_bytes"] += packet.bytes
        device_stats["packet_count"] += 1
        device_stats["payload_bytes"] += packet.bytes
    return {
        "packet_count": len(packets),
        "payload_bytes": total_bytes,
        "crc32_recorded_count": len(packets),
        "fragmentation_observed": any("fragment" in packet.packet_id.lower() for packet in packets),
        "bitstream_like_layers": int(
            sum(1 for packet in packets if packet.layer in {"roi_patch", "background_keyframe_summary"})
        ),
        "by_layer": by_layer,
        "by_device": by_device,
    }


def build_message_delivery_records(packets: list[Packet], results: list[SendResult]) -> list[dict[str, Any]]:
    packets_by_id = _packet_by_id(packets)
    groups: dict[str, list[SendResult]] = {}
    for result in results:
        groups.setdefault(base_packet_id(result.packet_id), []).append(result)

    records = []
    for base_id, group in groups.items():
        packet = packets_by_id.get(group[0].packet_id) or packets_by_id.get(base_id)
        acks = [normalize_application_ack(result) for result in group]
        observed_acks = [ack for ack in acks if ack["observed"]]
        confirmed_acks = [ack for ack in observed_acks if ack["confirmed"] is True]
        complete_acks = [ack for ack in observed_acks if ack["complete"] is True]
        failed_acks = [ack for ack in observed_acks if ack["confirmed"] is False]
        rtts = [ack["rtt_ms"] for ack in observed_acks if ack["rtt_ms"] is not None]
        errors = [
            error
            for error in (
                [getattr(result, "error", None) for result in group] + [ack["error"] for ack in observed_acks]
            )
            if error
        ]
        records.append(
            {
                "packet_id": base_id,
                "message_id": message_id_hex_for_packet(base_id),
                "device_id": packet.device_id if packet is not None else getattr(group[0], "device_id", None),
                "frame_idx": group[0].frame_idx,
                "layer": group[0].layer,
                "roi_idx": packet.roi_idx if packet is not None else getattr(group[0], "roi_idx", None),
                "rate_idx": packet.rate_idx if packet is not None else getattr(group[0], "rate_idx", None),
                "item_rate_idx": (packet.metadata or {}).get("item_rate_idx") if packet is not None else None,
                "quality_rank": (packet.metadata or {}).get("quality_rank") if packet is not None else None,
                "track_id": (packet.metadata or {}).get("track_id") if packet is not None else None,
                "fragment_count": len(group),
                "payload_bytes_sent": int(sum(result.bytes for result in group)),
                "wire_bytes_sent": int(sum(_result_wire_bytes(result) for result in group)),
                "module_confirmed": all(bool(result.confirmed) for result in group),
                "module_confirmed_fragment_count": int(sum(1 for result in group if result.confirmed)),
                "ack_observed": bool(observed_acks),
                "ack_confirmed": True if confirmed_acks else (False if failed_acks else None),
                "ack_complete": True if complete_acks else (False if failed_acks else None),
                "ack_rtt_ms": _mean(rtts),
                "ack_error": errors[0] if errors else None,
            }
        )
    return sorted(
        records, key=lambda item: (int(item.get("frame_idx") or 0), str(item.get("layer")), str(item.get("packet_id")))
    )


def build_compact_delivery_capabilities(args, packets, results, scheduler_summary, diagnostics) -> dict:
    ack_summary = summarize_end_to_end_ack(results)
    link = summarize_link_estimate(results, scheduler_summary)
    payload = summarize_payload_packetization(packets)
    metadata = summarize_metadata_schema(packets)
    scheduler_summary = scheduler_summary or {}
    return {
        "ack": {
            "status": ack_summary["status"],
            "message_count": ack_summary["message_count"],
            "observed": ack_summary["ack_observed_count"],
            "confirmed": ack_summary["ack_confirmed_count"],
            "complete": ack_summary["ack_complete_count"],
            "missing": ack_summary["ack_missing_count"],
            "observed_failed": ack_summary["ack_observed_failed_count"],
            "module_confirmed": ack_summary["module_confirmed_message_count"],
            "module_delivery_rate": ack_summary["module_delivery_rate"],
            "coverage_rate": ack_summary["ack_coverage_rate"],
            "success_rate": ack_summary["ack_success_rate"],
            "failure_rate": ack_summary["ack_failure_rate"],
            "observed_failure_rate": ack_summary["ack_observed_failure_rate"],
            "unobserved_rate": ack_summary["ack_unobserved_rate"],
            "required": ack_summary["ack_required"],
            "missing_is_delivery_failure": ack_summary["ack_missing_is_delivery_failure"],
            "rtt_ms_avg": ack_summary["rtt_ms_avg"],
        },
        "link": {
            "basis": link["basis"],
            "goodput_bps": link["application_goodput_bps"],
            "module_confirmed_goodput_bps": link["module_confirmed_goodput_bps"],
            "elapsed_seconds": link["elapsed_seconds"],
            "rtt_ms_p95": link["application_rtt_ms_p95"],
            "dynamic_quota_ready": link["dynamic_quota_ready"],
        },
        "scheduler_feedback": {
            "online": bool(scheduler_summary.get("online")),
            "round_count": int(scheduler_summary.get("round_count", 0) or 0),
            "missing_fragment_backlog": int(scheduler_summary.get("missing_fragment_backlog_packets", 0) or 0),
        },
        "payload": {
            "packets": payload["packet_count"],
            "bytes": payload["payload_bytes"],
            "fragmented": payload["fragmentation_observed"],
            "bitstream_packets": payload["bitstream_like_layers"],
            "by_layer": payload["by_layer"],
        },
        "metadata_schema": {
            "status": metadata["status"],
            "metadata_packets": metadata["metadata_packet_count"],
            "complete_packets": metadata["schema_complete_packet_count"],
            "missing_fields": sorted(metadata["missing_fields"]),
        },
    }


def summarize_diagnostics(diagnostics: list | None) -> dict:
    diagnostics = diagnostics or []
    ack_poll_errors = [
        item for item in diagnostics if isinstance(item, dict) and item.get("command") == "<tcp_round_ack_poll_error>"
    ]
    ack_poll_timeouts = [
        item for item in diagnostics if isinstance(item, dict) and item.get("command") == "<tcp_round_ack_poll_timeout>"
    ]
    socket_recoveries = [
        item
        for item in diagnostics
        if isinstance(item, dict) and ("socket_recovery" in item or "socket_recovery_trigger" in item)
    ]
    round_ack_queries = [item for item in diagnostics if isinstance(item, dict) and "round_ack_query_attempt" in item]
    fragment_retransmits = [
        item for item in diagnostics if isinstance(item, dict) and "round_ack_fragment_retransmit_attempt" in item
    ]
    critical_layer_retransmits = [
        item
        for item in diagnostics
        if isinstance(item, dict)
        and (
            "critical_layer_fragment_retransmit_attempt" in item
            or "round_ack_critical_layer_fragment_retransmit_attempt" in item
        )
    ]
    metadata_retransmits = [item for item in critical_layer_retransmits if "metadata" in set(item.get("layers") or [])]
    background_retransmits = [
        item for item in critical_layer_retransmits if "background_keyframe_summary" in set(item.get("layers") or [])
    ]
    roi_patch_retransmits = [
        item for item in critical_layer_retransmits if "roi_patch" in set(item.get("layers") or [])
    ]
    return {
        "tcp_ack_poll_error_count": len(ack_poll_errors),
        "tcp_ack_poll_timeout_count": len(ack_poll_timeouts),
        "round_ack_query_attempt_count": len(round_ack_queries),
        "round_ack_fragment_retransmit_attempt_count": len(fragment_retransmits),
        "critical_layer_fragment_retransmit_attempt_count": len(critical_layer_retransmits),
        "background_fragment_retransmit_attempt_count": len(background_retransmits),
        "metadata_fragment_retransmit_attempt_count": len(metadata_retransmits),
        "roi_patch_fragment_retransmit_attempt_count": len(roi_patch_retransmits),
        "socket_recovery_event_count": len(socket_recoveries),
    }


def write_delivery_log(
    output_root: Path,
    args,
    packets: list[Packet],
    results: list[SendResult],
    diagnostics=None,
    scheduler_summary=None,
) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / "delivery_log.json"
    payload = build_compact_delivery_log(args, packets, results, scheduler_summary, diagnostics)
    output_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    return output_path


def build_compact_delivery_log(args, packets, results, scheduler_summary, diagnostics) -> dict:
    capabilities = build_compact_delivery_capabilities(args, packets, results, scheduler_summary, diagnostics)
    summary = summarize_results(results)
    summary.update(
        {
            "source": args.source,
            "transport": "sim7080",
            "protocol": args.sim_protocol,
            "ack_status": capabilities["ack"]["status"],
            "ack_message_count": capabilities["ack"]["message_count"],
            "ack_confirmed_message_count": capabilities["ack"]["confirmed"],
            "ack_missing_message_count": capabilities["ack"]["missing"],
            "ack_coverage_rate": capabilities["ack"]["coverage_rate"],
            "ack_success_rate": capabilities["ack"]["success_rate"],
            "ack_failure_rate": capabilities["ack"]["failure_rate"],
        }
    )
    return {
        "summary": summary,
        "link": capabilities["link"],
        "scheduler": _compact_scheduler_summary(scheduler_summary or {}),
        "diagnostics": summarize_diagnostics(diagnostics),
        "messages": build_message_delivery_records(packets, results),
    }


def _compact_scheduler_summary(summary: dict) -> dict:
    keys = (
        "enabled",
        "online",
        "input_packets",
        "selected_packets",
        "input_bytes",
        "selected_bytes",
        "dropped_packets",
        "quota_bytes",
        "global_quota_bytes",
        "missing_fragment_backlog_packets",
        "missing_fragment_backlog_bytes",
        "round_count",
        "receiver_state_recovery_hint_count",
        "link_state",
        "missing_fragment_count",
        "node_link_states",
        "quota_allocation",
        "allocated_node_quotas",
        "global_quota_overflow_bytes",
        "selected_bytes_by_node",
    )
    compact = {key: summary[key] for key in keys if key in summary}
    if "rounds" in summary:
        compact["rounds"] = summary["rounds"]
    if "node_summaries" in summary:
        compact["node_summaries"] = {
            device_id: {
                key: value
                for key, value in node_summary.items()
                if key
                in {
                    "selected_packets",
                    "selected_bytes",
                    "dropped_packets",
                    "quota_bytes",
                    "selected_roi_group_count",
                    "required_background_group_count",
                    "optional_background_group_count",
                    "selected_optional_background_group_count",
                }
            }
            for device_id, node_summary in summary["node_summaries"].items()
        }
    return compact


def _result_wire_bytes(result) -> int:
    wire_bytes = getattr(result, "wire_bytes", None)
    if isinstance(wire_bytes, (int, float)) and wire_bytes >= 0:
        return int(wire_bytes)
    return int(getattr(result, "bytes", 0) or 0)


def _module_elapsed_seconds(results: list) -> float:
    elapsed = [float(getattr(result, "elapsed_ms", 0.0) or 0.0) for result in results]
    if not elapsed:
        return 0.0
    if any(getattr(result, "response", "") == "window_shared_header" for result in results):
        return max(elapsed) / 1000.0
    return sum(elapsed) / 1000.0


def _scheduler_elapsed_seconds(scheduler_summary: dict | None) -> float:
    if not isinstance(scheduler_summary, dict):
        return 0.0
    total = 0.0
    for round_record in scheduler_summary.get("rounds") or []:
        if not isinstance(round_record, dict):
            continue
        feedback = round_record.get("feedback")
        if not isinstance(feedback, dict):
            continue
        elapsed = feedback.get("round_elapsed_seconds")
        if isinstance(elapsed, (int, float)) and elapsed > 0:
            total += float(elapsed)
    return total
