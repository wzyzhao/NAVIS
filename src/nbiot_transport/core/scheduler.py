import math
from argparse import Namespace
from collections import defaultdict
from types import SimpleNamespace

from nbiot_transport.core.background_metadata import background_required
from nbiot_transport.core.packetizer import Packet

BACKGROUND_LAYER = "background_keyframe_summary"


def _clamp_rate(value, default: float = 0.0) -> float:
    try:
        return max(0.0, min(float(value), 1.0))
    except (TypeError, ValueError):
        return default


def _node_link_state(link_state: dict | None, device_id: str) -> dict:
    if not isinstance(link_state, dict):
        return {}
    node_states = link_state.get("nodes") or link_state.get("node_link_states") or {}
    if isinstance(node_states, dict) and isinstance(node_states.get(device_id), dict):
        merged = dict(link_state)
        merged.pop("nodes", None)
        merged.pop("node_link_states", None)
        merged.update(node_states[device_id])
        return merged
    return dict(link_state)


def _node_health_score(node_state: dict | None) -> float:
    node_state = node_state or {}
    loss_rate = _clamp_rate(node_state.get("loss_rate"))
    health = max(0.05, 1.0 - loss_rate)

    rtt_ms = node_state.get("rtt_ms_p95", node_state.get("rtt_ms", 0.0))
    try:
        rtt_ms = max(0.0, float(rtt_ms or 0.0))
    except (TypeError, ValueError):
        rtt_ms = 0.0
    if rtt_ms > 0:
        health /= 1.0 + min(rtt_ms, 60000.0) / 30000.0

    goodput_bps = _safe_float(node_state.get("goodput_bps"))
    if goodput_bps > 0:
        health *= math.sqrt(goodput_bps)

    basis = str(node_state.get("basis") or "")
    if basis and basis not in {"application_ack", "initial_config"}:
        return 0.0
    return max(0.05, health)


def _normalize_node_weights(scores: dict[str, float]) -> dict[str, float]:
    positive = [score for score in scores.values() if score > 0]
    if not positive:
        return {device_id: 1.0 for device_id in scores}
    average = sum(positive) / len(positive)
    return {
        device_id: max(0.05, min(4.0, score / average)) if score > 0 else 0.05 for device_id, score in scores.items()
    }


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value or default)
    except (TypeError, ValueError):
        return default


def layer_rank(layer: str) -> int:
    if layer == "metadata":
        return 0
    if layer == "roi_patch":
        return 1
    if layer == BACKGROUND_LAYER:
        return 2
    return 3


def packet_layer_rank(packet: Packet) -> int:
    return layer_rank(packet.layer)


def value_cost_score(packet: Packet, goodput_bps: float = 0.0, rtt_ms: float = 0.0, loss_rate: float = 0.0) -> float:
    cost = max(packet.bytes, 1)
    if goodput_bps > 0:
        cost = packet.bytes * 8 / goodput_bps
    cost += max(rtt_ms, 0.0) / 1000.0
    cost *= 1.0 + max(0.0, min(float(loss_rate or 0.0), 1.0))
    return (packet.priority + 1.0) / max(cost, 1e-6)


def semantic_priority(packet: Packet) -> float:
    metadata = packet.metadata or {}
    if packet.layer == "roi_patch":
        return roi_value_score(metadata)
    base = _clamp_rate(packet.priority)
    uncertainty = _clamp_rate(metadata.get("uncertainty"))
    return 0.90 * base + 0.10 * uncertainty


def _metadata_track_id(metadata: dict):
    return metadata.get("track_id", metadata.get("roi_idx"))


def _track_continuity_bonus(metadata: dict, recent_track_ids: set[str] | None, bonus: float) -> float:
    if not recent_track_ids or bonus <= 0:
        return 0.0
    track_id = _metadata_track_id(metadata)
    if track_id in (None, ""):
        return 0.0
    return float(bonus) if str(track_id) in recent_track_ids else 0.0


def semantic_priority_with_context(
    packet: Packet,
    recent_track_ids: set[str] | None = None,
    track_continuity_bonus: float = 0.0,
    utility_multiplier: float = 1.0,
) -> float:
    value = semantic_priority(packet)
    if packet.layer == "roi_patch":
        value += _track_continuity_bonus(packet.metadata or {}, recent_track_ids, track_continuity_bonus)
        value *= max(0.0, float(utility_multiplier or 1.0))
    return max(0.0, min(1.0, float(value)))


def _metadata_rate(metadata: dict, key: str) -> float:
    return _clamp_rate(metadata.get(key))


def _metadata_optional_rate(metadata: dict, key: str, default: float) -> float:
    if metadata.get(key) is None:
        return _clamp_rate(default)
    return _metadata_rate(metadata, key)


def roi_value_score(
    metadata: dict,
) -> float:
    motion_score = _clamp_rate(metadata.get("motion_score"))
    confidence = _metadata_optional_rate(metadata, "confidence", 0.5)
    continuity_score = _metadata_optional_rate(metadata, "continuity_score", 0.0)
    uncertainty = _metadata_rate(metadata, "uncertainty")

    # Keep the rule-based ROI value intentionally compact: motion indicates
    # current semantic change, track continuity prevents flicker, and detector
    # confidence avoids spending bytes on weak proposals. Mask statistics are
    # treated as quality penalties rather than positive value terms.
    semantic_score = 0.65 * motion_score + 0.20 * continuity_score + 0.15 * confidence
    if str(metadata.get("category") or "other") == "attached":
        semantic_score += 0.03
    if uncertainty > 0.65:
        semantic_score -= 0.08 * ((uncertainty - 0.65) / 0.35)
    semantic_score = max(0.0, min(1.0, float(semantic_score)))
    edge_mask_coverage = _metadata_rate(metadata, "edge_mask_coverage")
    masked_black_ratio = _metadata_rate(metadata, "masked_black_ratio")
    value = semantic_score
    value *= 1.0 - 0.25 * edge_mask_coverage
    value *= 1.0 - 0.50 * masked_black_ratio
    return max(0.0, min(1.0, float(value)))


def packet_item_id(packet: Packet) -> str:
    return packet.packet_id.split(":fragment:", 1)[0]


def group_packet_items(packets: list[Packet]) -> list[list[Packet]]:
    groups: dict[str, list[Packet]] = {}
    for packet in packets:
        groups.setdefault(packet_item_id(packet), []).append(packet)
    return [sorted(group, key=lambda packet: (packet.fragment_index, packet.packet_id)) for group in groups.values()]


def group_value_cost_score(
    packets: list[Packet],
    goodput_bps: float = 0.0,
    rtt_ms: float = 0.0,
    loss_rate: float = 0.0,
) -> float:
    cost = max(sum(packet.bytes for packet in packets), 1)
    if goodput_bps > 0:
        cost = cost * 8 / goodput_bps
    cost += max(rtt_ms, 0.0) / 1000.0
    cost *= 1.0 + max(0.0, min(float(loss_rate or 0.0), 1.0))
    priority = max((semantic_priority(packet) for packet in packets), default=0.0)
    return (priority + 1.0) / max(cost, 1e-6)


def group_value_cost_score_with_context(
    packets: list[Packet],
    goodput_bps: float = 0.0,
    rtt_ms: float = 0.0,
    loss_rate: float = 0.0,
    recent_track_ids: set[str] | None = None,
    track_continuity_bonus: float = 0.0,
) -> float:
    cost = max(sum(packet.bytes for packet in packets), 1)
    if goodput_bps > 0:
        cost = cost * 8 / goodput_bps
    cost += max(rtt_ms, 0.0) / 1000.0
    cost *= 1.0 + max(0.0, min(float(loss_rate or 0.0), 1.0))
    priority = max(
        (
            semantic_priority_with_context(
                packet,
                recent_track_ids=recent_track_ids,
                track_continuity_bonus=track_continuity_bonus,
            )
            for packet in packets
        ),
        default=0.0,
    )
    return (priority + 1.0) / max(cost, 1e-6)


def roi_quality_score_thresholds(args) -> list[float]:
    thresholds = getattr(args, "roi_quality_score_thresholds", None)
    if thresholds is None:
        thresholds = [0.35, 0.45, 0.60]
    if isinstance(thresholds, str):
        thresholds = [value.strip() for value in thresholds.split(",") if value.strip()]
    try:
        values = sorted(float(value) for value in thresholds)
    except (TypeError, ValueError):
        values = [0.35, 0.45, 0.60]
    return [max(0.0, min(1.0, value)) for value in values]


def _quality_rank(group: list[Packet]) -> int:
    metadata = group[0].metadata or {}
    return int(metadata.get("quality_rank", group[0].rate_idx or 0) or 0)


def _target_quality_rank(score: float, ranks: list[int], thresholds: list[float]) -> int:
    if not ranks:
        return 0
    sorted_ranks = sorted(set(int(rank) for rank in ranks))
    level = 0
    for threshold in thresholds:
        if score >= threshold:
            level += 1
    return sorted_ranks[min(level, len(sorted_ranks) - 1)]


def _select_roi_option_by_score(
    options: list[list[Packet]],
    remaining: int | None,
    score: float,
    thresholds: list[float],
) -> list[Packet] | None:
    ranked = sorted(options, key=lambda group: (_quality_rank(group), sum(packet.bytes for packet in group)))
    target = _target_quality_rank(score, [_quality_rank(group) for group in ranked], thresholds)
    preferred = [
        group
        for group in ranked
        if _quality_rank(group) <= target and (remaining is None or sum(packet.bytes for packet in group) <= remaining)
    ]
    if preferred:
        return max(preferred, key=lambda group: (_quality_rank(group), -sum(packet.bytes for packet in group)))
    fallback = [group for group in ranked if remaining is None or sum(packet.bytes for packet in group) <= remaining]
    if fallback:
        return max(fallback, key=lambda group: (_quality_rank(group), -sum(packet.bytes for packet in group)))
    return None


def _single_quality_option(options: list[list[Packet]]) -> list[list[Packet]]:
    """Keep the highest encoded quality for a physical ROI.

    This is the controlled Single-Q ablation: the scheduler still decides
    whether the ROI fits, but it cannot downgrade it to a smaller candidate.
    """
    if not options:
        return []
    return [
        max(
            options,
            key=lambda group: (_quality_rank(group), -sum(packet.bytes for packet in group)),
        )
    ]


def _option_min_bytes(options: list[list[Packet]]) -> int:
    return min((sum(packet.bytes for packet in group) for group in options), default=0)


def _affordable_roi_group_count(option_sets: list[tuple[int, int, list[list[Packet]]]], budget: int | None) -> int:
    if budget is None:
        return len(option_sets)
    remaining = max(0, int(budget))
    count = 0
    for _, _, options in option_sets:
        item_bytes = _option_min_bytes(options)
        if item_bytes <= 0:
            continue
        if item_bytes > remaining:
            continue
        remaining -= item_bytes
        count += 1
    return count


def _roi_reserve_bytes(
    option_sets: list[tuple[int, int, list[list[Packet]]]],
    start_index: int,
    required_count: int,
) -> int:
    if required_count <= 0:
        return 0
    future = [
        _option_min_bytes(options) for _, _, options in option_sets[start_index:] if _option_min_bytes(options) > 0
    ]
    if len(future) < required_count:
        return 0
    return sum(sorted(future)[:required_count])


def schedule_packets_for_nbiot(
    packets: list[Packet],
    args,
    quota_bytes: int | None = None,
    link_state: dict | None = None,
):
    if not args.scheduler_enabled:
        return packets, {"enabled": False, "input_packets": len(packets), "selected_packets": len(packets)}
    if getattr(args, "selection_policy", "semantic") == "static_fifo":
        selected, seen = [], set()
        used = 0
        groups = sorted(
            group_packet_items(packets),
            key=lambda g: (
                g[0].layer == BACKGROUND_LAYER,
                g[0].frame_idx,
                g[0].roi_idx if g[0].roi_idx is not None else -1,
                g[0].packet_id,
            ),
        )
        for group in groups:
            key = (group[0].metadata or {}).get("candidate_group_id")
            size = sum(p.bytes for p in group)
            if (key and key in seen) or used + size > int(quota_bytes or 0):
                continue
            selected.extend(group)
            used += size
            if key:
                seen.add(key)
        return selected, _summary(
            packets,
            selected,
            {
                "enabled": True,
                "policy": "static frame/ROI order; single quality; skip if not fitting",
                "selected_roi_group_count": sum(g[0].layer == "roi_patch" for g in group_packet_items(selected)),
            },
        )
    if getattr(args, "source", "") == "multi_schedule":
        selected, summary = centralized_multi_node_schedule(packets, args, quota_bytes, link_state)
    else:
        selected, summary = schedule_single_node_packets(
            packets,
            args,
            device_id="node_1",
            quota_bytes=quota_bytes,
            link_state=link_state,
        )
    return selected, summary


def schedule_single_node_packets(
    packets: list[Packet],
    args,
    device_id: str,
    quota_bytes: int | None = None,
    link_state: dict | None = None,
):
    if quota_bytes is None:
        quota_bytes = int(getattr(args, "quota_bytes", 0) or 0)
    recent_track_ids = {str(value) for value in getattr(args, "_recent_roi_track_ids", []) if value not in (None, "")}
    selected, summary = _schedule_node(
        packets=packets,
        device_id=device_id,
        quota_bytes=quota_bytes,
        max_roi_count=int(getattr(args, "max_roi_count", 0) or 0),
        roi_dynamic_count_enabled=bool(getattr(args, "roi_dynamic_count_enabled", False)),
        min_roi_count_per_window=int(getattr(args, "min_roi_count_per_window", 0) or 0),
        max_dynamic_roi_count_per_window=int(getattr(args, "max_dynamic_roi_count_per_window", 0) or 0),
        roi_budget_fill_enabled=bool(getattr(args, "roi_budget_fill_enabled", False)),
        max_budget_fill_roi_count_per_window=int(getattr(args, "max_budget_fill_roi_count_per_window", 0) or 0),
        allow_background_summary=bool(getattr(args, "allow_background_summary", True)),
        background_initialization_required=bool(getattr(args, "background_initialization_required", True)),
        metadata_only=bool(getattr(args, "metadata_only_on_congestion", False)),
        roi_quality_thresholds=roi_quality_score_thresholds(args),
        roi_single_quality=bool(getattr(args, "roi_single_quality", False)),
        link_state=link_state,
        recent_track_ids=recent_track_ids,
        track_continuity_bonus=float(getattr(args, "roi_track_continuity_bonus", 0.0) or 0.0),
    )
    memory = max(0, int(getattr(args, "roi_recent_track_memory", 32) or 0))
    if memory > 0:
        selected_tracks = [str(value) for value in summary.get("selected_roi_track_ids", []) if value not in (None, "")]
        if selected_tracks:
            merged = list(dict.fromkeys(selected_tracks + list(getattr(args, "_recent_roi_track_ids", []))))[:memory]
            setattr(args, "_recent_roi_track_ids", merged)
    return selected, summary


def schedule_with_mandatory_reservation(
    packets: list[Packet],
    args: Namespace | SimpleNamespace,
    quota_bytes: int,
    link_state: dict | None = None,
) -> tuple[list[Packet], dict]:
    """Reserve mandatory messages first; optional payload never creates overflow.

    Nominal quota is a control target. The sole exception is an indivisible
    mandatory set larger than that target, bounded by max_quota_bytes.
    """
    if getattr(args, "source", "") == "multi_schedule":
        return centralized_multi_node_schedule(packets, args, quota_bytes, link_state)
    metadata = [p for p in packets if p.layer == "metadata"]
    required = [p for p in packets if p.layer == BACKGROUND_LAYER and background_required(p.metadata)]
    optional = [
        p
        for p in packets
        if p.layer != "metadata" and not (p.layer == BACKGROUND_LAYER and background_required(p.metadata))
    ]
    mandatory_bytes = sum(p.bytes for p in metadata + required)
    # Required backgrounds exist only at manifest refresh events in this window.
    # A cached background/version referenced by metadata is not a new payload.
    nominal = int(quota_bytes)
    hard_limit = int(getattr(args, "max_quota_bytes", 40000) or 40000)
    if nominal <= 0 or nominal > hard_limit:
        raise ValueError(f"Nominal quota must be in 1..{hard_limit} bytes")
    if mandatory_bytes > hard_limit:
        raise ValueError(
            f"Mandatory messages require {mandatory_bytes} B, exceeding the {hard_limit} B hard limit; shrink the window or encode smaller messages before sending"
        )
    remaining = max(0, nominal - mandatory_bytes)
    if remaining > 0:
        selected_optional, summary = schedule_packets_for_nbiot(optional, args, remaining, link_state)
    else:
        selected_optional = []
        summary = {"enabled": True, "selected_roi_group_count": 0, "selected_roi_track_ids": []}
    selected = metadata + required + selected_optional
    summary.update(
        quota_bytes=nominal,
        nominal_quota_bytes=nominal,
        policy="reserve metadata and required background; schedule optional messages within remaining quota",
        mandatory_reserved_bytes=mandatory_bytes,
        optional_quota_bytes=remaining,
        metadata_reserved_bytes=sum(p.bytes for p in metadata),
        background_refresh_due=bool(required),
        required_background_reserved_bytes=sum(p.bytes for p in required),
        effective_quota_bytes=max(nominal, mandatory_bytes),
        hard_limit_bytes=hard_limit,
        mandatory_overflow_bytes=max(0, mandatory_bytes - nominal),
        total_selection_overflow_bytes=max(0, sum(p.bytes for p in selected) - nominal),
        forced_required_background_bytes=sum(p.bytes for p in required),
        budget_policy="mandatory_reserved_first; bounded_mandatory_only_exception",
    )
    assert sum(p.bytes for p in selected_optional) <= remaining
    assert sum(p.bytes for p in selected) <= hard_limit
    return selected, _summary(packets, selected, summary)


def centralized_multi_node_schedule(
    packets: list[Packet],
    args,
    quota_bytes: int | None = None,
    link_state: dict | None = None,
):
    packets_by_node: dict[str, list[Packet]] = defaultdict(list)
    for packet in packets:
        packets_by_node[packet.device_id].append(packet)

    node_config = {node["device_id"]: node for node in getattr(args, "nodes", [])}
    global_quota = int(
        quota_bytes
        if quota_bytes is not None
        else (getattr(args, "global_quota_bytes", 0) or getattr(args, "quota_bytes", 0) or 0)
    )
    hard_limit = int(getattr(args, "max_quota_bytes", 0) or 0)
    if hard_limit > 0 and global_quota > hard_limit:
        raise ValueError(f"Total link quota exceeds the {hard_limit} B hard limit")

    node_summaries = {}
    selected_by_node_packets = {}
    node_ids = sorted(packets_by_node)
    node_health_scores = {}
    node_link_states = {}
    node_runtime_args = {}
    for node_index, device_id in enumerate(node_ids):
        config = node_config.get(device_id, {})
        node_link_state = _node_link_state(link_state, device_id)
        node_link_states[device_id] = node_link_state
        node_health_scores[device_id] = _node_health_score(node_link_state)
        node_runtime_args[device_id] = SimpleNamespace(
            source="candidates",
            scheduler_enabled=getattr(args, "scheduler_enabled", True),
            selection_policy=getattr(args, "selection_policy", "semantic"),
            max_quota_bytes=getattr(args, "max_quota_bytes", 0),
            quota_bytes=0,
            max_roi_count=int(config.get("max_roi_count", getattr(args, "max_roi_count", 0)) or 0),
            roi_dynamic_count_enabled=bool(
                config.get("roi_dynamic_count_enabled", getattr(args, "roi_dynamic_count_enabled", False))
            ),
            min_roi_count_per_window=int(
                config.get("min_roi_count_per_window", getattr(args, "min_roi_count_per_window", 0)) or 0
            ),
            max_dynamic_roi_count_per_window=int(
                config.get(
                    "max_dynamic_roi_count_per_window",
                    getattr(args, "max_dynamic_roi_count_per_window", 0),
                )
                or 0
            ),
            roi_budget_fill_enabled=bool(
                config.get("roi_budget_fill_enabled", getattr(args, "roi_budget_fill_enabled", False))
            ),
            max_budget_fill_roi_count_per_window=int(
                config.get(
                    "max_budget_fill_roi_count_per_window",
                    getattr(args, "max_budget_fill_roi_count_per_window", 0),
                )
                or 0
            ),
            allow_background_summary=bool(
                config.get("allow_background_summary", getattr(args, "allow_background_summary", True))
            ),
            background_initialization_required=bool(
                config.get(
                    "background_initialization_required", getattr(args, "background_initialization_required", True)
                )
            ),
            metadata_only_on_congestion=bool(
                config.get("metadata_only_on_congestion", getattr(args, "metadata_only_on_congestion", False))
            ),
            roi_quality_score_thresholds=roi_quality_score_thresholds(args),
            roi_single_quality=bool(config.get("roi_single_quality", getattr(args, "roi_single_quality", False))),
            roi_track_continuity_bonus=float(
                config.get("roi_track_continuity_bonus", getattr(args, "roi_track_continuity_bonus", 0.0)) or 0.0
            ),
            roi_recent_track_memory=int(
                config.get("roi_recent_track_memory", getattr(args, "roi_recent_track_memory", 32)) or 0
            ),
            _recent_roi_track_ids=list((getattr(args, "_recent_roi_track_ids_by_device", {}) or {}).get(device_id, [])),
        )

    node_health_weights = _normalize_node_weights(node_health_scores)
    allocation_policy = getattr(args, "node_allocation_policy", "health_weighted")
    if allocation_policy == "equal_share":
        node_health_weights = {device_id: 1.0 for device_id in node_ids}
    # Mandatory messages belong to each node's full budget. Transfer budget
    # between nodes to satisfy these floors.  As in the single-node scheduler,
    # an indivisible mandatory set may exceed the nominal control target, but
    # never the configured hard link limit.
    mandatory_by_node = {
        device: sum(
            p.bytes
            for p in group
            if p.layer == "metadata" or (p.layer == BACKGROUND_LAYER and background_required(p.metadata))
        )
        for device, group in packets_by_node.items()
    }
    mandatory_total_bytes = sum(mandatory_by_node.values())
    effective_global_quota = max(global_quota, mandatory_total_bytes) if global_quota > 0 else 0
    if global_quota > 0:
        if mandatory_total_bytes > hard_limit:
            breakdown = ", ".join(f"{device}={value}" for device, value in sorted(mandatory_by_node.items()))
            raise ValueError(
                f"Mandatory messages require {mandatory_total_bytes} B ({breakdown}), exceeding the "
                f"{hard_limit} B hard limit; shrink the window or encode smaller messages"
            )
    allocated_node_quotas = _allocate_node_quotas(effective_global_quota, packets_by_node, node_health_weights)
    if effective_global_quota > 0:
        for device in node_ids:
            needed = max(0, mandatory_by_node[device] - allocated_node_quotas[device])
            for donor in node_ids:
                take = min(needed, max(0, allocated_node_quotas[donor] - mandatory_by_node[donor]))
                allocated_node_quotas[donor] -= take
                allocated_node_quotas[device] += take
                needed -= take

    def run_node_schedules() -> None:
        selected_by_node_packets.clear()
        node_summaries.clear()
        for current_device_id in node_ids:
            node_args = node_runtime_args[current_device_id]
            node_args.quota_bytes = allocated_node_quotas.get(current_device_id, 0)
            node_args._recent_roi_track_ids = list(
                (getattr(args, "_recent_roi_track_ids_by_device", {}) or {}).get(current_device_id, [])
            )
            if effective_global_quota > 0 and node_args.quota_bytes == 0:
                node_selected = []
                node_summary = _summary(
                    packets_by_node[current_device_id],
                    node_selected,
                    {
                        "quota_bytes": 0,
                        "selected_roi_track_ids": [],
                        "mandatory_overflow_bytes": sum(p.bytes for p in node_selected),
                        "dropped_by_reason": {"quota": len(packets_by_node[current_device_id]) - len(node_selected)},
                    },
                )
                selected_by_node_packets[current_device_id] = node_selected
                node_summaries[current_device_id] = node_summary
                continue
            if effective_global_quota > 0:
                node_selected, node_summary = schedule_with_mandatory_reservation(
                    packets_by_node[current_device_id],
                    node_args,
                    quota_bytes=node_args.quota_bytes,
                    link_state=node_link_states[current_device_id],
                )
            else:
                node_selected, node_summary = schedule_single_node_packets(
                    packets_by_node[current_device_id],
                    node_args,
                    current_device_id,
                    quota_bytes=0,
                    link_state=node_link_states[current_device_id],
                )
            selected_by_node_packets[current_device_id] = node_selected
            node_summaries[current_device_id] = node_summary

    run_node_schedules()

    top_up_rounds = 0
    if effective_global_quota > 0:
        while top_up_rounds < max(1, len(node_ids)):
            selected_bytes_by_device = {
                device_id: sum(packet.bytes for packet in node_packets)
                for device_id, node_packets in selected_by_node_packets.items()
            }
            selected_bytes = sum(selected_bytes_by_device.values())
            if selected_bytes >= effective_global_quota:
                break
            constrained_nodes = [
                device_id
                for device_id, summary in node_summaries.items()
                if int((summary.get("dropped_by_reason") or {}).get("quota", 0) or 0) > 0
            ]
            if not constrained_nodes:
                break
            if allocation_policy == "equal_share":
                grants = _allocate_node_quotas(
                    effective_global_quota - selected_bytes,
                    {device: packets_by_node[device] for device in constrained_nodes},
                    {device: 1.0 for device in constrained_nodes},
                )
                allocated_node_quotas = {
                    device: selected_bytes_by_device[device] + grants.get(device, 0) for device in node_ids
                }
                top_up_rounds += 1
                run_node_schedules()
                if sum(p.bytes for group in selected_by_node_packets.values() for p in group) <= selected_bytes:
                    break
                continue
            target = max(
                constrained_nodes,
                key=lambda device_id: (
                    float(node_health_weights.get(device_id, 1.0) or 1.0),
                    int((node_summaries[device_id].get("dropped_by_reason") or {}).get("quota", 0) or 0),
                    device_id,
                ),
            )
            demand = {
                device_id: int(sum(packet.bytes for packet in node_packets))
                for device_id, node_packets in packets_by_node.items()
            }
            target_need = max(0, demand.get(target, 0) - allocated_node_quotas.get(target, 0))
            unallocated_budget = max(0, effective_global_quota - sum(allocated_node_quotas.values()))
            unused_allocations = {
                device_id: max(
                    0,
                    allocated_node_quotas.get(device_id, 0) - selected_bytes_by_device.get(device_id, 0),
                )
                for device_id in node_ids
                if device_id != target
            }
            available = unallocated_budget + sum(unused_allocations.values())
            grant = min(target_need, available)
            if grant <= 0:
                break
            from_unallocated = min(unallocated_budget, grant)
            allocated_node_quotas[target] += from_unallocated
            remaining_grant = grant - from_unallocated
            for donor in sorted(
                unused_allocations,
                key=lambda device_id: (float(node_health_weights.get(device_id, 1.0) or 1.0), device_id),
            ):
                if remaining_grant <= 0:
                    break
                take = min(unused_allocations[donor], remaining_grant)
                allocated_node_quotas[donor] -= take
                allocated_node_quotas[target] += take
                remaining_grant -= take
            top_up_rounds += 1
            previous_selected_bytes = selected_bytes
            run_node_schedules()
            selected_bytes = sum(
                packet.bytes for node_packets in selected_by_node_packets.values() for packet in node_packets
            )
            if selected_bytes <= previous_selected_bytes:
                break

    selected = [packet for device_id in node_ids for packet in selected_by_node_packets.get(device_id, [])]
    recent_by_device = {}
    for device_id, summary in node_summaries.items():
        tracks = [str(value) for value in summary.get("selected_roi_track_ids", []) if value not in (None, "")]
        previous = list((getattr(args, "_recent_roi_track_ids_by_device", {}) or {}).get(device_id, []))
        memory = int(getattr(node_runtime_args[device_id], "roi_recent_track_memory", 32) or 0)
        if memory > 0:
            recent_by_device[device_id] = list(dict.fromkeys(tracks + previous))[:memory]
    if recent_by_device:
        setattr(args, "_recent_roi_track_ids_by_device", recent_by_device)
    fair_scheduling = (
        "link_state_budget_allocation_then_single_node_schedule" if global_quota > 0 else "not_needed_no_global_quota"
    )

    selected_by_node = _packet_counts_by_node(selected)
    selected_bytes_by_node = _packet_bytes_by_node(selected)
    global_quota_overflow_bytes = max(0, sum(selected_bytes_by_node.values()) - global_quota) if global_quota > 0 else 0
    summary = _summary(
        input_packets=packets,
        selected_packets=selected,
        extra={
            "enabled": True,
            "policy": "centralized multi-node quota; link-aware budget allocation, then single-node scheduling",
            "global_quota_bytes": global_quota,
            "effective_global_quota_bytes": effective_global_quota,
            "mandatory_global_overflow_bytes": max(0, mandatory_total_bytes - global_quota),
            "fair_scheduling": fair_scheduling,
            "quota_allocation": "proportional_to_node_health_and_capped_by_node_demand",
            "allocated_node_quotas": allocated_node_quotas,
            "quota_top_up_rounds": top_up_rounds,
            "global_quota_overflow_bytes": global_quota_overflow_bytes,
            "node_health_scores": node_health_scores,
            "node_allocation_policy": allocation_policy,
            "mandatory_bytes_by_node": mandatory_by_node,
            "node_health_weights": node_health_weights,
            "node_link_states": node_link_states,
            "input_packets_by_node": _packet_counts_by_node(packets),
            "selected_packets_by_node": selected_by_node,
            "selected_bytes_by_node": selected_bytes_by_node,
            "node_summaries": node_summaries,
        },
    )
    return selected, summary


def _packet_counts_by_node(packets: list[Packet]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for packet in packets:
        counts[packet.device_id] = counts.get(packet.device_id, 0) + 1
    return counts


def _packet_bytes_by_node(packets: list[Packet]) -> dict[str, int]:
    totals: dict[str, int] = {}
    for packet in packets:
        totals[packet.device_id] = totals.get(packet.device_id, 0) + int(packet.bytes)
    return totals


def _allocate_node_quotas(
    global_quota: int,
    packets_by_node: dict[str, list[Packet]],
    node_weights: dict[str, float],
) -> dict[str, int]:
    node_ids = sorted(packets_by_node)
    if global_quota <= 0:
        return {device_id: 0 for device_id in node_ids}

    demand = {device_id: int(sum(packet.bytes for packet in packets)) for device_id, packets in packets_by_node.items()}
    quotas = {device_id: 0 for device_id in node_ids}
    remaining = min(int(global_quota), sum(demand.values()))
    active = {device_id for device_id in node_ids if demand.get(device_id, 0) > 0}

    while remaining > 0 and active:
        weight_sum = sum(max(0.05, float(node_weights.get(device_id, 1.0) or 1.0)) for device_id in active)
        if weight_sum <= 0:
            weight_sum = float(len(active))
        shares = {
            device_id: remaining * max(0.05, float(node_weights.get(device_id, 1.0) or 1.0)) / weight_sum
            for device_id in active
        }
        progressed = 0
        for device_id in sorted(active):
            available = demand[device_id] - quotas[device_id]
            add = min(available, int(shares[device_id]))
            if add <= 0:
                continue
            quotas[device_id] += add
            progressed += add
        remaining -= progressed
        active = {device_id for device_id in active if quotas[device_id] < demand[device_id]}
        if progressed > 0:
            continue

        for device_id in sorted(active, key=lambda item: (-(shares[item] % 1.0), item)):
            if remaining <= 0:
                break
            quotas[device_id] += 1
            remaining -= 1
            if quotas[device_id] >= demand[device_id]:
                active.remove(device_id)
    return quotas


def _schedule_node(
    packets: list[Packet],
    device_id: str,
    quota_bytes: int,
    max_roi_count: int,
    roi_dynamic_count_enabled: bool,
    min_roi_count_per_window: int,
    max_dynamic_roi_count_per_window: int,
    roi_budget_fill_enabled: bool,
    max_budget_fill_roi_count_per_window: int,
    allow_background_summary: bool,
    background_initialization_required: bool,
    metadata_only: bool,
    roi_quality_thresholds: list[float] | None = None,
    roi_single_quality: bool = False,
    link_state: dict | None = None,
    recent_track_ids: set[str] | None = None,
    track_continuity_bonus: float = 0.0,
):
    link_state = link_state or {}
    roi_quality_thresholds = roi_quality_thresholds or [0.35, 0.45, 0.60]
    recent_track_ids = {str(value) for value in (recent_track_ids or set()) if value not in (None, "")}
    track_continuity_bonus = max(0.0, float(track_continuity_bonus or 0.0))
    goodput_bps = _safe_float(link_state.get("goodput_bps"))
    rtt_ms = _safe_float(link_state.get("rtt_ms_p95", link_state.get("rtt_ms")))
    loss_rate = _clamp_rate(link_state.get("loss_rate"))
    metadata_groups = group_packet_items([packet for packet in packets if packet.layer == "metadata"])
    roi_groups = group_packet_items([packet for packet in packets if packet.layer == "roi_patch"])
    background_groups = group_packet_items([packet for packet in packets if packet.layer == BACKGROUND_LAYER])
    unknown_groups = group_packet_items(
        [packet for packet in packets if packet.layer not in {"metadata", "roi_patch", BACKGROUND_LAYER}]
    )
    metadata_only_requested = bool(metadata_only)
    required_background_groups = [group for group in background_groups if background_required(group[0].metadata)]
    optional_background_groups = [group for group in background_groups if not background_required(group[0].metadata)]
    background_bytes = (
        sum(packet.bytes for group in optional_background_groups for packet in group) if allow_background_summary else 0
    )
    enhancement_groups = (optional_background_groups if allow_background_summary else []) + roi_groups
    smallest_enhancement_bytes = min(
        (sum(packet.bytes for packet in group) for group in enhancement_groups),
        default=0,
    )
    metadata_only_active = (
        metadata_only_requested
        and quota_bytes > 0
        and smallest_enhancement_bytes > 0
        and smallest_enhancement_bytes > quota_bytes
    )

    selected = []
    used_bytes = 0
    dropped_by_reason = {
        "quota": 0,
        "max_roi_count": 0,
        "background_disabled": 0,
        "background_not_due": 0,
        "metadata_only": 0,
        "unknown_layer": sum(len(group) for group in unknown_groups),
    }

    mandatory_overflow_bytes = 0
    selected_optional_background_group_count = 0
    selected_roi_group_count = 0
    selected_roi_track_ids: list[str] = []
    effective_max_roi_count = 0
    base_dynamic_roi_count_limit = 0
    budget_fill_expanded_roi_count = 0
    affordable_roi_group_count = 0
    min_roi_target = max(0, int(min_roi_count_per_window or 0))

    def try_add_group(group: list[Packet], reason: str = "quota", mandatory: bool = False) -> bool:
        nonlocal used_bytes
        nonlocal mandatory_overflow_bytes
        item_bytes = sum(packet.bytes for packet in group)
        if quota_bytes > 0 and used_bytes + item_bytes > quota_bytes and not mandatory:
            dropped_by_reason[reason] += len(group)
            return False
        selected.extend(group)
        used_bytes += item_bytes
        if mandatory and quota_bytes > 0:
            mandatory_overflow_bytes = max(mandatory_overflow_bytes, used_bytes - quota_bytes)
        return True

    for group in sorted(metadata_groups, key=lambda group: (group[0].frame_idx, packet_item_id(group[0]))):
        try_add_group(group, mandatory=True)

    if metadata_only_active:
        dropped_by_reason["metadata_only"] += sum(len(group) for group in background_groups)
    elif allow_background_summary:
        optional_background_groups = sorted(
            optional_background_groups,
            key=lambda group: (
                -max(semantic_priority(packet) for packet in group),
                group[0].frame_idx,
                packet_item_id(group[0]),
            ),
        )
    else:
        dropped_by_reason["background_disabled"] += sum(len(group) for group in background_groups)

    if metadata_only_active:
        dropped_by_reason["metadata_only"] += sum(len(group) for group in roi_groups)
        dropped_by_reason["metadata_only"] += sum(len(group) for group in optional_background_groups)
    else:
        roi_options_by_frame: dict[int, list[list[list[Packet]]]] = defaultdict(list)
        alternatives: dict[str, list[list[Packet]]] = defaultdict(list)
        for group in roi_groups:
            metadata = group[0].metadata or {}
            candidate_group_id = metadata.get("candidate_group_id") or packet_item_id(group[0])
            alternatives[str(candidate_group_id)].append(group)
        roi_option_sets: list[list[list[Packet]]] = []
        for options in alternatives.values():
            options.sort(
                key=lambda group: (
                    -_quality_rank(group),
                    sum(packet.bytes for packet in group),
                )
            )
            if roi_single_quality:
                options = _single_quality_option(options)
            roi_option_sets.append(options)
            roi_options_by_frame[options[0][0].frame_idx].append(options)
        for option_sets in roi_options_by_frame.values():
            option_sets.sort(
                key=lambda options: (
                    -max(
                        group_value_cost_score_with_context(
                            group,
                            goodput_bps=goodput_bps,
                            rtt_ms=rtt_ms,
                            loss_rate=loss_rate,
                            recent_track_ids=recent_track_ids,
                            track_continuity_bonus=track_continuity_bonus,
                        )
                        for group in options
                    ),
                    min(sum(packet.bytes for packet in group) for group in options),
                    str((options[0][0].metadata or {}).get("candidate_group_id") or packet_item_id(options[0][0])),
                )
            )

        ordered_roi_option_sets: list[tuple[int, int, list[list[Packet]]]] = []
        # Eq. (3) ranks every physical ROI in the window by semantic value
        # per estimated delivery cost.
        ordered_roi_option_sets = [(0, options[0][0].frame_idx, options) for options in roi_option_sets]
        ordered_roi_option_sets.sort(
            key=lambda entry: (
                -max(
                    group_value_cost_score_with_context(
                        group,
                        goodput_bps=goodput_bps,
                        rtt_ms=rtt_ms,
                        loss_rate=loss_rate,
                        recent_track_ids=recent_track_ids,
                        track_continuity_bonus=track_continuity_bonus,
                    )
                    for group in entry[2]
                ),
                min(sum(packet.bytes for packet in group) for group in entry[2]),
                entry[1],
                str((entry[2][0][0].metadata or {}).get("candidate_group_id") or packet_item_id(entry[2][0][0])),
            )
        )

        roi_budget = None if quota_bytes <= 0 else max(0, quota_bytes - used_bytes)
        affordable_roi_group_count = _affordable_roi_group_count(ordered_roi_option_sets, roi_budget)
        if roi_dynamic_count_enabled:
            dynamic_limit = len(ordered_roi_option_sets)
            if max_dynamic_roi_count_per_window > 0:
                dynamic_limit = min(dynamic_limit, max_dynamic_roi_count_per_window)
            if quota_bytes > 0:
                dynamic_limit = min(dynamic_limit, affordable_roi_group_count)
            base_dynamic_roi_count_limit = dynamic_limit
            # Preserve the configured semantic-diversity cap for the first pass,
            # then admit more already ranked ROI groups only when their minimum
            # representation fits in the residual quota.  This turns unused
            # capacity into extra evidence without exceeding the byte budget.
            if roi_budget_fill_enabled and quota_bytes > 0:
                fill_limit = min(len(ordered_roi_option_sets), affordable_roi_group_count)
                if max_budget_fill_roi_count_per_window > 0:
                    fill_limit = min(fill_limit, max_budget_fill_roi_count_per_window)
                dynamic_limit = max(dynamic_limit, fill_limit)
                budget_fill_expanded_roi_count = max(0, dynamic_limit - base_dynamic_roi_count_limit)
            effective_max_roi_count = dynamic_limit
        else:
            effective_max_roi_count = max_roi_count
        if max_roi_count > 0 and not roi_dynamic_count_enabled:
            effective_max_roi_count = max_roi_count
        if roi_dynamic_count_enabled and effective_max_roi_count <= 0 and max_roi_count > 0:
            effective_max_roi_count = min(max_roi_count, len(ordered_roi_option_sets))

        def remaining_roi_packets(start_index: int) -> int:
            return sum(len(group) for _, _, options in ordered_roi_option_sets[start_index:] for group in options)

        for option_index, (_round_idx, _frame_idx, options) in enumerate(ordered_roi_option_sets):
            if effective_max_roi_count > 0 and selected_roi_group_count >= effective_max_roi_count:
                dropped_by_reason["max_roi_count"] += remaining_roi_packets(option_index)
                break
            remaining = None if quota_bytes <= 0 else max(0, quota_bytes - used_bytes)
            if remaining is not None and selected_roi_group_count < min_roi_target:
                required_after_current = max(0, min_roi_target - selected_roi_group_count - 1)
                reserve = _roi_reserve_bytes(
                    ordered_roi_option_sets,
                    option_index + 1,
                    required_after_current,
                )
                remaining_for_quality = max(0, remaining - reserve)
            else:
                remaining_for_quality = remaining
            roi_score = max(
                semantic_priority_with_context(
                    packet,
                    recent_track_ids=recent_track_ids,
                    track_continuity_bonus=track_continuity_bonus,
                )
                for group in options
                for packet in group
            )
            selected_option = _select_roi_option_by_score(
                options,
                remaining=remaining_for_quality,
                score=roi_score,
                thresholds=roi_quality_thresholds,
            )
            if selected_option is None and remaining_for_quality != remaining:
                selected_option = _select_roi_option_by_score(
                    options,
                    remaining=remaining,
                    score=roi_score,
                    thresholds=roi_quality_thresholds,
                )
            if selected_option is None:
                dropped_by_reason["quota"] += sum(len(group) for group in options)
                continue
            if try_add_group(selected_option):
                selected_roi_group_count += 1
                selected_track_id = _metadata_track_id(selected_option[0].metadata or {})
                if selected_track_id not in (None, ""):
                    selected_roi_track_ids.append(str(selected_track_id))
                    recent_track_ids.add(str(selected_track_id))

        for group in optional_background_groups:
            if try_add_group(group):
                selected_optional_background_group_count += 1

    return selected, _summary(
        input_packets=packets,
        selected_packets=selected,
        extra={
            "enabled": True,
            "device_id": device_id,
            "policy": "scheduler handles metadata first, ROI by link-aware value/cost, then optional background if it fits quota; required background is forced by sender",
            "quota_bytes": quota_bytes,
            "max_roi_count": max_roi_count,
            "max_roi_count_scope": "per_window",
            "roi_dynamic_count_enabled": roi_dynamic_count_enabled,
            "min_roi_count_per_window": min_roi_target,
            "max_dynamic_roi_count_per_window": max_dynamic_roi_count_per_window,
            "roi_budget_fill_enabled": roi_budget_fill_enabled,
            "max_budget_fill_roi_count_per_window": max_budget_fill_roi_count_per_window,
            "base_dynamic_roi_count_limit": base_dynamic_roi_count_limit if not metadata_only_active else 0,
            "budget_fill_expanded_roi_count": budget_fill_expanded_roi_count if not metadata_only_active else 0,
            "selected_roi_groups_beyond_base_limit": (
                max(0, selected_roi_group_count - base_dynamic_roi_count_limit)
                if not metadata_only_active and roi_budget_fill_enabled
                else 0
            ),
            "effective_max_roi_count": effective_max_roi_count if not metadata_only_active else 0,
            "affordable_roi_group_count": affordable_roi_group_count if not metadata_only_active else 0,
            "selected_roi_group_count": selected_roi_group_count,
            "metadata_group_count": len(metadata_groups),
            "roi_ordering": "global_link_aware_value_per_cost",
            "roi_single_quality": roi_single_quality,
            "roi_single_quality_policy": "highest_available" if roi_single_quality else None,
            "roi_rate_selection": "score-thresholded target quality with quota fallback",
            "roi_quality_score_thresholds": roi_quality_thresholds,
            "roi_priority_basis": "nbiot_transport ROI value from scene metrics divided by bytes/goodput/rtt/loss cost",
            "roi_track_continuity_bonus": track_continuity_bonus,
            "recent_roi_track_ids": sorted(recent_track_ids),
            "selected_roi_track_ids": selected_roi_track_ids,
            "allow_background_summary": allow_background_summary,
            "forced_background_required": background_initialization_required,
            "required_background_group_count": len(required_background_groups),
            "optional_background_group_count": len(optional_background_groups),
            "selected_optional_background_group_count": selected_optional_background_group_count,
            "mandatory_overflow_bytes": mandatory_overflow_bytes,
            "metadata_only_on_congestion": metadata_only_requested,
            "metadata_only_active": metadata_only_active,
            "metadata_only_trigger": ("metadata_fits_but_no_enhancement_fits" if metadata_only_active else None),
            "smallest_enhancement_bytes": smallest_enhancement_bytes,
            "background_bytes": background_bytes,
            "link_state": link_state,
            "dropped_by_reason": dropped_by_reason,
        },
    )


def _summary(input_packets: list[Packet], selected_packets: list[Packet], extra: dict):
    summary = dict(extra)
    summary.update(
        {
            "input_packets": len(input_packets),
            "selected_packets": len(selected_packets),
            "input_bytes": int(sum(packet.bytes for packet in input_packets)),
            "selected_bytes": int(sum(packet.bytes for packet in selected_packets)),
            "metadata_packets": int(sum(1 for packet in selected_packets if packet.layer == "metadata")),
            "roi_packets": int(sum(1 for packet in selected_packets if packet.layer == "roi_patch")),
            "background_packets": int(sum(1 for packet in selected_packets if packet.layer == BACKGROUND_LAYER)),
            "dropped_packets": len(input_packets) - len(selected_packets),
        }
    )
    return summary
