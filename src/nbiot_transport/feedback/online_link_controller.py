import math
from collections import defaultdict

from nbiot_transport.feedback.ack_utils import normalize_application_ack


class OnlineLinkController:
    def __init__(self, args):
        self.current_quota_bytes = int(getattr(args, "initial_window_quota_bytes", 0) or 0)
        self.total_link_budget = (
            int(getattr(args, "global_quota_bytes", 0) or 0) if getattr(args, "source", "") == "multi_schedule" else 0
        )
        if self.total_link_budget > 0:
            self.current_quota_bytes = min(self.current_quota_bytes or self.total_link_budget, self.total_link_budget)
        self.initial_quota_bytes = self.current_quota_bytes
        self.adaptation_enabled = bool(getattr(args, "budget_adaptation_enabled", True))
        self.target_round_seconds = float(getattr(args, "scheduling_round_seconds", 30.0) or 30.0)
        self.safety_factor = float(getattr(args, "quota_safety_factor", 0.7) or 0.7)
        self.minimum_safety_factor = float(getattr(args, "quota_safety_factor_min", 0.35) or 0.35)
        self.loss_penalty = float(getattr(args, "quota_loss_penalty", 0.7) or 0.7)
        self.alpha = max(0.0, min(float(getattr(args, "quota_ewma_alpha", 0.3) or 0.3), 1.0))
        self.min_quota_bytes = int(getattr(args, "min_quota_bytes", 0) or 0)
        self.max_quota_bytes = int(getattr(args, "max_quota_bytes", 0) or 0)
        self.smoothed = {}

    @property
    def link_state(self) -> dict:
        return {
            "goodput_bps": float(self.smoothed.get("goodput_bps") or 0.0),
            "rtt_ms": float(self.smoothed.get("rtt_ms_p95") or 0.0),
            "loss_rate": float(self.smoothed.get("loss_rate") or 0.0),
            "basis": self.smoothed.get("basis", "initial_config"),
        }

    def observe_round(self, results: list, elapsed_seconds: float | None = None) -> dict:
        observed = summarize_round_results(results, elapsed_seconds=elapsed_seconds)
        return self.observe_observation(observed)

    def observe_observation(self, observed: dict) -> dict:
        """Update the controller from an application-ACK observation.

        Runtime transport supplies this observation through :meth:`observe_round`.
        Deterministic offline replays use the same update path after simulating
        complete application messages against a recorded capacity trace.
        """
        if observed.get("basis") != "application_ack":
            raise RuntimeError(
                "application round ACK is required for online scheduling; no quota or link-state fallback is allowed"
            )

        self.smoothed = smooth_link_state(self.smoothed, observed, self.alpha)
        dynamic_safety = max(
            self.minimum_safety_factor,
            self.safety_factor * (1.0 - self.loss_penalty * float(self.smoothed.get("loss_rate") or 0.0)),
        )
        goodput_bps = float(self.smoothed.get("goodput_bps") or 0.0)
        next_quota = int(goodput_bps * self.target_round_seconds * dynamic_safety / 8.0)
        if self.min_quota_bytes > 0:
            next_quota = max(self.min_quota_bytes, next_quota)
        if self.max_quota_bytes > 0:
            next_quota = min(self.max_quota_bytes, next_quota)
        if self.total_link_budget > 0:
            next_quota = min(self.total_link_budget, next_quota)
        self.current_quota_bytes = max(0, next_quota)
        if not self.adaptation_enabled:
            self.current_quota_bytes = self.initial_quota_bytes
        return {
            "observed": observed,
            "quota_observation": observed,
            "smoothed": dict(self.smoothed),
            "dynamic_safety_factor": dynamic_safety,
            "ack_required": True,
            "budget_adaptation_enabled": self.adaptation_enabled,
            "next_quota_bytes": self.current_quota_bytes,
        }


def summarize_round_results(results: list, elapsed_seconds: float | None = None) -> dict:
    grouped = defaultdict(list)
    for result in results:
        grouped[str(result.packet_id).split(":fragment:", 1)[0]].append(result)

    ack_observed = False
    delivered_groups = []
    rtts = []
    for group in grouped.values():
        acks = [normalize_application_ack(result) for result in group]
        ack_observed = ack_observed or any(ack["observed"] for ack in acks)
        complete = any(ack["confirmed"] is True and ack["complete"] is True for ack in acks)
        if complete:
            delivered_groups.append(group)
        rtts.extend(float(ack["rtt_ms"]) for ack in acks if ack["rtt_ms"] is not None)

    if elapsed_seconds is None:
        elapsed_seconds = (
            sum(
                max(
                    float(getattr(result, "elapsed_ms", 0.0) or 0.0),
                    float(normalize_application_ack(result).get("rtt_ms") or 0.0),
                    0.0,
                )
                for result in results
            )
            / 1000.0
        )
    # One byte contribution per confirmed fragment, not per retry attempt.
    delivered_wire_bytes = sum(
        sum(
            max(_result_wire_bytes(r) for r in group if r.packet_id == packet_id)
            for packet_id in {r.packet_id for r in group}
        )
        for group in delivered_groups
    )
    attempted = len(grouped)
    delivered = len(delivered_groups)
    return {
        "basis": "application_ack" if ack_observed else "application_ack_required",
        "ack_required": True,
        "attempted_message_count": attempted,
        "delivered_message_count": delivered,
        "failed_message_count": attempted - delivered,
        "delivered_wire_bytes": delivered_wire_bytes,
        "elapsed_seconds": elapsed_seconds,
        "goodput_bps": delivered_wire_bytes * 8.0 / elapsed_seconds if elapsed_seconds > 0 else 0.0,
        "rtt_ms_avg": _mean(rtts),
        "rtt_ms_p95": _percentile(rtts, 0.95),
        "loss_rate": (attempted - delivered) / attempted if attempted else 0.0,
    }


def _result_wire_bytes(result) -> int:
    wire_bytes = getattr(result, "wire_bytes", None)
    if isinstance(wire_bytes, (int, float)) and wire_bytes >= 0:
        return int(wire_bytes)
    return int(getattr(result, "bytes", 0) or 0)


def smooth_link_state(previous: dict, current: dict, alpha: float) -> dict:
    if not previous:
        return dict(current)
    smoothed = dict(current)
    for key in ("delivered_wire_bytes", "elapsed_seconds", "goodput_bps", "rtt_ms_avg", "rtt_ms_p95", "loss_rate"):
        old = previous.get(key)
        new = current.get(key)
        if old is not None and new is not None:
            smoothed[key] = (1.0 - alpha) * float(old) + alpha * float(new)
    return smoothed


def _mean(values: list[float]):
    return sum(values) / len(values) if values else None


def _percentile(values: list[float], percentile: float):
    if not values:
        return None
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    rank = (len(values) - 1) * percentile
    lower = int(math.floor(rank))
    upper = int(math.ceil(rank))
    if lower == upper:
        return values[lower]
    return values[lower] + (values[upper] - values[lower]) * (rank - lower)
