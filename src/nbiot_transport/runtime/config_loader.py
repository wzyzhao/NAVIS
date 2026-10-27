import json
import re
from pathlib import Path
from types import SimpleNamespace

from nbiot_transport.feedback.missing_fragment_queue import load_missing_fragment_queue
from common.paths import PROJECT_ROOT

PACKAGE_ROOT = PROJECT_ROOT / "src" / "nbiot_transport"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "experiments" / "results" / "navis" / "default" / "sender"
DEFAULT_CONFIG = PACKAGE_ROOT / "navis.json"


def _load_config(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing config file: {path}")
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _resolve_path(value: str | Path | None, base_dir: Path, default: Path | None = None) -> Path | None:
    if value in (None, ""):
        return default
    path = Path(value)
    if not path.is_absolute():
        path = base_dir / path
    return path


def _sequence_slug(sequence_filter: str | list[str] | None) -> str:
    value = sequence_filter[0] if isinstance(sequence_filter, list) and sequence_filter else sequence_filter
    match = re.search(r"MOT17-(\d+)", str(value or ""), flags=re.IGNORECASE)
    if match:
        return f"mot17_{match.group(1)}"
    return "default"


def load_config(config_path: Path = DEFAULT_CONFIG) -> SimpleNamespace:
    """Load transport defaults and resolve paths relative to the configuration file."""
    config_path = config_path.resolve()
    config = _load_config(config_path)
    base_dir = config_path.parent
    defaults = {
        "source": "payload",
        "schedule_json": None,
        "candidate_manifest_json": None,
        "sequence_filter": [],
        "nodes": [],
        "node_allocation_policy": "health_weighted",
        "receiver_session_isolation_confirmed": False,
        "max_frames": 0,
        "max_packets": 0,
        "payload_bytes": 64,
        "packet_count": 1,
        "output_root": DEFAULT_OUTPUT_ROOT,
        "serial_port": "",
        "serial_baud": 115200,
        "sim_apn": "",
        "remote_host": "",
        "remote_port": 0,
        "sim_protocol": "UDP",
        "sim_connect_id": 0,
        "sim_timeout_seconds": 8.0,
        "sim_send_timeout_seconds": 30.0,
        "network_registration_timeout_seconds": 120.0,
        "network_registration_poll_seconds": 3.0,
        "skip_sim_init": False,
        "enable_application_ack": True,
        "application_ack_timeout_seconds": 5.0,
        "ack_initial_delay_seconds": 2.0,
        "ack_poll_interval_seconds": 0.3,
        "round_ack_query_attempts": 5,
        "node_round_recovery_attempts": 1,
        "node_round_timeout_seconds": 180.0,
        "manual_modem_recovery_enabled": False,
        "manual_modem_recovery_poll_seconds": 5.0,
        "manual_modem_recovery_max_wait_seconds": 0.0,
        "carecv_read_bytes": 4096,
        "scheduler_enabled": True,
        "selection_policy": "semantic",
        "budget_adaptation_enabled": True,
        "quota_bytes": 0,
        "max_packet_payload_bytes": 0,
        "metadata_packet_payload_bytes": 0,
        "metadata_bundle_per_window": True,
        "window_shared_header_enabled": True,
        "window_bundle_packet_payload_bytes": 1024,
        "roi_packet_payload_bytes": 0,
        "background_packet_payload_bytes": 0,
        "global_quota_bytes": 0,
        "max_roi_count": 0,
        "roi_dynamic_count_enabled": False,
        "min_roi_count_per_window": 0,
        "max_dynamic_roi_count_per_window": 0,
        "roi_budget_fill_enabled": False,
        "max_budget_fill_roi_count_per_window": 0,
        "roi_quality_score_thresholds": [0.35, 0.45, 0.60],
        "roi_track_continuity_bonus": 0.08,
        "roi_recent_track_memory": 32,
        "allow_background_summary": True,
        "background_initialization_required": True,
        "metadata_only_on_congestion": False,
        "enable_receiver_state": True,
        "metadata_stale_after_frames": 30,
        "roi_cache_size": 5,
        "track_stale_after_frames": 10,
        "background_change_threshold": 0.6,
        "scheduling_window_frames": 3,
        "dynamic_window_enabled": True,
        "window_metadata_quota_ratio": 0.6,
        "max_scheduling_window_frames": 3,
        "scheduling_round_seconds": 30.0,
        "initial_window_quota_bytes": 50000,
        "quota_safety_factor": 0.7,
        "quota_safety_factor_min": 0.35,
        "quota_loss_penalty": 0.7,
        "quota_ewma_alpha": 0.3,
        "min_quota_bytes": 128,
        "max_quota_bytes": 0,
        "missing_fragment_queue_json": None,
        "missing_fragment_max_attempts": 5,
        "fragment_retransmit_enabled": True,
        "fragment_retransmit_attempts": 1,
        "roi_patch_fragment_retransmit_attempts": 2,
        "background_summary_fragment_retransmit_attempts": 3,
        "background_initialization_reliable_attempts": 3,
        "socket_reopen_on_send_failures": 1,
        "socket_reopen_attempts": 1,
    }

    merged = {key: config.get(key, default) for key, default in defaults.items()}
    if isinstance(merged["output_root"], str):
        merged["output_root"] = merged["output_root"].replace(
            "{sequence_slug}", _sequence_slug(merged["sequence_filter"])
        )
    merged["config"] = config_path
    merged["schedule_json"] = _resolve_path(merged["schedule_json"], base_dir)
    merged["candidate_manifest_json"] = _resolve_path(merged["candidate_manifest_json"], base_dir)
    merged["output_root"] = _resolve_path(merged["output_root"], base_dir, DEFAULT_OUTPUT_ROOT)
    merged["missing_fragment_queue_json"] = _resolve_path(
        merged["missing_fragment_queue_json"],
        base_dir,
        merged["output_root"] / "missing_fragment_queue.json",
    )
    merged["nodes"] = _resolve_nodes(merged["nodes"], base_dir, merged)
    merged["missing_fragment_queue"] = load_missing_fragment_queue(merged["missing_fragment_queue_json"])
    return SimpleNamespace(**merged)


def _resolve_nodes(nodes: list, base_dir: Path, inherited: dict | None = None) -> list:
    resolved = []
    for idx, node in enumerate(nodes):
        item = dict(node)
        for key in (
            "max_roi_count",
            "roi_dynamic_count_enabled",
            "min_roi_count_per_window",
            "max_dynamic_roi_count_per_window",
            "roi_budget_fill_enabled",
            "max_budget_fill_roi_count_per_window",
            "max_packet_payload_bytes",
            "metadata_packet_payload_bytes",
            "roi_packet_payload_bytes",
            "background_packet_payload_bytes",
            "roi_track_continuity_bonus",
            "roi_recent_track_memory",
            "metadata_only_on_congestion",
        ):
            if inherited is not None and key in inherited:
                item.setdefault(key, inherited[key])
        item.setdefault("device_id", f"node_{idx + 1}")
        item["schedule_json"] = _resolve_path(item.get("schedule_json"), base_dir)
        item.setdefault("sequence_filter", [])
        item.setdefault("max_roi_count", 0)
        item.setdefault("roi_dynamic_count_enabled", False)
        item.setdefault("min_roi_count_per_window", 0)
        item.setdefault("max_dynamic_roi_count_per_window", 0)
        item.setdefault("roi_budget_fill_enabled", False)
        item.setdefault("max_budget_fill_roi_count_per_window", 0)
        item.setdefault("roi_track_continuity_bonus", 0.08)
        item.setdefault("roi_recent_track_memory", 32)
        item.setdefault("allow_background_summary", True)
        item.setdefault("background_initialization_required", True)
        item.setdefault("metadata_only_on_congestion", False)
        item.setdefault("max_packet_payload_bytes", 0)
        item.setdefault("metadata_packet_payload_bytes", 0)
        item.setdefault("metadata_bundle_per_window", None)
        item.setdefault("roi_packet_payload_bytes", 0)
        item.setdefault("background_packet_payload_bytes", 0)
        resolved.append(item)
    return resolved


def validate_config(args: SimpleNamespace) -> None:
    """Reject inconsistent source, budget and transport options before opening devices."""
    if args.selection_policy not in {"semantic", "static_fifo"}:
        raise ValueError("selection_policy must be semantic or static_fifo")
    if args.source not in ("payload", "candidates", "multi_schedule"):
        raise ValueError("source must be 'payload', 'candidates', or 'multi_schedule'.")
    if args.source == "candidates" and args.candidate_manifest_json is None:
        raise ValueError("candidate_manifest_json is required when source is 'candidates'.")
    if args.source == "multi_schedule" and not args.nodes:
        raise ValueError("nodes are required when source is 'multi_schedule'.")
    if args.source == "multi_schedule":
        if getattr(args, "node_allocation_policy", "health_weighted") not in {"health_weighted", "equal_share"}:
            raise ValueError("node_allocation_policy must be health_weighted or equal_share")
        ids = [node.get("device_id") for node in args.nodes]
        if any(not value for value in ids) or len(set(ids)) != len(ids):
            raise ValueError("Each node requires a unique nonempty device_id.")
        if any(not node.get("schedule_json") for node in args.nodes):
            raise ValueError("Each node requires schedule_json.")
        if str(args.sim_protocol).upper() != "TCP":
            raise ValueError("Parallel sending requires TCP for ordered session initialization")
        ports = [str(node.get("serial_port", "")).strip().casefold() for node in args.nodes]
        if any(not port for port in ports) or len(set(ports)) != len(ports):
            raise ValueError("Parallel sending requires a distinct serial_port for each node")
        if int(args.global_quota_bytes) <= 0:
            raise ValueError("multi_schedule requires a positive global_quota_bytes total link budget")
    if not args.remote_host or int(args.remote_port) <= 0:
        raise ValueError("remote_host and remote_port are required.")
    if not args.serial_port and args.source != "multi_schedule":
        raise ValueError("serial_port is required. Set it in src/nbiot_transport/navis.json.")
    if not bool(getattr(args, "enable_application_ack", True)):
        raise ValueError("enable_application_ack must be true; scheduling requires strong application round ACK.")
    if int(args.scheduling_window_frames) <= 0:
        raise ValueError("scheduling_window_frames must be positive.")
    if int(getattr(args, "max_scheduling_window_frames", 0) or 0) < 0:
        raise ValueError("max_scheduling_window_frames must be non-negative.")
    if float(getattr(args, "window_metadata_quota_ratio", 0.0) or 0.0) <= 0:
        raise ValueError("window_metadata_quota_ratio must be positive.")
    if float(args.scheduling_round_seconds) <= 0:
        raise ValueError("scheduling_round_seconds must be positive.")
    if float(getattr(args, "node_round_timeout_seconds", 0.0)) <= 0:
        raise ValueError("node_round_timeout_seconds must be positive.")
    if float(getattr(args, "manual_modem_recovery_poll_seconds", 0.0)) <= 0:
        raise ValueError("manual_modem_recovery_poll_seconds must be positive.")
    if float(getattr(args, "manual_modem_recovery_max_wait_seconds", 0.0)) < 0:
        raise ValueError("manual_modem_recovery_max_wait_seconds must be non-negative.")
    if float(getattr(args, "network_registration_timeout_seconds", 0.0)) <= 0:
        raise ValueError("network_registration_timeout_seconds must be positive.")
    if float(getattr(args, "network_registration_poll_seconds", 0.0)) <= 0:
        raise ValueError("network_registration_poll_seconds must be positive.")
