"""Replay NAVIS and the Static ELIC+JPEG baseline under one frozen byte budget.

This is an offline *selection* experiment with sender-style dynamic windows.
NAVIS receives a per-window semantic budget; Static ELIC+JPEG remains at the
fixed comparison budget.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.ablations.scheduler import (
    dynamic_frame_windows,
    load_json,
    proxy_count,
    roi_groups_in_frames,
    selected_roi_group_ids,
    variant_args,
    window_packets,
)
from nbiot_transport.core.scheduler import (
    packet_item_id,
    schedule_packets_for_nbiot,
    schedule_with_mandatory_reservation,
)


def static_args() -> SimpleNamespace:
    """The paper baseline: single-quality, FIFO, frozen budget, no state reuse."""
    return SimpleNamespace(scheduler_enabled=True, selection_policy="static_fifo")


def validate_aligned_frames(navis_frames: list[dict], baseline_frames: list[dict]) -> None:
    """Ensure both codecs are evaluated on exactly the same physical ROIs."""

    def groups(frames: list[dict]) -> set[str]:
        return {
            str(item["candidate_group_id"])
            for frame in frames
            for item in frame.get("items", [])
            if item.get("layer") == "roi_patch"
        }

    navis_ids, baseline_ids = groups(navis_frames), groups(baseline_frames)
    if navis_ids != baseline_ids:
        only_navis = sorted(navis_ids - baseline_ids)[:3]
        only_baseline = sorted(baseline_ids - navis_ids)[:3]
        raise ValueError(
            "Candidate manifests do not describe the same physical ROIs; "
            f"only in NAVIS={only_navis}, only in Static ELIC+JPEG={only_baseline}"
        )
    if [int(f["frame_idx"]) for f in navis_frames] != [int(f["frame_idx"]) for f in baseline_frames]:
        raise ValueError("Candidate manifests do not have identical frame order")


def cap_complete_items(packets, budget_bytes: int):
    """Enforce the common hard budget while retaining whole logical items.

    Production NAVIS may let its mandatory metadata/background set exceed a
    nominal control target.  A fixed-budget codec comparison cannot use that
    exception, so this replay removes an item as a whole when it does not fit.
    """
    groups, order = {}, []
    for packet in packets:
        key = packet_item_id(packet)
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(packet)
    selected, used, dropped = [], 0, 0
    for key in order:
        group = groups[key]
        size = sum(packet.bytes for packet in group)
        if used + size <= budget_bytes:
            selected.extend(group)
            used += size
        else:
            dropped += 1
    return selected, dropped


def run_method(name: str, frames: list[dict], args: argparse.Namespace) -> dict:
    is_static = name == "static_elic_jpeg_fifo"
    navis_min_budget = int(getattr(args, "navis_min_budget_bytes", 6000))
    navis_max_budget = int(getattr(args, "navis_max_budget_bytes", 40000))
    scheduler_args = static_args() if is_static else variant_args(args, "navis_ack")
    totals = {
        "variant": name,
        "round_count": 0,
        "expected_roi_groups": 0,
        "current_roi_groups": 0,
        "proxy_roi_groups": 0,
        "available_roi_groups": 0,
        "selected_bytes": 0,
        "input_bytes": 0,
        "semantic_demand_bytes": 0,
        "budget_bytes_available": 0,
        "metadata_selected_windows": 0,
        "metadata_windows": 0,
        "budget_violation_windows": 0,
        "hard_cap_dropped_items": 0,
    }
    cache: dict[str, int] = {}
    # NAVIS adapts its budget to metadata pressure; the static baseline keeps
    # the declared fixed budget.  The resulting windows follow the same rule
    # as the production sender (metadata target is 60% of quota).
    if is_static:
        windows = dynamic_frame_windows(frames, args.budget_bytes, max_window_frames=args.window_frames)
        window_quotas = [args.budget_bytes] * len(windows)
    else:
        demand_quotas = []
        for frame in frames:
            metadata = sum(
                max(1, int(item.get("bytes", 1) or 1))
                for item in frame.get("items", [])
                if item.get("layer") == "metadata"
            )
            demand_quotas.append(
                max(navis_min_budget, min(navis_max_budget, int(metadata / 0.6) if metadata else args.budget_bytes))
            )
        windows = dynamic_frame_windows(frames, demand_quotas, max_window_frames=args.window_frames)
        window_quotas = []
        for window in windows:
            metadata = sum(
                max(1, int(item.get("bytes", 1) or 1))
                for frame in window
                for item in frame.get("items", [])
                if item.get("layer") == "metadata"
            )
            window_quotas.append(
                max(navis_min_budget, min(navis_max_budget, int(metadata / 0.6) if metadata else args.budget_bytes))
            )
    for window, window_budget in zip(windows, window_quotas):
        active_budget = args.budget_bytes if is_static else window_budget
        packets = window_packets(
            window,
            device_id="node_1",
            metadata_bundle=True,
            roi_payload_limit=args.roi_payload_limit,
            background_payload_limit=args.background_payload_limit,
        )
        if is_static:
            # Unlike NAVIS, Static ELIC+JPEG has no mandatory reservation or
            # feedback controller: it traverses packet groups in static FIFO order.
            selected, summary = schedule_packets_for_nbiot(packets, scheduler_args, active_budget)
        else:
            selected, summary = schedule_with_mandatory_reservation(
                packets, scheduler_args, active_budget, {"basis": "offline_dynamic_budget"}
            )
            selected, dropped_items = cap_complete_items(selected, active_budget)
            totals["hard_cap_dropped_items"] += dropped_items
        expected = roi_groups_in_frames(window)
        current_ids = selected_roi_group_ids(selected)
        proxy = 0 if is_static else proxy_count(expected, current_ids, args.roi_cache_ttl_frames, cache)
        selected_bytes = sum(packet.bytes for packet in selected)
        totals["round_count"] += 1
        totals["expected_roi_groups"] += len(expected)
        totals["current_roi_groups"] += len(current_ids)
        totals["proxy_roi_groups"] += proxy
        totals["selected_bytes"] += selected_bytes
        # Pre-admission encoded semantic-message demand, including the
        # alternative quality representations offered to the scheduler.
        candidate_demand = sum(packet.bytes for packet in packets)
        totals["input_bytes"] += candidate_demand
        totals["semantic_demand_bytes"] += candidate_demand
        totals["budget_bytes_available"] += int(active_budget)
        totals["metadata_windows"] += 1
        totals["metadata_selected_windows"] += int(any(packet.layer == "metadata" for packet in selected))
        totals["budget_violation_windows"] += int(selected_bytes > active_budget)
    expected = max(1, totals["expected_roi_groups"])
    totals["available_roi_groups"] = totals["current_roi_groups"] + totals["proxy_roi_groups"]
    totals["current_roi_ratio"] = totals["current_roi_groups"] / expected
    totals["roi_availability"] = min(1.0, totals["available_roi_groups"] / expected)
    totals["metadata_completeness"] = totals["metadata_selected_windows"] / max(1, totals["metadata_windows"])
    totals["semantic_admission_ratio"] = totals["selected_bytes"] / max(1, totals["semantic_demand_bytes"])
    totals["budget_utilization"] = totals["selected_bytes"] / max(1, totals["budget_bytes_available"])
    return totals


def write_markdown(path: Path, output: dict) -> None:
    rows = output["results"]
    lines = [
        "# Frozen-budget offline comparison: NAVIS vs. Static ELIC+JPEG",
        "",
        "Both methods use the same source sequence; sender-style windows are formed separately under each budget. "
        f"Static ELIC+JPEG uses a hard {output['settings']['budget_bytes']} B/window budget; NAVIS uses a per-window dynamic budget.",
        "NAVIS mandatory messages are also capped to that budget; this is candidate-size selection replay, not a measured cellular-delivery experiment.",
        "",
        "| Method | Selected MB | Current ROI | Proxy ROI | Available ROI | Availability | Metadata windows | Cap-dropped items | Budget violations |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {variant} | {mb:.2f} | {current_roi_groups}/{expected_roi_groups} | "
            "{proxy_roi_groups} | {available_roi_groups}/{expected_roi_groups} | "
            "{roi_availability:.3%} | {metadata_selected_windows}/{metadata_windows} | "
            "{hard_cap_dropped_items} | {budget_violation_windows} |".format(
                mb=row["selected_bytes"] / 1_000_000, **row
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Byte-controlled offline NAVIS/Static ELIC+JPEG replay.")
    parser.add_argument(
        "--navis-manifest", type=Path, default=REPO_ROOT / "experiments/results/encoding/dcvc/payloads/candidate_manifest.json"
    )
    parser.add_argument(
        "--baseline-manifest",
        type=Path,
        default=REPO_ROOT / "experiments/results/encoding/elic/payloads/MOT17-09-FRCNN/run01/candidate_manifest.json",
    )
    parser.add_argument("--sequence", default="MOT17-09-FRCNN")
    parser.add_argument(
        "--output-root", type=Path, default=REPO_ROOT / "experiments/results/baselines/static_elic_replay/mot17-09"
    )
    parser.add_argument("--window-frames", type=int, default=5)
    parser.add_argument("--budget-bytes", type=int, default=20_000)
    parser.add_argument("--roi-payload-limit", type=int, default=1024)
    parser.add_argument("--background-payload-limit", type=int, default=512)
    parser.add_argument("--roi-cache-ttl-frames", type=int, default=10)
    parser.add_argument("--max-roi-count", type=int, default=12)
    parser.add_argument("--min-roi-count-per-window", type=int, default=2)
    parser.add_argument("--max-dynamic-roi-count-per-window", type=int, default=24)
    parser.add_argument("--max-budget-fill-roi-count-per-window", type=int, default=0)
    parser.add_argument("--navis-min-budget-bytes", type=int, default=6000)
    parser.add_argument("--navis-max-budget-bytes", type=int, default=40000)
    parser.add_argument("--roi-track-continuity-bonus", type=float, default=0.08)
    parser.add_argument("--roi-recent-track-memory", type=int, default=32)
    parser.add_argument("--roi-quality-score-thresholds", type=float, nargs=3, default=[0.35, 0.45, 0.60])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.budget_bytes <= 0:
        raise ValueError("budget-bytes must be positive")
    navis = [f for f in load_json(args.navis_manifest).get("frames", []) if f.get("sequence") == args.sequence]
    baseline = [f for f in load_json(args.baseline_manifest).get("frames", []) if f.get("sequence") == args.sequence]
    if not navis or not baseline:
        raise ValueError("No frames for the requested sequence in one or both candidate manifests")
    validate_aligned_frames(navis, baseline)
    args.output_root.mkdir(parents=True, exist_ok=True)
    output = {
        "experiment": "offline_dynamic_budget_static_baseline",
        "model": "Same source sequence and physical ROIs with sender-style dynamic windows. NAVIS uses semantic scheduling, a per-window metadata-pressure budget, and a 10-frame proxy cache; Static ELIC+JPEG uses a fixed hard budget and FIFO with no proxy. No modem or ACK delivery is simulated.",
        "navis_manifest": str(args.navis_manifest),
        "baseline_manifest": str(args.baseline_manifest),
        "sequence": args.sequence,
        "frame_count": len(navis),
        "settings": {key: value for key, value in vars(args).items() if not isinstance(value, Path)},
        "results": [run_method("navis_dynamic_budget", navis, args), run_method("static_elic_jpeg_fifo", baseline, args)],
    }
    (args.output_root / "summary.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    write_markdown(args.output_root / "report.md", output)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
