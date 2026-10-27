from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from types import SimpleNamespace

from common.encoding_layout import load_artifact_json

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nbiot_transport.core.packetizer import Packet
from nbiot_transport.core.scheduler import schedule_with_mandatory_reservation

BACKGROUND_LAYER = "background_keyframe_summary"


def schedule_window(packets, args, quota, link_state):
    return schedule_with_mandatory_reservation(packets, args, quota, link_state)


def load_json(path: Path) -> dict:
    return load_artifact_json(path)


def fragments_for_item(
    *,
    device_id: str,
    sequence: str,
    frame_idx: int,
    item_index: int,
    item: dict,
    payload_limit: int,
) -> list[Packet]:
    layer = str(item.get("layer") or "payload")
    total_bytes = max(1, int(item.get("bytes", 1) or 1))
    fragment_count = max(1, math.ceil(total_bytes / max(1, payload_limit)))
    packets = []
    base_id = f"{device_id}:{sequence}:0:{frame_idx}:{item_index}:{layer}"
    for fragment_index in range(fragment_count):
        remaining = total_bytes - fragment_index * payload_limit
        fragment_bytes = min(payload_limit, remaining)
        packet_id = f"{base_id}:fragment:{fragment_index:04d}" if fragment_count > 1 else base_id
        metadata = {
            "sequence": sequence,
            "frame_idx": frame_idx,
            "frame_id": int(item.get("frame_id", frame_idx) or frame_idx),
            "layer": layer,
            "roi_idx": item.get("roi_idx"),
            "track_id": item.get("track_id"),
            "candidate_group_id": item.get("candidate_group_id"),
            "quality_rank": int(item.get("quality_rank", item.get("rate_idx", 0)) or 0),
            "rate_idx": item.get("rate_idx"),
            "motion_score": float(item.get("motion_score", 0.0) or 0.0),
            "background_change_score": float(item.get("background_change_score", 0.0) or 0.0),
            "uncertainty": float(item.get("uncertainty", 0.0) or 0.0),
            "confidence": item.get("confidence"),
            "continuity_score": item.get("continuity_score"),
            "mask_coverage": item.get("mask_coverage"),
            "edge_mask_coverage": item.get("edge_mask_coverage"),
            "masked_black_ratio": item.get("masked_black_ratio"),
            "category": item.get("category"),
            "background": item.get("background") or {},
        }
        packets.append(
            Packet(
                packet_id=packet_id,
                device_id=device_id,
                frame_idx=frame_idx,
                frame_id=frame_idx,
                layer=layer,
                payload=b"0" * fragment_bytes,
                roi_idx=item.get("roi_idx"),
                rate_idx=item.get("rate_idx"),
                priority=float(item.get("priority", 0.0) or 0.0),
                fragment_index=fragment_index,
                fragment_count=fragment_count,
                metadata=metadata,
            )
        )
    return packets


def metadata_bundle_packet(device_id: str, sequence: str, frames: list[dict]) -> Packet:
    frame_idx = int(frames[0]["frame_idx"])
    total_bytes = 0
    for frame in frames:
        for item in frame.get("items", []):
            if item.get("layer") == "metadata":
                total_bytes += max(1, int(item.get("bytes", 1) or 1))
    return Packet(
        packet_id=f"{device_id}:{sequence}:metadata_window:{frame_idx}",
        device_id=device_id,
        frame_idx=frame_idx,
        frame_id=frame_idx,
        layer="metadata",
        payload=b"0" * max(1, total_bytes),
        priority=1.0,
        metadata={
            "sequence": sequence,
            "frame_idx": frame_idx,
            "layer": "metadata",
            "bundled_frame_count": len(frames),
        },
    )


def window_packets(
    frames: list[dict],
    *,
    device_id: str,
    metadata_bundle: bool,
    roi_payload_limit: int,
    background_payload_limit: int,
) -> list[Packet]:
    packets: list[Packet] = []
    if not frames:
        return packets
    sequence = str(frames[0].get("sequence") or device_id)
    item_index = 0
    if metadata_bundle:
        packets.append(metadata_bundle_packet(device_id, sequence, frames))
    for frame in frames:
        frame_idx = int(frame["frame_idx"])
        for item in frame.get("items", []):
            layer = item.get("layer")
            if layer == "metadata" and metadata_bundle:
                continue
            limit = background_payload_limit if layer == BACKGROUND_LAYER else roi_payload_limit
            if layer == "metadata":
                limit = max(roi_payload_limit, background_payload_limit)
            packets.extend(
                fragments_for_item(
                    device_id=device_id,
                    sequence=sequence,
                    frame_idx=frame_idx,
                    item_index=item_index,
                    item=item,
                    payload_limit=max(1, limit),
                )
            )
            item_index += 1
    return packets


def dynamic_frame_windows(
    frames: list[dict], quota_bytes: int | list[int], *, max_window_frames: int = 5, metadata_quota_ratio: float = 0.6
) -> list[list[dict]]:
    """Form one-to-cap-frame windows using the same metadata pressure rule as the sender."""
    if not frames:
        return []
    quotas = quota_bytes if isinstance(quota_bytes, list) else [int(quota_bytes)]
    windows, pos, round_index = [], 0, 0
    cap = max(1, int(max_window_frames))
    while pos < len(frames):
        quota = int(quotas[min(round_index, len(quotas) - 1)] or 0)
        target = max(1, int(quota * float(metadata_quota_ratio)))
        selected, metadata_bytes = [], 0
        while pos < len(frames) and len(selected) < cap:
            frame = frames[pos]
            frame_metadata = sum(
                max(1, int(item.get("bytes", 1) or 1))
                for item in frame.get("items", [])
                if item.get("layer") == "metadata"
            )
            if selected and metadata_bytes + frame_metadata > target:
                break
            selected.append(frame)
            metadata_bytes += frame_metadata
            pos += 1
        if not selected:
            selected = [frames[pos]]
            pos += 1
        windows.append(selected)
        round_index += 1
    return windows


def roi_groups_in_frames(frames: list[dict]) -> list[dict]:
    groups: dict[str, dict] = {}
    for frame in frames:
        for item in frame.get("items", []):
            if item.get("layer") != "roi_patch":
                continue
            group_id = str(item.get("candidate_group_id") or f"{frame['frame_idx']}:{item.get('roi_idx')}")
            groups.setdefault(
                group_id,
                {
                    "candidate_group_id": group_id,
                    "frame_idx": int(frame["frame_idx"]),
                    "track_id": item.get("track_id"),
                    "roi_idx": item.get("roi_idx"),
                    "motion_score": float(item.get("motion_score", 0.0) or 0.0),
                },
            )
    return sorted(groups.values(), key=lambda item: (item["frame_idx"], str(item["candidate_group_id"])))


def selected_roi_group_ids(packets: list[Packet]) -> set[str]:
    ids = set()
    for packet in packets:
        if packet.layer != "roi_patch":
            continue
        metadata = packet.metadata or {}
        group_id = metadata.get("candidate_group_id")
        if group_id:
            ids.add(str(group_id))
    return ids


def proxy_count(expected_rois: list[dict], current_ids: set[str], ttl_frames: int, cache: dict | None = None) -> int:
    if ttl_frames <= 0:
        return 0
    cache = {} if cache is None else cache
    proxies = 0
    for roi in expected_rois:
        frame_idx = int(roi["frame_idx"])
        group_id = str(roi["candidate_group_id"])
        track_id = roi.get("track_id")
        if track_id in (None, ""):
            continue
        track_key = str(track_id)
        if group_id in current_ids:
            cache[track_key] = frame_idx
        elif track_key in cache and 0 < frame_idx - cache[track_key] <= ttl_frames:
            proxies += 1
    return proxies


def summarize_variant(name: str, rounds: list[dict], ttl_frames: int) -> dict:
    totals = {
        "variant": name,
        "round_count": len(rounds),
        "input_packets": 0,
        "selected_packets": 0,
        "input_bytes": 0,
        "selected_bytes": 0,
        "metadata_windows": 0,
        "metadata_selected_windows": 0,
        "expected_roi_groups": 0,
        "current_roi_groups": 0,
        "proxy_roi_groups": 0,
        "background_packets": 0,
        "mandatory_overflow_bytes": 0,
        "dropped_quota_packets": 0,
        "dropped_max_roi_packets": 0,
        "dropped_metadata_only_packets": 0,
    }
    selected_quality = []
    cache = {}
    quotas = []
    for item in rounds:
        summary = item["summary"]
        expected_rois = item["expected_rois"]
        current_ids = item["current_roi_group_ids"]
        totals["input_packets"] += int(summary.get("input_packets", 0) or 0)
        totals["selected_packets"] += int(summary.get("selected_packets", 0) or 0)
        totals["input_bytes"] += int(summary.get("input_bytes", 0) or 0)
        totals["selected_bytes"] += int(summary.get("selected_bytes", 0) or 0)
        totals["metadata_windows"] += 1
        totals["metadata_selected_windows"] += 1 if int(summary.get("metadata_packets", 0) or 0) > 0 else 0
        totals["expected_roi_groups"] += len(expected_rois)
        totals["current_roi_groups"] += len(current_ids)
        totals["proxy_roi_groups"] += proxy_count(expected_rois, current_ids, ttl_frames, cache)
        totals["background_packets"] += int(summary.get("background_packets", 0) or 0)
        totals["mandatory_overflow_bytes"] += int(summary.get("mandatory_overflow_bytes", 0) or 0)
        dropped = summary.get("dropped_by_reason") or {}
        totals["dropped_quota_packets"] += int(dropped.get("quota", 0) or 0)
        totals["dropped_max_roi_packets"] += int(dropped.get("max_roi_count", 0) or 0)
        totals["dropped_metadata_only_packets"] += int(dropped.get("metadata_only", 0) or 0)
        quotas.append(int(summary.get("quota_bytes", 0) or 0))
        selected_quality.extend(item["selected_quality_ranks"])
    expected = max(1, totals["expected_roi_groups"])
    totals["metadata_completeness"] = totals["metadata_selected_windows"] / max(1, totals["metadata_windows"])
    totals["current_roi_ratio"] = totals["current_roi_groups"] / expected
    totals["proxy_roi_ratio"] = totals["proxy_roi_groups"] / expected
    totals["roi_availability"] = min(1.0, (totals["current_roi_groups"] + totals["proxy_roi_groups"]) / expected)
    totals["available_roi_groups"] = totals["current_roi_groups"] + totals["proxy_roi_groups"]
    totals["roi_transmission_rate"] = totals["current_roi_groups"] / max(1, totals["round_count"])
    totals["roi_availability_rate"] = totals["available_roi_groups"] / max(1, totals["round_count"])
    totals["roi_per_selected_mb"] = totals["available_roi_groups"] / max(1e-9, totals["selected_bytes"] / 1_000_000.0)
    totals["selected_input_packet_ratio"] = totals["selected_packets"] / max(1, totals["input_packets"])
    totals["avg_selected_quality_rank"] = statistics.fmean(selected_quality) if selected_quality else None
    totals["avg_quota_bytes"] = statistics.fmean(quotas) if quotas else 0
    return totals


def round_link_state(round_record: dict) -> dict:
    state = round_record.get("link_state_before") or {}
    return {
        "basis": state.get("basis") or "application_ack",
        "goodput_bps": float(state.get("goodput_bps", 0.0) or 0.0),
        "rtt_ms_p95": float(state.get("rtt_ms", state.get("rtt_ms_p95", 0.0)) or 0.0),
        "loss_rate": float(state.get("loss_rate", 0.0) or 0.0),
    }


def variant_args(base: argparse.Namespace, variant: str) -> SimpleNamespace:
    args = SimpleNamespace(
        scheduler_enabled=True,
        source="candidates",
        quota_bytes=0,
        max_quota_bytes=40000,
        max_roi_count=base.max_roi_count,
        roi_dynamic_count_enabled=True,
        min_roi_count_per_window=base.min_roi_count_per_window,
        max_dynamic_roi_count_per_window=base.max_dynamic_roi_count_per_window,
        roi_budget_fill_enabled=False,
        max_budget_fill_roi_count_per_window=base.max_budget_fill_roi_count_per_window,
        allow_background_summary=True,
        background_initialization_required=True,
        metadata_only_on_congestion=True,
        roi_quality_score_thresholds=list(base.roi_quality_score_thresholds),
        roi_track_continuity_bonus=base.roi_track_continuity_bonus,
        roi_recent_track_memory=base.roi_recent_track_memory,
        roi_single_quality=False,
    )
    if variant == "budget_fill":
        args.roi_budget_fill_enabled = True
    elif variant == "single_q_roi":
        args.roi_single_quality = True
    elif variant not in {"navis_ack", "no_proxy"}:
        raise ValueError(f"Unsupported final ablation variant: {variant}")
    return args


def run_variant(
    variant: str,
    manifest_frames: list[dict],
    base: argparse.Namespace,
) -> dict:
    rounds = []
    args = variant_args(base, variant)
    demand_quotas = []
    for frame in manifest_frames:
        metadata_bytes = sum(
            max(1, int(item.get("bytes", 1) or 1)) for item in frame.get("items", []) if item.get("layer") == "metadata"
        )
        demand_quotas.append(
            max(
                base.offline_min_quota_bytes,
                min(base.offline_max_quota_bytes, int(metadata_bytes / 0.6)),
            )
        )
    dynamic_windows = dynamic_frame_windows(
        manifest_frames,
        demand_quotas or [base.offline_min_quota_bytes],
        max_window_frames=base.window_frames,
    )
    for round_index, frames in enumerate(dynamic_windows, start=1):
        if not frames:
            continue
        packets = window_packets(
            frames,
            device_id="node_1",
            metadata_bundle=True,
            roi_payload_limit=base.roi_payload_limit,
            background_payload_limit=base.background_payload_limit,
        )
        metadata_bytes = sum(
            max(1, int(item.get("bytes", 1) or 1))
            for frame in frames
            for item in frame.get("items", [])
            if item.get("layer") == "metadata"
        )
        quota = max(
            base.offline_min_quota_bytes,
            min(base.offline_max_quota_bytes, int(metadata_bytes / 0.6)),
        )
        selected, summary = schedule_window(packets, args, quota, {"basis": "offline_metadata_pressure"})
        current_ids = selected_roi_group_ids(selected)
        quality_ranks = [
            int((packet.metadata or {}).get("quality_rank", packet.rate_idx or 0) or 0)
            for packet in selected
            if packet.layer == "roi_patch" and packet.fragment_index == 0
        ]
        rounds.append(
            {
                "round_index": round_index,
                "frame_start": int(frames[0]["frame_idx"]),
                "frame_end": int(frames[-1]["frame_idx"]),
                "summary": summary,
                "expected_rois": roi_groups_in_frames(frames),
                "current_roi_group_ids": sorted(current_ids),
                "selected_quality_ranks": quality_ranks,
            }
        )
    return summarize_variant(variant, rounds, 0 if variant == "no_proxy" else base.roi_cache_ttl_frames)


def write_markdown(path: Path, rows: list[dict]) -> None:
    headers = [
        "Variant",
        "Sel. MB",
        "Meta",
        "Cur/Exp",
        "Avail/Exp",
        "ROI/round",
        "Avail %",
        "Avail/MB",
    ]
    lines = [
        "# Offline scheduler ablation",
        "",
        "| " + " | ".join(headers) + " |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {variant} | {selected_mb:.2f} | {metadata_completeness:.3f} | "
            "{current_roi_groups}/{expected_roi_groups} | {available_roi_groups}/{expected_roi_groups} | "
            "{roi_transmission_rate:.2f} | {roi_availability:.3f} | {roi_per_selected_mb:.1f} |".format(
                **row,
                selected_mb=float(row["selected_bytes"]) / 1_000_000.0,
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline metadata-pressure scheduler ablation.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPO_ROOT / "experiments/results/encoding/dcvc/payloads/candidate_manifest.json",
    )
    parser.add_argument(
        "--output-root", type=Path, default=REPO_ROOT / "experiments/results/ablations/scheduler/mot17-04"
    )
    parser.add_argument("--sequence", default="MOT17-04-FRCNN")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--window-frames", type=int, default=5)
    parser.add_argument("--offline-min-quota-bytes", type=int, default=6000)
    parser.add_argument("--offline-max-quota-bytes", type=int, default=40000)
    parser.add_argument("--roi-payload-limit", type=int, default=1024)
    parser.add_argument("--background-payload-limit", type=int, default=512)
    parser.add_argument("--max-roi-count", type=int, default=12)
    parser.add_argument("--min-roi-count-per-window", type=int, default=2)
    parser.add_argument("--max-dynamic-roi-count-per-window", type=int, default=24)
    parser.add_argument("--max-budget-fill-roi-count-per-window", type=int, default=96)
    parser.add_argument("--roi-cache-ttl-frames", type=int, default=10)
    parser.add_argument("--roi-track-continuity-bonus", type=float, default=0.08)
    parser.add_argument("--roi-recent-track-memory", type=int, default=32)
    parser.add_argument("--roi-quality-score-thresholds", type=float, nargs=3, default=[0.35, 0.45, 0.60])
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=["navis_ack", "budget_fill", "single_q_roi", "no_proxy"],
        default=[
            "navis_ack",
            "single_q_roi",
            "no_proxy",
        ],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = load_json(args.manifest)
    frames = list(manifest.get("frames", []))
    if args.sequence:
        frames = [frame for frame in frames if str(frame.get("sequence")) == args.sequence]
    if args.max_frames > 0:
        frames = frames[: args.max_frames]
    args.output_root.mkdir(parents=True, exist_ok=True)
    rows = [run_variant(variant, frames, args) for variant in args.variants]
    output = {
        "manifest": str(args.manifest),
        "sequence": args.sequence,
        "frame_count": len(frames),
        "window_frames": args.window_frames,
        "model": "Offline candidate-size replay with metadata-pressure quotas (6--40 kB), persistent track memory, and a 10-frame proxy cache. No ACK trace or physical delivery is simulated.",
        "settings": {key: value for key, value in vars(args).items() if not isinstance(value, Path)},
        "results": rows,
    }
    (args.output_root / "summary.json").write_text(
        json.dumps(output, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    write_markdown(args.output_root / "report.md", rows)
    print(json.dumps(output, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
