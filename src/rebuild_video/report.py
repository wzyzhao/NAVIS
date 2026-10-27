from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from rebuild_video.quality import compute_reference_quality
from rebuild_video.utils import frame_set, load_json, load_jsonl

ACK_RATE_EXCLUDED_LAYERS = {"session_start"}
BACKGROUND_LAYERS = {"background_keyframe_summary"}


def _safe_rate(numerator: float, denominator: float):
    if denominator <= 0:
        return None
    return numerator / denominator


def analyze_delivery_log(path: Path) -> dict:
    if not path.is_file():
        return {"available": False}
    delivery = load_json(path)
    messages = delivery.get("messages", [])
    scheduler = delivery.get("scheduler") if isinstance(delivery.get("scheduler"), dict) else {}
    diagnostics = delivery.get("diagnostics") if isinstance(delivery.get("diagnostics"), dict) else {}
    by_layer = {}
    for layer in sorted({str(message.get("layer")) for message in messages}):
        layer_messages = [message for message in messages if str(message.get("layer")) == layer]
        confirmed_count = sum(message.get("ack_confirmed") is True for message in layer_messages)
        by_layer[layer] = {
            "messages": len(layer_messages),
            "ack_confirmed": confirmed_count,
            "ack_success_rate": _safe_rate(confirmed_count, len(layer_messages)),
            "fragments": sum(int(message.get("fragment_count") or 0) for message in layer_messages),
            "payload_bytes_sent": sum(int(message.get("payload_bytes_sent") or 0) for message in layer_messages),
        }
    summary = dict(delivery.get("summary", {}))
    for key in (
        "ack_delivery_rate",
        "ack_loss_rate",
        "message_ack_delivery_rate",
        "message_ack_loss_rate",
    ):
        summary.pop(key, None)
    ack_messages = [message for message in messages if message.get("layer") not in ACK_RATE_EXCLUDED_LAYERS]
    observed = [message for message in ack_messages if message.get("ack_observed") is True]
    confirmed = [message for message in observed if message.get("ack_confirmed") is True]
    failed = [message for message in observed if message.get("ack_confirmed") is False]
    module_confirmed = [message for message in ack_messages if message.get("module_confirmed") is True]
    missing = len(ack_messages) - len(observed)
    message_count = len(ack_messages)
    summary.update(
        {
            "ack_message_count": message_count,
            "ack_observed_message_count": len(observed),
            "ack_confirmed_message_count": len(confirmed),
            "ack_failed_message_count": len(failed),
            "ack_missing_message_count": missing,
            "module_confirmed_message_count": len(module_confirmed),
            "module_delivery_rate": _safe_rate(len(module_confirmed), message_count),
            "ack_coverage_rate": _safe_rate(len(observed), message_count),
            "ack_success_rate": _safe_rate(len(confirmed), message_count),
            "ack_failure_rate": _safe_rate(len(failed), len(observed)),
            "ack_observed_failure_rate": _safe_rate(len(failed), len(observed)),
            "ack_unobserved_rate": _safe_rate(missing, message_count),
            "ack_required": True,
            "ack_missing_is_delivery_failure": True,
        }
    )
    return {
        "available": True,
        "summary": {
            "ack_message_count": summary.get("ack_message_count"),
            "ack_confirmed_message_count": summary.get("ack_confirmed_message_count"),
            "ack_missing_message_count": summary.get("ack_missing_message_count"),
            "ack_coverage_rate": summary.get("ack_coverage_rate"),
            "ack_success_rate": summary.get("ack_success_rate"),
            "ack_failure_rate": summary.get("ack_failure_rate"),
            "confirmed_payload_bytes": summary.get("confirmed_payload_bytes"),
            "confirmed_wire_bytes": summary.get("confirmed_wire_bytes"),
        },
        "link": delivery.get("link", {}),
        "scheduler": {
            "selected_packets": scheduler.get("selected_packets"),
            "selected_bytes": scheduler.get("selected_bytes"),
            "dropped_packets": scheduler.get("dropped_packets"),
            "missing_fragment_count": scheduler.get("missing_fragment_count"),
        },
        "diagnostics": {
            "critical_layer_fragment_retransmit_attempt_count": diagnostics.get(
                "critical_layer_fragment_retransmit_attempt_count"
            ),
            "metadata_fragment_retransmit_attempt_count": diagnostics.get("metadata_fragment_retransmit_attempt_count"),
            "background_fragment_retransmit_attempt_count": diagnostics.get(
                "background_fragment_retransmit_attempt_count"
            ),
            "roi_patch_fragment_retransmit_attempt_count": diagnostics.get(
                "roi_patch_fragment_retransmit_attempt_count"
            ),
            "socket_recovery_event_count": diagnostics.get("socket_recovery_event_count"),
        },
        "by_layer": by_layer,
    }


def group_missing_fragments(path: Path) -> dict:
    if not path.is_file():
        return {}
    grouped = defaultdict(list)
    for item in load_json(path).get("missing_fragments") or []:
        if item.get("frame_idx") is not None:
            grouped[(int(item.get("frame_idx")), str(item.get("layer")))].append(item)
    return grouped


def analyze_missing_frame_retransmission(
    root: Path,
    receiver_root: Path,
    missing_metadata_frames: list[int],
    filled_frames: set[int],
) -> dict:
    missing_by_frame = group_missing_fragments(receiver_root / "missing_fragment_queue.json")
    frames_in_queue = {
        int(frame_idx) for (frame_idx, layer), items in missing_by_frame.items() if layer == "metadata" and items
    }
    return {
        "missing_metadata_frames_in_missing_queue_count": len(set(missing_metadata_frames) & frames_in_queue),
        "metadata_frames_in_missing_queue_but_recovered_count": len(frames_in_queue - set(missing_metadata_frames)),
    }


def analyze_reconstruction(root: Path, expected_frames: int | None, receiver_root: Path | None = None) -> dict:
    receiver_root = receiver_root or root
    manifest = load_jsonl(root / "received_payload_manifest.jsonl")
    status = load_jsonl(root / "reconstruction_status.jsonl")
    errors = load_jsonl(root / "reconstruction_errors.jsonl")
    rendered_events = [row for row in status if row.get("event") == "frame_rendered"]
    filled_events = [row for row in status if row.get("event") == "frame_filled"]
    final_render = {int(row["frame_idx"]): row for row in rendered_events if row.get("frame_idx") is not None}
    expected = set(range(1, int(expected_frames) + 1)) if expected_frames else set()
    rendered_frames = set(final_render)
    filled_frames = {int(row["frame_idx"]) for row in filled_events if row.get("frame_idx") is not None}
    metadata_frames = manifest_metadata_frames(manifest)
    true_missing_metadata_frames = sorted(expected - metadata_frames) if expected else []
    background_ready = [
        row
        for row in status
        if row.get("layer") in BACKGROUND_LAYERS and row.get("reconstruction_status") == "background_ready"
    ]
    return {
        "manifest": _manifest_summary(manifest, expected),
        "decode": _decode_summary(root, manifest, errors, background_ready),
        "frames": _frame_summary(
            root, expected_frames, expected, rendered_frames, filled_frames, true_missing_metadata_frames
        ),
        "retransmission": analyze_missing_frame_retransmission(
            root,
            receiver_root,
            true_missing_metadata_frames,
            filled_frames,
        ),
        "rendering": _rendering_summary(final_render),
    }


def _manifest_summary(manifest: list[dict], expected: set[int]) -> dict:
    manifest_frames = frame_set(manifest)
    metadata_frames = manifest_metadata_frames(manifest)
    return {
        "payload_count": len(manifest),
        "layer_counts": dict(Counter(row.get("layer") for row in manifest)),
        "unique_frames": len(manifest_frames),
        "metadata_frames": len(metadata_frames),
        "roi_frames": len(frame_set(manifest, "roi_patch")),
        "background_frame_count": len(frame_set(manifest, "background_keyframe_summary")),
        "missing_metadata_count": len(expected - metadata_frames) if expected else 0,
    }


def manifest_metadata_frames(manifest: list[dict]) -> set[int]:
    frames = set()
    for row in manifest:
        if row.get("layer") != "metadata":
            continue
        if row.get("metadata_bundle") and row.get("frame_start") is not None and row.get("frame_end") is not None:
            frames.update(range(int(row["frame_start"]), int(row["frame_end"]) + 1))
        elif row.get("frame_idx") is not None:
            frames.add(int(row["frame_idx"]))
    return frames


def _decode_summary(
    root: Path,
    manifest: list[dict],
    errors: list[dict],
    background_ready: list[dict],
) -> dict:
    background_errors = [row for row in errors if row.get("layer") in BACKGROUND_LAYERS]
    return {
        "error_count": len(errors),
        "errors_by_layer": dict(Counter(row.get("layer") for row in errors)),
        "background_ready_frames": sorted(int(row["frame_idx"]) for row in background_ready),
        "background_decode_failure_count": len(background_errors),
    }


def _frame_summary(
    root: Path,
    expected_frames: int | None,
    expected: set[int],
    rendered_frames: set[int],
    filled_frames: set[int],
    true_missing_metadata_frames: list[int],
) -> dict:
    frame_files = {
        path.parent.name: sorted(int(item.stem) for item in path.parent.glob("*.png") if item.stem.isdigit())
        for path in (root / "reconstructed_frames").glob("*/*.png")
    }
    return {
        "expected_frames": int(expected_frames or 0),
        "rendered_event_frames": len(rendered_frames),
        "filled_frames": len(filled_frames),
        "missing_rendered_count": len(expected - rendered_frames) if expected else 0,
        "true_missing_metadata_count": len(true_missing_metadata_frames),
        "final_files_include_filled_frames": bool(filled_frames),
        "final_frame_files": {
            device_id: {
                "count": len(indices),
                "first": indices[0] if indices else None,
                "last": indices[-1] if indices else None,
                "missing_count": len(expected - set(indices)) if expected else 0,
            }
            for device_id, indices in frame_files.items()
        },
    }


def _rendering_summary(final_render: dict[int, dict]) -> dict:
    roi_applied = sum(int(row.get("roi_applied") or 0) for row in final_render.values())
    roi_current_applied = sum(int(row.get("roi_current_applied") or 0) for row in final_render.values())
    roi_metadata_proxy_applied = sum(int(row.get("roi_metadata_proxy_applied") or 0) for row in final_render.values())
    roi_expected = sum(int(row.get("roi_expected") or 0) for row in final_render.values())
    roi_missing = sum(int(row.get("roi_missing") or 0) for row in final_render.values())
    frames_with_current_roi = [
        int(frame_idx) for frame_idx, row in final_render.items() if int(row.get("roi_current_applied") or 0) > 0
    ]
    frames_with_metadata_roi_proxy = [
        int(frame_idx) for frame_idx, row in final_render.items() if int(row.get("roi_metadata_proxy_applied") or 0) > 0
    ]
    frames_with_missing_roi = [
        int(frame_idx) for frame_idx, row in final_render.items() if int(row.get("roi_missing") or 0) > 0
    ]
    frames_without_current_roi_applied = [
        int(frame_idx) for frame_idx, row in final_render.items() if int(row.get("roi_current_applied") or 0) == 0
    ]
    background_ages = [
        int(row.get("background_age_frames"))
        for row in final_render.values()
        if row.get("background_age_frames") is not None
    ]
    background_reused_frames = [
        int(frame_idx) for frame_idx, row in final_render.items() if row.get("background") == "reused"
    ]
    background_missing_frames = [
        int(frame_idx) for frame_idx, row in final_render.items() if row.get("background") == "missing"
    ]
    return {
        "background_status_counts": dict(Counter(row.get("background") for row in final_render.values())),
        "background_reused_frames": len(background_reused_frames),
        "background_missing_frames": len(background_missing_frames),
        "background_max_age_frames": max(background_ages) if background_ages else None,
        "background_avg_age_frames": (sum(background_ages) / len(background_ages)) if background_ages else None,
        "roi_expected_sum": roi_expected,
        "roi_current_applied_sum": roi_current_applied,
        "roi_metadata_proxy_applied_sum": roi_metadata_proxy_applied,
        "roi_applied_sum": roi_applied,
        "roi_missing_sum": roi_missing,
        "frames_with_current_roi": len(frames_with_current_roi),
        "frames_with_metadata_roi_proxy": len(frames_with_metadata_roi_proxy),
        "frames_with_missing_roi": len(frames_with_missing_roi),
        "frames_without_current_roi_applied": len(frames_without_current_roi_applied),
        "frame_image_quality": analyze_reconstructed_frames(next(iter(final_render.values()), {}).get("output_path")),
    }


def analyze_reconstructed_frames(example_output_path: str | None) -> dict:
    if not example_output_path:
        return {"available": False}
    try:
        import cv2
    except Exception as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    device_dir = Path(example_output_path).parent
    if not device_dir.is_dir():
        return {"available": False}
    rows = []
    for path in sorted(device_dir.glob("*.png")):
        if not path.stem.isdigit():
            continue
        frame_idx = int(path.stem)
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        mean = float(image.mean())
        std = float(image.std())
        black_ratio = float(((image[:, :, 0] < 8) & (image[:, :, 1] < 8) & (image[:, :, 2] < 8)).mean())
        rows.append(
            {
                "frame_idx": frame_idx,
                "mean": mean,
                "std": std,
                "black_ratio": black_ratio,
                "looks_black": mean < 5.0 or black_ratio > 0.75,
                "looks_flat": std < 3.0,
            }
        )
    if not rows:
        return {"available": False}
    return {
        "available": True,
        "frame_count": len(rows),
        "black_frame_count": sum(row["looks_black"] for row in rows),
        "flat_frame_count": sum(row["looks_flat"] for row in rows),
        "mean_min": min(row["mean"] for row in rows),
        "mean_max": max(row["mean"] for row in rows),
        "std_min": min(row["std"] for row in rows),
        "std_max": max(row["std"] for row in rows),
    }


def analyze_missing_queue(path: Path) -> dict:
    if not path.is_file():
        return {"available": False}
    fragments = load_json(path).get("missing_fragments") or []
    frames = [int(item.get("frame_idx") or 0) for item in fragments if item.get("frame_idx") is not None]
    return {
        "available": True,
        "missing_fragment_count": len(fragments),
        "layers": dict(Counter(item.get("layer") for item in fragments)),
        "errors": dict(Counter(item.get("last_error") for item in fragments)),
        "unique_frame_count": len(set(frames)),
        "min_frame": min(frames) if frames else None,
        "max_frame": max(frames) if frames else None,
    }


def _queue_frames(fragments: list[dict], layer: str) -> list[int]:
    return sorted(
        {
            int(item.get("frame_idx"))
            for item in fragments
            if item.get("layer") == layer and item.get("frame_idx") is not None
        }
    )


def analyze_receiver_state(path: Path, reconstruction: dict) -> dict:
    if not path.is_file():
        return {"available": False}
    data = load_json(path)
    recovery_hints = data.get("recovery_hints") or []
    background_state = data.get("background_state") or {}
    decoded_background_frames = set(reconstruction.get("decode", {}).get("background_ready_frames") or [])
    return {
        "available": True,
        "recovery_hint_count": data.get("recovery_hint_count", data.get("retransmission_request_count")),
        "recovery_hint_types": dict(Counter(item.get("request_type") for item in recovery_hints)),
        "background_decoder_conflict_count": sum(
            state.get("last_frame_idx") is not None
            and int(state.get("last_frame_idx")) not in decoded_background_frames
            for state in background_state.values()
        ),
    }


def write_rebuild_report(
    root: Path,
    expected_frames: int | None,
    reference_root: Path | None,
    *,
    receiver_root: Path | None = None,
    quality_workers: int = 1,
) -> Path:
    """Write delivery and reconstruction metrics; compare independent frames in parallel."""
    receiver_root = receiver_root or root
    reconstruction = analyze_reconstruction(root, expected_frames, receiver_root)
    report = {
        "output_root": str(root),
        "receiver_root": str(receiver_root),
        "expected_frames": int(expected_frames or 0),
        "delivery": analyze_delivery_log(receiver_root / "delivery_log.json"),
        "missing_queue": analyze_missing_queue(receiver_root / "missing_fragment_queue.json"),
        "receiver_state": analyze_receiver_state(receiver_root / "receiver_state.json", reconstruction),
        "reconstruction": reconstruction,
        "quality": compute_reference_quality(root, reference_root, workers=quality_workers),
    }
    path = root / "reconstruction_quality_report.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
