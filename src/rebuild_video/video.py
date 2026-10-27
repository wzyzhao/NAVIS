from __future__ import annotations

import shutil
import sys
from pathlib import Path

from nbiot_transport.feedback.missing_fragment_queue import (
    enqueue_missing_metadata_from_delivery,
    reconcile_missing_fragment_queue,
)
from rebuild_video.quality import infer_reference_root
from rebuild_video.report import write_rebuild_report
from rebuild_video.utils import clear_rebuild_outputs, load_json, schedule_path_from_config
from rebuild_video.video_reconstructor import VideoReconstructor
from common.paths import PROJECT_ROOT as WORKSPACE_ROOT

DEFAULT_REBUILD_CONFIG = WORKSPACE_ROOT / "src" / "rebuild_video" / "local_config.json"


def rebuild_video(config_path: Path = DEFAULT_REBUILD_CONFIG) -> dict:
    """Recover ordered receiver state, render frames and write delivery/quality reports."""
    config_path = config_path.resolve()
    raw_config = load_json(config_path)
    config = load_rebuild_config(config_path, raw_config)
    config["_config_path"] = str(config_path)
    config.setdefault("reconstruction_fill_missing_frames", True)

    root = Path(config["output_root"])
    receiver_root = Path(config["receiver_output_root"])
    clear_rebuild_outputs(root)
    sync_rebuild_inputs(receiver_root, root)
    reconstructor = VideoReconstructor(root, config)
    try:
        queued = reconstructor.recover_manifest()
        schedule_path = schedule_path_from_config(config)
        if not queued and schedule_path is not None:
            queued = reconstructor.recover_directory(schedule_path)
        reconstructor.close(wait=True, write_videos=True)
    except Exception:
        reconstructor.close(wait=True, write_videos=False)
        raise
    metadata_enqueue = enqueue_missing_metadata_from_delivery(
        receiver_root / "missing_fragment_queue.json",
        root / "received_payload_manifest.jsonl",
        receiver_root / "delivery_log.json",
        int(config.get("reconstruction_expected_frames") or 0),
    )
    queue_reconcile = reconcile_missing_fragment_queue(
        receiver_root / "missing_fragment_queue.json",
        root / "received_payload_manifest.jsonl",
    )

    videos = sorted(str(path) for path in (root / "reconstructed_videos").glob("*.mp4"))
    skip_quality = bool(raw_config.get("reconstruction_skip_quality", False))
    reference_root = raw_config.get("reconstruction_reference_root")
    reference_root = Path(reference_root) if reference_root else None
    if reference_root is not None and not reference_root.is_absolute():
        reference_root = (config_path.parent / reference_root).resolve()
    if reference_root is None and not skip_quality:
        reference_root = infer_reference_root(config)
    report_path = write_rebuild_report(
        root,
        int(config.get("reconstruction_expected_frames") or 0),
        None if skip_quality else reference_root,
        receiver_root=receiver_root,
        quality_workers=config["reconstruction_quality_workers"],
    )
    return {
        "queued_payloads": queued,
        "videos": videos,
        "output_root": str(root),
        "status_log": str(root / "reconstruction_status.jsonl"),
        "error_log": str(root / "reconstruction_errors.jsonl"),
        "report": str(report_path),
        "missing_queue_reconcile": {
            "available": queue_reconcile.get("available", False),
            "metadata_added_for_retransmission_count": metadata_enqueue.get("added_count", 0),
            "metadata_missing_frames": metadata_enqueue.get("missing_metadata_frames", []),
            "purged_by_received_manifest_count": queue_reconcile.get("purged_by_received_manifest_count", 0),
            "remaining_missing_fragments": len(queue_reconcile.get("missing_fragments", []) or []),
            "direct_receiver_output_root": True,
            "receiver_missing_fragment_queue": str(receiver_root / "missing_fragment_queue.json"),
        },
    }


def load_rebuild_config(config_path: Path, raw_config: dict) -> dict:
    """Resolve paths and validate worker limits before rebuilding any output."""
    quality_workers = int(raw_config.get("reconstruction_quality_workers", 1))
    if quality_workers < 1:
        raise ValueError("reconstruction_quality_workers must be at least 1")
    base_dir = config_path.parent
    output_root = _resolve_path(
        raw_config.get("output_root") or "../../experiments/results/transmission/navis/default/rebuild", base_dir
    )
    receiver_output_root = _resolve_path(
        raw_config.get("receiver_output_root") or "../../experiments/results/transmission/navis/default/receiver",
        base_dir,
    )
    schedule_json = _resolve_path(raw_config.get("schedule_json"), base_dir, required=False)
    if schedule_json and not schedule_json.is_file():
        print(f"warning: schedule_json is not a file and will be ignored: {schedule_json}")
        schedule_json = None
    return {
        "output_root": output_root,
        "receiver_output_root": receiver_output_root,
        "reconstruction_enabled": True,
        "reconstruction_quality_workers": quality_workers,
        "reconstruction_expected_frames": int(raw_config.get("reconstruction_expected_frames", 0) or 0),
        "reconstruction_width": int(raw_config.get("reconstruction_width", 1920)),
        "reconstruction_height": int(raw_config.get("reconstruction_height", 1080)),
        "reconstruction_fps": float(raw_config.get("reconstruction_fps", 30)),
        "reconstruction_fill_missing_frames": bool(raw_config.get("reconstruction_fill_missing_frames", True)),
        "reconstruction_roi_mask_erode_pixels": int(raw_config.get("reconstruction_roi_mask_erode_pixels", 1) or 0),
        "reconstruction_roi_mask_feather_pixels": int(raw_config.get("reconstruction_roi_mask_feather_pixels", 5) or 0),
        "reconstruction_roi_black_border_suppression": bool(
            raw_config.get("reconstruction_roi_black_border_suppression", True)
        ),
        "reconstruction_roi_black_threshold": int(raw_config.get("reconstruction_roi_black_threshold", 18) or 0),
        "reconstruction_roi_content_mask_dilate_pixels": int(
            raw_config.get("reconstruction_roi_content_mask_dilate_pixels", 2) or 0
        ),
        "reconstruction_roi_internal_black_suppression": bool(
            raw_config.get("reconstruction_roi_internal_black_suppression", True)
        ),
        "reconstruction_roi_internal_black_min_area_ratio": float(
            raw_config.get("reconstruction_roi_internal_black_min_area_ratio", 0.015) or 0.0
        ),
        "roi_cache_ttl_frames": int(raw_config.get("roi_cache_ttl_frames", 10) or 0),
        "roi_proxy_fade_frames": int(raw_config.get("roi_proxy_fade_frames", 5) or 0),
        "roi_proxy_min_alpha": float(raw_config.get("roi_proxy_min_alpha", 0.25) or 0.0),
        "roi_proxy_motion_compensation": bool(raw_config.get("roi_proxy_motion_compensation", True)),
        "roi_proxy_max_motion_scale": float(raw_config.get("roi_proxy_max_motion_scale", 1.5) or 0.0),
        "reconstruction_roi_min_effective_alpha_ratio": float(
            raw_config.get("reconstruction_roi_min_effective_alpha_ratio", 0.02) or 0.0
        ),
        "dcvc_decoder_command": raw_config.get("dcvc_decoder_command") or _default_decoder_command(raw_config),
        "recovery_schedule_json": schedule_json,
        "schedule_json": schedule_json,
    }


def sync_rebuild_inputs(source_root: Path, output_root: Path) -> None:
    if not source_root.is_dir():
        raise FileNotFoundError(f"receiver_output_root does not exist: {source_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    for name in (
        "received_payload_manifest.jsonl",
        "tcp_receiver_acks.jsonl",
    ):
        source = source_root / name
        if source.is_file():
            shutil.copy2(source, output_root / name)
    source_payloads = source_root / "reassembled_payloads"
    target_payloads = output_root / "reassembled_payloads"
    if source_payloads.is_dir():
        if target_payloads.exists():
            shutil.rmtree(target_payloads)
        shutil.copytree(source_payloads, target_payloads)


def _resolve_path(value, base_dir: Path, required: bool = True) -> Path | None:
    if value in (None, ""):
        if required:
            return base_dir
        return None
    path = Path(value)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def _default_decoder_command(config: dict) -> list[str]:
    project = WORKSPACE_ROOT / "src" / "dcvc_compression"
    return [
        str(config.get("reconstruction_python_executable") or sys.executable),
        "-m",
        "dcvc_compression.dcvc_codec.stream_decoder_worker",
        "--model-path-i",
        str(project / "models" / "dcvc_rt" / "cvpr2025_image.pth.tar"),
        "--model-path-p",
        str(project / "models" / "dcvc_rt" / "cvpr2025_video.pth.tar"),
        "--device",
        str(config.get("reconstruction_device", "auto")),
    ]


if __name__ == "__main__":
    import json

    print(json.dumps(rebuild_video(), ensure_ascii=True, indent=2))
