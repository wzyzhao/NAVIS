import json
import sys
from pathlib import Path
from types import SimpleNamespace

from common.encoding_layout import write_frame_index
from common.paths import PROJECT_ROOT as WORKSPACE_ROOT
from dcvc_compression.dcvc_codec.bundle_io import ensure_dir
from dcvc_compression.dcvc_codec.layered_candidate_codec import (
    build_elic_comparison,
    format_layered_metrics,
    load_dcvc_layered_controller,
    load_elic_jpg_points,
    prepare_layered_candidates,
    summarize_quantitative_metrics,
)
from dcvc_compression.dcvc_codec.offline_schedule_simulator import (
    compose_layered_roi_results,
    export_transport_candidate_manifest,
)

PROJECT_ROOT = WORKSPACE_ROOT / "src" / "dcvc_compression"
DEFAULT_BUNDLE_ROOT = WORKSPACE_ROOT / "src" / "scene_factorization" / "scene_factorization_output"
DEFAULT_OUTPUT_ROOT = WORKSPACE_ROOT / "experiments" / "results" / "encoding" / "dcvc"
DEFAULT_DCVC_REPO = PROJECT_ROOT / "vendor" / "DCVC"
DEFAULT_MODEL_PATH_I = PROJECT_ROOT / "models" / "dcvc_rt" / "cvpr2025_image.pth.tar"
DEFAULT_MODEL_PATH_P = PROJECT_ROOT / "models" / "dcvc_rt" / "cvpr2025_video.pth.tar"
DEFAULT_PYTHON = Path(sys.executable)
DEFAULT_ELIC_SUMMARY = WORKSPACE_ROOT / "experiments" / "results" / "encoding" / "elic" / "summary.json"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.json"
RUN_LEVEL_REFRESH_FULL = "refresh_full"
RUN_LEVEL_EXPORT_CANDIDATES = "export_candidates"
RUN_LEVELS = {RUN_LEVEL_REFRESH_FULL, RUN_LEVEL_EXPORT_CANDIDATES}


DEFAULT_CONFIG = {
    "run_level": RUN_LEVEL_REFRESH_FULL,
    "bundle_root": str(DEFAULT_BUNDLE_ROOT),
    "sequence_filter": [],
    "output_root": str(DEFAULT_OUTPUT_ROOT),
    "dcvc_repo": str(DEFAULT_DCVC_REPO),
    "model_path_i": str(DEFAULT_MODEL_PATH_I),
    "model_path_p": str(DEFAULT_MODEL_PATH_P),
    "python_executable": str(DEFAULT_PYTHON),
    "device": "auto",
    "fps": 30,
    "background_rate_num": 2,
    "background_rate_idx": 1,
    "background_qp_i": [0, 21],
    "background_qp_p": [0, 21],
    "background_required_interval_seconds": 0.0,
    "background_summary_change_threshold": 0.018,
    "background_summary_min_interval_frames": 300,
    "metadata_roi_mask_mode": "track_refresh",
    "metadata_roi_mask_refresh_interval": 15,
    "metadata_roi_mask_motion_threshold": 0.65,
    "metadata_roi_mask_uncertainty_threshold": 0.65,
    "foreground_rate_num": 4,
    "foreground_rate_idx": 2,
    "elic_summary": str(DEFAULT_ELIC_SUMMARY),
    "worker_num": 1,
    "force_frame_num": -1,
    "force_zero_thres": 0.12,
    "reset_interval": 64,
    "verbose": 0,
    "calc_ssim": False,
}


PATH_FIELDS = {
    "bundle_root",
    "output_root",
    "dcvc_repo",
    "model_path_i",
    "model_path_p",
    "python_executable",
    "elic_summary",
}


def resolve_config_path(value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def load_config(config_path: Path = DEFAULT_CONFIG_PATH) -> SimpleNamespace:
    """Merge pipeline defaults with JSON and resolve project-relative paths."""
    config = dict(DEFAULT_CONFIG)
    if config_path.exists():
        with open(config_path, encoding="utf-8") as f:
            config.update(json.load(f))
    for field in PATH_FIELDS:
        config[field] = resolve_config_path(config.get(field))
    run_level = str(config.get("run_level", "")).strip()
    config["run_level"] = run_level
    config["rerun_dcvc"] = run_level == RUN_LEVEL_REFRESH_FULL
    config["reuse_dcvc_outputs"] = run_level == RUN_LEVEL_EXPORT_CANDIDATES
    return SimpleNamespace(**config)


def validate_args(args: SimpleNamespace) -> None:
    if args.run_level not in RUN_LEVELS:
        raise ValueError(f"config run_level must be one of {sorted(RUN_LEVELS)}, got {args.run_level!r}.")
    if args.background_rate_num < 2:
        raise ValueError("DCVC-RT background layer requires background_rate_num >= 2.")
    if args.foreground_rate_num < 2:
        raise ValueError("DCVC-RT foreground layer requires foreground_rate_num >= 2.")
    if args.background_rate_idx < 0 or args.background_rate_idx >= args.background_rate_num:
        raise ValueError("background_rate_idx must be in [0, background_rate_num).")
    if len(getattr(args, "background_qp_i", []) or []) != args.background_rate_num:
        raise ValueError("background_qp_i length must match background_rate_num.")
    if len(getattr(args, "background_qp_p", []) or []) != args.background_rate_num:
        raise ValueError("background_qp_p length must match background_rate_num.")
    if args.foreground_rate_idx < 0 or args.foreground_rate_idx >= args.foreground_rate_num:
        raise ValueError("foreground_rate_idx must be in [0, foreground_rate_num).")
    if float(getattr(args, "background_required_interval_seconds", 0.0) or 0.0) < 0:
        raise ValueError("background_required_interval_seconds must be >= 0.")
    if int(getattr(args, "background_summary_min_interval_frames", 0) or 0) < 0:
        raise ValueError("background_summary_min_interval_frames must be >= 0.")


def compute_required_background_interval_frames(args) -> int:
    seconds = float(getattr(args, "background_required_interval_seconds", 0.0) or 0.0)
    if seconds <= 0:
        return 0
    fps = max(1.0, float(getattr(args, "fps", 30) or 30))
    return max(1, int(round(seconds * fps)))


def build_controller(args):
    controller_cls = load_dcvc_layered_controller(args.dcvc_repo.resolve())
    return controller_cls(
        fps=args.fps,
        background_rate_idx=args.background_rate_idx,
        foreground_rate_idx=args.foreground_rate_idx,
        background_update_seconds=1.0,
        background_update_frames=0,
    )


def write_summary(
    output_root: Path,
    args,
    controller,
    candidates: dict,
    quantitative_metrics: dict,
    comparison: dict,
    frame_metrics_json: Path,
    quantitative_metrics_json: Path,
    candidate_manifest_json: Path,
) -> dict:
    required_background_interval = compute_required_background_interval_frames(args)
    summary = {
        "project": "dcvc_compression",
        "scheme": "dcvc_rt_layered_candidate_export",
        "status": "completed",
        "device": "cuda" if candidates["use_cuda"] else "cpu",
        "input": {
            "bundle_root": str(args.bundle_root.resolve()),
            "rate_num": 1,
            "background_rate_num": args.background_rate_num,
            "background_rate_idx": args.background_rate_idx,
            "background_required_interval_seconds": float(
                getattr(args, "background_required_interval_seconds", 0.0) or 0.0
            ),
            "background_required_interval_derived_frames": required_background_interval,
            "background_summary_change_threshold": float(
                getattr(args, "background_summary_change_threshold", 0.0) or 0.0
            ),
            "background_summary_min_interval_frames": int(
                getattr(args, "background_summary_min_interval_frames", 0) or 0
            ),
            "metadata_roi_mask_mode": str(getattr(args, "metadata_roi_mask_mode", "track_refresh") or "track_refresh"),
            "metadata_roi_mask_refresh_interval": int(getattr(args, "metadata_roi_mask_refresh_interval", 15) or 15),
            "foreground_rate_num": args.foreground_rate_num,
            "foreground_rate_idx": args.foreground_rate_idx,
            "foreground_mode": "roi_crop",
            "reuse_dcvc_outputs": args.reuse_dcvc_outputs,
            "elic_summary": str(args.elic_summary.resolve()) if args.elic_summary is not None else None,
            "fps": args.fps,
        },
        "controller": {
            "background_layer": "background_keyframe_summary items carry independent DCVC background keyframes; optional summaries are scheduled after ROI",
            "foreground_layer": "per-ROI crop sequences encoded by DCVC-RT",
            "rate_count": 1,
            "rate_plan": controller.build_rate_plan(),
            "dcvc_rt_controller": controller.describe(),
            "selection_owner": "nbiot_transport",
        },
        "metrics": quantitative_metrics.get("overall"),
        "comparison_to_elic_jpeg": comparison.get("mean") if comparison.get("available") else None,
        "artifacts": {
            "dcvc_layered_metrics_json": str(candidates["layered_metrics_json"]),
            "background_stream_root": str(candidates["background_streams_root"]),
            "foreground_stream_root": str(candidates["foreground_streams_root"]),
            "frame_metrics_json": str(frame_metrics_json),
            "quantitative_metrics_json": str(quantitative_metrics_json),
            "candidate_manifest_json": str(candidate_manifest_json),
        },
    }
    with open(output_root / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=True, indent=2)
    return summary


def main() -> None:
    args = load_config()
    validate_args(args)
    controller = build_controller(args)
    required_background_interval = compute_required_background_interval_frames(args)
    candidates = prepare_layered_candidates(args, controller)
    rate_plan = controller.build_rate_plan()
    candidate_manifest_json = export_transport_candidate_manifest(
        grouped=candidates["grouped"],
        bg_points=candidates["background_points"],
        required_bg_points=candidates["required_background_points"],
        fg_points=candidates["foreground_points"],
        foreground_export=candidates["foreground_export"],
        bg_stream_dir=candidates["background_stream_dir"],
        required_bg_stream_dir=candidates["required_background_stream_dir"],
        fg_stream_dir=candidates["foreground_stream_dir"],
        output_root=candidates["output_root"],
        rate_plan=rate_plan,
        required_background_interval_derived_frames=required_background_interval,
        required_background_interval_seconds=float(getattr(args, "background_required_interval_seconds", 0.0) or 0.0),
        background_summary_change_threshold=float(getattr(args, "background_summary_change_threshold", 0.0) or 0.0),
        background_summary_min_interval_frames=int(getattr(args, "background_summary_min_interval_frames", 0) or 0),
        metadata_roi_mask_mode=str(getattr(args, "metadata_roi_mask_mode", "track_refresh") or "track_refresh"),
        metadata_roi_mask_refresh_interval=int(getattr(args, "metadata_roi_mask_refresh_interval", 15) or 15),
        metadata_roi_mask_motion_threshold=float(getattr(args, "metadata_roi_mask_motion_threshold", 0.65) or 0.65),
        metadata_roi_mask_uncertainty_threshold=float(
            getattr(args, "metadata_roi_mask_uncertainty_threshold", 0.65) or 0.65
        ),
    )

    composed = compose_layered_roi_results(
        grouped=candidates["grouped"],
        bg_points=candidates["background_points"],
        fg_points=candidates["foreground_points"],
        foreground_export=candidates["foreground_export"],
        bg_stream_dir=candidates["background_stream_dir"],
        fg_stream_dir=candidates["foreground_stream_dir"],
        rate_plan=rate_plan,
    )

    output_root = ensure_dir(args.output_root.resolve())
    frame_metrics_json = output_root / "frame_metrics.json"
    write_frame_index(frame_metrics_json, {"frames": composed["frame_metrics"]}, output_root / "decoded")

    elic_points = load_elic_jpg_points(args.elic_summary)
    comparison = build_elic_comparison(composed["points"], elic_points, args.elic_summary)
    quantitative_metrics = summarize_quantitative_metrics(
        composed["points"], composed["frame_metrics"], args.fps, comparison
    )
    quantitative_metrics_json = output_root / "quantitative_metrics.json"
    with open(quantitative_metrics_json, "w", encoding="utf-8") as f:
        json.dump(quantitative_metrics, f, ensure_ascii=True, indent=2)

    summary = write_summary(
        output_root=output_root,
        args=args,
        controller=controller,
        candidates=candidates,
        quantitative_metrics=quantitative_metrics,
        comparison=comparison,
        frame_metrics_json=frame_metrics_json,
        quantitative_metrics_json=quantitative_metrics_json,
        candidate_manifest_json=candidate_manifest_json,
    )

    print(format_layered_metrics(len(composed["points"]), summary["device"], quantitative_metrics))
    print(f"candidate_manifest: {candidate_manifest_json}")


if __name__ == "__main__":
    main()
