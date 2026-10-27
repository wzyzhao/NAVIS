import importlib.util
import json
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from common.encoding_layout import load_artifact_json, organize_dcvc_outputs, write_metric_index
from common.macro_metrics import seven_sequence_mean
from dcvc_compression.dcvc_codec.bundle_io import ensure_dir, load_bundle_frame
from dcvc_compression.dcvc_codec.dcvc_pipeline import (
    bgr_to_yuv420_bytes,
    build_dcvc_rt_command,
    inspect_python_environment,
    parse_dcvc_results,
    parse_layered_dcvc_metrics,
    write_layered_dcvc_metrics_json,
)

FOREGROUND_STREAM_CLASS = "layered_foreground_roi"
REQUIRED_BACKGROUND_STREAM_CLASS = "required_background"


class LazyBundleSequence:
    def __init__(self, frame_dirs: list[Path]):
        self.frame_dirs = list(frame_dirs)

    def __len__(self) -> int:
        return len(self.frame_dirs)

    def __iter__(self) -> Iterator:
        for frame_dir in self.frame_dirs:
            yield load_bundle_frame(frame_dir)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return LazyBundleSequence(self.frame_dirs[index])
        return load_bundle_frame(self.frame_dirs[index])

    def limit(self, frame_num: int) -> "LazyBundleSequence":
        if frame_num <= 0:
            return self
        return LazyBundleSequence(self.frame_dirs[:frame_num])


def resolve_device(requested: str) -> bool:
    if requested == "auto":
        return torch.cuda.is_available()
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")
    return requested == "cuda"


def load_dcvc_layered_controller(dcvc_repo: Path):
    module_path = dcvc_repo / "src" / "utils" / "layered_rate_control.py"
    spec = importlib.util.spec_from_file_location("dcvc_rt_layered_rate_control", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load DCVC-RT layered rate controller from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.LayeredRateController


def require_existing_dcvc_output(path: Path, layer_name: str) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing existing {layer_name} DCVC output: {path}. "
            "Use rerun_dcvc=True only when encoded candidates need to be regenerated."
        )


def remove_if_exists(path: Path) -> None:
    if path.exists():
        path.unlink()


def group_bundles(bundle_root: Path, frame_num: int = -1) -> dict[str, list]:
    grouped = {}
    for sequence_dir in sorted(path for path in bundle_root.iterdir() if path.is_dir()):
        frame_dirs = [
            frame_dir
            for frame_dir in sorted(path for path in sequence_dir.iterdir() if path.is_dir())
            if (frame_dir / "metadata.json").exists()
        ]
        if frame_num > 0:
            frame_dirs = frame_dirs[:frame_num]
        if frame_dirs:
            sequence = json.loads((frame_dirs[0] / "metadata.json").read_text(encoding="utf-8"))["sequence"]
            grouped[str(sequence)] = LazyBundleSequence(frame_dirs)
    return grouped


def limit_grouped_frames(grouped: dict[str, list], frame_num: int) -> dict[str, list]:
    if frame_num <= 0:
        return grouped
    return {
        sequence: bundles.limit(frame_num) if hasattr(bundles, "limit") else bundles[:frame_num]
        for sequence, bundles in grouped.items()
    }


def filter_grouped_sequences(grouped: dict[str, list], sequence_filter) -> dict[str, list]:
    if not sequence_filter:
        return grouped
    requested = [sequence_filter] if isinstance(sequence_filter, str) else list(sequence_filter)
    requested = {str(value).casefold() for value in requested if str(value).strip()}
    selected = {sequence: bundles for sequence, bundles in grouped.items() if sequence.casefold() in requested}
    missing = requested - {sequence.casefold() for sequence in selected}
    if missing:
        raise ValueError(f"Requested sequences are unavailable in bundle_root: {sorted(missing)}")
    return selected


def export_low_frequency_backgrounds(
    grouped: dict[str, list],
    output_root: Path,
    intra_period: int,
    default_fps: int,
) -> dict:
    yuv_dir = ensure_dir(output_root / "background_yuv420")
    config = {
        "root_path": str(output_root.resolve()),
        "test_classes": {
            "layered_background": {
                "test": 1,
                "base_path": "background_yuv420",
                "src_type": "yuv420",
                "sequences": {},
            }
        },
    }
    sequence_meta = {}
    for sequence_name, bundles in tqdm(grouped.items(), desc="Export background YUV420", unit="seq"):
        first = bundles[0]
        height, width = first.background.shape[:2]
        file_name = f"{sequence_name}_bg_lf_{width}x{height}_{default_fps}.yuv"
        yuv_path = yuv_dir / file_name
        with open(yuv_path, "wb") as f:
            for frame_idx in range(len(bundles)):
                f.write(bgr_to_yuv420_bytes(bundles[frame_idx].background))
        config["test_classes"]["layered_background"]["sequences"][file_name] = {
            "width": width,
            "height": height,
            "frames": len(bundles),
            "intra_period": intra_period,
        }
        sequence_meta[sequence_name] = {"width": width, "height": height, "frame_count": len(bundles)}

    config_path = output_root / "background_dataset_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=True, separators=(",", ":"))
    return {"config_path": config_path, "sequence_meta": sequence_meta}


def required_background_sequence_name(sequence_name: str, frame_idx: int) -> str:
    return f"{sequence_name}_bg_keyframe_f{int(frame_idx):06d}"


def compute_required_background_interval_frames(args) -> int:
    seconds = float(getattr(args, "background_required_interval_seconds", 0.0) or 0.0)
    if seconds <= 0:
        return 0
    fps = max(1.0, float(getattr(args, "fps", 30) or 30))
    return max(1, int(round(seconds * fps)))


def export_required_background_keyframes(
    grouped: dict[str, list],
    output_root: Path,
    required_interval: int,
    summary_change_threshold: float,
    summary_min_interval: int,
    default_fps: int,
) -> dict:
    yuv_dir = ensure_dir(output_root / "required_background_yuv420")
    config = {
        "root_path": str(output_root.resolve()),
        "test_classes": {
            REQUIRED_BACKGROUND_STREAM_CLASS: {
                "test": 1,
                "base_path": "required_background_yuv420",
                "src_type": "yuv420",
                "sequences": {},
            }
        },
    }
    records = []
    interval = int(required_interval or 0)
    change_threshold = float(summary_change_threshold or 0.0)
    min_summary_interval = max(0, int(summary_min_interval or 0))
    if interval <= 0 and change_threshold <= 0:
        config_path = output_root / "required_background_dataset_config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=True, separators=(",", ":"))
        return {"config_path": config_path, "records": records}

    for sequence_name, bundles in tqdm(grouped.items(), desc="Export required background YUV420", unit="seq"):
        first = bundles[0]
        height, width = first.background.shape[:2]
        last_summary_frame_idx = None
        for frame_idx, bundle in enumerate(bundles, start=1):
            if frame_idx == 1:
                continue
            is_required = interval > 0 and frame_idx % interval == 0
            change_score = float(getattr(bundle, "background_change_score", 0.0) or 0.0)
            is_summary = change_threshold > 0 and change_score >= change_threshold
            if is_summary and min_summary_interval > 0 and last_summary_frame_idx is not None:
                is_summary = frame_idx - last_summary_frame_idx >= min_summary_interval
            if not is_required and not is_summary:
                continue
            if is_summary:
                last_summary_frame_idx = frame_idx
            required_sequence = required_background_sequence_name(sequence_name, frame_idx)
            file_name = f"{required_sequence}_{width}x{height}_{default_fps}.yuv"
            yuv_path = yuv_dir / file_name
            with open(yuv_path, "wb") as f:
                f.write(bgr_to_yuv420_bytes(bundle.background))
            config["test_classes"][REQUIRED_BACKGROUND_STREAM_CLASS]["sequences"][file_name] = {
                "width": width,
                "height": height,
                "frames": 1,
                "intra_period": -1,
            }
            records.append(
                {
                    "sequence": sequence_name,
                    "frame_idx": frame_idx,
                    "required_sequence": required_sequence,
                    "width": width,
                    "height": height,
                }
            )

    config_path = output_root / "required_background_dataset_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=True, separators=(",", ":"))
    return {"config_path": config_path, "records": records}


def pad_frame_to_even(frame: np.ndarray, pad_value: int = 0) -> np.ndarray:
    height, width = frame.shape[:2]
    pad_h = height % 2
    pad_w = width % 2
    if pad_h == 0 and pad_w == 0:
        return frame
    border_value = [pad_value] * frame.shape[2] if frame.ndim == 3 else pad_value
    return cv2.copyMakeBorder(frame, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=border_value)


def foreground_roi_sequence_name(sequence_name: str, frame_id: int, roi_idx: int) -> str:
    return f"{sequence_name}_fgroi_f{frame_id:06d}_r{roi_idx:02d}"


def export_foreground_roi_crops(
    grouped: dict[str, list],
    output_root: Path,
    intra_period: int,
    default_fps: int,
) -> dict:
    yuv_dir = ensure_dir(output_root / "foreground_roi_yuv420")
    config = {
        "root_path": str(output_root.resolve()),
        "test_classes": {
            "layered_foreground_roi": {
                "test": 1,
                "base_path": "foreground_roi_yuv420",
                "src_type": "yuv420",
                "sequences": {},
            }
        },
    }
    roi_records = []
    for sequence_name, bundles in tqdm(grouped.items(), desc="Export foreground ROI YUV420", unit="seq"):
        for frame_idx, bundle in enumerate(bundles, start=1):
            for roi_idx, roi in enumerate(bundle.selected_rois):
                x1, y1, x2, y2 = roi.roi_bbox
                crop = pad_frame_to_even(roi.image, pad_value=0)
                padded_height, padded_width = crop.shape[:2]
                if padded_width <= 0 or padded_height <= 0:
                    continue
                roi_sequence = foreground_roi_sequence_name(sequence_name, bundle.frame_id, roi_idx)
                file_name = f"{roi_sequence}_{padded_width}x{padded_height}_{default_fps}.yuv"
                yuv_path = yuv_dir / file_name
                with open(yuv_path, "wb") as f:
                    f.write(bgr_to_yuv420_bytes(crop))
                config["test_classes"]["layered_foreground_roi"]["sequences"][file_name] = {
                    "width": padded_width,
                    "height": padded_height,
                    "frames": 1,
                    "intra_period": intra_period,
                }
                roi_records.append(
                    {
                        "sequence": sequence_name,
                        "frame_idx": frame_idx,
                        "frame_id": bundle.frame_id,
                        "roi_idx": roi_idx,
                        "roi_sequence": roi_sequence,
                        "bbox": [int(value) for value in roi.bbox],
                        "roi_bbox": [x1, y1, x2, y2],
                        "track_id": int(roi.track_id),
                        "class_id": int(roi.class_id),
                        "category": str(getattr(roi, "category", "other")),
                        "width": x2 - x1,
                        "height": y2 - y1,
                        "padded_width": padded_width,
                        "padded_height": padded_height,
                        "motion_score": float(getattr(roi, "motion_score", 0.0)),
                        "uncertainty": float(getattr(roi, "uncertainty", 0.0)),
                        "confidence": getattr(roi, "confidence", None),
                        "continuity_score": getattr(roi, "continuity_score", None),
                        "mask_pixels": getattr(roi, "mask_pixels", None),
                        "mask_coverage": getattr(roi, "mask_coverage", None),
                        "edge_mask_coverage": getattr(roi, "edge_mask_coverage", None),
                        "masked_black_ratio": getattr(roi, "masked_black_ratio", None),
                        "mask": roi.mask,
                    }
                )

    config_path = output_root / "foreground_dataset_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=True, separators=(",", ":"))

    metadata_path = output_root / "foreground_roi_metadata.json"
    serializable_records = [{key: value for key, value in record.items() if key != "mask"} for record in roi_records]
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(
            {"foreground_mode": "roi_crop", "rois": serializable_records}, f, ensure_ascii=True, separators=(",", ":")
        )
    return {"config_path": config_path, "roi_records": roi_records, "metadata_path": metadata_path}


def build_foreground_roi_records(grouped: dict[str, list]) -> list[dict]:
    roi_records = []
    for sequence_name, bundles in grouped.items():
        for frame_idx, bundle in enumerate(bundles, start=1):
            for roi_idx, roi in enumerate(bundle.selected_rois):
                x1, y1, x2, y2 = roi.roi_bbox
                padded_height, padded_width = pad_frame_to_even(roi.image, pad_value=0).shape[:2]
                if padded_width <= 0 or padded_height <= 0:
                    continue
                roi_records.append(
                    {
                        "sequence": sequence_name,
                        "frame_idx": frame_idx,
                        "frame_id": bundle.frame_id,
                        "roi_idx": roi_idx,
                        "roi_sequence": foreground_roi_sequence_name(sequence_name, bundle.frame_id, roi_idx),
                        "bbox": [int(value) for value in roi.bbox],
                        "roi_bbox": [x1, y1, x2, y2],
                        "track_id": int(roi.track_id),
                        "class_id": int(roi.class_id),
                        "category": str(getattr(roi, "category", "other")),
                        "width": x2 - x1,
                        "height": y2 - y1,
                        "padded_width": padded_width,
                        "padded_height": padded_height,
                        "motion_score": float(getattr(roi, "motion_score", 0.0)),
                        "uncertainty": float(getattr(roi, "uncertainty", 0.0)),
                        "confidence": getattr(roi, "confidence", None),
                        "continuity_score": getattr(roi, "continuity_score", None),
                        "mask_pixels": getattr(roi, "mask_pixels", None),
                        "mask_coverage": getattr(roi, "mask_coverage", None),
                        "edge_mask_coverage": getattr(roi, "edge_mask_coverage", None),
                        "masked_black_ratio": getattr(roi, "masked_black_ratio", None),
                        "mask": roi.mask,
                    }
                )
    return roi_records


def attach_raw_metrics(points: list[dict], metrics_json: Path) -> list[dict]:
    raw_results = load_artifact_json(metrics_json)
    for point in points:
        point["raw_metrics"] = raw_results[point["dataset"]][point["sequence"]][f"{point['rate_idx']:03d}"]
    return points


def attach_raw_metrics_from_points(points: list[dict]) -> list[dict]:
    for point in points:
        point["raw_metrics"] = point.get("raw_metrics", {})
    return points


def load_elic_jpg_points(summary_path: Path | None) -> list[dict]:
    if summary_path is None or not summary_path.exists():
        return []
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    points = []
    for result in payload.get("results", []):
        if result.get("scheme") not in {"elic_jpeg", "elic_background_jpeg_foreground"}:
            continue
        for point in result.get("points", []):
            points.extend(point.get("per_sequence", []) or [point])
    return points


def build_elic_comparison(layered_points: list[dict], elic_points: list[dict], summary_path: Path | None) -> dict:
    comparison = {
        "reference_scheme": "elic_jpeg",
        "reference_summary": str(summary_path) if summary_path is not None else None,
        "available": bool(elic_points),
        "points": [],
    }
    if not elic_points:
        comparison["notes"] = ["ELIC+JPEG summary was not found or did not contain comparable points."]
        return comparison

    elic_by_sequence = {point.get("sequence"): point for point in elic_points}
    for layered in layered_points:
        reference = elic_by_sequence.get(layered.get("sequence"))
        if reference is None or reference.get("bpp") is None or reference.get("avg_psnr") is None:
            continue

        layered_bpp = float(layered["core_video_bpp"])
        layered_estimated_bpp = float(layered.get("estimated_total_bpp", layered_bpp))
        reference_bpp = float(reference["bpp"])
        layered_psnr = float(layered["avg_psnr"])
        reference_psnr = float(reference["avg_psnr"])
        comparison["points"].append(
            {
                "sequence": layered["sequence"],
                "layered_core_video_bpp": layered_bpp,
                "layered_estimated_total_bpp": layered_estimated_bpp,
                "elic_jpg_bpp": reference_bpp,
                "core_bpp_delta": layered_bpp - reference_bpp,
                "core_bpp_delta_percent": ((layered_bpp / reference_bpp) - 1.0) * 100.0 if reference_bpp else None,
                "estimated_bpp_delta": layered_estimated_bpp - reference_bpp,
                "estimated_bpp_delta_percent": ((layered_estimated_bpp / reference_bpp) - 1.0) * 100.0
                if reference_bpp
                else None,
                "layered_avg_psnr": layered_psnr,
                "elic_jpg_avg_psnr": reference_psnr,
                "psnr_delta_db": layered_psnr - reference_psnr,
            }
        )

    if comparison["points"]:
        comparison["mean"] = {
            key: seven_sequence_mean(comparison["points"], key)
            for key in (
                "core_bpp_delta",
                "core_bpp_delta_percent",
                "estimated_bpp_delta",
                "estimated_bpp_delta_percent",
                "psnr_delta_db",
            )
        }
    else:
        comparison["available"] = False
        comparison["notes"] = ["ELIC+JPEG points exist, but no sequence names matched layered output."]
    return comparison


def _minimal_metric_point(point: dict) -> dict:
    frames = int(point["frame_count"])
    roi_count = int(point.get("foreground_roi_count", 0) or 0)
    bpp_visual = float(point["core_video_bpp"])
    bpp_side = float(point.get("mask_bpp", 0.0) or 0.0) + float(point.get("metadata_bpp", 0.0) or 0.0)
    return {
        "sequence": point["sequence"],
        "frames": frames,
        "roi_per_frame": float(roi_count / frames) if frames else 0.0,
        "bpp": float(point.get("estimated_total_bpp", bpp_visual + bpp_side)),
        "psnr": float(point["avg_psnr"]),
    }


def summarize_quantitative_metrics(points: list[dict], frame_metrics: list[dict], fps: int, comparison: dict) -> dict:
    if not points:
        return {"available": False}
    minimal_points = [_minimal_metric_point(point) for point in points]
    total_frames = int(sum(point["frames"] for point in minimal_points))
    overall = {
        "frames": total_frames,
        "roi_per_frame": seven_sequence_mean(minimal_points, "roi_per_frame"),
        "bpp": seven_sequence_mean(minimal_points, "bpp"),
        "psnr": seven_sequence_mean(minimal_points, "psnr"),
    }
    return {
        "available": True,
        "fps": fps,
        "aggregation": "macro: seven equally weighted MOT17 sequences; incomplete metrics are null",
        "overall": overall,
        "per_sequence": minimal_points,
        "metric_definitions": {
            "roi_per_frame": "foreground ROI groups divided by frame count",
            "bpp": "total BPP, including visual payload, masks, and metadata",
            "psnr": "average reconstructed-frame PSNR in dB",
        },
    }


def format_metric(value, digits: int = 6, suffix: str = "") -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.{digits}f}{suffix}"
    return f"{value}{suffix}"


def format_layered_metrics(point_count: int, device: str, quantitative_metrics: dict) -> str:
    overall = quantitative_metrics.get("overall") or {}
    comparison = quantitative_metrics.get("comparison_to_elic_jpeg")
    lines = [
        "Layered DCVC-RT metrics",
        f"points: {point_count}",
        f"device: {device}",
        f"frames: {format_metric(overall.get('frames'))}",
        f"roi_per_frame: {format_metric(overall.get('roi_per_frame'))}",
        f"bpp: {format_metric(overall.get('bpp'))}",
        f"psnr: {format_metric(overall.get('psnr'))}",
    ]
    if comparison:
        lines.extend(
            [
                "",
                "Comparison to ELIC+JPEG",
                f"core_bpp_delta_percent: {format_metric(comparison.get('core_bpp_delta_percent'), suffix='%')}",
                f"estimated_bpp_delta_percent: {format_metric(comparison.get('estimated_bpp_delta_percent'), suffix='%')}",
                f"psnr_delta_db: {format_metric(comparison.get('psnr_delta_db'))}",
            ]
        )
    return "\n".join(lines)


def prepare_layered_candidates(args, controller) -> dict:
    python_executable = str(args.python_executable.resolve())
    env_info = inspect_python_environment(python_executable)
    version = tuple(env_info["version"])
    if version < (3, 12, 0):
        raise RuntimeError(f"DCVC-RT requires Python >= 3.12, got {version} from {python_executable}")
    if not env_info["mlcodec_ext"]:
        raise RuntimeError(
            "Missing MLCodec_extensions_cpp in the selected environment. "
            "Run `python -m pip install --no-build-isolation .\\vendor\\DCVC\\src\\cpp` first."
        )

    output_root = ensure_dir(args.output_root.resolve())
    background_streams_root = ensure_dir(output_root / "background_streams")
    foreground_streams_root = ensure_dir(output_root / "foreground_streams")
    use_cuda = resolve_device(args.device)
    grouped = limit_grouped_frames(
        group_bundles(args.bundle_root.resolve(), args.force_frame_num), args.force_frame_num
    )
    grouped = filter_grouped_sequences(grouped, getattr(args, "sequence_filter", []))

    bg_output_json = output_root / "background_dcvc_output.json"
    fg_output_json = output_root / "foreground_dcvc_output.json"
    required_bg_output_json = output_root / "required_background_dcvc_output.json"
    layered_metrics_json = output_root / "dcvc_layered_metrics.json"
    required_background_interval = compute_required_background_interval_frames(args)
    background_summary_change_threshold = float(getattr(args, "background_summary_change_threshold", 0.0) or 0.0)
    background_summary_min_interval = int(getattr(args, "background_summary_min_interval_frames", 0) or 0)
    has_required_background_frames = any(
        any(
            frame_idx != 1
            and (
                (required_background_interval > 0 and frame_idx % required_background_interval == 0)
                or (
                    background_summary_change_threshold > 0
                    and float(getattr(bundle, "background_change_score", 0.0) or 0.0)
                    >= background_summary_change_threshold
                )
            )
            for frame_idx, bundle in enumerate(bundles, start=1)
        )
        for bundles in grouped.values()
    )

    if args.reuse_dcvc_outputs:
        if not layered_metrics_json.exists():
            require_existing_dcvc_output(bg_output_json, "background")
            require_existing_dcvc_output(fg_output_json, "foreground")
            write_layered_dcvc_metrics_json(bg_output_json, fg_output_json, layered_metrics_json)
        background_export = {
            "config_path": output_root / "background_dataset_config.json",
            "sequence_meta": {},
        }
        foreground_export = {
            "config_path": output_root / "foreground_dataset_config.json",
            "roi_records": build_foreground_roi_records(grouped),
            "metadata_path": output_root / "foreground_roi_metadata.json",
        }
        required_background_export = {
            "config_path": output_root / "required_background_dataset_config.json",
            "records": [],
        }
        if has_required_background_frames and not required_bg_output_json.exists():
            raise FileNotFoundError(
                f"Missing existing required-background DCVC output: {required_bg_output_json}. "
                "Use run_level=refresh_full after enabling required background keyframes."
            )
    else:
        background_export = export_low_frequency_backgrounds(
            grouped,
            output_root,
            intra_period=-1,
            default_fps=args.fps,
        )
        required_background_export = export_required_background_keyframes(
            grouped,
            output_root,
            required_interval=required_background_interval,
            summary_change_threshold=background_summary_change_threshold,
            summary_min_interval=background_summary_min_interval,
            default_fps=args.fps,
        )
        foreground_export = export_foreground_roi_crops(
            grouped,
            output_root,
            intra_period=-1,
            default_fps=args.fps,
        )
        bg_command = build_dcvc_rt_command(
            python_executable=python_executable,
            dcvc_repo=args.dcvc_repo.resolve(),
            dataset_config=background_export["config_path"],
            model_path_i=args.model_path_i.resolve(),
            model_path_p=args.model_path_p.resolve(),
            output_json=bg_output_json,
            stream_path=background_streams_root,
            rate_num=args.background_rate_num,
            worker_num=args.worker_num,
            use_cuda=use_cuda,
            force_zero_thres=args.force_zero_thres,
            force_frame_num=args.force_frame_num,
            reset_interval=args.reset_interval,
            verbose=args.verbose,
            calc_ssim=args.calc_ssim,
            qp_i=getattr(args, "background_qp_i", None),
            qp_p=getattr(args, "background_qp_p", None),
        )
        fg_command = build_dcvc_rt_command(
            python_executable=python_executable,
            dcvc_repo=args.dcvc_repo.resolve(),
            dataset_config=foreground_export["config_path"],
            model_path_i=args.model_path_i.resolve(),
            model_path_p=args.model_path_p.resolve(),
            output_json=fg_output_json,
            stream_path=foreground_streams_root,
            rate_num=args.foreground_rate_num,
            worker_num=args.worker_num,
            use_cuda=use_cuda,
            force_zero_thres=args.force_zero_thres,
            force_frame_num=args.force_frame_num,
            reset_interval=args.reset_interval,
            verbose=args.verbose,
            calc_ssim=args.calc_ssim,
        )
        subprocess.run(bg_command, cwd=args.dcvc_repo.resolve(), check=True)
        if required_background_export["records"]:
            required_bg_command = build_dcvc_rt_command(
                python_executable=python_executable,
                dcvc_repo=args.dcvc_repo.resolve(),
                dataset_config=required_background_export["config_path"],
                model_path_i=args.model_path_i.resolve(),
                model_path_p=args.model_path_p.resolve(),
                output_json=required_bg_output_json,
                stream_path=background_streams_root,
                rate_num=args.background_rate_num,
                worker_num=args.worker_num,
                use_cuda=use_cuda,
                force_zero_thres=args.force_zero_thres,
                force_frame_num=-1,
                reset_interval=args.reset_interval,
                verbose=args.verbose,
                calc_ssim=args.calc_ssim,
                qp_i=getattr(args, "background_qp_i", None),
                qp_p=getattr(args, "background_qp_p", None),
            )
            subprocess.run(required_bg_command, cwd=args.dcvc_repo.resolve(), check=True)
        subprocess.run(fg_command, cwd=args.dcvc_repo.resolve(), check=True)
        write_layered_dcvc_metrics_json(bg_output_json, fg_output_json, layered_metrics_json)
        remove_if_exists(bg_output_json)
        remove_if_exists(fg_output_json)
        remove_if_exists(background_export["config_path"])
        remove_if_exists(required_background_export["config_path"])
        remove_if_exists(foreground_export["config_path"])
        remove_if_exists(foreground_export["metadata_path"])

    organize_dcvc_outputs(output_root, discard_sidecars=True, replace_existing=not args.reuse_dcvc_outputs)
    if not args.reuse_dcvc_outputs:
        write_metric_index(layered_metrics_json, output_root / "encoded")
        if required_bg_output_json.exists():
            write_metric_index(required_bg_output_json, output_root / "encoded")
    bg_points = attach_raw_metrics_from_points(parse_layered_dcvc_metrics(layered_metrics_json, "background"))
    fg_points = attach_raw_metrics_from_points(parse_layered_dcvc_metrics(layered_metrics_json, "foreground"))
    required_bg_points = parse_dcvc_results(required_bg_output_json) if required_bg_output_json.exists() else []
    return {
        "grouped": grouped,
        "background_export": background_export,
        "required_background_export": required_background_export,
        "foreground_export": foreground_export,
        "background_points": bg_points,
        "required_background_points": required_bg_points,
        "foreground_points": fg_points,
        "background_stream_dir": output_root / "encoded",
        "required_background_stream_dir": output_root / "encoded",
        "foreground_stream_dir": output_root / "encoded",
        "background_output_json": bg_output_json,
        "foreground_output_json": fg_output_json,
        "layered_metrics_json": layered_metrics_json,
        "background_streams_root": output_root / "encoded",
        "foreground_streams_root": output_root / "encoded",
        "output_root": output_root,
        "use_cuda": use_cuda,
    }
