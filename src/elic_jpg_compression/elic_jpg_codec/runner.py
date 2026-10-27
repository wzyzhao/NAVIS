import itertools
import json
import statistics
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
from tqdm import tqdm

from common.encoding_layout import load_artifact_json, write_frame_index
from common.images import read_image
from common.macro_metrics import seven_sequence_mean
from common.paths import PROJECT_ROOT as WORKSPACE_ROOT
from elic_jpg_compression.elic_jpg_codec.bundle_io import ensure_dir, iter_bundle_frames
from elic_jpg_compression.elic_jpg_codec.elic_pipeline import (
    ELICBackgroundCodec,
    compute_ms_ssim,
    compute_psnr,
    deserialize_rois,
    dump_pickle,
    serialize_rois,
)

PROJECT_ROOT = WORKSPACE_ROOT / "src" / "elic_jpg_compression"
DEFAULT_BUNDLE_ROOT = WORKSPACE_ROOT / "src" / "scene_factorization" / "scene_factorization_output"
DEFAULT_OUTPUT_ROOT = WORKSPACE_ROOT / "experiments" / "results" / "encoding" / "elic"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "models" / "elic" / "ELIC_0016_ft_3980_Plateau.pth.tar"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.json"
COMPARISON_ROLE = "factorized_full_reconstruction_reference"


DEFAULT_CONFIG = {
    "bundle_root": str(DEFAULT_BUNDLE_ROOT),
    "output_root": str(DEFAULT_OUTPUT_ROOT),
    "elic_checkpoint": str(DEFAULT_CHECKPOINT),
    "jpeg_quality": 85,
    "max_foregrounds": 12,
    "sequence_filter": ["MOT17-04-FRCNN"],
    "fps": 30,
    "device": "auto",
    "max_frames": -1,
    "candidate_export_only": False,
    "candidate_source_manifest": None,
}

PATH_FIELDS = {"bundle_root", "output_root", "elic_checkpoint", "candidate_source_manifest"}


def resolve_config_path(value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def load_config(config_path: Path = DEFAULT_CONFIG_PATH):
    config = dict(DEFAULT_CONFIG)
    if config_path.exists():
        with open(config_path, encoding="utf-8") as f:
            config.update(json.load(f))
    for field in PATH_FIELDS:
        config[field] = resolve_config_path(config.get(field))
    return SimpleNamespace(**config)


def validate_config(config) -> None:
    if config.device not in {"auto", "cpu", "cuda"}:
        raise ValueError("config device must be one of ['auto', 'cpu', 'cuda'].")
    if int(config.jpeg_quality) < 1 or int(config.jpeg_quality) > 100:
        raise ValueError("config jpeg_quality must be in [1, 100].")
    if int(config.max_foregrounds) < 0:
        raise ValueError("config max_foregrounds must be >= 0.")
    if int(config.fps) <= 0:
        raise ValueError("config fps must be > 0.")


def find_probe_image(bundle_root: Path) -> Path | None:
    for sequence_dir in sorted(path for path in bundle_root.iterdir() if path.is_dir()):
        for frame_dir in sorted(path for path in sequence_dir.iterdir() if path.is_dir()):
            image_path = frame_dir / "background.png"
            if image_path.exists():
                return image_path
    return None


def probe_elic_cuda_roundtrip(checkpoint_path: Path, probe_image_path: Path | None) -> tuple[bool, str]:
    script_lines = [
        "from pathlib import Path",
        "import cv2",
        "import numpy as np",
        "from elic_jpg_compression.elic_jpg_codec.elic_pipeline import ELICBackgroundCodec, compute_psnr",
        f'checkpoint = Path(r"{checkpoint_path}")',
    ]
    if probe_image_path is None:
        script_lines.append("image = np.zeros((64, 64, 3), dtype=np.uint8)")
    else:
        script_lines.extend(
            [
                f'image = cv2.imread(r"{probe_image_path}", cv2.IMREAD_COLOR)',
                "if image is None:",
                f'    raise RuntimeError(r"Failed to read probe image: {probe_image_path}")',
            ]
        )
    script_lines.extend(
        [
            'codec = ELICBackgroundCodec(str(checkpoint), device="cuda")',
            "payload = codec.compress(image)",
            "decoded = codec.decompress(payload)",
            "score = compute_psnr(image, decoded)",
            "if score < 20.0:",
            '    raise RuntimeError(f"roundtrip PSNR too low: {score:.4f} dB")',
            'print(f"ok psnr={score:.4f}")',
        ]
    )
    script = "\n".join(script_lines)
    completed = subprocess.run(
        [sys.executable, "-u", "-c", script],
        cwd=str(WORKSPACE_ROOT),
        capture_output=True,
        text=True,
    )
    if completed.returncode == 0:
        return True, completed.stdout.strip()
    details = (completed.stderr or completed.stdout or "").strip()
    return False, details or f"subprocess exit code {completed.returncode}"


def resolve_device(
    requested: str,
    checkpoint_path: Path,
    bundle_root: Path,
    *,
    candidate_export_only: bool = False,
) -> str:
    probe_image_path = find_probe_image(bundle_root)
    if requested == "auto":
        if not torch.cuda.is_available():
            return "cpu"
        if candidate_export_only:
            return "cuda"
        ok, details = probe_elic_cuda_roundtrip(checkpoint_path, probe_image_path)
        if ok:
            return "cuda"
        print(
            "Warning: CUDA is available but ELIC entropy roundtrip is incompatible "
            f"with the current torch/compressai stack; falling back to CPU. Details: {details}",
            file=sys.stderr,
        )
        return "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")
    if requested == "cuda":
        ok, details = probe_elic_cuda_roundtrip(checkpoint_path, probe_image_path)
        if not ok:
            raise RuntimeError(
                "CUDA requested, but ELIC entropy roundtrip is incompatible with the current "
                f"torch/compressai stack. Details: {details}"
            )
    return requested


def directory_size_bits(path: Path) -> int:
    total = 0
    for file_path in path.rglob("*"):
        if file_path.is_file():
            total += file_path.stat().st_size * 8
    return total


def write_video_from_png_dir(frames_dir: Path, output_path: Path, fps: int) -> None:
    frame_paths = sorted(frames_dir.glob("*.png"))
    if not frame_paths:
        return
    first = cv2.imread(str(frame_paths[0]))
    height, width = first.shape[:2]
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height))
    try:
        for frame_path in frame_paths:
            writer.write(cv2.imread(str(frame_path)))
    finally:
        writer.release()


def load_existing_progress(output_root: Path) -> tuple[list[dict], set[tuple[str, int]]]:
    frame_metrics_path = output_root / "frame_metrics.json"
    if not frame_metrics_path.exists():
        return [], set()
    payload = load_artifact_json(frame_metrics_path)
    frame_metrics = sorted(
        list(payload.get("frames", [])),
        key=lambda item: (str(item["sequence"]), int(item["frame_id"])),
    )
    for item in frame_metrics:
        frame_path = Path(str(item.get("reconstructed_frame", "")))
        if not frame_path.exists():
            sequence = str(item["sequence"])
            frame_id = int(item["frame_id"])
            repaired_path = output_root / "decoded" / sequence / "frames" / f"{frame_id:06d}.png"
            if repaired_path.exists():
                item["reconstructed_frame"] = str(repaired_path)
    completed = {(str(item["sequence"]), int(item["frame_id"])) for item in frame_metrics}
    return frame_metrics, completed


def load_candidate_background_frames(source_manifest: Path | None) -> set[tuple[str, int]]:
    if source_manifest is None:
        return set()
    source = load_artifact_json(source_manifest)
    return {
        (str(frame["sequence"]), int(frame.get("frame_id", frame["frame_idx"])))
        for frame in source.get("frames", [])
        if any(item.get("layer") == "background_keyframe_summary" for item in frame.get("items", []))
    }


def build_summary(
    output_root: Path,
    bundle_root: Path,
    device: str,
    jpeg_quality: int,
    max_foregrounds: int,
    sequence_filter,
    comparison_role: str,
    fps: int,
    frame_metrics: list[dict],
    encoded_root: Path,
    decoded_root: Path,
    videos: list[str],
    status: str,
) -> dict:
    by_sequence = {}
    for item in frame_metrics:
        by_sequence.setdefault(str(item["sequence"]), []).append(item)
    per_sequence = []
    for sequence, rows in sorted(by_sequence.items()):
        source_frame_count = sum(1 for _ in (bundle_root / sequence).glob("*/original.png"))
        complete_frames = source_frame_count > 0 and len(rows) == source_frame_count
        pixels = 0
        source_complete = True
        for row in rows:
            source_path = bundle_root / sequence / f"{int(row['frame_id']):06d}" / "original.png"
            image = read_image(source_path)
            if image is None:
                source_complete = False
            else:
                pixels += image.shape[0] * image.shape[1]
        psnr_values = [row["psnr"] for row in rows if row.get("psnr") is not None]
        ssim_values = [row["ms_ssim"] for row in rows if row.get("ms_ssim") is not None]
        per_sequence.append(
            {
                "sequence": sequence,
                "frame_count": len(rows),
                "source_frame_count": source_frame_count,
                "evaluated_psnr_frames": len(psnr_values),
                "evaluated_ms_ssim_frames": len(ssim_values),
                "source_pixels": pixels if source_complete else None,
                "avg_psnr": statistics.fmean(psnr_values)
                if complete_frames and len(psnr_values) == len(rows)
                else None,
                "avg_ms_ssim": statistics.fmean(ssim_values)
                if complete_frames and len(ssim_values) == len(rows)
                else None,
                "bpp": sum(directory_size_bits(encoded_root / sequence / f"{int(row['frame_id']):06d}") for row in rows)
                / pixels
                if pixels and source_complete and complete_frames
                else None,
            }
        )
    point = {
        "rate_idx": 0,
        "label": f"jpeg_q{jpeg_quality}",
        "aggregation": "macro: seven equally weighted MOT17 sequences; incomplete metrics are null",
        "avg_psnr": seven_sequence_mean(per_sequence, "avg_psnr"),
        "avg_ms_ssim": seven_sequence_mean(per_sequence, "avg_ms_ssim"),
        "bpp": seven_sequence_mean(per_sequence, "bpp"),
        "bpp_basis": "local encoded pickle bytes / full source pixels; not application payload BPP",
        "frame_count": len(frame_metrics),
        "evaluated_psnr_frames": sum(row["evaluated_psnr_frames"] for row in per_sequence),
        "evaluated_ms_ssim_frames": sum(row["evaluated_ms_ssim_frames"] for row in per_sequence),
        "per_sequence": per_sequence,
    }
    return {
        "project": "elic_jpg_compression",
        "scheme": "elic_jpeg",
        "comparison_role": comparison_role,
        "status": status,
        "device": device,
        "input": {
            "bundle_root": str(bundle_root),
            "jpeg_quality": jpeg_quality,
            "max_foregrounds": max_foregrounds,
            "sequence_filter": sequence_filter,
            "fps": fps,
        },
        "results": [
            {
                "scheme": "elic_jpeg",
                "points": [point],
                "artifacts": {
                    "encoded_root": str(encoded_root),
                    "decoded_frames_root": str(decoded_root),
                    "frame_metrics_json": str(output_root / "frame_metrics.json"),
                    "videos": videos,
                },
            }
        ],
        "format_review": {
            "reasonable_for_metrics": True,
            "notes": [
                "Single-rate ELIC+JPEG output is normalized to a one-element points list.",
                "PSNR, MS-SSIM, BPP, and frame_count are scalar values suitable for CSV or plotting.",
            ],
        },
    }


def persist_progress(
    output_root: Path,
    bundle_root: Path,
    device: str,
    jpeg_quality: int,
    max_foregrounds: int,
    sequence_filter,
    comparison_role: str,
    fps: int,
    frame_metrics: list[dict],
    encoded_root: Path,
    decoded_root: Path,
    videos: list[str],
    status: str,
) -> None:
    write_frame_index(output_root / "frame_metrics.json", {"frames": frame_metrics}, output_root / "decoded")
    summary = build_summary(
        output_root=output_root,
        bundle_root=bundle_root,
        device=device,
        jpeg_quality=jpeg_quality,
        max_foregrounds=max_foregrounds,
        sequence_filter=sequence_filter,
        comparison_role=comparison_role,
        fps=fps,
        frame_metrics=frame_metrics,
        encoded_root=encoded_root,
        decoded_root=decoded_root,
        videos=videos,
        status=status,
    )
    with open(output_root / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=True, indent=2)


def format_metric(value, digits: int = 6) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def format_elic_metrics(summary: dict) -> str:
    point = summary["results"][0]["points"][0]
    lines = [
        "ELIC+JPEG metrics",
        f"status: {summary['status']}",
        f"device: {summary['device']}",
        f"frames: {point['frame_count']}",
        f"bpp: {format_metric(point['bpp'])}",
        f"avg_psnr_db: {format_metric(point['avg_psnr'])}",
        f"avg_ms_ssim: {format_metric(point['avg_ms_ssim'])}",
    ]
    return "\n".join(lines)


def main():
    args = load_config()
    validate_config(args)
    bundle_root = args.bundle_root.resolve()
    output_root = ensure_dir(args.output_root.resolve())
    encoded_root = ensure_dir(output_root / "encoded")
    decoded_root = ensure_dir(output_root / "decoded")

    checkpoint_path = args.elic_checkpoint.resolve()
    candidate_export_only = bool(getattr(args, "candidate_export_only", False))
    candidate_background_frames = load_candidate_background_frames(
        args.candidate_source_manifest if candidate_export_only else None
    )
    device = resolve_device(
        args.device,
        checkpoint_path,
        bundle_root,
        candidate_export_only=candidate_export_only,
    )
    codec = ELICBackgroundCodec(str(checkpoint_path), device=device)

    bundle_iter = iter_bundle_frames(
        bundle_root,
        sequence_filter=getattr(args, "sequence_filter", []),
        roi_count=int(args.max_foregrounds),
    )
    if args.max_frames > 0:
        bundle_iter = itertools.islice(bundle_iter, args.max_frames)
    first_bundle = next(bundle_iter, None)
    if first_bundle is None:
        raise RuntimeError(f"No bundle frames found under {bundle_root}")
    bundles = itertools.chain([first_bundle], bundle_iter)

    frame_metrics, completed_frames = load_existing_progress(output_root)
    selected_sequences = args.sequence_filter
    if isinstance(selected_sequences, str):
        selected_sequences = [selected_sequences]
    if selected_sequences:
        frame_metrics = [row for row in frame_metrics if row["sequence"] in selected_sequences]
        completed_frames = {(str(row["sequence"]), int(row["frame_id"])) for row in frame_metrics}
    videos: list[str] = []
    processed_sequences: set[str] = set()
    status = "running"

    try:
        for bundle in tqdm(bundles, desc="ELIC+JPEG", unit="frame"):
            processed_sequences.add(bundle.sequence)
            if (bundle.sequence, bundle.frame_id) in completed_frames:
                continue
            frame_dir = ensure_dir(encoded_root / bundle.sequence / f"{bundle.frame_id:06d}")
            rois = bundle.selected_rois[: args.max_foregrounds]

            has_candidate_background = (bundle.sequence, bundle.frame_id) in candidate_background_frames
            background_payload = (
                codec.compress(bundle.background) if not candidate_export_only or has_candidate_background else None
            )
            foreground_payload = serialize_rois(rois, args.jpeg_quality)
            if background_payload is not None:
                dump_pickle(frame_dir / "background.pkl", background_payload)
            dump_pickle(frame_dir / "foreground.pkl", foreground_payload)

            if candidate_export_only:
                frame_metrics.append(
                    {
                        "sequence": bundle.sequence,
                        "frame_id": bundle.frame_id,
                        "psnr": None,
                        "ms_ssim": None,
                        "foreground_count_used": len(rois),
                        "foreground_count_available": len(bundle.selected_rois),
                        "reconstructed_frame": "",
                    }
                )
                completed_frames.add((bundle.sequence, bundle.frame_id))
                if len(frame_metrics) % 100 == 0:
                    write_frame_index(output_root / "frame_metrics.json", {"frames": frame_metrics}, decoded_root)
                continue

            decoded_background = codec.decompress(background_payload)
            decoded_rois = deserialize_rois(foreground_payload)
            reconstructed = decoded_background.copy()
            for roi in decoded_rois:
                x1, y1, x2, y2 = roi.roi_bbox
                alpha = (roi.mask > 0)[..., None]
                reconstructed[y1:y2, x1:x2] = np.where(alpha, roi.image, reconstructed[y1:y2, x1:x2])

            sequence_dir = ensure_dir(decoded_root / bundle.sequence / "frames")
            reconstructed_path = sequence_dir / f"{bundle.frame_id:06d}.png"
            cv2.imwrite(str(reconstructed_path), reconstructed)

            psnr = compute_psnr(bundle.original_frame, reconstructed)
            ms_ssim = compute_ms_ssim(bundle.original_frame, reconstructed)
            frame_metrics.append(
                {
                    "sequence": bundle.sequence,
                    "frame_id": bundle.frame_id,
                    "psnr": psnr,
                    "ms_ssim": ms_ssim,
                    "foreground_count_used": len(rois),
                    "foreground_count_available": len(bundle.selected_rois),
                    "reconstructed_frame": str(reconstructed_path),
                }
            )
            completed_frames.add((bundle.sequence, bundle.frame_id))
            persist_progress(
                output_root=output_root,
                bundle_root=bundle_root,
                device=device,
                jpeg_quality=args.jpeg_quality,
                max_foregrounds=args.max_foregrounds,
                sequence_filter=args.sequence_filter,
                comparison_role=COMPARISON_ROLE,
                fps=args.fps,
                frame_metrics=frame_metrics,
                encoded_root=encoded_root,
                decoded_root=decoded_root,
                videos=[],
                status=status,
            )

        if not processed_sequences:
            processed_sequences = {path.name for path in decoded_root.iterdir() if path.is_dir()}
        if not candidate_export_only:
            for sequence in sorted(processed_sequences):
                video_path = ensure_dir(decoded_root / sequence) / "reconstruction.mp4"
                write_video_from_png_dir(decoded_root / sequence / "frames", video_path, args.fps)
                if video_path.exists():
                    videos.append(str(video_path))
        status = "completed"
    finally:
        persist_progress(
            output_root=output_root,
            bundle_root=bundle_root,
            device=device,
            jpeg_quality=args.jpeg_quality,
            max_foregrounds=args.max_foregrounds,
            sequence_filter=args.sequence_filter,
            comparison_role=COMPARISON_ROLE,
            fps=args.fps,
            frame_metrics=frame_metrics,
            encoded_root=encoded_root,
            decoded_root=decoded_root,
            videos=videos,
            status=status,
        )

    summary = json.loads((output_root / "summary.json").read_text(encoding="utf-8"))
    print(format_elic_metrics(summary))
