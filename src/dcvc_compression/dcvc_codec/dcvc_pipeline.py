import json
import subprocess
from collections.abc import Iterable
from pathlib import Path

import cv2
import numpy as np

from common.encoding_layout import load_artifact_json, metric_stream_directory

DCVC_REQUIRED_METRIC_KEYS = {
    "ave_all_frame_psnr",
    "ave_all_frame_msssim",
    "ave_all_frame_bpp",
    "i_frame_num",
    "p_frame_num",
    "avg_frame_encoding_time",
    "avg_frame_decoding_time",
    "qp_i",
    "qp_p",
    "frame_bpp",
}


def bgr_to_yuv420_bytes(frame):
    yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV_I420)
    return yuv.tobytes()


def build_dcvc_rt_command(
    python_executable: str,
    dcvc_repo: Path,
    dataset_config: Path,
    model_path_i: Path,
    model_path_p: Path,
    output_json: Path,
    stream_path: Path,
    rate_num: int,
    worker_num: int,
    use_cuda: bool,
    force_zero_thres: float,
    force_frame_num: int,
    reset_interval: int,
    verbose: int,
    calc_ssim: bool,
    qp_i: Iterable[int] | None = None,
    qp_p: Iterable[int] | None = None,
) -> list[str]:
    command = [
        python_executable,
        str(dcvc_repo / "test_video.py"),
        "--model_path_i",
        str(model_path_i),
        "--model_path_p",
        str(model_path_p),
        "--rate_num",
        str(rate_num),
        "--test_config",
        str(dataset_config),
        "--cuda",
        str(use_cuda),
        "-w",
        str(worker_num),
        "--write_stream",
        "True",
        "--stream_path",
        str(stream_path),
        "--save_decoded_frame",
        "True",
        "--calc_ssim",
        str(calc_ssim),
        "--verbose_json",
        "True",
        "--force_zero_thres",
        str(force_zero_thres),
        "--output_path",
        str(output_json),
        "--force_intra_period",
        "-1",
        "--reset_interval",
        str(reset_interval),
        "--force_frame_num",
        str(force_frame_num),
        "--check_existing",
        "False",
        "--verbose",
        str(verbose),
    ]
    if qp_i is not None:
        command.extend(["--qp_i", *(str(int(value)) for value in qp_i)])
    if qp_p is not None:
        command.extend(["--qp_p", *(str(int(value)) for value in qp_p)])
    return command


def inspect_python_environment(python_executable: str) -> dict:
    probe = subprocess.run(
        [
            python_executable,
            "-c",
            (
                "import importlib.util, json, sys; "
                "print(json.dumps({"
                "'version': list(sys.version_info[:3]), "
                "'torch': importlib.util.find_spec('torch') is not None, "
                "'pybind11': importlib.util.find_spec('pybind11') is not None, "
                "'mlcodec_ext': importlib.util.find_spec('MLCodec_extensions_cpp') is not None"
                "}))"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(probe.stdout.strip())


def parse_dcvc_results(path: Path) -> list[dict]:
    payload = load_artifact_json(path)
    points = []
    for dataset_name, dataset_result in payload.items():
        for sequence_name, sequence_result in dataset_result.items():
            for rate_idx, metrics in sorted(sequence_result.items(), key=lambda item: int(item[0])):
                points.append(
                    {
                        "dataset": dataset_name,
                        "sequence": sequence_name,
                        "rate_idx": int(rate_idx),
                        "avg_psnr": metrics["ave_all_frame_psnr"],
                        "avg_ms_ssim": metrics.get("ave_all_frame_msssim"),
                        "bpp": metrics["ave_all_frame_bpp"],
                        "frame_count": metrics["i_frame_num"] + metrics["p_frame_num"],
                        "avg_frame_encoding_time": metrics.get("avg_frame_encoding_time"),
                        "avg_frame_decoding_time": metrics.get("avg_frame_decoding_time"),
                        "raw_metrics": metrics,
                    }
                )
    return points


def normalize_dcvc_output_json(source_path: Path, output_path: Path) -> Path:
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    slim = {}
    for dataset_name, dataset_result in payload.items():
        slim_dataset = {}
        for sequence_name, sequence_result in dataset_result.items():
            slim_sequence = {}
            for rate_idx, metrics in sequence_result.items():
                slim_sequence[rate_idx] = {key: metrics[key] for key in DCVC_REQUIRED_METRIC_KEYS if key in metrics}
            slim_dataset[sequence_name] = slim_sequence
        slim[dataset_name] = slim_dataset
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(slim, ensure_ascii=True, separators=(",", ":")),
        encoding="utf-8",
    )
    return output_path


def compact_dcvc_output_payload(source_path: Path) -> dict:
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    slim = {}
    for dataset_name, dataset_result in payload.items():
        slim_dataset = {}
        for sequence_name, sequence_result in dataset_result.items():
            slim_sequence = {}
            for rate_idx, metrics in sequence_result.items():
                slim_sequence[rate_idx] = {key: metrics[key] for key in DCVC_REQUIRED_METRIC_KEYS if key in metrics}
            slim_dataset[sequence_name] = slim_sequence
        slim[dataset_name] = slim_dataset
    return slim


def write_layered_dcvc_metrics_json(
    background_output_json: Path,
    foreground_output_json: Path,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "format": "dcvc_layered_metrics_v1",
                "background": compact_dcvc_output_payload(background_output_json),
                "foreground": compact_dcvc_output_payload(foreground_output_json),
            },
            ensure_ascii=True,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    return output_path


def parse_layered_dcvc_metrics(path: Path, layer: str) -> list[dict]:
    payload = load_artifact_json(path)
    if payload.get("format") != "dcvc_layered_metrics_v1":
        raise ValueError(f"Unsupported layered DCVC metrics JSON format: {path}")
    layer_payload = payload.get(layer)
    if not isinstance(layer_payload, dict):
        raise ValueError(f"Missing {layer!r} metrics in {path}")
    points = []
    for dataset_name, dataset_result in layer_payload.items():
        for sequence_name, sequence_result in dataset_result.items():
            for rate_idx, metrics in sorted(sequence_result.items(), key=lambda item: int(item[0])):
                points.append(
                    {
                        "dataset": dataset_name,
                        "sequence": sequence_name,
                        "rate_idx": int(rate_idx),
                        "avg_psnr": metrics["ave_all_frame_psnr"],
                        "avg_ms_ssim": metrics.get("ave_all_frame_msssim"),
                        "bpp": metrics["ave_all_frame_bpp"],
                        "frame_count": metrics["i_frame_num"] + metrics["p_frame_num"],
                        "avg_frame_encoding_time": metrics.get("avg_frame_encoding_time"),
                        "avg_frame_decoding_time": metrics.get("avg_frame_decoding_time"),
                        "raw_metrics": metrics,
                    }
                )
    return points


def iter_yuv420_frames(yuv_path: Path, width: int, height: int) -> Iterable[np.ndarray]:
    frame_size = width * height * 3 // 2
    with open(yuv_path, "rb") as f:
        while True:
            raw = f.read(frame_size)
            if not raw:
                break
            yuv = np.frombuffer(raw, dtype=np.uint8).reshape((height * 3 // 2, width))
            yield cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)


def strip_dcvc_sequence_suffix(sequence_name: str) -> str:
    name = Path(sequence_name).stem
    parts = name.rsplit("_", 2)
    if len(parts) == 3 and "x" in parts[1] and parts[2].isdigit():
        return parts[0]
    return name


def find_reconstructed_yuv(stream_dir: Path, sequence_name: str, qp_i: int) -> Path | None:
    if stream_dir.name == "encoded":
        stream_dir = metric_stream_directory(stream_dir.parent / "decoded", sequence_name)
    base_name = Path(sequence_name).stem
    matches = sorted(stream_dir.glob(f"{base_name}*_q{qp_i}_*kbps.yuv"))
    return matches[-1] if matches else None
