import argparse
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from .background import BackgroundModel
from .config import (
    BACKGROUND_MEDIAN_SAMPLE_COUNT,
    DEFAULT_OUTPUT_ROOT,
    MAX_FOREGROUNDS,
    MOT17_ROOT,
    MOT17_SEQUENCES,
    MOT17_SPLIT,
)
from .detector import Detector
from .exporter import export_factorized_frame
from .factorization import SceneFactorization
from .mot17_loader import MOT17Loader
from .tracker import Tracker
from .types import Image
from .visualizer import DEFAULT_PREVIEW_FPS, VisualizationVideoWriter, close_preview, show_preview


def build_factorizer(max_foregrounds: int) -> SceneFactorization:
    """Construct the stateful detector, background model, and tracker pipeline."""
    background_model = BackgroundModel()
    detector = Detector()
    tracker = Tracker()
    return SceneFactorization(background_model, detector, tracker, max_foregrounds=max_foregrounds)


def build_median_background(
    frame_paths: Sequence[Path],
    sample_count: int = BACKGROUND_MEDIAN_SAMPLE_COUNT,
    detector: Detector | None = None,
    io_workers: int = 4,
) -> Image | None:
    """Decode sampled frames concurrently, then remove detected objects in order."""
    if not frame_paths:
        return None
    if sample_count < 1 or io_workers < 1:
        raise ValueError("sample_count and io_workers must be positive")

    indices = np.linspace(0, len(frame_paths) - 1, min(sample_count, len(frame_paths)), dtype=int)
    frames = []
    paths = [frame_paths[index] for index in indices]
    with ThreadPoolExecutor(max_workers=io_workers) as pool:
        for offset in range(0, len(paths), io_workers):
            batch = paths[offset : offset + io_workers]
            for frame in pool.map(lambda path: cv2.imread(str(path)), batch):
                if frame is not None:
                    if detector is not None:
                        frame = remove_detected_objects(frame, detector)
                    frames.append(frame)

    if not frames:
        return None

    return np.median(np.stack(frames, axis=0), axis=0).astype(np.uint8)


def remove_detected_objects(frame: Image, detector: Detector) -> Image:
    """Inpaint detected objects before estimating a static scene background."""
    detections = detector.detect(frame)
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    for detection in detections:
        instance_mask = detection.get("instance_mask")
        if instance_mask is not None:
            mask[instance_mask > 0] = 255
            continue

        x1, y1, x2, y2 = [int(value) for value in detection["bbox"]]
        x1 = max(0, min(x1, frame.shape[1]))
        y1 = max(0, min(y1, frame.shape[0]))
        x2 = max(0, min(x2, frame.shape[1]))
        y2 = max(0, min(y2, frame.shape[0]))
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 255

    if np.count_nonzero(mask) == 0:
        return frame

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.dilate(mask, kernel, iterations=2)
    return cv2.inpaint(frame, mask, 3, cv2.INPAINT_TELEA)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run scene factorization and export background/foreground results."
    )
    parser.add_argument("--mot17-root", default=MOT17_ROOT)
    parser.add_argument("--split", default=MOT17_SPLIT)
    parser.add_argument("--sequences", nargs="*", default=MOT17_SEQUENCES)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--max-foregrounds", type=int, default=MAX_FOREGROUNDS)
    parser.add_argument("--background-io-workers", type=int, default=4)
    parser.add_argument(
        "--no-preview", action="store_true", help="Disable the realtime preview window."
    )
    parser.add_argument(
        "--no-visualization-video",
        action="store_true",
        help="Disable writing the 2x2 visualization video.",
    )
    return parser.parse_args(argv)


def main() -> None:
    """Process configured MOT17 sequences and export reconstruction artifacts."""
    args = parse_args()
    output_root = Path(args.output_root)
    loader = MOT17Loader(args.mot17_root, split=args.split, sequences=args.sequences)
    factorizer = build_factorizer(args.max_foregrounds)
    current_sequence = None
    visualization_writer = None
    exported_frames = 0

    print(f"Detector device: {factorizer.detector.device}")
    print(f"Export root: {output_root}")

    try:
        for sample in loader:
            sequence_name = sample["sequence"]
            if sample["is_new_sequence"] or sequence_name != current_sequence:
                if visualization_writer is not None:
                    visualization_writer.close()
                    visualization_writer = None

                current_sequence = sequence_name
                initial_background = build_median_background(
                    sample.get("sequence_frame_paths", []),
                    detector=factorizer.detector,
                    io_workers=args.background_io_workers,
                )
                factorizer.reset(initial_background=initial_background)
                sequence_fps = sample["seqinfo"].get("frameRate", DEFAULT_PREVIEW_FPS)
                print(f"Processing sequence: {sequence_name}")
                if initial_background is not None:
                    print(f"Initialized background from {BACKGROUND_MEDIAN_SAMPLE_COUNT} frames")
                if not args.no_visualization_video:
                    video_path = output_root / sequence_name / "visualization.mp4"
                    video_path.parent.mkdir(parents=True, exist_ok=True)
                    visualization_writer = VisualizationVideoWriter(video_path, fps=sequence_fps)
                    print(f"Visualization video: {video_path} ({sequence_fps:g} fps)")

            frame = sample["frame"]
            sf_out = factorizer.process(frame)
            export_factorized_frame(
                output_root,
                sample,
                frame,
                sf_out,
                max_foregrounds=args.max_foregrounds,
            )
            exported_frames += 1

            if visualization_writer is not None:
                visualization_writer.write(frame, sf_out, sample=sample)

            if not args.no_preview:
                preview_fps = sample["seqinfo"].get("frameRate", DEFAULT_PREVIEW_FPS)
                keep_running = show_preview(frame, sf_out, sample=sample, fps=preview_fps)
                if not keep_running:
                    print("Preview closed by user.")
                    break

            if exported_frames % 20 == 0:
                print(f"Exported {exported_frames} frames")
    finally:
        if visualization_writer is not None:
            visualization_writer.close()
        if not args.no_preview:
            close_preview()

    print(f"Completed. Exported {exported_frames} frames to {output_root}")


if __name__ == "__main__":
    main()
