"""Evaluate receiver-visible AI track state from ACK-confirmed reconstruction artifacts.

This offline task measures NAVIS's core contribution with a fixed
receiver-state model: whether AI-generated object tracks remain task-ready when
current ROI appearance is not delivered. Source metadata is the fixed
detector/tracker reference. A state is current when its ROI patch arrived and
proxy when a previous patch for the same track remains within the configured TTL.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

LAYER_NAME = re.compile(r"frame_(\d+)_roi_(\d+)\.png$", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        required=True,
        help="Sequence folder containing six-digit frame folders and metadata.json.",
    )
    parser.add_argument(
        "--method",
        action="append",
        required=True,
        metavar="NAME=ARTIFACT",
        help="A reconstruction manifest (.jsonl) or decoded_layers directory. Repeat per method.",
    )
    parser.add_argument(
        "--no-proxy-method",
        action="append",
        default=[],
        metavar="NAME",
        help="Method whose receiver does not reuse historical ROI appearance. Repeat as needed.",
    )
    parser.add_argument("--proxy-ttl-frames", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_source(source_root: Path) -> dict[int, list[dict[str, int]]]:
    frames: dict[int, list[dict[str, int]]] = {}
    for metadata_path in sorted(source_root.glob("*/metadata.json")):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        frame_idx = int(metadata["frame_id"])
        frames[frame_idx] = [
            {"roi_idx": index, "track_id": int(roi["track_id"])}
            for index, roi in enumerate(metadata.get("selected_rois", []))
        ]
    if not frames:
        raise ValueError(f"No metadata.json files found under {source_root}")
    return frames


def current_from_manifest(path: Path) -> set[tuple[int, int]]:
    current: set[tuple[int, int]] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        frame_idx = int(entry["frame_idx"])
        for roi in entry.get("rois", []):
            current.add((frame_idx, int(roi["roi_idx"])))
    return current


def current_from_decoded_layers(path: Path) -> set[tuple[int, int]]:
    current: set[tuple[int, int]] = set()
    for image_path in path.rglob("*.png"):
        match = LAYER_NAME.search(image_path.name)
        if match:
            current.add((int(match.group(1)), int(match.group(2))))
    return current


def read_current(path: Path) -> set[tuple[int, int]]:
    if path.is_file() and path.suffix.lower() == ".jsonl":
        return current_from_manifest(path)
    if path.is_dir():
        return current_from_decoded_layers(path)
    raise ValueError(f"Artifact must be a .jsonl manifest or decoded_layers directory: {path}")


def evaluate(source: dict[int, list[dict[str, int]]], current: set[tuple[int, int]], ttl: int) -> dict[str, Any]:
    counts = defaultdict(int)
    track_states: dict[int, list[tuple[int, str]]] = defaultdict(list)
    last_current: dict[int, int] = {}

    for frame_idx in sorted(source):
        for roi in source[frame_idx]:
            roi_idx, track_id = roi["roi_idx"], roi["track_id"]
            if (frame_idx, roi_idx) in current:
                state = "current"
                last_current[track_id] = frame_idx
            elif track_id in last_current and frame_idx - last_current[track_id] <= ttl:
                state = "proxy"
            else:
                state = "missing"
            counts[state] += 1
            track_states[track_id].append((frame_idx, state))

    expected = sum(counts.values())
    ready = counts["current"] + counts["proxy"]
    adjacent_pairs = ready_pairs = 0
    episodes = 0
    track_coverage = []
    for observations in track_states.values():
        ready_observations = sum(state != "missing" for _, state in observations)
        track_coverage.append(ready_observations / len(observations))
        previously_ready = False
        for index, (frame_idx, state) in enumerate(observations):
            is_ready = state != "missing"
            if is_ready and not previously_ready:
                episodes += 1
            previously_ready = is_ready
            if index:
                previous_frame, previous_state = observations[index - 1]
                if frame_idx == previous_frame + 1:
                    adjacent_pairs += 1
                    ready_pairs += int(is_ready and previous_state != "missing")

    return {
        "expected_roi_observations": expected,
        "current_roi_observations": counts["current"],
        "proxy_roi_observations": counts["proxy"],
        "missing_roi_observations": counts["missing"],
        "task_ready_recall": ready / expected if expected else 0.0,
        "current_only_recall": counts["current"] / expected if expected else 0.0,
        "state_recovery_gain": counts["proxy"] / expected if expected else 0.0,
        "active_track_count": len(track_states),
        "mean_track_ready_coverage": sum(track_coverage) / len(track_coverage) if track_coverage else 0.0,
        "adjacent_track_pair_count": adjacent_pairs,
        "task_ready_continuity": ready_pairs / adjacent_pairs if adjacent_pairs else 0.0,
        "ready_track_episodes": episodes,
        "proxy_ttl_frames": ttl,
    }


def main() -> None:
    args = parse_args()
    if args.proxy_ttl_frames < 0:
        raise ValueError("--proxy-ttl-frames must be non-negative")
    source = read_source(args.source_root)
    no_proxy_methods = set(args.no_proxy_method)
    methods: dict[str, Any] = {}
    for item in args.method:
        name, separator, raw_path = item.partition("=")
        if not separator or not name or not raw_path:
            raise ValueError("Each --method must be NAME=ARTIFACT")
        artifact = Path(raw_path)
        ttl = 0 if name in no_proxy_methods else args.proxy_ttl_frames
        methods[name] = evaluate(source, read_current(artifact), ttl)
        methods[name]["artifact"] = str(artifact)

    result = {
        "task": "metadata_guided_stateful_tracking",
        "proxy_state": "deterministic_track_ttl_model",
        "reference": "source metadata track IDs and physical ROI groups",
        "source_root": str(args.source_root),
        "frame_count": len(source),
        "methods": methods,
        "interpretation": (
            "task_ready_recall counts current and TTL-modeled cached proxy appearance; "
            "task_ready_continuity measures consecutive source-track observations that remain task-ready."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
