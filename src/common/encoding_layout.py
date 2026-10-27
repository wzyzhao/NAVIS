"""Sequence-based artifact layout and transparent JSON indexes."""

import json
import os
import re
from pathlib import Path

SEQUENCE = re.compile(r"^(MOT17-\d+-FRCNN)")


def sequence_name(name):
    match = SEQUENCE.match(name)
    if not match:
        raise ValueError(f"Cannot identify MOT sequence: {name}")
    return match.group(1)


def dump_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")


def merge_dict(target, source):
    for key, value in source.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            merge_dict(target[key], value)
        elif key in target and target[key] != value:
            raise ValueError(f"Conflicting indexed metric: {key}")
        else:
            target[key] = value


def load_artifact_json(path):
    """Read ordinary JSON or a sequence index; candidate paths become absolute."""
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict) or data.get("format") != "artifact_index_v1":
        return data
    result = dict(data["metadata"])
    if data["kind"] == "frames":
        result["frames"] = []
    for entry in data["entries"]:
        child_path = path.parent / entry["path"]
        child = load_artifact_json(child_path)
        if data["kind"] == "frames":
            for frame in child["frames"]:
                if frame["sequence"] != entry["sequence"]:
                    raise ValueError(f"Wrong sequence in {child_path}")
                for item in frame.get("items", []):
                    if item.get("payload_path"):
                        item["payload_path"] = str((child_path.parent / item["payload_path"]).resolve())
                result["frames"].append(frame)
        elif data["kind"] == "metrics":
            merge_dict(result, child)
        else:
            raise ValueError(f"Unknown artifact index kind: {data['kind']}")
    return result


def write_frame_index(path, data, category_root, source_base=None):
    """Store frames once per sequence and write a small combined index."""
    path, category_root = Path(path), Path(category_root)
    metadata = {k: v for k, v in data.items() if k != "frames"}
    grouped = {}
    for frame in data["frames"]:
        grouped.setdefault(frame["sequence"], []).append(frame)
    entries = []
    for sequence, frames in sorted(grouped.items()):
        child_path = category_root / sequence / path.name
        if source_base is not None:
            for frame in frames:
                for item in frame.get("items", []):
                    original = Path(source_base) / item["payload_path"]
                    item["payload_path"] = Path(os.path.relpath(original.resolve(), child_path.parent)).as_posix()
        dump_json(child_path, dict(metadata, frames=frames))
        entries.append(
            {
                "sequence": sequence,
                "path": Path(os.path.relpath(child_path, path.parent)).as_posix(),
                "frame_count": len(frames),
            }
        )
    dump_json(path, {"format": "artifact_index_v1", "kind": "frames", "metadata": metadata, "entries": entries})
    return path


def write_candidate_manifest(output_root, data):
    root = Path(output_root)
    return write_frame_index(root / "payloads/candidate_manifest.json", data, root / "payloads", source_base=root)


def metric_stream_directory(root, sequence):
    """Narrow DCVC lookup to one sequence and one layer, avoiding full-tree scans."""
    layer = "roi" if "_fgroi_" in sequence else "required_background" if "_bg_keyframe_" in sequence else "background"
    return Path(root) / sequence_name(sequence) / layer


def organize_dcvc_outputs(root, discard_sidecars=False, replace_existing=False):
    """Move DCVC tool outputs into encoded/decoded; preserve source YUV separately."""
    root = Path(root).resolve()
    removed = {"files": 0, "bytes": 0}
    sources = {
        "background_yuv420": "background",
        "foreground_roi_yuv420": "roi",
        "required_background_yuv420": "required_background",
    }
    streams = {
        "background_streams/layered_background": "background",
        "background_streams/required_background": "required_background",
        "foreground_streams/layered_foreground_roi": "roi",
    }
    for folder, layer in {**sources, **streams}.items():
        origin = root / folder
        if not origin.exists():
            continue
        for file in origin.iterdir():
            if not file.is_file():
                raise ValueError(f"Unexpected nested DCVC output: {file}")
            if file.suffix == ".json" and discard_sidecars:
                if not (root / "dcvc_layered_metrics.json").is_file():
                    raise ValueError("Cannot discard sidecars without consolidated metrics")
                removed["files"] += 1
                removed["bytes"] += file.stat().st_size
                file.unlink()
                continue
            sequence = sequence_name(file.name)
            if folder in sources:
                target = root / "encoded" / sequence / "source" / layer / file.name
            else:
                category = "decoded" if file.suffix == ".yuv" else "encoded"
                target = root / category / sequence / layer / file.name
            if target.exists() and not replace_existing:
                raise FileExistsError(target)
            target.parent.mkdir(parents=True, exist_ok=True)
            file.replace(target)
        origin.rmdir()
    for folder in ("background_streams", "foreground_streams"):
        p = root / folder
        if p.exists() and not any(p.iterdir()):
            p.rmdir()
    return removed


def write_metric_index(path, category_root):
    """Split consolidated DCVC metrics without changing any metric values."""
    path, category_root = Path(path), Path(category_root)
    data = load_artifact_json(path)
    metadata = {k: v for k, v in data.items() if not isinstance(v, dict)}
    grouped = {}
    layered = data.get("format") == "dcvc_layered_metrics_v1"
    layers = {k: v for k, v in data.items() if isinstance(v, dict)}
    for layer, datasets in layers.items():
        # Required-background output has dataset -> sequence, rather than layer -> dataset.
        for dataset, sequences in datasets.items() if layered else [(layer, datasets)]:
            for name, rates in sequences.items():
                seq = sequence_name(name)
                target = grouped.setdefault(seq, {})
                if layered:
                    target.setdefault(layer, {}).setdefault(dataset, {})[name] = rates
                else:
                    target.setdefault(dataset, {})[name] = rates
    entries = []
    for seq, metrics in sorted(grouped.items()):
        child = category_root / seq / path.name
        dump_json(child, metrics)
        entries.append({"sequence": seq, "path": Path(os.path.relpath(child, path.parent)).as_posix()})
    dump_json(path, {"format": "artifact_index_v1", "kind": "metrics", "metadata": metadata, "entries": entries})
