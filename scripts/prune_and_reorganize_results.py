"""Compact completed experiments and migrate encoding artifacts by sequence."""

import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from common.encoding_layout import (
    dump_json,
    load_artifact_json,
    organize_dcvc_outputs,
    write_frame_index,
    write_metric_index,
)

RESULTS = ROOT / "experiments/results"
DCVC = RESULTS / "encoding/dcvc"
ELIC = RESULTS / "encoding/elic"
REPORT = {"deleted": [], "moved": [], "validation": {}}


def safe(path):
    if not path.resolve().is_relative_to(ROOT / "experiments"):
        raise ValueError(f"Operation outside experiments: {path}")


def remove(path, reason):
    safe(path)
    if not path.exists():
        return
    if path.is_dir():
        files = [p for p in path.rglob("*") if p.is_file()]
        info = {"files": len(files), "bytes": sum(p.stat().st_size for p in files)}
        shutil.rmtree(path)
    else:
        info = {"files": 1, "bytes": path.stat().st_size}
        path.unlink()
    REPORT["deleted"].append({"path": path.relative_to(ROOT).as_posix(), "reason": reason, **info})


def move(old, new):
    safe(old)
    safe(new)
    if new.exists():
        raise FileExistsError(new)
    new.parent.mkdir(parents=True, exist_ok=True)
    old.rename(new)
    REPORT["moved"].append({"from": old.relative_to(ROOT).as_posix(), "to": new.relative_to(ROOT).as_posix()})


def mapped_payload(old):
    old = old.resolve()
    if old.is_relative_to(DCVC / "payloads"):
        rel = old.relative_to(DCVC / "payloads")
        if rel.parts[0] in {"background", "metadata", "roi"}:
            return DCVC / "payloads" / rel.parts[1] / rel.parts[0] / Path(*rel.parts[2:])
    if old.is_relative_to(ELIC):
        rel = old.relative_to(ELIC)
        if rel.parts[0].startswith("mot17_"):
            seq = "MOT17-" + rel.parts[0][-2:] + "-FRCNN"
            tail = rel.parts[2:]
            if tail[0] == "payloads":
                tail = tail[1:]
            return ELIC / "payloads" / seq / rel.parts[1] / Path(*tail)
    return old


def binary_inventory(root):
    files = list(root.rglob("*.bin"))
    return {"files": len(files), "bytes": sum(p.stat().st_size for p in files)}


def main():
    if (RESULTS / "cleanup_report.json").exists():
        raise ValueError("This one-time migration already ran")
    original = DCVC / "candidate_manifest.json"
    duplicate = DCVC / "candidate_manifest_seven_mot17.json"
    old_source_hash = hashlib.sha256(original.read_bytes()).hexdigest()
    if hashlib.sha256(duplicate.read_bytes()).hexdigest() != old_source_hash:
        raise ValueError("DCVC candidate manifests differ; do not deduplicate")
    candidates = json.loads(original.read_text(encoding="utf-8"))
    before = {"dcvc": binary_inventory(DCVC / "payloads"), "elic": {"files": 0, "bytes": 0}}
    baseline_data = []
    for manifest in sorted(ELIC.glob("mot*/run01/candidate_manifest.json")):
        data = json.loads(manifest.read_text(encoding="utf-8"))
        for frame in data["frames"]:
            for item in frame["items"]:
                payload = (manifest.parent / item["payload_path"]).resolve()
                if not payload.is_file() or payload.stat().st_size != item["bytes"]:
                    raise ValueError(f"Invalid original ELIC payload: {payload}")
        inv = binary_inventory(manifest.parent / "payloads")
        for key in inv:
            before["elic"][key] += inv[key]
        baseline_data.append((manifest, data))
    if len(baseline_data) != 7:
        raise ValueError("Expected seven ELIC source manifests")
    for frame in candidates["frames"]:
        for item in frame["items"]:
            payload = (DCVC / item["payload_path"]).resolve()
            if not payload.is_file() or payload.stat().st_size != item["bytes"]:
                raise ValueError(f"Invalid original DCVC payload: {payload}")
            item["payload_path"] = str(mapped_payload(payload))
    # Move payload layer/sequence folders to sequence/layer.
    for layer in ("background", "metadata", "roi"):
        folder = DCVC / "payloads" / layer
        for seq_dir in sorted(folder.iterdir()):
            move(seq_dir, DCVC / "payloads" / seq_dir.name / layer)
        folder.rmdir()
    write_frame_index(DCVC / "payloads/candidate_manifest.json", candidates, DCVC / "payloads", source_base=DCVC)
    REPORT["dcvc_source_sha256_before"] = old_source_hash
    remove(duplicate, "Exact duplicate of the canonical DCVC candidate manifest")
    remove(original, "Replaced by one manifest per sequence and a combined index")
    # ELIC flat payloads and manifest now share a sequence/run directory.
    for old_manifest, data in baseline_data:
        new_parent = ELIC / "payloads" / data["sequence"] / "run01"
        for frame in data["frames"]:
            for item in frame["items"]:
                target = mapped_payload(old_manifest.parent / item["payload_path"])
                item["payload_path"] = Path(os.path.relpath(target, new_parent)).as_posix()
        source = DCVC / "payloads" / data["sequence"] / "candidate_manifest.json"
        data["source_manifest_provenance_sha256"] = data["source_manifest_sha256"]
        data["source_manifest_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
        data["source_manifest"] = Path(os.path.relpath(source, new_parent)).as_posix()
        move(old_manifest.parent / "payloads", new_parent)
        move(old_manifest, new_parent / "candidate_manifest.json")
        dump_json(new_parent / "candidate_manifest.json", data)
    for folder in (ELIC / "decoded_frames").iterdir():
        move(folder, ELIC / "decoded" / folder.name / "frames")
    (ELIC / "decoded_frames").rmdir()
    for video in (ELIC / "videos").iterdir():
        move(video, ELIC / "decoded" / video.stem / "reconstruction.mp4")
    (ELIC / "videos").rmdir()
    # Final consolidated metrics retain the values used by quality analysis.
    sidecars = organize_dcvc_outputs(DCVC, discard_sidecars=True)
    REPORT["deleted"].append(
        {
            "path": "experiments/results/encoding/dcvc/*_streams/**/*.json",
            "reason": "Intermediate per-stream tool JSON; consolidated final metrics retained",
            **sidecars,
        }
    )
    for name in ("dcvc_layered_metrics.json", "required_background_dcvc_output.json"):
        write_metric_index(DCVC / name, DCVC / "encoded")
    for codec in (DCVC, ELIC):
        path = codec / "frame_metrics.json"
        data = load_artifact_json(path)
        for row in data["frames"]:
            value = row.get("reconstructed_frame")
            if value and "decoded_frames" in value:
                seq = row["sequence"]
                row["reconstructed_frame"] = str(codec / "decoded" / seq / "frames" / Path(value).name)
        write_frame_index(path, data, codec / "decoded")
    # Completed empty queues, diagnostic text, and superseded migration copies.
    for path in sorted((ROOT / "experiments").rglob("*.log")):
        remove(path, "Installation/codec fallback/progress log; final results retained")
    for path in sorted(RESULTS.rglob("missing_fragment_queue.json")):
        d = json.loads(path.read_text(encoding="utf-8"))
        if not d.get("missing_fragments"):
            remove(path, "Completed run has an empty retransmission queue")
    remove(ROOT / "experiments/layout_backup_20260929", "Superseded temporary text backup from previous layout")
    for name in ("layout_migration.json", "layout_validation.json"):
        remove(RESULTS / name, "Superseded by this cleanup report and validation")
    for folder in sorted(ELIC.glob("mot17_*")):
        for child in sorted(folder.rglob("*"), key=lambda p: -len(p.parts)):
            if child.is_dir():
                child.rmdir()
        folder.rmdir()
    REPORT["validation"]["binary_inventory_before"] = before
    after = {"dcvc": binary_inventory(DCVC / "payloads"), "elic": binary_inventory(ELIC / "payloads")}
    if before != after:
        raise ValueError(f"Binary inventory changed: {before} -> {after}")
    REPORT["validation"]["binary_inventory_after"] = after
    dump_json(RESULTS / "cleanup_report.json", REPORT)
    print(
        json.dumps(
            {
                "deleted_files": sum(r["files"] for r in REPORT["deleted"]),
                "deleted_bytes": sum(r["bytes"] for r in REPORT["deleted"]),
                "binary_inventory": after,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
