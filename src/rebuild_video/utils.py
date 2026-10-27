from __future__ import annotations

import json
import shutil
from pathlib import Path


def clear_rebuild_outputs(root: Path) -> None:
    for name in ("reconstructed_frames", "reconstructed_videos", "decoded_layers"):
        path = root / name
        if path.is_dir():
            shutil.rmtree(path)
    for name in (
        "reconstruction_status.jsonl",
        "reconstruction_errors.jsonl",
        "reconstruction_quality_report.json",
        "dcvc_decoder_stderr.log",
    ):
        path = root / name
        if path.exists():
            path.unlink()


def load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def load_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8-sig"))


def frame_set(rows: list[dict], layer: str | None = None) -> set[int]:
    values = set()
    for row in rows:
        if layer is not None and row.get("layer") != layer:
            continue
        frame_idx = row.get("frame_idx")
        if frame_idx is not None:
            values.add(int(frame_idx))
    return values


def resolve_payload_path(root: Path, record: dict) -> Path | None:
    value = record.get("payload_path")
    if not value:
        return None
    path = Path(str(value))
    if path.is_file():
        return path
    fallback = root / "reassembled_payloads" / path.name
    return fallback if fallback.is_file() else None


def schedule_path_from_config(config: dict) -> Path | None:
    value = config.get("recovery_schedule_json") or config.get("schedule_json")
    if not value:
        return None
    path = Path(value)
    if path.is_file():
        return path
    config_path = Path(config.get("_config_path", ""))
    if config_path:
        candidate = config_path.parent / path
        if candidate.is_file():
            return candidate
    return path
