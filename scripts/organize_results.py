"""One-time results layout migration; dry run by default, --apply to migrate."""

import argparse
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments/results"
MOVES = {
    "elic": "transmission/elic",
    "navis": "transmission/navis",
    "multinode": "transmission/multinode",
    "encoding/elic/mot17_04/payloads": "encoding/elic/mot17_04/run01/payloads",
    "encoding/elic/candidate_manifest.json": "encoding/elic/mot17_04/run01/candidate_manifest.json",
}


def relocated(path):
    for old, new in sorted(MOVES.items(), key=lambda pair: -len(pair[0])):
        source = RESULTS / old
        if path == source or source in path.parents:
            return RESULTS / new / path.relative_to(source)
    return path


def update_references(raw):
    for sep in ("/", "\\", "\\\\"):
        for old, new in MOVES.items():
            before = "experiments/results/" + old
            after = "experiments/results/" + new
            raw = raw.replace(before.replace("/", sep), after.replace("/", sep))
        # Derived indexes used the previous workspace's absolute path.
        before = "D:/video_compression/experiments/results/".replace("/", sep)
        after = (ROOT.as_posix() + "/experiments/results/").replace("/", sep)
        raw = raw.replace(before, after)
    return raw


def inventory(path):
    files = [p for p in path.rglob("*") if p.is_file()] if path.is_dir() else [path]
    return {"files": len(files), "bytes": sum(p.stat().st_size for p in files)}


def verify_candidates():
    rows = []
    for manifest in sorted((RESULTS / "encoding/elic").glob("mot*/run01/candidate_manifest.json")):
        data = json.loads(manifest.read_text(encoding="utf-8"))
        count = 0
        for frame in data["frames"]:
            for item in frame["items"]:
                payload = manifest.parent / item["payload_path"]
                if not payload.is_file() or payload.stat().st_size != item["bytes"]:
                    raise ValueError(f"Missing payload or byte mismatch: {payload}")
                if item["layer"] != "metadata":
                    with payload.open("rb") as stream:
                        if stream.read(4) != b"NEJ1":
                            raise ValueError(f"Invalid ELIC/JPEG envelope: {payload}")
                count += 1
        source = manifest.parent / data["source_manifest"]
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        if digest.lower() != data["source_manifest_sha256"].lower():
            raise ValueError(f"Source fingerprint mismatch: {manifest}")
        rows.append({"sequence": data["sequence"], "frames": len(data["frames"]), "items": count})
    if len(rows) != 7:
        raise ValueError(f"Expected seven candidate manifests, got {len(rows)}")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    for old, new in MOVES.items():
        source, target = RESULTS / old, RESULTS / new
        if not source.exists() or target.exists():
            raise ValueError(f"Migration requires source present and target absent: {source} -> {target}")
        if not source.resolve().is_relative_to(ROOT) or not target.resolve().is_relative_to(ROOT):
            raise ValueError("Move must stay inside workspace")
    old_manifest = RESULTS / "encoding/elic/candidate_manifest.json"
    data = json.loads(old_manifest.read_text(encoding="utf-8"))
    new_parent = relocated(old_manifest).parent
    for frame in data["frames"]:
        for item in frame["items"]:
            source = (old_manifest.parent / item["payload_path"]).resolve()
            if not source.is_file() or source.stat().st_size != item["bytes"]:
                raise ValueError(f"Pre-migration candidate mismatch: {source}")
            item["payload_path"] = Path(os.path.relpath(relocated(source), new_parent)).as_posix()
    source = (old_manifest.parent / data["source_manifest"]).resolve()
    data["source_manifest"] = Path(os.path.relpath(source, new_parent)).as_posix()
    plan = [{"from": old, "to": new, **inventory(RESULTS / old)} for old, new in MOVES.items()]
    if not args.apply:
        print(json.dumps(plan, indent=2))
        return
    # Save exact text originals for reversibility; preserve historical sender logs/snapshots.
    backup = ROOT / "experiments/layout_backup_20260929"
    if backup.exists():
        raise ValueError(f"Backup already exists: {backup}")
    backup.mkdir()
    edits = []
    for folder in ("src", "experiments", "docs", "tests", "deploy", "scripts"):
        for path in (ROOT / folder).rglob("*"):
            if not path.is_file() or path.suffix not in {".py", ".json", ".jsonl", ".md"}:
                continue
            if "vendor" in path.parts or backup in path.parents or path == Path(__file__).resolve():
                continue
            if RESULTS in path.parents:
                derived = path.name in {"reconstruction_manifest.jsonl", "received_payload_manifest.jsonl"}
                evaluation = RESULTS / "evaluation" in path.parents
                if path.suffix != ".md" and not derived and not evaluation:
                    continue
            original = path.read_bytes()
            try:
                raw = original.decode("utf-8")
            except UnicodeDecodeError:
                continue
            updated = update_references(raw)
            if updated != raw:
                saved = backup / path.relative_to(ROOT)
                saved.parent.mkdir(parents=True, exist_ok=True)
                saved.write_bytes(original)
                path.write_bytes(updated.encode("utf-8"))
                edits.append(path.relative_to(ROOT).as_posix())
    saved = backup / old_manifest.relative_to(ROOT)
    saved.parent.mkdir(parents=True, exist_ok=True)
    saved.write_bytes(old_manifest.read_bytes())
    before_move = {old: inventory(RESULTS / old) for old in MOVES}
    for old, new in MOVES.items():
        target = RESULTS / new
        target.parent.mkdir(parents=True, exist_ok=True)
        (RESULTS / old).rename(target)
    relocated(old_manifest).write_text(json.dumps(data, indent=2), encoding="utf-8")
    for row in plan:
        if row["from"].endswith("candidate_manifest.json"):
            continue
        if inventory(RESULTS / row["to"]) != before_move[row["from"]]:
            raise ValueError(f"Move inventory mismatch: {row}")
    report = {"moves": plan, "updated_references": edits, "text_backup": backup.relative_to(ROOT).as_posix(),
              "candidate_validation": verify_candidates(),
              "historical_logs": "Sender logs and config snapshots retain original provenance paths."}
    (RESULTS / "layout_migration.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
