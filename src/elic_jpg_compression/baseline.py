"""Export, send, and reconstruct the Static ELIC+JPEG transport baseline."""

from __future__ import annotations

import json
import pickle
from hashlib import sha256
from pathlib import Path

from common.encoding_layout import load_artifact_json
from common.paths import PROJECT_ROOT as ROOT
from nbiot_transport.core.elic_jpeg_payload import decode, encode

EXPORT = ROOT / "experiments/results/encoding/elic"
CONFIG = ROOT / "src/nbiot_transport/elic.json"


def _configured_path(settings, key):
    value = settings.get(key)
    if not value:
        raise ValueError(f"{key} is required in {CONFIG}")
    path = Path(value)
    return path if path.is_absolute() else (CONFIG.parent / path).resolve()


def _source_fingerprint(source_path):
    return sha256(source_path.read_bytes()).hexdigest()


def export(settings, output=None):
    output = output or _configured_path(settings, "candidate_manifest_json").parent
    source_path = _configured_path(settings, "candidate_source_manifest")
    source = load_artifact_json(source_path)
    seq = str(settings["candidate_source_sequence"])
    jpeg_quality = int(settings["static_roi_jpeg_quality"])
    frames = [f for f in source["frames"] if f["sequence"] == seq]
    if not frames:
        raise ValueError(f"No frames for {seq} in {source_path}")
    expected_fingerprint = settings.get("candidate_source_manifest_sha256")
    fingerprint = _source_fingerprint(source_path)
    if expected_fingerprint and fingerprint.lower() != str(expected_fingerprint).lower():
        raise ValueError("candidate_source_manifest_sha256 does not match the configured source manifest")
    encoded_root = (
        _configured_path(settings, "elic_encoded_root")
        if settings.get("elic_encoded_root")
        else ROOT / "experiments/results/encoding/elic"
    )
    encoded = encoded_root / "encoded" / seq
    output.mkdir(parents=True, exist_ok=True)
    output.mkdir(exist_ok=True)
    result = []
    count = 0
    repaired = 0
    for frame in frames:
        fid = int(frame.get("frame_id", frame["frame_idx"]))
        # Existing local experiment pickles are trusted input; not sent as pickle.
        with (encoded / f"{fid:06d}" / "foreground.pkl").open("rb") as file:
            existing = pickle.load(file)
        lookup = {(int(r["track_id"]), tuple(r["roi_bbox"])): r for r in existing}
        items = []
        seen = set()
        bundle = None
        for original in frame["items"]:
            item = dict(original)
            layer = item["layer"]
            if layer == "metadata":
                path = Path(item["payload_path"])
                item["payload_path"] = str(path if path.is_absolute() else (source_path.parent / path).resolve())
                items.append(item)
                continue
            if layer == "background_keyframe_summary":
                with (encoded / f"{fid:06d}" / "background.pkl").open("rb") as file:
                    data = pickle.load(file)
                data.update(sequence=seq, frame_idx=frame["frame_idx"], background=item.get("background", {}))
                payload = encode("elic_background", data)
                name = f"{fid:06d}_background.bin"
            elif layer == "roi_patch":
                group = str(item["candidate_group_id"])
                if group in seen:
                    continue
                seen.add(group)
                bbox = item.get("roi_bbox") or item.get("bbox")
                key = (int(item["track_id"]), tuple(bbox))
                roi = lookup.get(key)
                if roi is None:
                    from elic_jpg_compression.elic_jpg_codec.bundle_io import load_bundle_frame
                    from elic_jpg_compression.elic_jpg_codec.elic_pipeline import serialize_rois

                    if bundle is None:
                        directory = ROOT / "src" / "scene_factorization" / "scene_factorization_output" / seq
                        frame_dir = directory / f"{fid:06d}"
                        if not (frame_dir / "metadata.json").exists():
                            raise ValueError(f"Missing source frame {fid}")
                        bundle = load_bundle_frame(frame_dir, 12)
                    choices = [r for r in bundle.selected_rois if r.track_id == key[0] and tuple(r.roi_bbox) == key[1]]
                    if len(choices) != 1:
                        raise ValueError(f"ROI set mismatch: {fid} {key}")
                    roi = serialize_rois(choices, jpeg_quality)[0]
                    repaired += 1
                data = dict(
                    roi, sequence=seq, frame_idx=frame["frame_idx"], roi_idx=item["roi_idx"], candidate_group_id=group
                )
                payload = encode("jpeg_roi", data)
                name = f"{fid:06d}_roi{int(item['roi_idx']):03d}.bin"
                count += 1
            else:
                continue
            (output / name).write_bytes(payload)
            item.update(
                payload_path=name,
                bytes=len(payload),
                rate_idx=0,
                quality_rank=0,
                codec="elic_background" if layer == "background_keyframe_summary" else "jpeg_roi",
            )
            items.append(item)
        result.append(dict(frame, rate_idx=0, items=items))
        if len(result) % 100 == 0:
            print(f"Exported {len(result)}/{len(frames)} frames", flush=True)
    manifest = {
        "scheme": "static_elic_jpeg",
        "sequence": seq,
        "jpeg_quality": jpeg_quality,
        "source_manifest": str(source_path),
        "source_manifest_sha256": fingerprint,
        "source_manifest_sequence": seq,
        "metadata_background_policy": "copied_from_source_manifest",
        "roi_selection": "one_static_jpeg_candidate_per_source_candidate_group",
        "frames": result,
        "roi_count": count,
        "reencoded_missing_rois": repaired,
    }
    (output / "candidate_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "frames": len(result),
                "physical_rois": count,
                "reencoded_missing_rois": repaired,
                "manifest": str(output / "candidate_manifest.json"),
            }
        )
    )
    return manifest


def validate_export(settings):
    """Reject a send when its static candidates came from another source revision."""
    manifest_path = _configured_path(settings, "candidate_manifest_json")
    if not manifest_path.is_file():
        raise ValueError(f"Missing ELIC candidate manifest: {manifest_path}; set baseline_action to export first")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_path = _configured_path(settings, "candidate_source_manifest")
    required = {
        "scheme": "static_elic_jpeg",
        "sequence": str(settings["candidate_source_sequence"]),
        "jpeg_quality": int(settings["static_roi_jpeg_quality"]),
        "source_manifest_sha256": _source_fingerprint(source_path),
    }
    mismatch = {key: (manifest.get(key), value) for key, value in required.items() if manifest.get(key) != value}
    if mismatch:
        raise ValueError(
            f"ELIC candidates are stale for this configuration: {mismatch}; set baseline_action to export first"
        )
    return manifest


def rebuild(receiver_root, output, checkpoint=None):
    """Reconstruct only CRC-verified received payloads. No ROI proxy or sender files."""
    import cv2
    import numpy as np

    from elic_jpg_compression.elic_jpg_codec.elic_pipeline import ELICBackgroundCodec, decode_jpeg, decode_png
    from nbiot_transport.core.metadata_bundle import expand_metadata_bundle
    from rebuild_video.side_info_codec import decode_metadata

    checkpoint = checkpoint or ROOT / "src/elic_jpg_compression/models/elic/ELIC_0016_ft_3980_Plateau.pth.tar"
    records = [
        json.loads(line)
        for line in (receiver_root / "received_payload_manifest.jsonl").read_text().splitlines()
        if line.strip()
    ]
    states = {}
    codec = None
    for record in records:
        if not record.get("ack_confirmed") or not record.get("ack_complete"):
            continue
        path = Path(record["payload_path"])
        if not path.is_absolute():
            path = receiver_root / path
        if not path.is_file():
            path = receiver_root / "reassembled_payloads" / path.name
        raw = path.read_bytes()
        import zlib

        if (zlib.crc32(raw) & 0xFFFFFFFF) != record["payload_crc32"]:
            raise ValueError("Received payload CRC mismatch")
        device = str(record["device_id"])
        state = states.setdefault(device, dict(metadata={}, backgrounds={}, rois={}))
        if record["layer"] == "metadata":
            meta = decode_metadata(raw)
            for frame in expand_metadata_bundle(meta) or [meta]:
                state["metadata"][int(frame["frame_idx"])] = frame
        elif raw.startswith(b"NEJ1"):
            kind, data = decode(raw)
            frame = int(data["frame_idx"])
            if kind == "elic_background":
                if codec is None:
                    codec = ELICBackgroundCodec(str(checkpoint), device="cpu")
                state["backgrounds"][frame] = codec.decompress(data)
            else:
                state["rois"].setdefault(frame, []).append(data)
    output.mkdir(parents=True, exist_ok=True)
    rendered = []
    expected = current = missing_bg = 0
    for device, state in states.items():
        frames_dir = output / "frames" / device
        frames_dir.mkdir(parents=True, exist_ok=True)
        background = None
        writer = None
        for frame, meta in sorted(state["metadata"].items()):
            usable = [f for f in state["backgrounds"] if f <= frame]
            if usable:
                background = state["backgrounds"][max(usable)]
            expected += len(meta.get("rois", []))
            if background is None:
                missing_bg += 1
                continue
            image = background.copy()
            rendered_rois = []
            for roi in state["rois"].get(frame, []):
                patch = decode_jpeg(roi["image_jpeg"])
                mask = decode_png(roi["mask_png"])
                x1, y1, x2, y2 = map(int, roi["roi_bbox"])
                if not (0 <= x1 < x2 <= image.shape[1] and 0 <= y1 < y2 <= image.shape[0]):
                    raise ValueError("ROI bbox out of bounds")
                if patch.shape[:2] != (y2 - y1, x2 - x1) or mask.shape[:2] != patch.shape[:2]:
                    raise ValueError("ROI dimensions mismatch")
                image[y1:y2, x1:x2] = np.where((mask > 0)[..., None], patch, image[y1:y2, x1:x2])
                rendered_rois.append(
                    {k: roi[k] for k in ["roi_idx", "track_id", "class_id", "roi_bbox", "candidate_group_id"]}
                )
            current += len(rendered_rois)
            path = frames_dir / f"{frame:06d}.png"
            if not cv2.imwrite(str(path), image):
                raise OSError(path)
            if writer is None:
                writer = cv2.VideoWriter(
                    str(output / f"{device}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30, (image.shape[1], image.shape[0])
                )
                if not writer.isOpened():
                    raise OSError("Cannot create reconstruction video")
            writer.write(image)
            rendered.append(
                dict(
                    device_id=device,
                    frame_idx=frame,
                    sequence=meta.get("sequence"),
                    image_path=str(path),
                    rois=rendered_rois,
                    proxy_roi_count=0,
                )
            )
        if writer is not None:
            writer.release()
    (output / "reconstruction_manifest.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rendered), encoding="utf-8"
    )
    summary = dict(
        scheme="static_elic_jpeg",
        metadata_frames=sum(len(s["metadata"]) for s in states.values()),
        rendered_frames=len(rendered),
        expected_roi_on_received_metadata=expected,
        current_roi=current,
        proxy_roi=0,
        frames_missing_background=missing_bg,
        roi_availability=current / max(1, expected),
        source="received payloads only",
    )
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary))
    return summary


def main():
    settings = json.loads(CONFIG.read_text(encoding="utf-8-sig"))
    action = settings.get("baseline_action", "send")
    if action == "export":
        export(settings)
    elif action == "send":
        validate_export(settings)
        from nbiot_transport.runtime.sender import run

        run(CONFIG)
    elif action == "rebuild":
        rebuild(
            _configured_path(settings, "baseline_receiver_root"),
            _configured_path(settings, "baseline_rebuild_output_root"),
        )
    else:
        raise ValueError(f"Unknown baseline_action: {action!r}; use export, send or rebuild in {CONFIG}")


if __name__ == "__main__":
    main()
