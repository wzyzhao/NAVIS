def expand_metadata_bundle(metadata: dict) -> list[dict]:
    """Expand compact_v4 metadata in wire order, skipping malformed frame/ROI rows."""
    if not bool(metadata.get("metadata_bundle")):
        return []
    if metadata.get("bundle_format") != "compact_v4" or not isinstance(metadata.get("frames_compact"), list):
        return []
    frame_start = int(metadata.get("frame_start", 0) or 0)
    common = metadata.get("common") if isinstance(metadata.get("common"), dict) else {}
    frames = []
    for item in metadata["frames_compact"]:
        if not isinstance(item, list) or len(item) < 6:
            continue
        frame_idx = frame_start + int(item[0])
        frame_id = int(item[1])
        compact_rois = item[5]
        rois = []
        for roi in compact_rois:
            if not isinstance(roi, list) or len(roi) < 6:
                continue
            bbox = [int(value) for value in (roi[2] or [0, 0, 0, 0])[:4]]
            decoded_roi = {
                "roi_idx": int(roi[0]),
                "track_id": int(roi[1]),
                "bbox": bbox,
                "roi_bbox": bbox,
                "class_id": int(roi[3]),
                "motion_score": float(roi[4] or 0.0),
                "uncertainty": float(roi[5] or 0.0),
            }
            if len(roi) >= 7 and isinstance(roi[6], dict):
                decoded_roi["mask_rle"] = roi[6]
            rois.append(decoded_roi)
        primary = rois[0] if rois else {}
        frames.append(
            {
                "device_id": common.get("device_id"),
                "sequence": common.get("sequence"),
                "timestamp": float(frame_idx),
                "frame_idx": frame_idx,
                "frame_id": frame_id,
                "track_id": primary.get("track_id", -1),
                "bbox": primary.get("bbox", [0, 0, 0, 0]),
                "motion_score": float(item[2] or 0.0),
                "background_change_score": float(item[3] or 0.0),
                "uncertainty": float(item[4] or 0.0),
                "background_version": common.get("background_version"),
                "foreground_count": len(rois),
                "rois": rois,
            }
        )
    return frames
