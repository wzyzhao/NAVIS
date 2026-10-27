"""Strict seven-sequence means used by experiment and codec reports."""

import math
import statistics

MOT17_SEQUENCES = tuple(f"MOT17-{number}-FRCNN" for number in ("02", "04", "05", "09", "10", "11", "13"))


def seven_sequence_mean(rows: list[dict], metric: str) -> float | None:
    if len(rows) != 7 or {row.get("sequence") for row in rows} != set(MOT17_SEQUENCES):
        return None
    values = [row.get(metric) for row in rows]
    if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in values):
        return None
    return statistics.fmean(float(value) for value in values if isinstance(value, (int, float)))
