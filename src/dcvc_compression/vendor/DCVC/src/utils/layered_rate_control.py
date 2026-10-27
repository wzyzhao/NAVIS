# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class LayeredRateController:
    """Dual-timescale rate controller for layered DCVC-RT evaluation.

    The controller keeps DCVC-RT as the background-layer neural codec and
    schedules the background at an ultra-low temporal frequency. The foreground
    layer is explicitly high-frequency and is represented by per-frame ROI
    quality control at the integration layer.
    """

    fps: int
    background_rate_idx: int
    foreground_rate_idx: int
    background_update_seconds: float = 10.0
    background_update_frames: int = 0

    def __post_init__(self):
        if self.fps <= 0:
            raise ValueError("fps must be > 0.")
        if self.background_update_seconds <= 0 and self.background_update_frames <= 0:
            raise ValueError("background_update_seconds must be > 0.")
        if self.background_update_frames < 0:
            raise ValueError("background_update_frames must be >= 0.")
        if self.background_rate_idx < 0:
            raise ValueError("background_rate_idx must be >= 0.")
        if self.foreground_rate_idx < 0:
            raise ValueError("foreground_rate_idx must be >= 0.")

    @property
    def background_update_interval_frames(self) -> int:
        if self.background_update_frames > 0:
            return int(self.background_update_frames)
        return max(1, int(round(self.background_update_seconds * self.fps)))

    def background_source_index(self, frame_idx: int) -> int:
        interval = self.background_update_interval_frames
        return (frame_idx // interval) * interval

    def build_rate_plan(self) -> List[Dict]:
        return [
            {
                "rate_idx": 0,
                "background_rate_idx": int(self.background_rate_idx),
                "foreground_rate_idx": int(self.foreground_rate_idx),
                "background_update_seconds": float(self.background_update_seconds),
                "background_update_interval_frames": self.background_update_interval_frames,
            }
        ]

    def describe(self) -> Dict:
        return {
            "controller": "dcvc_rt_layered_rate_control",
            "background_timescale": "low_frequency_seconds",
            "background_update_seconds": float(self.background_update_seconds),
            "background_update_interval_frames": self.background_update_interval_frames,
            "foreground_timescale": "per_frame_high_frequency",
            "foreground_update_interval_frames": 1,
            "background_rate_idx": int(self.background_rate_idx),
            "foreground_rate_idx": int(self.foreground_rate_idx),
        }
