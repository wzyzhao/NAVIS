"""Aggregate all seven MOT17 scheduler summaries with equal sequence weights."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

SEQUENCES = tuple(f"MOT17-{number}-FRCNN" for number in ("02", "04", "05", "09", "10", "11", "13"))
VARIANTS = ("navis_ack", "no_proxy", "single_q_roi")
METRICS = (
    "selected_bytes",
    "metadata_completeness",
    "current_roi_ratio",
    "proxy_roi_ratio",
    "roi_availability",
    "roi_transmission_rate",
    "roi_availability_rate",
    "roi_per_selected_mb",
    "avg_selected_quality_rank",
    "avg_quota_bytes",
)


def aggregate(summaries: list[dict]) -> dict:
    by_sequence = {summary["sequence"]: summary for summary in summaries}
    if len(summaries) != 7 or set(by_sequence) != set(SEQUENCES):
        raise ValueError("Exactly one summary for each of the seven MOT17 sequences is required")
    macro = []
    for variant in VARIANTS:
        rows = []
        for sequence in SEQUENCES:
            matches = [row for row in by_sequence[sequence]["results"] if row["variant"] == variant]
            if len(matches) != 1:
                raise ValueError(f"Expected exactly one {variant} result for {sequence}")
            rows.append(matches[0])
        result = {"variant": variant}
        for metric in METRICS:
            values = [row.get(metric) for row in rows]
            # Missing measurements invalidate the seven-sequence metric.
            result[metric] = (
                statistics.fmean(values)
                if all(isinstance(value, (int, float)) and math.isfinite(value) for value in values)
                else None
            )
        macro.append(result)
    return {
        "aggregation": "macro: arithmetic mean of seven per-sequence metrics",
        "sequence_count": 7,
        "model": "Offline candidate admission and TTL state model; no ACK or physical delivery",
        "per_sequence": [by_sequence[sequence] for sequence in SEQUENCES],
        "macro": macro,
    }


def write_report(path: Path, output: dict) -> None:
    lines = [
        "# 七序列调度消融（macro）",
        "",
        "七序列等权；比例先在序列内计算，再取七序列算术平均。载荷为每序列平均 MB。",
        "离线候选录取和 TTL 状态模型，不代表真实交付或视觉重建质量。",
        "",
        "| 变体 | 平均载荷 MB | 当前 ROI 覆盖 | TTL 可用率 | metadata 完整率 | 可用 ROI/MB |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    def display(value: float | None, style: str, scale: float = 1.0) -> str:
        return "NA" if value is None else format(value / scale, style)

    for row in output["macro"]:
        lines.append(
            f"| {row['variant']} | {display(row['selected_bytes'], '.4f', 1_000_000)} | "
            f"{display(row['current_roi_ratio'], '.2%')} | {display(row['roi_availability'], '.2%')} | "
            f"{display(row['metadata_completeness'], '.2%')} | {display(row['roi_per_selected_mb'], '.2f')} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1] / "results/ablations/scheduler"
    )
    args = parser.parse_args()
    summaries = [
        json.loads((args.root / f"mot17-{sequence[6:8]}" / "summary.json").read_text(encoding="utf-8"))
        for sequence in SEQUENCES
    ]
    output = aggregate(summaries)
    (args.root / "summary.json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_report(args.root / "report.md", output)


if __name__ == "__main__":
    main()
