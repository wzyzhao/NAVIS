from pathlib import Path

from nbiot_transport.logging.delivery_log import summarize_results


def format_scheduler_summary(summary: dict) -> str:
    if not summary:
        return ""
    if not summary.get("enabled"):
        return f"nbiot_scheduler: disabled, packets={summary['selected_packets']}"
    background_packets = summary.get("background_packets", 0)
    return (
        "nbiot_scheduler: "
        f"selected={summary['selected_packets']}/{summary['input_packets']}, "
        f"bytes={summary['selected_bytes']}/{summary['input_bytes']}, "
        f"metadata={summary['metadata_packets']}, roi={summary['roi_packets']}, "
        f"background={background_packets}, dropped={summary['dropped_packets']}, "
        f"rounds={summary.get('round_count', 1)}"
    )


def format_summary(summary: dict, output_path: Path) -> str:
    transport = summary.get("transport", "sim7080")
    lines = [
        f"NB-IoT sender ({transport})",
        f"confirmed_send: packets={summary['confirmed_packet_count']}/{summary['packet_count']}, "
        f"bytes={summary['confirmed_payload_bytes']}, failed={summary['failed_packet_count']}",
    ]
    lines.append(f"delivery_log: {output_path}")
    return "\n".join(lines)


def print_run_summary(run, output_path: Path) -> None:
    if not run.args.remote_host or int(run.args.remote_port) <= 0:
        print("remote endpoint not configured; only SIM7080G initialization was verified, packets were not sent")
    print(format_scheduler_summary(run.scheduler_summary))
    summary = summarize_results(run.results)
    summary["transport"] = "sim7080"
    print(format_summary(summary, output_path))
