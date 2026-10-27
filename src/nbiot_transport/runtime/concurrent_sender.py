"""One coordinator, independent modem workers, and a barrier between rounds."""

from __future__ import annotations

import json
import os
import threading
import time
from argparse import Namespace
from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor, wait
from copy import copy
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Protocol

from nbiot_transport.core.packetizer import Packet, build_session_start_packet
from nbiot_transport.core.scheduler import schedule_with_mandatory_reservation
from nbiot_transport.feedback.online_link_controller import (
    OnlineLinkController,
    smooth_link_state,
    summarize_round_results,
)
from nbiot_transport.runtime import sender
from nbiot_transport.runtime.config_loader import validate_config

if TYPE_CHECKING:
    from nbiot_transport.transport.sim7080_transport import SendResult


class PacketTransport(Protocol):
    """Minimal worker interface; optional recovery hooks are detected at runtime."""

    def send_packets(self, packets: Iterable[Packet]) -> list[SendResult]: ...


type NodeResults = dict[str, tuple[list[SendResult], float]]
type NodeErrors = dict[str, str]
type EventLogger = Callable[..., None]


def validate_concurrent(args: Namespace | SimpleNamespace) -> None:
    """Require distinct modem ports and isolated receiver sessions before workers start."""
    validate_config(args)
    if args.source != "multi_schedule" or len(args.nodes) < 2:
        raise ValueError("Concurrent transmission requires at least two nodes in multi_schedule")
    ports = [str(n.get("serial_port", "")).strip().upper() for n in args.nodes]
    if any(not p for p in ports) or len(set(ports)) != len(ports):
        raise ValueError("Each node requires its own distinct modem serial_port")
    if not getattr(args, "receiver_session_isolation_confirmed", False):
        raise ValueError("Deploy receiver isolate_sessions=true, then set receiver_session_isolation_confirmed=true")
    if getattr(args, "missing_fragment_queue", []):
        raise ValueError("Use a fresh concurrent run output; stale-session retries are not admitted")


def send_parallel(
    pool: ThreadPoolExecutor,
    transports: Mapping[str, PacketTransport],
    groups: Mapping[str, list[Packet]],
    timeout_seconds: float | None = None,
    event_logger: EventLogger | None = None,
    stage: str = "round",
    round_index: int | None = None,
) -> tuple[NodeResults, NodeErrors, float]:
    """Send one group per modem, with a bounded wall-clock wait per node round."""

    def log(device: str, event: str, **detail: object) -> None:
        if event_logger:
            event_logger(device, event, stage=stage, round_index=round_index, **detail)

    def send(device: str, packets: list[Packet]) -> tuple[list[SendResult], float]:
        started = time.perf_counter()
        log(
            device,
            "send_started",
            packet_count=len(packets),
            selected_bytes=sum(getattr(p, "bytes", 0) for p in packets),
        )
        try:
            results = transports[device].send_packets(packets)
        except Exception as exc:
            log(device, "send_failed", elapsed_seconds=time.perf_counter() - started, error=repr(exc))
            raise
        elapsed = time.perf_counter() - started
        log(device, "send_completed", elapsed_seconds=elapsed, result_count=len(results))
        return results, elapsed

    started = time.perf_counter()
    futures = {d: pool.submit(send, d, p) for d, p in groups.items() if p}
    completed = {}
    errors = {}
    timeout = float(timeout_seconds) if timeout_seconds else None
    _, pending = wait(futures.values(), timeout=timeout)
    timed_out = {device for device, future in futures.items() if future in pending}
    for device in timed_out:
        error = f"node round timeout after {timeout:.1f}s"
        errors[device] = error
        log(device, "node_round_timeout", timeout_seconds=timeout, error=error)
        abort = getattr(transports[device], "abort_current_operation", None)
        if callable(abort):
            try:
                abort(error)
                log(device, "abort_requested", reason=error)
            except Exception as exc:
                log(device, "abort_request_failed", error=repr(exc))
    if pending:
        # SIM7080Transport interrupts its serial read/write in abort_current_operation.
        # A short grace period lets that worker return before socket recovery starts.
        wait(pending, timeout=10.0)
    for device, future in futures.items():
        if device in timed_out and not future.done():
            errors[device] = f"{errors[device]}; worker did not stop after abort request"
            log(device, "worker_still_running", error=errors[device])
            continue
        try:
            item = future.result()
            if device not in timed_out:
                completed[device] = item
        except Exception as exc:
            if device not in timed_out:
                errors[device] = repr(exc)
    return completed, errors, time.perf_counter() - started


def observe_node_rounds(completed: NodeResults, states: dict[str, dict], alpha: float) -> tuple[dict, NodeErrors]:
    """Return application-ACK observations and only the nodes that still fail."""
    observations = {}
    errors = {}
    for device, (node_results, service_time) in completed.items():
        observed = summarize_round_results(node_results, service_time)
        observations[device] = observed
        if observed["basis"] != "application_ack":
            errors[device] = "Application ACK missing"
        else:
            states[device] = smooth_link_state(states.get(device, {}), observed, alpha)
    return observations, errors


def recover_failed_nodes(
    pool: ThreadPoolExecutor,
    transports: Mapping[str, PacketTransport],
    groups: Mapping[str, list[Packet]],
    completed: NodeResults,
    errors: NodeErrors,
    states: dict[str, dict],
    alpha: float,
    attempts: int,
    timeout_seconds: float | None = None,
    event_logger: EventLogger | None = None,
    round_index: int | None = None,
) -> tuple[dict, NodeErrors, float, list[dict]]:
    """Reconnect and resend only nodes without an application ACK."""
    recovery_events = []
    elapsed = 0.0
    observations, ack_errors = observe_node_rounds(completed, states, alpha)
    errors = {**errors, **ack_errors}
    for attempt in range(1, max(0, int(attempts)) + 1):
        failed = [device for device in groups if device in errors and groups[device]]
        if not failed:
            break
        recovery_events.append({"attempt": attempt, "devices": failed})
        recovered_devices = []
        for device in failed:
            if event_logger:
                event_logger(
                    device,
                    "socket_recovery_started",
                    stage="recovery",
                    round_index=round_index,
                    attempt=attempt,
                    prior_error=errors.get(device),
                )
            recover = getattr(transports[device], "_recover_socket", None)
            if callable(recover):
                try:
                    recovery_events[-1][f"{device}_socket_recovered"] = bool(recover())
                except Exception as exc:
                    recovery_events[-1][f"{device}_socket_recovery_error"] = repr(exc)
            else:
                recovery_events[-1][f"{device}_socket_recovered"] = True
            if recovery_events[-1].get(f"{device}_socket_recovered"):
                recovered_devices.append(device)
            else:
                details = getattr(transports[device], "diagnostics", [])
                latest_failure = next(
                    (
                        item
                        for item in reversed(details)
                        if isinstance(item, dict) and item.get("socket_recovery") == "failed"
                    ),
                    None,
                )
                if latest_failure:
                    recovery_events[-1][f"{device}_socket_recovery_detail"] = latest_failure
            if event_logger:
                event_logger(
                    device,
                    "socket_recovery_finished",
                    stage="recovery",
                    round_index=round_index,
                    attempt=attempt,
                    recovered=recovery_events[-1].get(f"{device}_socket_recovered"),
                    error=recovery_events[-1].get(f"{device}_socket_recovery_error"),
                    detail=recovery_events[-1].get(f"{device}_socket_recovery_detail"),
                )
        unrecovered = set(failed) - set(recovered_devices)
        terminal_errors = {}
        for device in unrecovered:
            terminal_errors[device] = "socket recovery failed; retry was not sent"
            errors[device] = terminal_errors[device]
        if not recovered_devices:
            break
        retry_groups = {device: groups[device] for device in recovered_devices}
        retry_completed, retry_errors, retry_elapsed = send_parallel(
            pool,
            transports,
            retry_groups,
            timeout_seconds=timeout_seconds,
            event_logger=event_logger,
            stage="recovery_retry",
            round_index=round_index,
        )
        elapsed += retry_elapsed
        for device, item in retry_completed.items():
            prior_results, prior_elapsed = completed.get(device, ([], 0.0))
            completed[device] = (prior_results + item[0], prior_elapsed + item[1])
        errors = {**terminal_errors, **retry_errors}
        observations, ack_errors = observe_node_rounds(completed, states, alpha)
        errors.update(ack_errors)
    return observations, errors, elapsed, recovery_events


def resume_after_manual_modem_restart(
    pool,
    transports,
    sessions,
    groups,
    completed,
    errors,
    states,
    alpha,
    timeout_seconds,
    poll_seconds,
    max_wait_seconds,
    event_logger,
    round_index,
):
    """Pause a failed round until manually restarted modems can restore their original sessions."""
    devices = [device for device in groups if device in errors and groups[device]]
    manual_events = []
    elapsed = 0.0
    started = time.monotonic()
    for device in devices:
        event = {"device_id": device, "event": "manual_restart_required", "prior_error": errors[device]}
        manual_events.append(event)
        event_logger(
            device,
            "manual_restart_required",
            stage="manual_recovery",
            round_index=round_index,
            prior_error=errors[device],
        )
    print(
        f"[PAUSED] Round {round_index}: manually restart modem(s) {', '.join(devices)}; "
        "the coordinator will detect the modem and resume this same round.",
        flush=True,
    )
    ready = set()
    while len(ready) < len(devices):
        if max_wait_seconds and time.monotonic() - started >= max_wait_seconds:
            waiting = sorted(set(devices) - ready)
            raise RuntimeError(f"Manual modem recovery wait expired for {waiting}")
        for device in devices:
            if device in ready:
                continue
            event_logger(device, "manual_reinitialize_started", stage="manual_recovery", round_index=round_index)
            recover = getattr(transports[device], "reinitialize_after_manual_restart", None)
            try:
                recovered = bool(recover()) if callable(recover) else bool(transports[device]._recover_socket())
            except Exception as exc:
                recovered = False
                recovery_error = repr(exc)
            else:
                recovery_error = None
            if not recovered:
                details = getattr(transports[device], "diagnostics", [])
                detail = details[-1] if details else None
                event_logger(
                    device,
                    "manual_reinitialize_failed",
                    stage="manual_recovery",
                    round_index=round_index,
                    error=recovery_error,
                    detail=detail,
                    recent_transport_diagnostics=details[-12:],
                )
                continue
            session_completed, session_errors, session_elapsed = send_parallel(
                pool,
                transports,
                {device: sessions[device]},
                timeout_seconds=timeout_seconds,
                event_logger=event_logger,
                stage="manual_session_restore",
                round_index=round_index,
            )
            elapsed += session_elapsed
            session_item = session_completed.get(device)
            session_ok = (
                bool(session_item)
                and not session_errors
                and all(
                    result.confirmed
                    and isinstance(getattr(result, "application_ack", None), dict)
                    and result.application_ack.get("complete") is True
                    for result in session_item[0]
                )
            )
            if session_ok:
                ready.add(device)
                event_logger(device, "manual_session_restored", stage="manual_recovery", round_index=round_index)
            else:
                event_logger(
                    device,
                    "manual_session_restore_failed",
                    stage="manual_recovery",
                    round_index=round_index,
                    error=session_errors or "session ACK missing",
                )
        if len(ready) < len(devices):
            time.sleep(poll_seconds)

    resumed, resume_errors, resume_elapsed = send_parallel(
        pool,
        transports,
        {device: groups[device] for device in devices},
        timeout_seconds=timeout_seconds,
        event_logger=event_logger,
        stage="manual_round_resume",
        round_index=round_index,
    )
    elapsed += resume_elapsed
    new_results = []
    for device, item in resumed.items():
        completed[device] = item
        new_results.extend(item[0])
    observations, ack_errors = observe_node_rounds(completed, states, alpha)
    errors = {**resume_errors, **ack_errors}
    for device in devices:
        if device not in errors:
            event_logger(device, "manual_round_resumed", stage="manual_recovery", round_index=round_index)
    manual_events.append({"devices": devices, "event": "manual_recovery_completed", "resumed_devices": sorted(resumed)})
    return observations, errors, elapsed, manual_events, new_results


def send_packets(args: Namespace | SimpleNamespace, packets: list[Packet]) -> sender.SendRun:
    """Execute one isolated multi-node run and return the common SendRun result.

    Every entry point uses this coordinator. Node transports own their serial
    ports; only the coordinator mutates budgets, scheduling state and logs.
    """
    validate_concurrent(args)
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise ValueError("This output contains a prior run; choose a fresh output_root")
    devices = {p.device_id for p in packets}
    if devices != {n["device_id"] for n in args.nodes}:
        raise ValueError("Every configured node must have candidate packets")
    args.output_root.mkdir(parents=True, exist_ok=True)
    journal = args.output_root / "rounds.jsonl"
    # Exclusive creation prevents two coordinators from using one output path.
    with journal.open("x", encoding="utf-8"):
        pass
    diagnostic_root = args.output_root / "node_diagnostics"
    diagnostic_root.mkdir()
    diagnostic_logs = {node["device_id"]: diagnostic_root / f"{node['device_id']}.jsonl" for node in args.nodes}
    diagnostic_lock = threading.Lock()

    def log_node_event(device, event, **detail):
        record = {"timestamp": time.time(), "device_id": device, "event": event, **detail}
        with diagnostic_lock:
            with diagnostic_logs[device].open("a", encoding="utf-8") as file:
                file.write(json.dumps(record, default=str) + "\n")
                file.flush()
                os.fsync(file.fileno())

    (args.output_root / "config_snapshot.json").write_text(
        json.dumps(vars(args), default=str, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    controller = OnlineLinkController(args)
    transports = {}
    sent = []
    results = []
    rounds = []
    states = {}
    diagnostics = []
    status = "aborted"
    session_ids = {}
    try:
        for node in args.nodes:
            node_args = copy(args)
            node_args.source = "candidates"
            for key in (
                "serial_port",
                "serial_baud",
                "sim_apn",
                "remote_host",
                "remote_port",
                "sim_connect_id",
                "sim_protocol",
                "sim_timeout_seconds",
                "sim_send_timeout_seconds",
            ):
                if key in node:
                    setattr(node_args, key, node[key])
            transports[node["device_id"]] = sender.create_transport(node_args)
        with ThreadPoolExecutor(max_workers=len(transports)) as pool:
            sessions = {d: [build_session_start_packet(d)] for d in transports}
            session_ids = {d: group[0].metadata["session_id"] for d, group in sessions.items()}
            sent.extend(p for group in sessions.values() for p in group)
            completed, errors, _ = send_parallel(
                pool,
                transports,
                sessions,
                timeout_seconds=args.node_round_timeout_seconds,
                event_logger=log_node_event,
                stage="session_start",
                round_index=0,
            )
            if errors:
                _, errors, _, _ = recover_failed_nodes(
                    pool,
                    transports,
                    sessions,
                    completed,
                    errors,
                    {},
                    controller.alpha,
                    getattr(args, "node_round_recovery_attempts", 1),
                    timeout_seconds=args.node_round_timeout_seconds,
                    event_logger=log_node_event,
                    round_index=0,
                )
            results.extend(r for rr, _ in completed.values() for r in rr)
            session_failures = {}
            for device, (node_results, _) in completed.items():
                rejected = [
                    r
                    for r in node_results
                    if (
                        not r.confirmed
                        or not isinstance(getattr(r, "application_ack", None), dict)
                        or r.application_ack.get("complete") is not True
                    )
                ]
                if rejected:
                    session_failures[device] = [
                        {
                            "packet_id": r.packet_id,
                            "module_confirmed": bool(r.confirmed),
                            "application_ack": getattr(r, "application_ack", None),
                        }
                        for r in rejected
                    ]
            if errors or session_failures:
                raise RuntimeError(
                    f"Session startup failed: transport_errors={errors}; session_failures={session_failures}"
                )
            frames = sender._frame_indices(packets)
            position = 0
            while position < len(frames):
                quota = controller.current_quota_bytes
                first, last, window, next_position = sender.next_dynamic_packet_window(
                    packets, frames, position, args, quota
                )
                if args.metadata_bundle_per_window:
                    window = sender._bundle_window_metadata_packets(
                        window, first, last, args.metadata_packet_payload_bytes
                    )
                before = dict(controller.link_state, nodes=dict(states))
                selected, schedule = schedule_with_mandatory_reservation(window, args, quota, before)
                groups = {d: [p for p in selected if p.device_id == d] for d in transports}
                sent.extend(selected)
                round_index = len(rounds) + 1
                completed, errors, elapsed = send_parallel(
                    pool,
                    transports,
                    groups,
                    timeout_seconds=args.node_round_timeout_seconds,
                    event_logger=log_node_event,
                    round_index=round_index,
                )
                round_results = [r for rr, _ in completed.values() for r in rr]
                results.extend(round_results)
                observations, errors, recovery_elapsed, recovery_events = recover_failed_nodes(
                    pool,
                    transports,
                    groups,
                    completed,
                    errors,
                    states,
                    controller.alpha,
                    getattr(args, "node_round_recovery_attempts", 1),
                    timeout_seconds=args.node_round_timeout_seconds,
                    event_logger=log_node_event,
                    round_index=round_index,
                )
                if recovery_elapsed:
                    retry_results = [r for rr, _ in completed.values() for r in rr]
                    results.extend(retry_results[len(round_results) :])
                    round_results = retry_results
                    elapsed += recovery_elapsed
                if errors and getattr(args, "manual_modem_recovery_enabled", False):
                    while errors:
                        observations, errors, manual_elapsed, manual_events, manual_results = (
                            resume_after_manual_modem_restart(
                                pool,
                                transports,
                                sessions,
                                groups,
                                completed,
                                errors,
                                states,
                                controller.alpha,
                                timeout_seconds=args.node_round_timeout_seconds,
                                poll_seconds=args.manual_modem_recovery_poll_seconds,
                                max_wait_seconds=args.manual_modem_recovery_max_wait_seconds,
                                event_logger=log_node_event,
                                round_index=round_index,
                            )
                        )
                        elapsed += manual_elapsed
                        results.extend(manual_results)
                        recovery_events.extend(manual_events)
                    round_results = [r for rr, _ in completed.values() for r in rr]
                feedback = {"error": errors, "smoothed": dict(controller.smoothed)}
                if not errors:
                    feedback = controller.observe_round(round_results, elapsed_seconds=elapsed)
                feedback.update(
                    nodes=observations,
                    node_link_states=dict(states),
                    round_elapsed_seconds=elapsed,
                    node_recovery=recovery_events,
                )
                record = dict(
                    round_index=len(rounds) + 1,
                    frame_start=first,
                    frame_end=last,
                    quota_bytes=quota,
                    link_state_before=before,
                    scheduler=schedule,
                    feedback=feedback,
                    node_selected_bytes={d: sum(p.bytes for p in g) for d, g in groups.items()},
                    execution="independent_modems_concurrent_barrier",
                )
                rounds.append(record)
                with journal.open("a", encoding="utf-8") as file:
                    file.write(json.dumps(record, default=str) + "\n")
                    file.flush()
                    os.fsync(file.fileno())
                print(
                    f"Round {len(rounds)} frames {first}-{last}: B={quota}, node bytes={record['node_selected_bytes']}, wall={elapsed:.2f}s",
                    flush=True,
                )
                if errors:
                    raise RuntimeError(f"Concurrent round aborted: {errors}")
                position = next_position
        status = "complete"
    finally:
        for device, transport in transports.items():
            diagnostics.extend({"device_id": device, "detail": x} for x in transport.diagnostics)
            try:
                transport.close()
            except Exception as exc:
                diagnostics.append({"device_id": device, "close_error": repr(exc)})
        summary = sender._aggregate_round_summaries(packets, [p for p in sent if p.layer != "session_start"], rounds)
        summary.update(
            execution="independent_modems_concurrent_barrier",
            status=status,
            session_ids=session_ids,
            node_allocation_policy=args.node_allocation_policy,
        )
        outcome = sender.SendRun(args, sent, results, diagnostics, summary)
        # Successful runs use the common caller's writer; failures still retain
        # completed-node results instead of discarding the partially sent run.
        if status != "complete":
            sender.write_run_artifacts(outcome)
    return outcome


def run(config_path: Path) -> None:
    sender.run(config_path)
