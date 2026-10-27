import re
import threading
import time
from collections.abc import Iterable
from dataclasses import dataclass

from nbiot_transport.core.packetizer import Packet, build_round_ack_packet, build_window_bundle_packets
from nbiot_transport.feedback.ack_utils import missing_fragment_packet_ids
from nbiot_transport.feedback.round_ack import (
    ack_matches_message_ids,
    apply_round_ack_to_results,
    base_packet_id,
    expected_message_ids_for_packets,
    message_id_hex_for_packet,
)

BACKGROUND_LAYER = "background_keyframe_summary"


def _retransmit_layer_rank(layer: str) -> int:
    if layer in {"metadata", BACKGROUND_LAYER}:
        return 0
    if layer == "roi_patch":
        return 1
    return 2


def _print_carecv_response(source: str, response: str) -> None:
    text = str(response or "").strip()
    if len(text) > 1000:
        text = text[:1000] + "...<truncated>"
    print(f"[CARECV:{source}] {text}", flush=True)


def _print_tcp_state(source: str, state: dict) -> None:
    parts = []
    for command, response in state.items():
        text = str(response or "").replace("\r", "\\r").replace("\n", "\\n")
        if len(text) > 300:
            text = text[:300] + "...<truncated>"
        parts.append(f"{command}={text}")
    print(f"[TCP_STATE:{source}] " + " | ".join(parts), flush=True)


@dataclass
class SendResult:
    packet_id: str
    frame_idx: int
    layer: str
    bytes: int
    confirmed: bool
    elapsed_ms: float
    roi_idx: int | None = None
    rate_idx: int | None = None
    response: str = ""
    error: str | None = None
    application_ack: dict | None = None
    wire_bytes: int | None = None

    def to_json(self) -> dict:
        return {
            "packet_id": self.packet_id,
            "frame_idx": self.frame_idx,
            "layer": self.layer,
            "bytes": self.bytes,
            "confirmed": self.confirmed,
            "elapsed_ms": self.elapsed_ms,
            "roi_idx": self.roi_idx,
            "rate_idx": self.rate_idx,
            "response": self.response,
            "error": self.error,
            "application_ack": self.application_ack,
            "wire_bytes": self.wire_bytes,
        }


class NullTransport:
    def send_packets(self, packets: Iterable[Packet]) -> list[SendResult]:
        results = []
        for packet in packets:
            results.append(
                SendResult(
                    packet_id=packet.packet_id,
                    frame_idx=packet.frame_idx,
                    layer=packet.layer,
                    bytes=packet.bytes,
                    confirmed=True,
                    elapsed_ms=0.0,
                    roi_idx=packet.roi_idx,
                    rate_idx=packet.rate_idx,
                    response="offline",
                )
            )
        return results

    def close(self) -> None:
        return None


class Sim7080Transport:
    def __init__(
        self,
        serial_port: str,
        baudrate: int,
        apn: str,
        remote_host: str,
        remote_port: int,
        protocol: str = "UDP",
        connect_id: int = 0,
        timeout_seconds: float = 8.0,
        send_confirm_timeout_seconds: float = 30.0,
        network_registration_timeout_seconds: float = 120.0,
        network_registration_poll_seconds: float = 3.0,
        application_ack_enabled: bool = False,
        application_ack_timeout_seconds: float = 5.0,
        ack_initial_delay_seconds: float = 2.0,
        ack_poll_interval_seconds: float = 1.0,
        round_ack_query_attempts: int = 2,
        carecv_read_bytes: int = 256,
        fragment_retransmit_enabled: bool = True,
        fragment_retransmit_attempts: int = 1,
        roi_patch_fragment_retransmit_attempts: int = 2,
        background_summary_fragment_retransmit_attempts: int = 3,
        socket_reopen_on_send_failures: int = 1,
        socket_reopen_attempts: int = 1,
        window_shared_header_enabled: bool = False,
        window_bundle_packet_payload_bytes: int = 1024,
        initialize: bool = True,
    ):
        try:
            import serial
        except ImportError as exc:
            raise RuntimeError(
                "pyserial is required for --transport sim7080. Install it with pip install pyserial."
            ) from exc

        self.serial = serial.Serial(serial_port, baudrate=baudrate, timeout=0.2, write_timeout=timeout_seconds)
        self.apn = apn
        self.remote_host = remote_host
        self.remote_port = int(remote_port)
        self.protocol = protocol.upper()
        self.connect_id = int(connect_id)
        self.timeout_seconds = float(timeout_seconds)
        self.send_confirm_timeout_seconds = float(send_confirm_timeout_seconds)
        self.network_registration_timeout_seconds = max(1.0, float(network_registration_timeout_seconds))
        self.network_registration_poll_seconds = max(0.2, float(network_registration_poll_seconds))
        self.application_ack_enabled = bool(application_ack_enabled)
        self.application_ack_timeout_seconds = float(application_ack_timeout_seconds)
        self.ack_initial_delay_seconds = max(0.0, float(ack_initial_delay_seconds))
        self.ack_poll_interval_seconds = max(0.05, float(ack_poll_interval_seconds))
        self.round_ack_query_attempts = max(1, int(round_ack_query_attempts))
        self.carecv_read_bytes = min(1024, max(128, int(carecv_read_bytes)))
        self.fragment_retransmit_enabled = bool(fragment_retransmit_enabled)
        self.fragment_retransmit_attempts = max(0, int(fragment_retransmit_attempts))
        self.background_summary_fragment_retransmit_attempts = max(
            self.fragment_retransmit_attempts,
            int(background_summary_fragment_retransmit_attempts),
        )
        self.roi_patch_fragment_retransmit_attempts = max(
            self.fragment_retransmit_attempts,
            min(int(roi_patch_fragment_retransmit_attempts), self.background_summary_fragment_retransmit_attempts),
        )
        self.socket_reopen_on_send_failures = max(0, int(socket_reopen_on_send_failures))
        self.socket_reopen_attempts = max(0, int(socket_reopen_attempts))
        self.window_shared_header_enabled = bool(window_shared_header_enabled)
        self.window_bundle_packet_payload_bytes = max(256, int(window_bundle_packet_payload_bytes or 1024))
        self.consecutive_send_failures = 0
        self.tcp_round_ack_buffer = ""
        self.diagnostics = []
        self._abort_event = threading.Event()
        self._abort_reason = None
        if initialize:
            self.initialize()

    def initialize(self, verify_at: bool = True) -> None:
        """Bring up packet data and the application socket from a known AT state."""
        if verify_at:
            self._require_at_ready()
        # Do not discard input between initialization commands.  A delayed
        # socket URC from a cancelled operation is evidence for recovery and
        # must not silently become the response to an unrelated command.
        self.command("ATE0", reset_input=False)
        self.command("AT+CPIN?", expected=("READY", "OK"), reset_input=False)
        self.command("AT+CSQ", reset_input=False)
        self._wait_for_network_registration()
        self.command("AT+COPS?", reset_input=False)
        self.command("AT+CGATT?", reset_input=False)
        self.command("AT+CGNAPN", expected=("OK", "ERROR"), raise_on_error=False, reset_input=False)
        if self.apn:
            self.command(f'AT+CGDCONT=1,"IP","{self.apn}"', reset_input=False)
            self.command(f'AT+CNCFG=0,1,"{self.apn}"', reset_input=False)
        self._ensure_app_pdp_active()
        if not self.remote_host or self.remote_port <= 0:
            return
        self.command(
            f'AT+CASSLCFG={self.connect_id},"SSL",0', expected=("OK", "ERROR"), raise_on_error=False, reset_input=False
        )
        self.command(f"AT+CACLOSE={self.connect_id}", expected=("OK", "ERROR"), raise_on_error=False, reset_input=False)
        self._open_socket()

    def _open_socket(self) -> None:
        response = self.command(
            f'AT+CAOPEN={self.connect_id},0,"{self.protocol}","{self.remote_host}",{self.remote_port}',
            timeout_seconds=10.0,
            expected=("OK",),
            reset_input=False,
        )
        if "+CAOPEN:" not in response:
            response += self._read_until(60.0, ("+CAOPEN:", "ERROR"))
        if "ERROR" in response:
            raise RuntimeError(f"AT+CAOPEN failed: {response.strip()}")
        if "+CAOPEN:" not in response:
            state = self._socket_open_failure_state()
            raise RuntimeError(
                f"AT+CAOPEN did not return socket status: {response.strip()}; diagnostics: {' | '.join(state)}"
            )
        if f"+CAOPEN: {self.connect_id},0" not in response and f"+CAOPEN:{self.connect_id},0" not in response:
            state = []
            for command in ("AT+CSQ", "AT+CEREG?", "AT+CNACT?"):
                state.append(self.command(command, expected=("OK", "ERROR"), raise_on_error=False).strip())
            raise RuntimeError(
                f"AT+CAOPEN returned non-zero status: {response.strip()}; diagnostics: {' | '.join(state)}"
            )

    def _socket_open_failure_state(self) -> list[str]:
        """Collect bounded modem state after a missing CAOPEN completion URC."""
        state = []
        for command in ("AT", "AT+CSQ", "AT+CEREG?", "AT+CNACT?", "AT+CASTATE?"):
            try:
                response = self.command(
                    command,
                    timeout_seconds=2.0,
                    expected=("OK", "ERROR"),
                    raise_on_error=False,
                    reset_input=False,
                ).strip()
            except Exception as exc:
                response = f"{type(exc).__name__}: {exc}"
            state.append(f"{command}={response or '<no response>'}")
        self.diagnostics.append({"socket_open_missing_status": state})
        return state

    def _drain_serial_input(self, quiet_seconds: float = 0.4, max_seconds: float = 3.0) -> str:
        """Record and drain delayed URCs before beginning a new command sequence."""
        deadline = time.monotonic() + max(quiet_seconds, max_seconds)
        quiet_deadline = time.monotonic() + quiet_seconds
        chunks = []
        while time.monotonic() < deadline:
            data = self.serial.read(4096)
            if data:
                chunks.append(data.decode("utf-8", errors="replace"))
                quiet_deadline = time.monotonic() + quiet_seconds
                continue
            if time.monotonic() >= quiet_deadline:
                break
        drained = "".join(chunks)
        if drained:
            self.diagnostics.append({"serial_recovery_drain": drained[:4000]})
        return drained

    def _require_at_ready(self, attempts: int = 3, interval_seconds: float = 0.5) -> None:
        """Require repeated clean AT responses before touching PDP or sockets."""
        responses = []
        consecutive_ok = 0
        for attempt in range(1, max(1, attempts) + 1):
            response = self.command(
                "AT",
                timeout_seconds=2.0,
                expected=("OK",),
                raise_on_error=False,
                reset_input=False,
            )
            responses.append(response.strip() or "<no response>")
            if "OK" in response:
                consecutive_ok += 1
                if consecutive_ok >= 2:
                    self.diagnostics.append({"at_preflight": "ready", "attempt": attempt, "responses": responses})
                    return
            else:
                consecutive_ok = 0
            if attempt < attempts:
                time.sleep(interval_seconds)
        self.diagnostics.append({"at_preflight": "not_ready", "responses": responses})
        raise RuntimeError("modem AT channel is not responsive; wait for a physical restart before socket recovery")

    def _ensure_app_pdp_active(self) -> None:
        state = self.command("AT+CNACT?", expected=("OK", "ERROR"), raise_on_error=False, reset_input=False)
        if self._app_pdp_is_active(state):
            return
        response = self.command(
            "AT+CNACT=0,1",
            timeout_seconds=60.0,
            expected=("OK", "+APP PDP:", "ERROR"),
            raise_on_error=False,
            reset_input=False,
        )
        if "ERROR" in response:
            state = self.command("AT+CNACT?", expected=("OK", "ERROR"), raise_on_error=False, reset_input=False)
            if self._app_pdp_is_active(state):
                return
            raise RuntimeError(f"AT+CNACT=0,1 failed: {response.strip()}; state: {state.strip()}")
        state = self.command("AT+CNACT?", expected=("OK", "ERROR"), raise_on_error=False, reset_input=False)
        if not self._app_pdp_is_active(state):
            raise RuntimeError(f"APP PDP is not active after AT+CNACT=0,1: {state.strip()}")

    def _wait_for_network_registration(self) -> None:
        deadline = time.monotonic() + self.network_registration_timeout_seconds
        last_response = ""
        while time.monotonic() < deadline:
            last_response = self.command(
                "AT+CEREG?",
                timeout_seconds=min(5.0, max(0.2, deadline - time.monotonic())),
                expected=("OK", "ERROR"),
                raise_on_error=False,
                reset_input=False,
            )
            if self._is_network_registered(last_response):
                self.diagnostics.append({"network_registration": "ready", "response": last_response.strip()})
                return
            time.sleep(self.network_registration_poll_seconds)
        raise RuntimeError(
            "Network registration timeout before PDP/socket initialization; "
            f"last AT+CEREG? response: {last_response.strip() or '<no response>'}"
        )

    @staticmethod
    def _is_network_registered(response: str) -> bool:
        for line in response.replace("\r", "\n").split("\n"):
            line = line.strip()
            if not line.startswith("+CEREG:"):
                continue
            fields = [part.strip() for part in line.split(":", 1)[1].split(",")]
            try:
                status = int(fields[1] if len(fields) >= 2 else fields[0])
            except (IndexError, ValueError):
                continue
            if status in {1, 5}:
                return True
        return False

    @staticmethod
    def _app_pdp_is_active(response: str) -> bool:
        for line in response.replace("\r", "\n").split("\n"):
            line = line.strip()
            if line.startswith("+CNACT:") and ",1" in line:
                return True
        return False

    def command(
        self,
        command: str,
        timeout_seconds: float | None = None,
        expected: tuple[str, ...] = ("OK",),
        raise_on_error: bool = True,
        reset_input: bool = True,
    ) -> str:
        self._raise_if_aborted()
        self._write_line(command, reset_input=reset_input)
        response = self._read_until(timeout_seconds or self.timeout_seconds, expected + ("ERROR",))
        self.diagnostics.append({"command": command, "response": response.strip()})
        if raise_on_error and "ERROR" in response and not any(token in response for token in expected):
            raise RuntimeError(f"{command} failed: {response.strip()}")
        return response

    def send_packets(self, packets: Iterable[Packet]) -> list[SendResult]:
        self._abort_event.clear()
        self._abort_reason = None
        packets = list(packets)
        if len({packet.device_id for packet in packets}) > 1:
            raise ValueError("A SIM7080 transport accepts only one node per send; use independent node transports")
        if not packets:
            return []
        return self._send_single_device_packets(packets)

    def abort_current_operation(self, reason: str = "coordinator timeout") -> None:
        """Interrupt a bounded coordinator round without sharing serial I/O across workers."""
        self._abort_reason = reason
        self._abort_event.set()
        self.diagnostics.append({"operation_abort": reason})
        for name in ("cancel_read", "cancel_write"):
            cancel = getattr(self.serial, name, None)
            if callable(cancel):
                try:
                    cancel()
                except Exception as exc:
                    self.diagnostics.append({"operation_abort_error": name, "error": str(exc)})
        # Closing releases a driver write that cancel_write cannot interrupt on
        # some Windows serial implementations. Recovery reopens it after the
        # worker observes the abort and exits.
        try:
            self.serial.close()
        except Exception as exc:
            self.diagnostics.append({"operation_abort_error": "serial_close", "error": str(exc)})

    def reinitialize_after_manual_restart(self) -> bool:
        """Resume only after a manually restarted modem has a clean AT channel."""
        self._abort_event.clear()
        self._abort_reason = None
        self.tcp_round_ack_buffer = ""
        try:
            if self.serial.is_open:
                self.serial.close()
            self.serial.open()
            self.serial.reset_output_buffer()
            self._drain_serial_input()
            self._require_at_ready()
            self.initialize(verify_at=False)
            self.diagnostics.append({"manual_modem_reinitialize": "ok"})
            return True
        except Exception as exc:
            self.diagnostics.append({"manual_modem_reinitialize": "failed", "error": str(exc)})
            # Keep an open COM port when possible.  The next polling pass can
            # observe boot URCs and AT recovery instead of repeatedly tearing
            # down the UART while the user is restarting the module.
            return False

    def _raise_if_aborted(self) -> None:
        if self._abort_event.is_set():
            raise TimeoutError(self._abort_reason or "SIM7080 operation aborted")

    def _send_single_device_packets(self, packets: list[Packet]) -> list[SendResult]:
        if self.application_ack_enabled:
            return self._send_packets_with_round_ack(packets)
        results = []
        for packet in packets:
            if self._abort_event.is_set():
                break
            base_id = packet.packet_id.split(":fragment:", 1)[0]
            if self._base_packet_complete(results, base_id):
                continue
            results.append(self.send_packet(packet))
        if not self.fragment_retransmit_enabled or self.fragment_retransmit_attempts <= 0:
            return results

        packets_by_id = {packet.packet_id: packet for packet in packets}
        for attempt in range(1, self.fragment_retransmit_attempts + 1):
            missing_ids = missing_fragment_packet_ids(results)
            retry_packets = [packets_by_id[packet_id] for packet_id in missing_ids if packet_id in packets_by_id]
            if not retry_packets:
                break
            self.diagnostics.append(
                {
                    "fragment_retransmit_attempt": attempt,
                    "missing_packet_ids": [packet.packet_id for packet in retry_packets],
                }
            )
            for packet in retry_packets:
                base_id = packet.packet_id.split(":fragment:", 1)[0]
                if self._base_packet_complete(results, base_id):
                    break
                result = self.send_packet(packet)
                results.append(result)
                if self._base_packet_complete(results, base_id):
                    break
        for attempt in range(
            self.fragment_retransmit_attempts + 1, self.background_summary_fragment_retransmit_attempts + 1
        ):
            retry_packets = self._extended_missing_packets(packets_by_id, results, attempt)
            if not retry_packets:
                break
            self.diagnostics.append(
                {
                    "critical_layer_fragment_retransmit_attempt": attempt,
                    "missing_packet_ids": [packet.packet_id for packet in retry_packets],
                    "layers": sorted({packet.layer for packet in retry_packets}),
                }
            )
            for packet in retry_packets:
                base_id = packet.packet_id.split(":fragment:", 1)[0]
                if self._base_packet_complete(results, base_id):
                    break
                result = self.send_packet(packet)
                results.append(result)
                if self._base_packet_complete(results, base_id):
                    break
        return results

    def _send_packets_with_round_ack(self, packets: list[Packet]) -> list[SendResult]:
        wire_packets = self._wire_packets_for_round(packets)
        wire_results = []
        for packet in wire_packets:
            if self._abort_event.is_set():
                break
            wire_results.append(self.send_packet(packet))
        results = self._logical_results_for_round(packets, wire_packets, wire_results)
        if not packets or self._abort_event.is_set():
            return results
        self.query_round_ack(packets, results)
        if not any(
            isinstance(getattr(result, "application_ack", None), dict)
            and result.application_ack.get("observed") is True
            for result in results
        ):
            self.diagnostics.append(
                {
                    "socket_recovery_trigger": "round_ack_not_observed",
                    "packet_count": len(packets),
                }
            )
            self._recover_socket()
        if not self.fragment_retransmit_enabled or self.fragment_retransmit_attempts <= 0:
            return results

        packets_by_id = {packet.packet_id: packet for packet in packets}
        for attempt in range(1, self.fragment_retransmit_attempts + 1):
            missing_ids = missing_fragment_packet_ids(results)
            retry_packets = [packets_by_id[packet_id] for packet_id in missing_ids if packet_id in packets_by_id]
            if not retry_packets:
                break
            self.diagnostics.append(
                {
                    "round_ack_fragment_retransmit_attempt": attempt,
                    "missing_packet_ids": [packet.packet_id for packet in retry_packets],
                }
            )
            retry_results = [self.send_packet(packet) for packet in retry_packets]
            results.extend(retry_results)
            self.query_round_ack(retry_packets, retry_results)
        for attempt in range(
            self.fragment_retransmit_attempts + 1, self.background_summary_fragment_retransmit_attempts + 1
        ):
            retry_packets = self._extended_missing_packets(packets_by_id, results, attempt)
            if not retry_packets:
                break
            self.diagnostics.append(
                {
                    "round_ack_critical_layer_fragment_retransmit_attempt": attempt,
                    "missing_packet_ids": [packet.packet_id for packet in retry_packets],
                    "layers": sorted({packet.layer for packet in retry_packets}),
                }
            )
            retry_results = [self.send_packet(packet) for packet in retry_packets]
            results.extend(retry_results)
            self.query_round_ack(retry_packets, retry_results)
        return results

    def _wire_packets_for_round(self, packets: list[Packet]) -> list[Packet]:
        if len({packet.device_id for packet in packets}) > 1:
            raise ValueError("Window bundles must contain exactly one device; use send_packets to partition nodes.")
        if not self.window_shared_header_enabled or len(packets) <= 1:
            return packets
        if any(packet.layer in {BACKGROUND_LAYER, "session_start", "round_ack"} for packet in packets):
            return packets
        device_id = packets[0].device_id if packets else "node"
        round_index = int(time.time() * 1000)
        bundled = build_window_bundle_packets(
            device_id,
            round_index,
            packets,
            max_payload_bytes=self.window_bundle_packet_payload_bytes,
        )
        if not bundled:
            return packets
        self.diagnostics.append(
            {
                "window_shared_header": True,
                "logical_packet_count": len(packets),
                "wire_packet_count": len(bundled),
                "logical_bytes": sum(packet.bytes for packet in packets),
                "wire_bytes": sum(packet.bytes for packet in bundled),
                "saved_bytes": sum(packet.bytes for packet in packets) - sum(packet.bytes for packet in bundled),
            }
        )
        return bundled

    @staticmethod
    def _logical_results_for_round(
        logical_packets: list[Packet],
        wire_packets: list[Packet],
        wire_results: list[SendResult],
    ) -> list[SendResult]:
        if wire_packets == logical_packets:
            return wire_results
        confirmed = all(result.confirmed for result in wire_results)
        error = next((result.error for result in wire_results if result.error), None)
        elapsed_ms = max((float(result.elapsed_ms or 0.0) for result in wire_results), default=0.0)
        wire_bytes = sum(int(result.bytes or 0) for result in wire_results)
        logical_bytes = max(1, sum(packet.bytes for packet in logical_packets))
        results = []
        allocated = 0
        for index, packet in enumerate(logical_packets):
            if index + 1 == len(logical_packets):
                packet_wire_bytes = max(0, wire_bytes - allocated)
            else:
                packet_wire_bytes = max(1, round(wire_bytes * packet.bytes / logical_bytes))
                allocated += packet_wire_bytes
            results.append(
                SendResult(
                    packet_id=packet.packet_id,
                    frame_idx=packet.frame_idx,
                    layer=packet.layer,
                    bytes=packet.bytes,
                    confirmed=confirmed,
                    elapsed_ms=elapsed_ms,
                    roi_idx=packet.roi_idx,
                    rate_idx=packet.rate_idx,
                    response="window_shared_header",
                    error=error,
                    wire_bytes=packet_wire_bytes,
                )
            )
        return results

    def _extended_missing_packets(
        self,
        packets_by_id: dict[str, Packet],
        results: list[SendResult],
        attempt: int,
    ) -> list[Packet]:
        layers = {"metadata", BACKGROUND_LAYER}
        if attempt <= self.roi_patch_fragment_retransmit_attempts:
            layers.add("roi_patch")
        return self._missing_packets_for_layers(packets_by_id, results, layers)

    @staticmethod
    def _missing_packets_for_layers(
        packets_by_id: dict[str, Packet],
        results: list[SendResult],
        layers: set[str],
    ) -> list[Packet]:
        retry_ids = set(missing_fragment_packet_ids(results))
        complete_bases = set()
        for result in results:
            ack = getattr(result, "application_ack", None)
            if isinstance(ack, dict) and ack.get("complete") is True:
                complete_bases.add(str(ack.get("packet_id") or result.packet_id).split(":fragment:", 1)[0])
        for result in results:
            if bool(getattr(result, "confirmed", False)):
                continue
            packet_id = str(result.packet_id)
            base_id = packet_id.split(":fragment:", 1)[0]
            if base_id not in complete_bases:
                retry_ids.add(packet_id)
        retry_packets = []
        for packet_id in sorted(retry_ids):
            packet = packets_by_id.get(packet_id)
            if packet is not None and packet.layer in layers:
                retry_packets.append(packet)
        return sorted(retry_packets, key=lambda packet: (_retransmit_layer_rank(packet.layer), packet.packet_id))

    def query_round_ack(self, packets: list[Packet], results: list[SendResult] | None = None) -> dict:
        if not packets:
            return {
                "observed": False,
                "type": "round_ack",
                "confirmed": True,
                "complete": True,
                "packet_id": "",
                "messages": {},
                "expected_message_count": 0,
                "received_message_count": 0,
                "missing_message_count": 0,
                "missing_message_ids": [],
                "rtt_ms": None,
                "error": "empty packet list; round ACK query skipped locally",
            }
        round_ack = self._send_round_ack(packets)
        if results is not None:
            apply_round_ack_to_results(results, round_ack)
        return round_ack

    def _send_round_ack(self, packets: list[Packet]) -> dict:
        if not packets:
            return {
                "observed": False,
                "type": "round_ack",
                "confirmed": True,
                "complete": True,
                "packet_id": "",
                "messages": {},
                "expected_message_count": 0,
                "received_message_count": 0,
                "missing_message_count": 0,
                "missing_message_ids": [],
                "rtt_ms": None,
                "error": "empty packet list; round ACK query skipped locally",
            }
        expected_message_ids = expected_message_ids_for_packets(packets)
        if not expected_message_ids:
            self.diagnostics.append(
                {
                    "round_ack_query_skipped": "no expected message ids",
                    "packet_count": len(packets),
                }
            )
            return {
                "observed": False,
                "type": "round_ack",
                "confirmed": True,
                "complete": True,
                "packet_id": "",
                "messages": {},
                "expected_message_count": 0,
                "received_message_count": 0,
                "missing_message_count": 0,
                "missing_message_ids": [],
                "rtt_ms": None,
                "error": "no expected message ids; round ACK query skipped locally",
            }
        last_ack = None
        for attempt in range(1, self.round_ack_query_attempts + 1):
            self.tcp_round_ack_buffer = ""
            packet = build_round_ack_packet(
                packets[0].device_id if packets else "node", int(time.time() * 1000), packets
            )
            started = time.monotonic()
            response = self._send_payload(packet.payload, reset_input=False)
            confirmed = "ERROR" not in response and ("OK" in response or "SEND" in response or "+CASEND:" in response)
            if not confirmed:
                unsolicited = self._read_available_text(0.5)
                if unsolicited:
                    self._record_tcp_unsolicited(
                        "round_ack_send_failed_unsolicited",
                        unsolicited,
                        expected_message_ids,
                    )
                    ack = self._select_round_ack(unsolicited, packet.packet_id, started, expected_message_ids)
                    if ack is not None:
                        ack["round_ack_query_attempt"] = attempt
                        ack["source"] = "serial_unsolicited_after_round_ack_send_failed"
                        return ack
                last_ack = {
                    "observed": False,
                    "confirmed": False,
                    "complete": False,
                    "packet_id": packet.packet_id,
                    "rtt_ms": (time.monotonic() - started) * 1000.0,
                    "round_ack_query_attempt": attempt,
                    "error": self._socket_state_error(),
                }
                self.diagnostics.append(
                    {
                        "round_ack_query_attempt": attempt,
                        "confirmed": False,
                        "error": last_ack["error"],
                        "expected_message_ids": sorted(expected_message_ids),
                    }
                )
                continue
            ack = self._select_round_ack(response, packet.packet_id, started, expected_message_ids)
            if ack is not None:
                ack["round_ack_query_attempt"] = attempt
                ack["source"] = "round_ack_send_response"
                self.diagnostics.append(
                    {
                        "command": "<round_ack_send_response_ack>",
                        "round_ack_query_attempt": attempt,
                        "expected_message_ids": sorted(expected_message_ids),
                        "response_preview": response.strip()[:500],
                    }
                )
                return ack
            self._record_tcp_unsolicited("round_ack_send_response", response, expected_message_ids)
            ack = self._read_round_ack(packet.packet_id, started, expected_message_ids)
            ack["round_ack_query_attempt"] = attempt
            if ack.get("observed") is True:
                return ack
            last_ack = ack
            self.diagnostics.append(
                {
                    "round_ack_query_attempt": attempt,
                    "confirmed": True,
                    "ack_observed": False,
                    "error": ack.get("error"),
                    "expected_message_ids": sorted(expected_message_ids),
                }
            )
            if attempt < self.round_ack_query_attempts:
                time.sleep(self.ack_initial_delay_seconds)
        return last_ack or {
            "observed": False,
            "confirmed": None,
            "complete": None,
            "packet_id": "",
            "rtt_ms": None,
            "error": "round ACK timeout",
        }

    @staticmethod
    def _base_packet_complete(results: list[SendResult], base_id: str) -> bool:
        for result in results:
            ack = result.application_ack
            if not isinstance(ack, dict) or ack.get("complete") is not True or ack.get("confirmed") is not True:
                continue
            ack_id = str(ack.get("packet_id") or result.packet_id)
            if base_packet_id(ack_id) == base_id:
                return True
        return False

    def send_packet(self, packet: Packet) -> SendResult:
        started = time.monotonic()
        if not self.remote_host or self.remote_port <= 0:
            return SendResult(
                packet_id=packet.packet_id,
                frame_idx=packet.frame_idx,
                layer=packet.layer,
                bytes=packet.bytes,
                confirmed=False,
                elapsed_ms=0.0,
                roi_idx=packet.roi_idx,
                rate_idx=packet.rate_idx,
                error="remote_host/remote_port not configured; SIM initialized but packet was not sent",
            )
        try:
            response = self._send_payload(packet.payload)
            confirmed = "ERROR" not in response and ("OK" in response or "SEND" in response or "+CASEND:" in response)
            error = None
            application_ack = None
            if not confirmed:
                error = self._socket_state_error()
                self._note_send_failure(error)
            else:
                self.consecutive_send_failures = 0
            elapsed_ms = (time.monotonic() - started) * 1000.0
            return SendResult(
                packet_id=packet.packet_id,
                frame_idx=packet.frame_idx,
                layer=packet.layer,
                bytes=packet.bytes,
                confirmed=confirmed,
                elapsed_ms=elapsed_ms,
                roi_idx=packet.roi_idx,
                rate_idx=packet.rate_idx,
                response=response.strip(),
                error=error,
                application_ack=application_ack,
            )
        except Exception as exc:
            elapsed_ms = (time.monotonic() - started) * 1000.0
            self._note_send_failure(str(exc))
            return SendResult(
                packet_id=packet.packet_id,
                frame_idx=packet.frame_idx,
                layer=packet.layer,
                bytes=packet.bytes,
                confirmed=False,
                elapsed_ms=elapsed_ms,
                roi_idx=packet.roi_idx,
                rate_idx=packet.rate_idx,
                error=str(exc),
            )

    def close(self) -> None:
        if not self.remote_host or self.remote_port <= 0:
            self.serial.close()
            return
        try:
            self.command(f"AT+CACLOSE={self.connect_id}", expected=("OK", "ERROR"), raise_on_error=False)
        finally:
            self.serial.close()

    def _send_payload(self, payload: bytes, reset_input: bool = True) -> str:
        self._raise_if_aborted()
        self._write_line(f"AT+CASEND={self.connect_id},{len(payload)}", reset_input=reset_input)
        prompt = self._read_until(self.timeout_seconds, (">", "ERROR"))
        self.diagnostics.append({"command": f"AT+CASEND={self.connect_id},{len(payload)}", "response": prompt.strip()})
        if "ERROR" in prompt:
            return prompt
        self.serial.write(payload)
        self.serial.flush()
        response = self._read_until(self.send_confirm_timeout_seconds, ("OK", "SEND", "+CASEND:", "ERROR"))
        self.diagnostics.append({"command": "<payload>", "response": response.strip()})
        return prompt + response

    def _note_send_failure(self, error: str | None = None) -> None:
        if self._abort_event.is_set():
            return
        self.consecutive_send_failures += 1
        threshold = self.socket_reopen_on_send_failures
        if threshold <= 0 or self.consecutive_send_failures < threshold:
            return
        self.diagnostics.append(
            {
                "socket_recovery_trigger": "consecutive_send_failures",
                "count": self.consecutive_send_failures,
                "last_error": error,
            }
        )
        recovered = self._recover_socket()
        if recovered:
            self.consecutive_send_failures = 0

    def _recover_socket(self) -> bool:
        if not self.remote_host or self.remote_port <= 0:
            return False
        self._abort_event.clear()
        self._abort_reason = None
        for attempt in range(1, self.socket_reopen_attempts + 1):
            try:
                if not self.serial.is_open:
                    self.serial.open()
                self._drain_serial_input()
                self._require_at_ready()
                self.command(
                    f"AT+CACLOSE={self.connect_id}", expected=("OK", "ERROR"), raise_on_error=False, reset_input=False
                )
                self._ensure_app_pdp_active()
                self._open_socket()
                self.diagnostics.append({"socket_recovery": "ok", "attempt": attempt})
                return True
            except Exception as exc:
                self.diagnostics.append({"socket_recovery": "failed", "attempt": attempt, "error": str(exc)})
                time.sleep(1.0)
        return False

    def _read_round_ack(self, packet_id: str, started: float, expected_message_ids: set[str]) -> dict:
        if self.protocol == "TCP":
            ack = self._poll_tcp_round_ack(packet_id, started, expected_message_ids)
            if ack is not None:
                return ack
            return {
                "observed": False,
                "confirmed": None,
                "complete": None,
                "packet_id": packet_id,
                "rtt_ms": None,
                "error": "round ACK timeout",
            }

        response = self._read_round_ack_line()
        if response:
            self.diagnostics.append({"command": "<round_ack_wait>", "response": response.strip()})
            ack = self._select_round_ack(response, packet_id, started, expected_message_ids)
            if ack is not None:
                return ack

        return {
            "observed": False,
            "confirmed": None,
            "complete": None,
            "packet_id": packet_id,
            "rtt_ms": None,
            "error": "round ACK timeout",
        }

    def _poll_tcp_round_ack(
        self,
        packet_id: str,
        started: float,
        expected_message_ids: set[str],
    ) -> dict | None:
        if self.ack_initial_delay_seconds > 0:
            time.sleep(self.ack_initial_delay_seconds)
        deadline = time.monotonic() + self.application_ack_timeout_seconds
        poll_count = 0
        last_error = None
        while time.monotonic() < deadline:
            poll_count += 1
            unsolicited = self._read_available_text(0.15)
            if unsolicited:
                self._record_tcp_unsolicited("before_carecv", unsolicited, expected_message_ids, poll_count)
                ack = self._select_round_ack(unsolicited, packet_id, started, expected_message_ids)
                if ack is not None:
                    ack["tcp_ack_poll_count"] = poll_count
                    ack["source"] = "serial_unsolicited_before_carecv"
                    return ack
                if "+CADATAIND" in unsolicited or "+CARECV" in unsolicited:
                    ack = self._drain_tcp_carecv(
                        packet_id,
                        started,
                        expected_message_ids,
                        poll_count,
                        deadline,
                        source="cadatind_drain",
                    )
                    if ack is not None:
                        return ack
            try:
                response = self._carecv_once(deadline)
                _print_carecv_response(f"poll:{poll_count}", response)
            except Exception as exc:
                last_error = str(exc)
                self.diagnostics.append(
                    {
                        "command": "<tcp_round_ack_poll_error>",
                        "poll": poll_count,
                        "error": last_error,
                    }
                )
                time.sleep(self.ack_poll_interval_seconds)
                continue

            self.diagnostics.append(
                {
                    "command": "<tcp_round_ack_poll>",
                    "poll": poll_count,
                    "response": response.strip(),
                    "expected_message_ids": sorted(expected_message_ids),
                }
            )
            self._record_tcp_unsolicited("carecv_response", response, expected_message_ids, poll_count)
            ack = self._select_round_ack(response, packet_id, started, expected_message_ids)
            if ack is not None:
                ack["tcp_ack_poll_count"] = poll_count
                ack["source"] = "carecv_response"
                return ack
            if "ERROR" in response:
                last_error = response.strip()
                if poll_count == 1 or poll_count % 5 == 0:
                    self._record_tcp_receive_error_state(poll_count, response, expected_message_ids)
                time.sleep(max(0.5, self.ack_poll_interval_seconds))
            else:
                if self._carecv_has_payload(response):
                    ack = self._drain_tcp_carecv(
                        packet_id,
                        started,
                        expected_message_ids,
                        poll_count,
                        deadline,
                        source="carecv_payload_drain",
                    )
                    if ack is not None:
                        return ack
                time.sleep(self.ack_poll_interval_seconds)

        if last_error:
            self.diagnostics.append(
                {
                    "command": "<tcp_round_ack_poll_timeout>",
                    "poll_count": poll_count,
                    "last_error": last_error,
                    "expected_message_ids": sorted(expected_message_ids),
                }
            )
        return None

    def _carecv_once(self, deadline: float) -> str:
        return self.command(
            f"AT+CARECV={self.connect_id},{self.carecv_read_bytes}",
            timeout_seconds=min(3.0, max(0.2, deadline - time.monotonic())),
            expected=("OK", "ERROR"),
            raise_on_error=False,
            reset_input=False,
        )

    def _drain_tcp_carecv(
        self,
        packet_id: str,
        started: float,
        expected_message_ids: set[str],
        poll_count: int,
        deadline: float,
        source: str,
        max_reads: int = 8,
    ) -> dict | None:
        for read_index in range(1, max_reads + 1):
            if time.monotonic() >= deadline:
                return None
            response = self._carecv_once(deadline)
            _print_carecv_response(f"{source}:poll:{poll_count}:read:{read_index}", response)
            self.diagnostics.append(
                {
                    "command": "<tcp_round_ack_drain>",
                    "source": source,
                    "poll": poll_count,
                    "read": read_index,
                    "response": response.strip(),
                    "expected_message_ids": sorted(expected_message_ids),
                }
            )
            self._record_tcp_unsolicited(source, response, expected_message_ids, poll_count)
            ack = self._select_round_ack(response, packet_id, started, expected_message_ids)
            if ack is not None:
                ack["tcp_ack_poll_count"] = poll_count
                ack["source"] = source
                return ack
            if "ERROR" in response:
                self._record_tcp_receive_error_state(poll_count, response, expected_message_ids)
            if not self._carecv_has_payload(response):
                return None
        return None

    @staticmethod
    def _carecv_has_payload(response: str) -> bool:
        for chunk in response.replace("\r", "\n").split("\n"):
            line = chunk.strip()
            if not line.startswith("+CARECV:"):
                continue
            after = line.split(":", 1)[1].strip()
            size_text = after.split(",", 1)[0].strip()
            try:
                if int(size_text) > 0:
                    return True
            except ValueError:
                return "," in after
        return False

    def _record_tcp_unsolicited(
        self,
        source: str,
        response: str,
        expected_message_ids: set[str],
        poll_count: int | None = None,
    ) -> None:
        text = response.strip()
        if not text:
            return
        compact = text.replace("\r", "").replace("\n", "").replace(" ", "")
        if source == "carecv_response" and compact in {"+CARECV:0OK", "+CARECV:0,0OK"}:
            return
        markers = ("+CADATAIND", "+CARECV", "+CAURC", "+CASTATE", "{")
        if not any(marker in text for marker in markers):
            return
        item = {
            "command": "<tcp_round_ack_unsolicited>",
            "source": source,
            "response_preview": text[:1000],
            "expected_message_ids": sorted(expected_message_ids),
        }
        if poll_count is not None:
            item["poll"] = poll_count
        self.diagnostics.append(item)

    def _record_tcp_receive_error_state(
        self,
        poll_count: int,
        carecv_response: str,
        expected_message_ids: set[str],
    ) -> None:
        state = {}
        for command in ("AT+CASTATE?", "AT+CNACT?", "AT+CACFG?"):
            try:
                state[command] = self.command(
                    command,
                    timeout_seconds=2.0,
                    expected=("OK", "ERROR"),
                    raise_on_error=False,
                    reset_input=False,
                ).strip()
            except Exception as exc:
                state[command] = f"{type(exc).__name__}: {exc}"
        _print_tcp_state(f"carecv_error:poll:{poll_count}", state)
        self.diagnostics.append(
            {
                "command": "<tcp_round_ack_receive_error_state>",
                "poll": poll_count,
                "carecv_response": carecv_response.strip(),
                "expected_message_ids": sorted(expected_message_ids),
                "state": state,
            }
        )

    def _read_available_text(self, duration_seconds: float) -> str:
        deadline = time.monotonic() + max(0.0, float(duration_seconds))
        chunks = []
        while time.monotonic() < deadline:
            data = self.serial.read(4096)
            if data:
                chunks.append(data.decode("utf-8", errors="replace"))
            else:
                time.sleep(0.02)
        return "".join(chunks)

    def _read_round_ack_line(self) -> str:
        deadline = time.monotonic() + self.application_ack_timeout_seconds
        chunks = []
        while time.monotonic() < deadline:
            data = self.serial.read(4096)
            if data:
                chunks.append(data.decode("utf-8", errors="replace"))
                response = "".join(chunks)
                if "\n" in response:
                    return response
            else:
                time.sleep(0.02)
        return "".join(chunks)

    def _select_round_ack(
        self,
        response: str,
        packet_id: str,
        started: float,
        expected_message_ids: set[str] | None = None,
    ) -> dict | None:
        self._append_tcp_round_ack_payload(response)
        stale_acks = []
        seen_ack_keys = set()
        for text in (response, self.tcp_round_ack_buffer):
            if not text:
                continue
            for ack in self._parse_round_ack_candidates(text, packet_id, started):
                ack_key = (ack.get("round_index"), tuple(sorted((ack.get("messages") or {}).keys())))
                if ack_key in seen_ack_keys:
                    continue
                seen_ack_keys.add(ack_key)
                matches = ack["confirmed"] is not None and self._ack_matches_packet(
                    ack, packet_id, expected_message_ids
                )
                if matches:
                    self.tcp_round_ack_buffer = ""
                    return ack
                if ack["confirmed"] is not None:
                    stale_acks.append(ack)
        for ack in stale_acks:
            self.diagnostics.append(
                {
                    "command": "<stale_round_ack>",
                    "expected_packet_id": packet_id,
                    "expected_message_ids": sorted(expected_message_ids or []),
                    "ack_type": ack.get("type"),
                    "ack_message_ids": sorted((ack.get("messages") or {}).keys()),
                }
            )
        return None

    def _append_tcp_round_ack_payload(self, response: str) -> None:
        chunks = self._extract_carecv_payload_chunks(response)
        if not chunks:
            return
        self.tcp_round_ack_buffer += "".join(chunks)
        if len(self.tcp_round_ack_buffer) > 65536:
            self.tcp_round_ack_buffer = self.tcp_round_ack_buffer[-65536:]
        self.diagnostics.append(
            {
                "command": "<tcp_round_ack_buffer>",
                "appended_bytes": sum(len(chunk.encode("utf-8", errors="replace")) for chunk in chunks),
                "buffer_chars": len(self.tcp_round_ack_buffer),
            }
        )

    @staticmethod
    def _extract_carecv_payload_chunks(response: str) -> list[str]:
        chunks = []
        marker = "+CARECV:"
        cursor = 0
        while True:
            start = response.find(marker, cursor)
            if start < 0:
                break
            comma = response.find(",", start + len(marker))
            if comma < 0:
                cursor = start + len(marker)
                continue
            payload_start = comma + 1
            next_marker = response.find(marker, payload_start)
            end_candidates = [
                index
                for index in (
                    response.find("\r\n\r\nOK", payload_start),
                    response.find("\r\nOK", payload_start),
                    next_marker,
                )
                if index >= 0
            ]
            payload_end = min(end_candidates) if end_candidates else len(response)
            chunk = response[payload_start:payload_end]
            if chunk:
                chunks.append(chunk)
            cursor = payload_end if next_marker < 0 else next_marker
        return chunks

    @staticmethod
    def _ack_matches_packet(ack: dict, packet_id: str, expected_message_ids: set[str] | None = None) -> bool:
        expected_message_ids = expected_message_ids or {message_id_hex_for_packet(packet_id)}
        if not ack_matches_message_ids(ack, expected_message_ids):
            return False
        if ack.get("ack_format") != "summary_v1":
            return True
        match = re.search(r":round:(\d+):round_ack$", str(packet_id))
        return match is not None and int(ack.get("round_index") or -1) == int(match.group(1))

    @staticmethod
    def _parse_round_ack_candidates(response: str, packet_id: str, started: float) -> list[dict]:
        import json

        decoder = json.JSONDecoder()
        decoded_objects = []
        for index, char in enumerate(response):
            if char != "{":
                continue
            try:
                data, _ = decoder.raw_decode(response[index:])
            except json.JSONDecodeError:
                continue
            decoded_objects.append(data)
        candidates = []
        for data in decoded_objects:
            if data.get("ack_type") == "round_ack":
                confirmed = data.get("round_confirmed")
                messages = data.get("messages")
                if confirmed is None and isinstance(messages, dict):
                    confirmed = True
                ack = {
                    "observed": True,
                    "type": "round_ack",
                    "ack_format": data.get("ack_format"),
                    "confirmed": bool(confirmed) if confirmed is not None else None,
                    "complete": data.get("round_complete"),
                    "packet_id": packet_id,
                    "fragment_count": data.get("fragment_count"),
                    "missing_fragment_ranges": data.get("missing_fragment_ranges"),
                    "received_bytes": data.get("received_bytes"),
                    "messages": messages,
                    "round_index": data.get("round_index"),
                    "expected_message_count": data.get("expected_message_count"),
                    "received_message_count": data.get("received_message_count"),
                    "missing_message_count": data.get("missing_message_count"),
                    "missing_message_ids": data.get("missing_message_ids"),
                    "rtt_ms": (time.monotonic() - started) * 1000.0,
                    "sequence": data.get("ack_sequence"),
                    "error": data.get("error"),
                }
                candidates.append(ack)
        return candidates

    def _socket_state_error(self) -> str:
        try:
            state = self.command("AT+CASTATE?", expected=("OK", "ERROR"), raise_on_error=False)
        except Exception as exc:
            return f"send failed; socket state query failed: {exc}"
        return f"send failed; socket state: {state.strip()}"

    def _write_line(self, command: str, reset_input: bool = True) -> None:
        self._raise_if_aborted()
        if reset_input:
            self.serial.reset_input_buffer()
        self.serial.write((command + "\r\n").encode("ascii"))
        self.serial.flush()

    def _read_until(self, timeout_seconds: float, tokens: tuple[str, ...]) -> str:
        deadline = time.monotonic() + timeout_seconds
        chunks = []
        while time.monotonic() < deadline:
            self._raise_if_aborted()
            data = self.serial.read(4096)
            if data:
                chunks.append(data.decode("utf-8", errors="replace"))
                response = "".join(chunks)
                if any(token in response for token in tokens):
                    return response
            else:
                time.sleep(0.02)
        return "".join(chunks)
