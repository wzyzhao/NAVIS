import argparse
import json
import re
import shutil
import socket
import socketserver
import struct
import threading
import time
import zlib
from pathlib import Path

if __package__:
    from .packetizer import CODE_LAYERS, HEADER_SIZE, parse_envelope_header
else:
    from packetizer import CODE_LAYERS, HEADER_SIZE, parse_envelope_header

PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = PACKAGE_ROOT / "config.json"
DEFAULT_OUTPUT_ROOT = Path("/var/lib/nbiot_receiver")
WINDOW_BUNDLE_MAGIC = b"NBW1"
WINDOW_BUNDLE_VERSION = 1
WINDOW_BUNDLE_ENTRY_STRUCT = struct.Struct("!QBBIIHhhHHIII")


def parse_payload(data: bytes) -> tuple[dict, bytes]:
    header = parse_envelope_header(data)
    payload = data[HEADER_SIZE : HEADER_SIZE + header["payload_size"]]
    header["packet_id"] = _wire_packet_id(header)
    return header, payload


def build_ack(header: dict, payload: bytes, client: str, complete: bool = True, error: str | None = None) -> dict:
    expected_crc = header.get("payload_crc32")
    actual_crc = zlib.crc32(payload) & 0xFFFFFFFF
    confirmed = error is None and (not complete or expected_crc is None or int(expected_crc) == actual_crc)
    return {
        "type": "ack",
        "packet_id": header.get("packet_id"),
        "message_id": header.get("message_id_hex"),
        "confirmed": confirmed,
        "complete": complete,
        "received_bytes": len(payload),
        "payload_crc32": actual_crc,
        "expected_payload_crc32": expected_crc,
        "device_id": header.get("device_id"),
        "frame_idx": header.get("frame_idx"),
        "layer": header.get("layer"),
        "fragment_index": header.get("fragment_index", 0),
        "fragment_count": header.get("fragment_count", 1),
        "client": client,
        "received_at": time.time(),
        "error": error if error is not None else (None if confirmed else "payload_crc32_mismatch"),
    }


def build_round_ack_error(header: dict, payload: bytes, client: str, error: str) -> dict:
    return {
        "ack_type": "round_ack",
        "round_index": None,
        "round_confirmed": False,
        "round_complete": False,
        "messages": {},
        "expected_message_count": 0,
        "received_message_count": 0,
        "missing_message_count": 0,
        "missing_message_ids": [],
        "error": error,
    }


def missing_fragment_ranges(fragment_count: int, received: dict) -> list[list[int]]:
    ranges = []
    start = None
    previous = None
    for idx in range(fragment_count):
        if idx in received:
            continue
        if start is None:
            start = previous = idx
            continue
        if idx == previous + 1:
            previous = idx
            continue
        ranges.append([start, previous])
        start = previous = idx
    if start is not None:
        ranges.append([start, previous])
    return ranges


def parse_window_bundle_payload(outer_header: dict, payload: bytes) -> list[tuple[dict, bytes]]:
    if len(payload) < 7 or payload[:4] != WINDOW_BUNDLE_MAGIC:
        raise ValueError("missing NBW1 magic")
    version = payload[4]
    if version != WINDOW_BUNDLE_VERSION:
        raise ValueError(f"unsupported NBW1 version {version}")
    count = int.from_bytes(payload[5:7], "big")
    table_start = 7
    table_size = count * WINDOW_BUNDLE_ENTRY_STRUCT.size
    payload_offset = table_start + table_size
    if len(payload) < payload_offset:
        raise ValueError("truncated NBW1 table")
    entries = []
    cursor = payload_offset
    for index in range(count):
        entry_offset = table_start + index * WINDOW_BUNDLE_ENTRY_STRUCT.size
        (
            message_id,
            layer_code,
            _flags,
            frame_idx,
            frame_id,
            item_index,
            roi_idx,
            rate_idx,
            fragment_index,
            fragment_count,
            payload_size,
            payload_crc32,
            fragment_crc32,
        ) = WINDOW_BUNDLE_ENTRY_STRUCT.unpack(payload[entry_offset : entry_offset + WINDOW_BUNDLE_ENTRY_STRUCT.size])
        next_cursor = cursor + int(payload_size)
        if next_cursor > len(payload):
            raise ValueError("truncated NBW1 item payload")
        inner_payload = payload[cursor:next_cursor]
        cursor = next_cursor
        layer = CODE_LAYERS.get(int(layer_code), "payload")
        inner_header = {
            "version": outer_header.get("version"),
            "flags": 0,
            "message_id": int(message_id),
            "message_id_hex": f"{int(message_id):016x}",
            "device_id": outer_header.get("device_id"),
            "frame_idx": int(frame_idx),
            "frame_id": int(frame_id),
            "item_index": int(item_index),
            "layer": layer,
            "roi_idx": None if int(roi_idx) < 0 else int(roi_idx),
            "rate_idx": None if int(rate_idx) < 0 else int(rate_idx),
            "fragment_index": int(fragment_index),
            "fragment_count": int(fragment_count),
            "payload_size": int(payload_size),
            "payload_crc32": int(payload_crc32),
            "fragment_crc32": int(fragment_crc32),
            "header_size": HEADER_SIZE,
        }
        inner_header["packet_id"] = _wire_packet_id(inner_header)
        entries.append((inner_header, inner_payload))
    if cursor != len(payload):
        raise ValueError("NBW1 trailing bytes")
    return entries


class AckHandler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        contexts = {}
        state = self.server
        registered_device = None
        try:
            self.request.settimeout(self.server.client_idle_timeout_seconds)
        except OSError:
            pass
        while True:
            close_after_ack = False
            try:
                data = self._receive_packet()
                if not data:
                    self.server.release_device_connection(registered_device, self.request)
                    return
                header, payload = parse_payload(data)
                if self.server.config.get("isolate_sessions", False):
                    device = str(header.get("device_id"))
                    if header.get("layer") == "session_start" or device not in contexts:
                        contexts[device] = self.server.session_context(header, payload)
                    if header.get("layer") == "session_start":
                        self.server.replace_device_connection(device, self.request)
                        registered_device = device
                    state = contexts[device]
                with state.reset_lock:
                    ack = state.accept_fragment(header, payload, self.client_address[0])
            except Exception as exc:
                close_after_ack = True
                ack = {
                    "type": "ack",
                    "confirmed": False,
                    "complete": False,
                    "client": self.client_address[0],
                    "received_at": time.time(),
                    "close_connection": True,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            state.log_ack(ack)
            if self.server.should_send_ack(ack):
                payload = (json.dumps(ack, ensure_ascii=True, separators=(",", ":")) + "\n").encode("utf-8")
                repeat = max(1, int(self.server.config.get("round_ack_repeat_count", 2) or 2))
                for index in range(repeat):
                    self.request.sendall(payload)
                    if index + 1 < repeat:
                        time.sleep(
                            max(0.0, float(self.server.config.get("round_ack_repeat_interval_seconds", 0.05) or 0.05))
                        )
            if close_after_ack or ack.get("close_connection"):
                self.server.release_device_connection(registered_device, self.request)
                return

    def _receive_packet(self) -> bytes:
        header_raw = self._receive_exact(HEADER_SIZE)
        if not header_raw:
            return b""
        header = parse_envelope_header(header_raw)
        payload = self._receive_exact(header["payload_size"])
        if len(payload) != int(header["payload_size"]):
            raise ValueError(f"incomplete payload: expected {header['payload_size']} bytes, got {len(payload)}")
        return header_raw + payload

    def _receive_exact(self, size: int) -> bytes:
        data = bytearray()
        while len(data) < size:
            chunk = self.request.recv(size - len(data))
            if not chunk:
                break
            data.extend(chunk)
        return bytes(data)


class AckServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True
    request_queue_size = 32

    def __init__(self, server_address, handler_class, output_root: Path, config: dict | None = None):
        super().__init__(server_address, handler_class)
        self.output_root = output_root
        self.config = config or {}
        self.reset_lock = threading.RLock()
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.log_path = self.output_root / "tcp_receiver_acks.jsonl"
        self.log_path.write_text("", encoding="utf-8")
        self.manifest_path = self.output_root / "received_payload_manifest.jsonl"
        self.manifest_path.write_text("", encoding="utf-8")
        self.reassembly_root = self.output_root / "reassembled_payloads"
        self.reassembly_root.mkdir(parents=True, exist_ok=True)
        self.fragments = {}
        self.completed = {}
        self.session_id = None
        self.session_contexts = {}
        self.latest_device_context = {}
        self.client_idle_timeout_seconds = max(1.0, float(self.config.get("client_idle_timeout_seconds", 300.0)))
        self.active_device_connections = {}
        self.connection_lock = threading.Lock()

    def replace_device_connection(self, device: str, request) -> None:
        """A resumed node replaces its stale TCP connection, if one remains."""
        with self.connection_lock:
            previous = self.active_device_connections.get(device)
            self.active_device_connections[device] = request
        if previous is None or previous is request:
            return
        try:
            previous.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            previous.close()
        except OSError:
            pass

    def release_device_connection(self, device: str | None, request) -> None:
        if not device:
            return
        with self.connection_lock:
            if self.active_device_connections.get(device) is request:
                self.active_device_connections.pop(device, None)

    def session_context(self, header, payload):
        """Separate ACK, fragment and payload state for every modem/session pair."""
        device = str(header.get("device_id"))
        with self.reset_lock:
            if header.get("layer") != "session_start":
                if device not in self.latest_device_context:
                    raise ValueError("session_start required before node data")
                return self.latest_device_context[device]
            if len(payload) != 20 or not payload.startswith(b"NBS1"):
                raise ValueError("invalid session_start payload")
            session = payload[4:].hex()
            key = (device, session)
            if key not in self.session_contexts:
                # A storage context uses the existing parser/ACK methods, not a socket.
                context = object.__new__(AckServer)
                context.config = self.config
                context.reset_lock = threading.RLock()
                token = re.sub(r"[^A-Za-z0-9_.-]+", "_", device).strip("._") or "unknown_device"
                # Node directories are stable across sessions so that the
                # receiver artifacts are directly available as
                # sessions/node_1 and sessions/node_2.  A new session for the
                # same node replaces only that node's previous artifacts.
                context.output_root = self.output_root / "sessions" / token
                if context.output_root.exists():
                    shutil.rmtree(context.output_root)
                context.output_root.mkdir(parents=True, exist_ok=False)
                context.log_path = context.output_root / "tcp_receiver_acks.jsonl"
                context.manifest_path = context.output_root / "received_payload_manifest.jsonl"
                context.reassembly_root = context.output_root / "reassembled_payloads"
                context.reassembly_root.mkdir()
                context.fragments = {}
                context.completed = {}
                context.session_id = session
                (context.output_root / "session.json").write_text(
                    json.dumps(
                        {
                            "session_id": session,
                            "device_id": device,
                            "created_at": time.time(),
                            "isolate_sessions": True,
                        },
                        ensure_ascii=True,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                self.session_contexts[key] = context
            self.latest_device_context[device] = self.session_contexts[key]
            return self.session_contexts[key]

    def finish_request(self, request, client_address):
        try:
            request.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except OSError:
            pass
        return super().finish_request(request, client_address)

    def reset_session(self, payload: bytes) -> str:
        if len(payload) != 20 or not payload.startswith(b"NBS1"):
            raise ValueError("invalid session_start payload")
        session_id = payload[4:].hex()
        if self.config.get("isolate_sessions", False):
            if self.session_id != session_id:
                raise ValueError("session does not match isolated context")
            return session_id
        with self.reset_lock:
            for child in list(self.output_root.iterdir()):
                if child.is_dir():
                    shutil.rmtree(child)
                else:
                    child.unlink()
            self.output_root.mkdir(parents=True, exist_ok=True)
            self.log_path = self.output_root / "tcp_receiver_acks.jsonl"
            self.log_path.write_text("", encoding="utf-8")
            self.manifest_path = self.output_root / "received_payload_manifest.jsonl"
            self.manifest_path.write_text("", encoding="utf-8")
            self.reassembly_root = self.output_root / "reassembled_payloads"
            self.reassembly_root.mkdir(parents=True, exist_ok=True)
        self.fragments.clear()
        self.completed.clear()
        self.session_id = session_id
        return session_id

    def should_send_ack(self, ack: dict) -> bool:
        return ack.get("ack_type") == "round_ack"

    def log_ack(self, ack: dict) -> None:
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(ack, ensure_ascii=True, separators=(",", ":")) + "\n")

    def accept_fragment(self, header: dict, payload: bytes, client: str) -> dict:
        if header.get("layer") == "session_start":
            session_id = self.reset_session(payload)
            result = build_ack(header, payload, client, complete=True)
            result["session_id"] = session_id
            result["output_cleared"] = not self.config.get("isolate_sessions", False)
            # The sender confirms a session start through the normal round-query
            # path.  Keep this acknowledgement in the same completed-message
            # index as data packets so that query can find it.
            key = (
                str(header.get("device_id", "")),
                str(header.get("frame_idx", "")),
                str(header.get("layer", "")),
                int(header.get("message_id", 0)),
                int(header.get("payload_crc32", 0) or 0),
            )
            if result["confirmed"]:
                self.completed[key] = dict(result)
            return result
        if header.get("layer") == "round_ack":
            return self.build_round_ack(header, payload, client)
        fragment_count = int(header.get("fragment_count", 1) or 1)
        fragment_index = int(header.get("fragment_index", 0) or 0)
        base_packet_id = _base_packet_id(str(header.get("packet_id", "")))
        key = (
            str(header.get("device_id", "")),
            str(header.get("frame_idx", "")),
            str(header.get("layer", "")),
            int(header.get("message_id", 0)),
            int(header.get("payload_crc32", 0) or 0),
        )

        completed_ack = self.completed.get(key)
        if completed_ack is not None:
            ack = dict(completed_ack)
            ack["received_at"] = time.time()
            ack["duplicate"] = True
            return ack

        if fragment_index < 0 or fragment_index >= fragment_count:
            return build_ack(header, payload, client, complete=False, error="fragment_index_out_of_range")
        fragment_crc = header.get("fragment_crc32")
        if fragment_crc is not None and int(fragment_crc) != (zlib.crc32(payload) & 0xFFFFFFFF):
            return build_ack(header, payload, client, complete=False, error="fragment_crc32_mismatch")

        if fragment_count <= 1:
            if header.get("layer") == "window_bundle":
                return self.accept_window_bundle(header, payload, client)
            ack = build_ack(header, payload, client, complete=True)
            if ack["confirmed"]:
                payload_path = self._write_reassembled_payload(header, payload)
                ack["reassembled_path"] = str(payload_path)
                self.completed[key] = dict(ack)
            return ack

        state = self.fragments.setdefault(
            key,
            {
                "header": dict(header),
                "fragment_count": fragment_count,
                "fragments": {},
                "first_seen": time.time(),
            },
        )
        state["fragments"][fragment_index] = payload

        if len(state["fragments"]) < fragment_count:
            ack = build_ack(header, payload, client, complete=False)
            ack["received_fragment_count"] = len(state["fragments"])
            ack["missing_fragment_ranges"] = missing_fragment_ranges(fragment_count, state["fragments"])
            return ack

        reassembled = b"".join(state["fragments"][idx] for idx in range(fragment_count))
        expected_crc = int(header.get("payload_crc32", 0) or 0)
        actual_crc = zlib.crc32(reassembled) & 0xFFFFFFFF
        complete_header = dict(state["header"])
        complete_header["packet_id"] = base_packet_id
        complete_header["fragment_index"] = 0
        complete_header["fragment_count"] = fragment_count
        ack = build_ack(
            complete_header,
            reassembled,
            client,
            complete=True,
            error=None if actual_crc == expected_crc else "reassembled_payload_crc32_mismatch",
        )
        ack["received_fragment_count"] = fragment_count
        if ack["confirmed"]:
            if complete_header.get("layer") == "window_bundle":
                bundle_ack = self.accept_window_bundle(complete_header, reassembled, client)
                self.fragments.pop(key, None)
                self.completed[key] = dict(bundle_ack)
                return bundle_ack
            payload_path = self._write_reassembled_payload(complete_header, reassembled)
            ack["reassembled_path"] = str(payload_path)
            self.fragments.pop(key, None)
            self.completed[key] = dict(ack)
        return ack

    def accept_window_bundle(self, header: dict, payload: bytes, client: str) -> dict:
        try:
            entries = parse_window_bundle_payload(header, payload)
        except Exception as exc:
            return build_ack(header, payload, client, complete=True, error=f"invalid_window_bundle: {exc}")
        inner_acks = []
        for inner_header, inner_payload in entries:
            inner_acks.append(self.accept_fragment(inner_header, inner_payload, client))
        failed = [ack for ack in inner_acks if not ack.get("confirmed")]
        ack = build_ack(
            header,
            payload,
            client,
            complete=True,
            error=None if not failed else "window_bundle_inner_failure",
        )
        ack["bundled_packet_count"] = len(entries)
        ack["bundled_confirmed_count"] = len(entries) - len(failed)
        if failed:
            ack["bundled_failed_message_ids"] = [ack.get("message_id") for ack in failed if ack.get("message_id")]
        return ack

    def build_round_ack(self, header: dict, payload: bytes, client: str) -> dict:
        try:
            request = json.loads(payload.decode("utf-8"))
        except Exception as exc:
            return build_round_ack_error(header, payload, client, f"invalid_round_ack_payload: {exc}")
        if request.get("ack_type") != "round_query":
            return build_round_ack_error(header, payload, client, "unsupported_round_ack_query")
        expected = request.get("expected_message_ids") or []
        completed_by_message_id = {
            str(ack.get("message_id")): ack for ack in self.completed.values() if ack.get("message_id")
        }
        fragments_by_message_id = {}
        for state in self.fragments.values():
            state_header = state.get("header") or {}
            message_id = f"{int(state_header.get('message_id', 0) or 0):016x}"
            fragments_by_message_id[message_id] = state

        messages = {}
        for item in expected:
            message_id = str(item or "").lower()
            completed = completed_by_message_id.get(message_id)
            if completed is not None:
                messages[message_id] = {
                    "confirmed": 1 if completed.get("confirmed") else 0,
                    "complete": 1 if completed.get("complete") else 0,
                    "received_bytes": completed.get("received_bytes"),
                    "fragment_count": completed.get("fragment_count"),
                }
                if completed.get("error"):
                    messages[message_id]["error"] = completed.get("error")
                continue
            state = fragments_by_message_id.get(message_id)
            if state is not None:
                fragment_count = int(state.get("fragment_count", 1) or 1)
                received = state.get("fragments") or {}
                messages[message_id] = {
                    "confirmed": 0,
                    "complete": 0,
                    "received_fragment_count": len(received),
                    "fragment_count": fragment_count,
                    "missing_fragment_ranges": missing_fragment_ranges(fragment_count, received),
                    "error": "incomplete_fragments",
                }
                continue
            messages[message_id] = {
                "confirmed": 0,
                "complete": 0,
                "error": "not_received",
            }

        missing = [mid for mid, item in messages.items() if item.get("confirmed") != 1 or item.get("complete") != 1]
        if not missing:
            # The sender already knows every message ID in its query.  A fixed
            # summary is sufficient on success and keeps NB-IoT downlink ACKs
            # below a single CARECV read even for a large scheduling window.
            return {
                "ack_type": "round_ack",
                "ack_format": "summary_v1",
                "round_index": request.get("round_index"),
                "round_confirmed": True,
                "round_complete": True,
                "expected_message_count": len(messages),
                "received_message_count": len(messages),
                "missing_message_count": 0,
                "missing_message_ids": [],
                "error": None,
            }
        return {
            "ack_type": "round_ack",
            "ack_format": "detail_v1",
            "round_index": request.get("round_index"),
            "round_confirmed": not missing,
            "round_complete": not missing,
            "messages": messages,
            "expected_message_count": len(messages),
            "received_message_count": len(messages) - len(missing),
            "missing_message_count": len(missing),
            "missing_message_ids": missing,
            "error": None if not missing else "missing_messages",
        }

    def _write_reassembled_payload(self, header: dict, payload: bytes) -> Path:
        safe_name = _payload_file_stem(header)
        path = self.reassembly_root / f"{safe_name}.bin"
        path.write_bytes(payload)
        self._append_received_manifest(header, path, payload)
        return path

    def _append_received_manifest(self, header: dict, path: Path, payload: bytes) -> None:
        try:
            relative_path = str(path.resolve().relative_to(self.output_root.resolve())).replace("\\", "/")
        except ValueError:
            relative_path = str(path)
        record = {
            "session_id": self.session_id,
            "packet_id": _base_packet_id(str(header.get("packet_id", ""))),
            "message_id": header.get("message_id_hex"),
            "message_id_hex": header.get("message_id_hex"),
            "device_id": header.get("device_id"),
            "frame_idx": int(header.get("frame_idx", 0) or 0),
            "frame_id": int(header.get("frame_id", header.get("frame_idx", 0)) or 0),
            "item_index": int(header.get("item_index", 0) or 0),
            "layer": header.get("layer"),
            "roi_idx": header.get("roi_idx"),
            "rate_idx": header.get("rate_idx"),
            "fragment_count": int(header.get("fragment_count", 1) or 1),
            "payload_path": relative_path,
            "bytes": len(payload),
            "payload_crc32": zlib.crc32(payload) & 0xFFFFFFFF,
            "ack_confirmed": True,
            "ack_complete": True,
            "received_at": time.time(),
        }
        self._add_metadata_bundle_fields(record, payload)
        with self.manifest_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=True, separators=(",", ":")) + "\n")

    def _add_metadata_bundle_fields(self, record: dict, payload: bytes) -> None:
        if record.get("layer") != "metadata" or not payload.startswith(b"NBM1"):
            return
        try:
            metadata = json.loads(zlib.decompress(payload[4:]).decode("utf-8"))
        except Exception:
            return
        if not isinstance(metadata, dict) or not metadata.get("metadata_bundle"):
            return
        record["metadata_bundle"] = True
        record["frame_start"] = metadata.get("frame_start")
        record["frame_end"] = metadata.get("frame_end")
        record["frame_count"] = metadata.get("frame_count")
        record["bundle_format"] = metadata.get("bundle_format")


def _base_packet_id(packet_id: str) -> str:
    marker = ":fragment:"
    if marker in packet_id:
        return packet_id.split(marker, 1)[0]
    return packet_id


def _wire_packet_id(header: dict) -> str:
    base = _readable_packet_id(header)
    if int(header.get("fragment_count", 1)) <= 1:
        return base
    return f"{base}:fragment:{int(header['fragment_index']) + 1}-of-{header['fragment_count']}"


def _readable_packet_id(header: dict) -> str:
    device_id = str(header.get("device_id") or "node")
    rate_idx = header.get("rate_idx")
    if rate_idx in (None, ""):
        rate_idx = 0
    parts = [
        device_id,
        "wire",
        str(int(rate_idx)),
        str(int(header.get("frame_idx", 0) or 0)),
        str(int(header.get("item_index", 0) or 0)),
        str(header.get("layer") or "payload"),
    ]
    return ":".join(parts)


def _payload_file_stem(header: dict) -> str:
    device_id = _compact_stem_token(header.get("device_id") or "node")
    rate_idx = header.get("rate_idx")
    if rate_idx in (None, ""):
        rate_idx = 0
    frame_idx = int(header.get("frame_idx", 0) or 0)
    layer = str(header.get("layer") or "payload")
    roi_idx = header.get("roi_idx")
    if layer == "roi_patch" and roi_idx not in (None, ""):
        suffix = f"rate{int(rate_idx)}_roi{int(roi_idx):03d}"
    elif layer == "background_keyframe_summary":
        suffix = f"rate{int(rate_idx)}_background"
    else:
        suffix = _compact_stem_token(layer)
    return f"{device_id}_frame{frame_idx}_{suffix}"


def _compact_stem_token(value) -> str:
    text = str(value or "")
    token = "".join(ch for ch in text if ch.isalnum())
    return token or "x"


def load_receiver_config(config_path: Path = DEFAULT_CONFIG) -> dict:
    if not config_path.exists():
        return {"host": "0.0.0.0", "port": 9000, "output_root": DEFAULT_OUTPUT_ROOT}
    config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    output_root = Path(config.get("receiver_output_root") or config.get("output_root") or DEFAULT_OUTPUT_ROOT)
    if not output_root.is_absolute():
        output_root = PACKAGE_ROOT / output_root
    return {
        "host": config.get("receiver_host") or config.get("host") or "0.0.0.0",
        "port": int(config.get("receiver_port") or config.get("port") or config.get("remote_port") or 9000),
        "output_root": output_root,
        "round_ack_repeat_count": int(config.get("round_ack_repeat_count", 2) or 2),
        "round_ack_repeat_interval_seconds": float(config.get("round_ack_repeat_interval_seconds", 0.05) or 0.05),
        "isolate_sessions": bool(config.get("isolate_sessions", False)),
        "client_idle_timeout_seconds": float(config.get("client_idle_timeout_seconds", 300.0) or 300.0),
    }


def run_receiver(config_path: Path | None = None) -> None:
    config = load_receiver_config(config_path or DEFAULT_CONFIG)
    with AckServer((config["host"], config["port"]), AckHandler, config["output_root"], config) as server:
        print(f"tcp_receiver listening on {config['host']}:{config['port']}; ACK log: {server.log_path}")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("tcp_receiver stopping")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the NBT1 ECS receiver for one or more devices.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Receiver configuration JSON")
    args = parser.parse_args()
    run_receiver(args.config)


if __name__ == "__main__":
    main()
