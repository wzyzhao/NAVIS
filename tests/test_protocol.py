"""Verify compressed side-information contracts independently of model weights."""

import struct
import zlib

import numpy as np
import pytest

from dcvc_compression.dcvc_codec.side_information import decode_binary_mask, encode_binary_mask, encode_metadata
from nbiot_transport.core.packetizer import HEADER_SIZE, envelope_payload, fragment_payload
from rebuild_video.side_info_codec import decode_metadata
from deploy.ecs_receiver.packetizer import parse_envelope_header


def test_sender_metadata_is_accepted_by_rebuilder() -> None:
    metadata = {"frame_idx": 7, "rois": [{"track_id": 3}], "sequence": "测试"}
    assert decode_metadata(encode_metadata(metadata)) == metadata


@pytest.mark.parametrize("shape", [(0, 3), (2, 3), (50, 70)])
def test_binary_mask_roundtrip(shape: tuple[int, int]) -> None:
    mask = np.random.default_rng(7).integers(0, 2, shape, dtype=np.uint8) * 255
    np.testing.assert_array_equal(decode_binary_mask(encode_binary_mask(mask)), mask)


def test_metadata_rejects_wrong_magic() -> None:
    with pytest.raises(ValueError, match="NBM1"):
        decode_metadata(b"invalid")


@pytest.mark.parametrize("runs", [b"\x02\x06\x00\x00\x00", b"\x00\x01", b"\x00\x07\x00\x00\x00"])
def test_mask_rejects_invalid_values_truncation_and_length(runs: bytes) -> None:
    payload = b"NBK1" + struct.pack("<II", 3, 2) + zlib.compress(runs)
    with pytest.raises(ValueError):
        decode_binary_mask(payload)


def test_sender_fragments_are_readable_by_standalone_receiver() -> None:
    payload = bytes(range(256)) * 3
    header = {
        "packet_id": "node1:frame:7",
        "device_id": "node1",
        "frame_idx": 7,
        "frame_id": 7,
        "layer": "roi_patch",
        "payload_crc32": zlib.crc32(payload),
    }
    restored = []
    for packet_id, fragment_header, fragment, index, count in fragment_payload(
        header["packet_id"], header, payload, 100
    ):
        fragment_header["packet_id"] = packet_id
        wire = envelope_payload(fragment_header, fragment)
        decoded = parse_envelope_header(wire)
        assert decoded["fragment_index"] == index
        assert decoded["fragment_count"] == count
        assert decoded["payload_crc32"] == zlib.crc32(payload)
        restored.append(wire[HEADER_SIZE:])
    assert b"".join(restored) == payload
