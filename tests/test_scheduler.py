"""Mandatory-budget behavior must survive general source modernization."""

from types import SimpleNamespace

import pytest

from nbiot_transport.core.packetizer import Packet
from nbiot_transport.core.scheduler import schedule_with_mandatory_reservation


def test_only_mandatory_messages_can_exceed_nominal_budget() -> None:
    metadata = Packet("metadata", 1, 1, "metadata", b"m" * 20)
    background = Packet(
        "background", 1, 1, "background_keyframe_summary", b"b" * 30, metadata={"background": {"mode": "required"}}
    )
    roi = Packet("roi", 1, 1, "roi_patch", b"r" * 10)
    selected, summary = schedule_with_mandatory_reservation(
        [roi, background, metadata],
        SimpleNamespace(max_quota_bytes=60),
        40,
    )
    assert selected == [metadata, background]
    assert summary["mandatory_overflow_bytes"] == 10
    assert summary["optional_quota_bytes"] == 0


def test_mandatory_overflow_cannot_exceed_hard_limit() -> None:
    packet = Packet("metadata", 1, 1, "metadata", b"m" * 61)
    with pytest.raises(ValueError, match="hard limit"):
        schedule_with_mandatory_reservation([packet], SimpleNamespace(max_quota_bytes=60), 40)
