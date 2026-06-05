from __future__ import annotations

from mcp_server import build_demo_frame


def test_build_demo_frame_shape() -> None:
    frame = build_demo_frame()
    assert list(frame.columns) == ["x", "y"]
    assert len(frame) == 5
