"""Locate project resources kept outside the installable ``src`` tree."""

from __future__ import annotations

import os
from pathlib import Path


def find_project_root() -> Path:
    """Find the NAVIS checkout, or use ``NAVIS_ROOT`` if set."""
    override = os.environ.get("NAVIS_ROOT") or os.environ.get("VIDEO_COMPRESSION_ROOT")
    starts = [Path(override)] if override else []
    starts.extend((Path(__file__).resolve(), Path.cwd()))
    for start in starts:
        path = start.resolve()
        for candidate in (path, *path.parents):
            if (candidate / "pyproject.toml").is_file() and (candidate / "src" / "common").is_dir():
                return candidate
    raise RuntimeError("Cannot locate NAVIS checkout; set NAVIS_ROOT")


PROJECT_ROOT = find_project_root()
