from __future__ import annotations

import argparse
import json
from pathlib import Path

from rebuild_video.video import DEFAULT_REBUILD_CONFIG, rebuild_video


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline NB-IoT payload video rebuild")
    parser.add_argument("--config", type=Path, default=DEFAULT_REBUILD_CONFIG, help="rebuild configuration JSON")
    args = parser.parse_args()
    print(
        json.dumps(
            rebuild_video(args.config),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
