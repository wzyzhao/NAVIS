from __future__ import annotations

import argparse
import faulthandler
import io
import json
import sys
import traceback
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image

from common.paths import PROJECT_ROOT

ROOT = PROJECT_ROOT / "src" / "dcvc_compression"
sys.path.insert(0, str(ROOT / "vendor" / "DCVC"))
from src.models.image_model import DMCI
from src.utils.common import get_state_dict, set_torch_env
from src.utils.stream_helper import NalType, SPSHelper, read_header, read_ip_remaining, read_sps_remaining
from src.utils.transforms import ycbcr2rgb

faulthandler.enable(file=sys.stderr, all_threads=True)


class Decoder:
    """Persistent intra-frame decoder reused across JSON-lines requests."""

    def __init__(self, mi: Path, mp: Path, device: str, zero: float) -> None:
        set_torch_env()
        device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
        self.device, self.zero = torch.device(device), zero
        self.i = DMCI()
        self.i.load_state_dict(get_state_dict(str(mi)))
        self.i = self.i.to(self.device).eval()
        self.i.update(zero)
        if self.device.type == "cuda":
            self.i.half()

    def rgb(self, x: torch.Tensor) -> NDArray[np.uint8]:
        """Convert batched YCbCr model output to one HWC RGB uint8 image."""
        x = torch.clamp(ycbcr2rgb(x) * 255, 0, 255).round().to(torch.uint8)
        return x.squeeze(0).permute(1, 2, 0).cpu().numpy()

    def roi(self, data: bytes) -> NDArray[np.uint8]:
        """Decode the last intra frame, rejecting predicted or empty payloads."""
        f, h, x = io.BytesIO(data), SPSHelper(), None
        with torch.no_grad():
            while f.tell() < len(data):
                n = read_header(f)
                if n["nal_type"] == NalType.NAL_SPS:
                    h.add_sps_by_id(read_sps_remaining(f, n["sps_id"]))
                    continue
                if n["nal_type"] != NalType.NAL_I:
                    raise ValueError("ROI is not intra-coded")
                s = h.get_sps_by_id(n["sps_id"])
                q, b = read_ip_remaining(f)
                x = self.i.decompress(b, s, q)["x_hat"][:, :, : s["height"], : s["width"]]
        if x is None:
            raise ValueError("no ROI frame")
        return self.rgb(x)


def main() -> None:
    """Serve ordered decode requests on stdin and flush one JSON reply per line."""
    a = argparse.ArgumentParser()
    a.add_argument("--model-path-i", type=Path, required=True)
    a.add_argument("--model-path-p", type=Path, required=True)
    a.add_argument("--device", default="auto")
    a.add_argument("--force-zero-thres", type=float, default=0.12)
    a = a.parse_args()
    d = Decoder(a.model_path_i, a.model_path_p, a.device, a.force_zero_thres)
    print(json.dumps({"ready": True, "device": str(d.device)}), flush=True)
    for line in sys.stdin:
        r = json.loads(line)
        out = {"request_id": r.get("request_id"), "ok": False}
        try:
            data = Path(r["input_path"]).read_bytes()
            if r["layer"] not in {"roi_patch", "background_keyframe_summary"}:
                raise ValueError(
                    "stream decoder worker only supports roi_patch and background_keyframe_summary DCVC intra payloads"
                )
            x = d.roi(data)
            p = Path(r["output_path"])
            p.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(x).save(p)
            out.update(ok=True, output_path=str(p), width=x.shape[1], height=x.shape[0])
        except Exception as e:
            traceback.print_exc(file=sys.stderr)
            frame = r.get("frame_idx")
            prefix = f"frame {frame}: " if frame is not None else ""
            out["error"] = f"{prefix}{type(e).__name__}: {e}"
        print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
