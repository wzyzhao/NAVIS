import pickle
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

try:
    from pytorch_msssim import ms_ssim
except ImportError:  # pragma: no cover
    ms_ssim = None

from elic_jpg_compression.elic_jpg_codec.bundle_io import BundleROI


def pad_to_multiple(tensor: torch.Tensor, multiple: int = 64) -> tuple[torch.Tensor, tuple[int, int]]:
    _, _, height, width = tensor.shape
    pad_h = (multiple - height % multiple) % multiple
    pad_w = (multiple - width % multiple) % multiple
    padded = F.pad(tensor, (0, pad_w, 0, pad_h), mode="replicate")
    return padded, (pad_h, pad_w)


def image_to_tensor(image: np.ndarray) -> torch.Tensor:
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    return tensor.unsqueeze(0)


def tensor_to_bgr_image(tensor: torch.Tensor) -> np.ndarray:
    tensor = tensor.detach().cpu().clamp(0.0, 1.0)
    image = (tensor.squeeze(0).permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


class ELICBackgroundCodec:
    def __init__(self, checkpoint_path: str, device: str = "cpu"):
        self.device = torch.device(device)
        self.model = self._load_model(checkpoint_path)

    def _load_model(self, checkpoint_path: str):
        from compressai.models import Elic2022Chandelier

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)
        model = Elic2022Chandelier()
        state_dict = adapt_chandelier_state_dict(state_dict, model.state_dict())
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        model.update(force=True)
        return model.to(self.device)

    @torch.inference_mode()
    def compress(self, image: np.ndarray) -> dict:
        x = image_to_tensor(image).to(self.device)
        x_padded, (pad_h, pad_w) = pad_to_multiple(x, multiple=64)
        payload = self.model.compress(x_padded)
        return {
            "strings": payload["strings"],
            "shape": payload["shape"],
            "orig_size": list(image.shape[:2]),
            "pad": [pad_h, pad_w],
        }

    @torch.inference_mode()
    def decompress(self, payload: dict) -> np.ndarray:
        out = self.model.decompress(payload["strings"], payload["shape"])
        x_hat = out["x_hat"]
        orig_h, orig_w = payload["orig_size"]
        x_hat = x_hat[:, :, :orig_h, :orig_w]
        return tensor_to_bgr_image(x_hat)


def adapt_chandelier_state_dict(
    state_dict: dict[str, torch.Tensor], model_state_dict: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    if set(state_dict.keys()) == set(model_state_dict.keys()):
        return state_dict

    adapted = {key: value for key, value in state_dict.items() if key in model_state_dict}

    direct_prefixes = {
        "h_a.": "latent_codec.hyper.h_a.",
        "h_s.": "latent_codec.hyper.h_s.",
    }
    entropy_bottleneck_prefixes = {
        "entropy_bottleneck._matrix": "latent_codec.hyper.entropy_bottleneck.matrices.",
        "entropy_bottleneck._bias": "latent_codec.hyper.entropy_bottleneck.biases.",
        "entropy_bottleneck._factor": "latent_codec.hyper.entropy_bottleneck.factors.",
        "entropy_bottleneck.quantiles": "latent_codec.hyper.entropy_bottleneck.quantiles",
        "entropy_bottleneck._offset": "latent_codec.hyper.entropy_bottleneck._offset",
        "entropy_bottleneck._quantized_cdf": "latent_codec.hyper.entropy_bottleneck._quantized_cdf",
        "entropy_bottleneck._cdf_length": "latent_codec.hyper.entropy_bottleneck._cdf_length",
        "entropy_bottleneck.target": "latent_codec.hyper.entropy_bottleneck.target",
        "entropy_bottleneck.likelihood_lower_bound.bound": "latent_codec.hyper.entropy_bottleneck.likelihood_lower_bound.bound",
    }

    for key, value in list(state_dict.items()):
        for old_prefix, new_prefix in direct_prefixes.items():
            if key.startswith(old_prefix):
                adapted[new_prefix + key[len(old_prefix) :]] = value
                break
        for old_prefix, new_prefix in entropy_bottleneck_prefixes.items():
            if key.startswith(old_prefix):
                suffix = key[len(old_prefix) :]
                adapted[new_prefix + suffix] = value
                break
        if key.startswith("cc_transforms."):
            parts = key.split(".")
            index = int(parts[1]) + 1
            remainder = ".".join(parts[2:])
            adapted[f"latent_codec.y.channel_context.y{index}.{remainder}"] = value
        if key.startswith("context_prediction."):
            parts = key.split(".")
            index = int(parts[1])
            remainder = ".".join(parts[2:])
            adapted[f"latent_codec.y.latent_codec.y{index}.context_prediction.{remainder}"] = value
        if key.startswith("ParamAggregation."):
            parts = key.split(".")
            index = int(parts[1])
            remainder = ".".join(parts[2:])
            adapted[f"latent_codec.y.latent_codec.y{index}.entropy_parameters.{remainder}"] = value
        if key.startswith("gaussian_conditional."):
            suffix = key[len("gaussian_conditional.") :]
            for index in range(5):
                adapted[f"latent_codec.y.latent_codec.y{index}.y.gaussian_conditional.{suffix}"] = value

    adapted = {key: adapted[key] for key in model_state_dict.keys() if key in adapted}

    missing = sorted(set(model_state_dict.keys()) - set(adapted.keys()))
    unexpected = sorted(set(adapted.keys()) - set(model_state_dict.keys()))
    if missing or unexpected:
        details = []
        if missing:
            details.append(f"missing={len(missing)} sample={missing[:10]}")
        if unexpected:
            details.append(f"unexpected={len(unexpected)} sample={unexpected[:10]}")
        raise RuntimeError("ELIC checkpoint does not match Elic2022Chandelier after adaptation: " + "; ".join(details))

    return adapted


def encode_jpeg(image: np.ndarray, quality: int) -> bytes:
    ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("JPEG encoding failed.")
    return encoded.tobytes()


def decode_jpeg(blob: bytes) -> np.ndarray:
    arr = np.frombuffer(blob, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError("JPEG decoding failed.")
    return image


def encode_png(image: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("PNG encoding failed.")
    return encoded.tobytes()


def decode_png(blob: bytes) -> np.ndarray:
    arr = np.frombuffer(blob, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError("PNG decoding failed.")
    return image


def serialize_rois(rois: list[BundleROI], jpeg_quality: int) -> list[dict]:
    payload = []
    for roi in rois:
        payload.append(
            {
                "bbox": roi.bbox,
                "roi_bbox": roi.roi_bbox,
                "track_id": roi.track_id,
                "class_id": roi.class_id,
                "priority_score": roi.priority_score,
                "image_jpeg": encode_jpeg(roi.image, jpeg_quality),
                "mask_png": encode_png(roi.mask),
            }
        )
    return payload


def deserialize_rois(payload: list[dict]) -> list[BundleROI]:
    rois = []
    for item in payload:
        rois.append(
            BundleROI(
                bbox=list(item["bbox"]),
                roi_bbox=list(item["roi_bbox"]),
                track_id=int(item["track_id"]),
                class_id=int(item["class_id"]),
                priority_score=float(item["priority_score"]),
                image=decode_jpeg(item["image_jpeg"]),
                mask=decode_png(item["mask_png"]),
            )
        )
    return rois


def compute_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    mse = np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)
    if mse == 0:
        return 100.0
    return float(10 * np.log10(255 * 255 / mse))


def compute_ms_ssim(img1: np.ndarray, img2: np.ndarray) -> float | None:
    if ms_ssim is None:
        return None
    t1 = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    t2 = torch.from_numpy(img2).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    return float(ms_ssim(t1, t2, data_range=1.0).item())


def dump_pickle(path: Path, payload: dict) -> None:
    with open(path, "wb") as f:
        pickle.dump(payload, f)
