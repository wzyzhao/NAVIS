"""Compare vectorized component selection against the original per-label rule."""

import numpy as np
import pytest

from rebuild_video.video_reconstructor import VideoReconstructor


@pytest.mark.parametrize("case", ["white", "black", "components"])
def test_content_alpha_matches_original_component_loop(case: str) -> None:
    import cv2

    decoder = VideoReconstructor.__new__(VideoReconstructor)
    decoder.np, decoder.cv2 = np, cv2
    decoder.roi_black_border_suppression = True
    decoder.roi_black_threshold = 18
    decoder.roi_internal_black_min_area_ratio = 0.015
    decoder.roi_content_mask_dilate_pixels = 2
    decoder.roi_mask_feather_pixels = 5
    mask = np.ones((48, 64), dtype=np.uint8)
    alpha = np.random.default_rng(1).random(mask.shape).astype(np.float32)
    image = np.full((*mask.shape, 3), 255, dtype=np.uint8)
    if case == "black":
        image[:] = 0
    elif case == "components":
        image[0:3, 3:6] = 0  # boundary-connected component
        image[10:12, 10:12] = 0  # small interior detail retained
        image[20:35, 20:35] = 0  # large interior black block suppressed
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dark = ((gray <= 18) & (mask > 0)).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
    invalid = np.zeros_like(mask)
    minimum_area = max(8, int(mask.sum() * 0.015))
    for label in range(1, count):
        x, y, width, height, area = stats[label]
        if x == 0 or y == 0 or x + width >= 64 or y + height >= 48 or area >= minimum_area:
            invalid[labels == label] = 1
    if invalid.any():
        invalid = cv2.dilate(invalid, np.ones((5, 5), dtype=np.uint8), iterations=1)
        soft = cv2.GaussianBlur(invalid * 255, (5, 5), 0).astype(np.float32) / 255.0
        expected = (alpha * (1.0 - soft)).clip(0.0, 1.0)
    else:
        expected = alpha
    actual = decoder._apply_content_validity_alpha(alpha, mask, image)
    np.testing.assert_array_equal(actual, expected)
