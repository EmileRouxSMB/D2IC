from __future__ import annotations

import numpy as np
import pytest

from d2ic.io import bin_image, bin_mask, downsample_image, downsample_mask
from d2ic.mesh.generation import downsample_mask as mesh_downsample_mask


def test_bin_image_averages_complete_blocks_and_trims_boundaries() -> None:
    image = np.arange(15, dtype=np.float32).reshape(3, 5)

    result = bin_image(image, 2)

    expected = np.array([[3.0, 5.0]], dtype=np.float32)
    np.testing.assert_allclose(result, expected)


def test_bin_mask_uses_conservative_fe_rule_and_keeps_boundary_blocks() -> None:
    mask = np.zeros((3, 5), dtype=bool)
    mask[1, 1] = True
    mask[2, 4] = True

    result = bin_mask(mask, 2)

    expected = np.array(
        [[True, False, False], [False, False, True]], dtype=bool
    )
    np.testing.assert_array_equal(result, expected)


def test_legacy_fe_and_io_names_share_the_common_implementation() -> None:
    image = np.arange(16).reshape(4, 4)
    mask = image == 0

    np.testing.assert_array_equal(downsample_image(image, 2), bin_image(image, 2))
    np.testing.assert_array_equal(downsample_mask(mask, 2), bin_mask(mask, 2))
    np.testing.assert_array_equal(mesh_downsample_mask(mask, 2), bin_mask(mask, 2))


@pytest.mark.parametrize("factor", [0, -1])
def test_binning_rejects_non_positive_factor(factor: int) -> None:
    with pytest.raises(ValueError, match=">= 1"):
        bin_image(np.ones((2, 2)), factor)


def test_imread_gray_ignores_alpha_channel(monkeypatch, tmp_path):
    """Opaque black pixels must remain black in grayscale+alpha images."""
    from d2ic.io import app_utils

    gray_alpha = np.array([[[0, 255], [255, 255]]], dtype=np.uint8)
    monkeypatch.setattr(app_utils, "_try_imageio", lambda path: gray_alpha)
    monkeypatch.setattr(app_utils, "_try_tifffile", lambda path: None)
    monkeypatch.setattr(app_utils, "_try_matplotlib", lambda path: None)

    gray = app_utils.imread_gray(tmp_path / "roi.tif")

    np.testing.assert_array_equal(gray, np.array([[0.0, 255.0]], dtype=np.float32))
    np.testing.assert_array_equal(app_utils.make_roi_mask(gray), [[False, True]])
