"""Common block-binning operators for DIC images and ROI masks."""

from __future__ import annotations

import numpy as np


def bin_image(image: np.ndarray, factor: int = 1) -> np.ndarray:
    """Average non-overlapping pixel blocks of a 2D image.

    Trailing rows and columns that do not form a complete block are discarded,
    as in the historical FE pipeline.
    """
    factor = _validate_factor(factor)
    image = np.asarray(image)
    if image.ndim != 2:
        raise ValueError(f"image must be 2D, got shape {image.shape}.")
    if factor == 1:
        return image
    h, w = image.shape
    new_h, new_w = h // factor, w // factor
    if new_h == 0 or new_w == 0:
        raise ValueError("Binning factor too large for the input image size.")
    blocks = image[: new_h * factor, : new_w * factor].reshape(
        new_h, factor, new_w, factor
    )
    return blocks.mean(axis=(1, 3))


def bin_mask(mask: np.ndarray, factor: int = 1) -> np.ndarray:
    """Bin a 2D ROI mask using the conservative FE occupancy rule.

    A binned pixel belongs to the ROI when any source pixel in its block
    belongs to it. Incomplete boundary blocks are padded with ``False`` so the
    complete spatial support of the input mask is retained.
    """
    factor = _validate_factor(factor)
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2D, got shape {mask.shape}.")
    if factor == 1:
        return mask
    h, w = mask.shape
    pad_h, pad_w = (-h) % factor, (-w) % factor
    if pad_h or pad_w:
        mask = np.pad(
            mask, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=False
        )
    new_h, new_w = mask.shape[0] // factor, mask.shape[1] // factor
    blocks = mask.reshape(new_h, factor, new_w, factor)
    return np.any(blocks, axis=(1, 3))


def _validate_factor(factor: int) -> int:
    """Return a validated integer binning factor."""
    if isinstance(factor, (bool, np.bool_)) or not isinstance(factor, (int, np.integer)):
        raise TypeError("binning factor must be an integer.")
    factor = int(factor)
    if factor < 1:
        raise ValueError("binning factor must be >= 1.")
    return factor


# Compatibility names used by existing scripts.
downsample_image = bin_image
downsample_mask = bin_mask

__all__ = ["bin_image", "bin_mask", "downsample_image", "downsample_mask"]
