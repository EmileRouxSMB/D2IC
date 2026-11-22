"""Regular pixel selection without spatial filtering."""

from __future__ import annotations

import numpy as np


def regular_decimation_mask(
    image_shape: tuple[int, int], factor: int = 1, *, offset: tuple[int, int] = (0, 0)
) -> np.ndarray:
    """Select one observation every ``factor`` pixels along both axes.

    The boolean mask keeps the original image shape: no image value is
    averaged, interpolated or discarded. ``offset`` uses ``(row, column)``.
    """
    if len(image_shape) != 2:
        raise ValueError("image_shape must contain (rows, columns).")
    rows, columns = (int(size) for size in image_shape)
    if rows < 1 or columns < 1:
        raise ValueError("image_shape dimensions must be positive.")
    if isinstance(factor, (bool, np.bool_)) or not isinstance(factor, (int, np.integer)):
        raise TypeError("decimation factor must be an integer.")
    factor = int(factor)
    if factor < 1:
        raise ValueError("decimation factor must be >= 1.")
    if len(offset) != 2:
        raise ValueError("offset must contain (row, column).")
    row_offset, column_offset = (int(value) for value in offset)
    if not (0 <= row_offset < factor and 0 <= column_offset < factor):
        raise ValueError("offset values must lie in [0, factor).")

    mask = np.zeros((rows, columns), dtype=bool)
    mask[row_offset::factor, column_offset::factor] = True
    return mask


__all__ = ["regular_decimation_mask"]
