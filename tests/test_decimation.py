from __future__ import annotations

import numpy as np
import pytest

from d2ic.io import regular_decimation_mask


def test_regular_decimation_keeps_original_grid_and_one_pixel_per_block() -> None:
    mask = regular_decimation_mask((5, 7), factor=2)
    assert mask.shape == (5, 7)
    expected = np.zeros((5, 7), dtype=bool)
    expected[::2, ::2] = True
    np.testing.assert_array_equal(mask, expected)


def test_regular_decimation_offset_selects_requested_lattice() -> None:
    mask = regular_decimation_mask((6, 8), factor=4, offset=(1, 3))
    expected = np.zeros((6, 8), dtype=bool)
    expected[1::4, 3::4] = True
    np.testing.assert_array_equal(mask, expected)


@pytest.mark.parametrize("factor", [0, -1])
def test_regular_decimation_rejects_non_positive_factor(factor: int) -> None:
    with pytest.raises(ValueError, match=">= 1"):
        regular_decimation_mask((4, 4), factor)
