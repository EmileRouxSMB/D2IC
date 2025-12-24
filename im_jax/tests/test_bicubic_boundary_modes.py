import jax.numpy as jnp
import pytest

from im_jax import flat_nd_cubic_interpolate


def _reflect_index(idx, size):
    if size <= 1:
        return 0
    period = 2 * size - 2
    idx = idx % period
    if idx >= size:
        idx = period - idx
    return idx


def _wrap_index(idx, size):
    if size <= 0:
        return 0
    return idx % size


def _map_index(idx, size, mode):
    if mode == "reflect":
        return _reflect_index(idx, size)
    if mode == "nearest":
        return max(0, min(size - 1, idx))
    if mode == "wrap":
        return _wrap_index(idx, size)
    if mode == "constant":
        if idx < 0 or idx >= size:
            return None
        return idx
    raise ValueError(f"Unsupported mode: {mode}")


@pytest.mark.parametrize("mode", ["reflect", "nearest", "wrap", "constant"])
def test_boundary_modes_exact_values(mode):
    h, w = 4, 5
    xs = jnp.arange(h)[:, None]
    ys = jnp.arange(w)[None, :]
    image = xs + 10 * ys
    cval = -3.25

    locs = jnp.array(
        [
            [-1.0, 0.0, 3.0, 4.0, 2.0],
            [0.0, -2.0, 4.0, 1.0, 6.0],
        ]
    )
    out = flat_nd_cubic_interpolate(image, locs, mode=mode, cval=cval)

    expected = []
    for x, y in zip(locs[0].tolist(), locs[1].tolist()):
        xi = _map_index(int(x), h, mode)
        yi = _map_index(int(y), w, mode)
        if xi is None or yi is None:
            expected.append(cval)
        else:
            expected.append(float(xi + 10 * yi))

    expected = jnp.array(expected, dtype=out.dtype)
    assert jnp.allclose(out, expected, atol=0.0)
