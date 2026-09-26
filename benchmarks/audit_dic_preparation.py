"""Compare equivalent node-pixel gathers: serial GPU scan versus host stable sort."""
import json
import time
import jax
import jax.numpy as jnp
import numpy as np
from d2ic.mesh.pixel_sampling import build_node_pixel_dense

def run():
    jax.config.update('jax_enable_x64', True)
    rng = np.random.default_rng(2)
    n, p = 512, 4096
    ids = rng.integers(0, n, (p, 4), dtype=np.int32)
    weights = rng.random((p, 4))
    def sorted_host():
        flat = ids.ravel()
        order = np.argsort(flat, kind='stable')
        degree = np.bincount(flat, minlength=n)
        starts = np.cumsum(degree) - degree
        slots = np.arange(len(flat)) - np.repeat(starts, degree)
        index = np.full((n, degree.max()), -1, dtype=np.int32)
        w = np.zeros(index.shape)
        index[flat[order], slots] = (order // 4).astype(np.int32)
        w[flat[order], slots] = weights.ravel()[order]
        return tuple(jax.device_put(x) for x in (index, w, degree.astype(np.int32)))
    a = build_node_pixel_dense(ids, jnp.asarray(weights), n)
    b = sorted_host()
    jax.block_until_ready((a, b))
    for x, y in zip(a, b):
        np.testing.assert_allclose(x, y)
    def timing(f):
        samples = []
        for _ in range(3):
            start = time.perf_counter()
            jax.block_until_ready(f())
            samples.append(time.perf_counter() - start)
        return float(np.median(samples))
    result = dict(device=str(jax.devices()[0]), pixels=p, nodes=n, incidences=p*4,
                  current_seconds=timing(lambda: build_node_pixel_dense(ids, jnp.asarray(weights), n)),
                  stable_sort_host_and_upload_seconds=timing(sorted_host))
    result['speedup'] = result['current_seconds'] / result['stable_sort_host_and_upload_seconds']
    return result

if __name__ == '__main__':
    print(json.dumps(run(), indent=2))
