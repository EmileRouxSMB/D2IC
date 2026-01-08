import numpy as np

from d2ic.pixel_assets import build_pixel_to_element_mapping_numpy


def test_build_pixel_to_element_mapping_numpy_quads_with_chunking():
    # Simple 1x1 quad covering [0, 1]x[0, 1]
    nodes = np.array(
        [
            [0.0, 0.0],  # 0
            [1.0, 0.0],  # 1
            [1.0, 1.0],  # 2
            [0.0, 1.0],  # 3
        ],
        dtype=float,
    )
    elements = np.array([[0, 1, 2, 3]], dtype=np.int32)

    # Pixels: two inside, one outside (forces padding when chunk_size=2)
    pixel_coords = np.array(
        [
            [0.5, 0.5],  # inside
            [0.1, 0.9],  # inside
            [1.5, 1.5],  # outside
        ],
        dtype=float,
    )

    pixel_elts, pixel_nodes = build_pixel_to_element_mapping_numpy(
        pixel_coords, nodes, elements, chunk_size=2
    )

    assert pixel_elts.tolist() == [0, 0, -1]
    assert pixel_nodes.shape == (3, 4)
    assert pixel_nodes[0].tolist() == [0, 1, 2, 3]
    assert pixel_nodes[2].tolist() == [0, 0, 0, 0]
