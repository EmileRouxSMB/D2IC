"""Separate geometric mesh support from eroded photometric observations."""
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from d2ic.io.app_utils import erode_correlation_roi
from d2ic.io.binning import bin_mask
from d2ic.io.config import load_sequence_config
from d2ic.io.result_store import ResultStore
from d2ic.mesh import Mesh
from d2ic.mesh.sampling import build_pixel_assets


def test_euclidean_erosion_preserves_input_and_holes():
    roi = np.ones((11, 11), dtype=bool)
    roi[5, 5] = False
    original = roi.copy()
    photo = erode_correlation_roi(roi, 2)
    np.testing.assert_array_equal(roi, original)
    assert not photo[:2].any() and not photo[:, -2:].any()
    assert not photo[5, 7]  # exactly two pixels from the hole
    assert photo[7, 7]  # sqrt(8)>2: Euclidean, not a square footprint
    np.testing.assert_array_equal(erode_correlation_roi(roi, 0), roi)


def test_erosion_uses_original_pixel_units_after_binning():
    roi = np.ones((10, 10), dtype=bool)
    photo = erode_correlation_roi(roi, 3, image_binning=2)
    assert not photo[0].any()
    assert photo[1, 1]  # distance 4 original pixels, not 2


@pytest.mark.parametrize("radius", [-1, np.nan, np.inf, 20])
def test_invalid_or_empty_erosion_rejected(radius):
    with pytest.raises(ValueError):
        erode_correlation_roi(np.ones((8, 8), bool), radius)


def test_pixel_sampling_eroded_but_mesh_unchanged():
    nodes = jnp.array([[0., 0.], [12., 0.], [12., 12.], [0., 12.]])
    elements = jnp.array([[0, 1, 2, 3]])
    mesh = Mesh(nodes, elements)
    image = jnp.arange(144.).reshape(12, 12)
    roi = np.ones((12, 12), bool)
    full = build_pixel_assets(mesh, image, 1., roi)
    photo = erode_correlation_roi(roi, 2)
    reduced = build_pixel_assets(mesh, image, 1., photo)
    assert len(reduced.pixel_coords_ref) == 64
    assert len(full.pixel_coords_ref) == 144
    xy = np.asarray(reduced.pixel_coords_ref)
    assert photo[np.floor(xy[:, 1]).astype(int), np.floor(xy[:, 0]).astype(int)].all()
    np.testing.assert_array_equal(mesh.nodes_xy, nodes)
    np.testing.assert_array_equal(mesh.elements, elements)


def test_config_default_option_and_validation(tmp_path):
    template = Path(__file__).resolve().parents[1]/"examples/configs/sequential_dic.toml"
    source = template.read_text().replace("roi_erosion_px = 0.0\n", "")
    path = tmp_path/"sequence.toml"
    path.write_text(source)
    assert load_sequence_config(path).roi_erosion_px == 0
    path.write_text(source.replace("[dic]", "[dic]\nroi_erosion_px = 4.5"))
    assert load_sequence_config(path).roi_erosion_px == 4.5
    path.write_text(source.replace("[dic]", "[dic]\nroi_erosion_px = -1"))
    with pytest.raises(ValueError, match="roi_erosion_px"):
        load_sequence_config(path)


def test_result_store_exports_both_masks(tmp_path):
    store = ResultStore(tmp_path/"store")
    roi = np.ones((8, 8), bool)
    photo = erode_correlation_roi(roi, 2)
    store.write_roi_mask(roi, correlation_mask=photo)
    with np.load(store.path/"roi.npz") as archive:
        np.testing.assert_array_equal(archive["roi_mask"], roi)
        np.testing.assert_array_equal(archive["correlation_mask"], photo)


def test_historical_pipeline_meshes_original_roi_then_passes_eroded_mask(tmp_path, monkeypatch):
    from dataclasses import replace
    from PIL import Image
    from d2ic.mesh.assets import make_mesh_assets
    from d2ic.pipelines import sequence_platehole as pipeline

    image = np.arange(144, dtype=np.uint8).reshape(12, 12)
    Image.fromarray(image).save(tmp_path/"ref.tiff")
    Image.fromarray(image).save(tmp_path/"def.tiff")
    roi = np.ones((12, 12), bool)
    Image.fromarray(roi.astype(np.uint8)*255).save(tmp_path/"roi.png")
    template = Path(__file__).resolve().parents[1]/"examples/configs/sequential_dic.toml"
    cfg = replace(load_sequence_config(template), img_dir=tmp_path,
                  ref_image_name="ref.tiff", mask_path=tmp_path/"roi.png",
                  image_paths=(tmp_path/"def.tiff",), out_dir=tmp_path/"out",
                  roi_erosion_px=2, image_binning=1, save_mesh=False,
                  save_npz=False, export_png=False, export_vtu=False,
                  abaqus_comparison_enabled=False)
    mesh = Mesh(jnp.array([[0., 0.], [12., 0.], [12., 12.], [0., 12.]]),
                jnp.array([[0, 1, 2, 3]]))

    def generate(**kwargs):
        np.testing.assert_array_equal(kwargs["mask"], roi)
        return mesh, replace(make_mesh_assets(mesh), roi_mask=roi)

    class StopBeforeSolving(Exception):
        pass

    def capture_batch(**kwargs):
        assets = kwargs["assets"]
        np.testing.assert_array_equal(assets.roi_mask, erode_correlation_roi(roi, 2))
        assert assets.mesh is mesh
        raise StopBeforeSolving

    monkeypatch.setattr(pipeline, "load_sequence_config", lambda _: cfg)
    monkeypatch.setattr(pipeline, "_configure_runtime", lambda _: None)
    monkeypatch.setattr(pipeline, "mask_to_mesh_assets_gmsh", generate)
    monkeypatch.setattr(pipeline, "_save_mesh_overlay_png", lambda *args, **kwargs: None)
    monkeypatch.setattr(pipeline, "BatchMeshBased", capture_batch)
    with pytest.raises(StopBeforeSolving):
        pipeline.run_sequence_from_config(template)


def test_pipeline_clips_photometric_mask_to_binned_image_shape(tmp_path, monkeypatch):
    from dataclasses import replace
    from PIL import Image
    from d2ic.mesh.assets import make_mesh_assets
    from d2ic.pipelines import sequence_platehole as pipeline

    image = np.arange(25, dtype=np.uint8).reshape(5, 5)
    Image.fromarray(image).save(tmp_path / "ref.tiff")
    Image.fromarray(image).save(tmp_path / "def.tiff")
    roi = np.ones((5, 5), dtype=bool)
    Image.fromarray(roi.astype(np.uint8) * 255).save(tmp_path / "roi.png")
    template = Path(__file__).resolve().parents[1] / "examples/configs/sequential_dic.toml"
    cfg = replace(load_sequence_config(template), img_dir=tmp_path,
                  ref_image_name="ref.tiff", mask_path=tmp_path / "roi.png",
                  image_paths=(tmp_path / "def.tiff",), out_dir=tmp_path / "out",
                  image_binning=2, roi_erosion_px=0, save_mesh=False,
                  save_npz=False, export_png=False, export_vtu=False,
                  abaqus_comparison_enabled=False)
    mesh = Mesh(jnp.array([[0., 0.], [6., 0.], [6., 6.], [0., 6.]]),
                jnp.array([[0, 1, 2, 3]]))
    ceil_roi = bin_mask(roi, 2)
    assert ceil_roi.shape == (3, 3)

    def generate(**kwargs):
        np.testing.assert_array_equal(kwargs["mask"], roi)
        return mesh, replace(make_mesh_assets(mesh), roi_mask=ceil_roi)

    class StopBeforeSolving(Exception):
        pass

    def capture_batch(**kwargs):
        assert kwargs["ref_image"].shape == (2, 2)
        observation_mask = kwargs["assets"].roi_mask
        assert observation_mask.shape == kwargs["ref_image"].shape
        np.testing.assert_array_equal(observation_mask, np.ones((2, 2), dtype=bool))
        assert kwargs["assets"].mesh is mesh
        raise StopBeforeSolving

    monkeypatch.setattr(pipeline, "load_sequence_config", lambda _: cfg)
    monkeypatch.setattr(pipeline, "_configure_runtime", lambda _: None)
    monkeypatch.setattr(pipeline, "mask_to_mesh_assets_gmsh", generate)
    monkeypatch.setattr(pipeline, "_save_mesh_overlay_png", lambda *args, **kwargs: None)
    monkeypatch.setattr(pipeline, "BatchMeshBased", capture_batch)
    with pytest.raises(StopBeforeSolving):
        pipeline.run_sequence_from_config(template)
