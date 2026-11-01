import numpy as np
import pytest

from d2ic.io.result_store import ResultStore, open_result_store


def test_versioned_result_store_roundtrip_and_npz_compatibility(tmp_path):
    store = ResultStore(tmp_path / "dic_store", metadata={"def_paths": ["a.tif", "b.tif"]})
    nodes = np.array([[0.0, 0.0], [1.0, 0.0]])
    elements = np.array([[0, 1, 0]], dtype=np.int32)
    store.write_mesh(nodes, elements)
    for k in range(2):
        store.write_frame(k, u_nodal=np.full((2, 2), k), strain=np.full((2, 3), k))
        store.save_checkpoint(fingerprint="same", u_prev=np.full((2, 2), k),
                              u_prevprev=None, completed_frames=k + 1)
        store.update_progress(status="running", total_frames=2, completed_frames=k + 1,
                              last_frame=k)
    assert store.read_node_history(1).shape == (2, 2)
    assert store.progress["completed_frames"] == 2
    target = tmp_path / "dic_results.npz"
    store.export_npz(target)
    legacy = open_result_store(target)
    np.testing.assert_array_equal(legacy.read_history(), store.read_history())
    assert legacy.frame_count == 2


def test_result_store_validates_field_shapes_and_checkpoint_fingerprint(tmp_path):
    store = ResultStore(tmp_path / "store")
    store.write_mesh(np.array([[0., 0.], [1., 0.], [0., 1.]]),
                      np.array([[0, 1, 2]], dtype=np.int32))
    with pytest.raises(ValueError, match="strain"):
        store.write_frame(0, u_nodal=np.zeros((3, 2)), strain=np.zeros((2, 3)))
    store.save_checkpoint(fingerprint="source-a", u_prev=np.ones((3, 2)),
                          u_prevprev=None, completed_frames=1)
    assert store.load_checkpoint("source-b") is None
    assert store.load_checkpoint("source-a")["completed_frames"] == 1


def test_uncheckpointed_frame_is_not_visible(tmp_path):
    store = ResultStore(tmp_path / "store")
    store.write_frame(0, u_nodal=np.zeros((2, 2)), strain=np.zeros((2, 3)))
    store.save_checkpoint(fingerprint="same", u_prev=np.zeros((2, 2)),
                          u_prevprev=None, completed_frames=0)
    store.update_progress(status="running", total_frames=1, completed_frames=0)
    assert store.frame_count == 0


def test_resume_refuses_regenerated_mesh_with_changed_node_order(tmp_path):
    store = ResultStore(tmp_path / "store")
    nodes = np.array([[0., 0.], [1., 0.], [0., 1.]])
    elements = np.array([[0, 1, 2]], dtype=np.int32)
    store.write_mesh(nodes, elements)
    store.save_checkpoint(fingerprint="same", u_prev=np.zeros((3, 2)),
                          u_prevprev=None, completed_frames=1)

    changed_nodes = nodes[[1, 0, 2]]
    changed_elements = np.array([[1, 0, 2]], dtype=np.int32)
    with pytest.raises(ValueError, match="Regenerated mesh differs"):
        store.write_mesh(changed_nodes, changed_elements)

    np.testing.assert_array_equal(store.read_mesh()["nodes_xy"], nodes)
    assert store.load_checkpoint("same")["completed_frames"] == 1


def test_interrupted_frame_export_is_not_committed_for_resume(tmp_path):
    store_dir = tmp_path / "store"
    store = ResultStore(store_dir)
    store.write_mesh(np.array([[0., 0.], [1., 0.], [0., 1.]]),
                     np.array([[0, 1, 2]], dtype=np.int32))
    # Simulate interruption after a field file was written but before VTU/PNG
    # exports and the checkpoint/progress commit.
    store.write_frame(0, u_nodal=np.zeros((3, 2)), strain=np.zeros((3, 3)))

    reopened = ResultStore(store_dir)
    assert reopened.frame_count == 0
    assert reopened.load_checkpoint("same") is None


def test_npz_export_preserves_previous_target_if_write_fails(tmp_path, monkeypatch):
    store = ResultStore(tmp_path / "store")
    store.write_mesh(np.array([[0., 0.], [1., 0.], [0., 1.]]),
                     np.array([[0, 1, 2]], dtype=np.int32))
    store.write_frame(0, u_nodal=np.zeros((3, 2)), strain=np.zeros((3, 3)))
    store.save_checkpoint(fingerprint="same", u_prev=np.zeros((3, 2)),
                          u_prevprev=None, completed_frames=1)
    store.update_progress(status="running", total_frames=1, completed_frames=1)
    target = tmp_path / "dic_results.npz"
    target.write_bytes(b"previous complete export")

    def fail_export(*args, **kwargs):
        raise OSError("simulated disk failure")

    monkeypatch.setattr(np, "savez_compressed", fail_export)
    with pytest.raises(OSError, match="simulated disk failure"):
        store.export_npz(target)
    assert target.read_bytes() == b"previous complete export"
