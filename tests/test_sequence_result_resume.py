import numpy as np

from d2ic.io.result_store import ResultStore
from d2ic.pipelines.sequence_platehole import _restore_comparison_results


def test_resume_restores_prior_frames_for_abaqus_sequence_alignment(tmp_path):
    store = ResultStore(tmp_path / "dic_store")
    store.write_mesh(np.array([[0., 0.], [1., 0.], [0., 1.]]),
                     np.array([[0, 1, 2]], dtype=np.int32))
    for k in range(2):
        store.write_frame(k, u_nodal=np.full((3, 2), k), strain=np.full((3, 3), k))
        store.save_checkpoint(fingerprint="same", u_prev=np.full((3, 2), k),
                              u_prevprev=None, completed_frames=k + 1)
        store.update_progress(status="running", total_frames=3, completed_frames=k + 1,
                              last_frame=k)

    new_result = type("Result", (), {"displacement": np.full((3, 2), 2),
                                     "strain": np.full((3, 3), 2)})()
    results = _restore_comparison_results(store, [new_result], resume_offset=2)

    assert len(results) == 3
    np.testing.assert_array_equal(results[0].displacement, 0)
    np.testing.assert_array_equal(results[1].displacement, 1)
    assert results[2] is new_result
