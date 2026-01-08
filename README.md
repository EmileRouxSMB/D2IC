# D²IC

D²IC (A differentiable framework for full-field kinematic identification) is a Digital Image Correlation (DIC) engine built on top of [JAX](https://github.com/google/jax). It combines fully-jittable pipelines, a high-level Python API to process ROIs on CPU or GPU. This README summarizes how to get started, follow the workflow, and reuse the provided tutorials.

The codebase ships a single API: the `d2ic` package, a modular architecture with
mask-to-mesh, batch execution, and strain utilities.

## Why D²IC?
- **Accelerated pixelwise DIC**: Gauss–Newton/CG written in `jax.numpy`, auto-differentiated gradients, identical CPU/GPU execution.
- **Native JAX preprocessing**: node neighborhoods, pixel→element mapping, and node→pixel CSR structures can run entirely in JAX (`use_jax_precompute=True`) to keep data on device.
- **Robust initialization**: ORB/SIFT detection, RANSAC filters, sparse matching strategies tuned for large motions.
- **Reproducible scripts**: step-by-step tutorials (see `doc/` directory) that export meshes, figures, and field summaries automatically.


## Repository layout
- `d2ic/`: package (mask2mesh, batch runner, solvers, strain).
- `doc/`: scripted tutorials, notebooks, example outputs.
- `img/`: demo datasets (PlateHole, ButterFly, ...).

## Installation
```bash
git clone https://github.com/EmileRouxSMB/D2IC.git
cd D2IC
python -m venv .venv
source .venv/bin/activate  # on Windows: .\.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

### GPU support (optional)
Install the JAX wheel matching your CUDA toolkit:
```bash
pip install "jax[cuda12]"
```
See the [official docs](https://jax.readthedocs.io/en/latest/installation.html) for other CUDA/cuDNN pairs.

### System dependencies (meshing)
On Debian/Ubuntu systems, the `gmsh` Python wheel also requires `libGLU`:
```bash
sudo apt-get install -y libglu1-mesa
```

## Typical workflow
1. **Prepare the ROI**: binary mask (`.tif/.bmp`) under `img/<case>/roi.*`.
2. **Generate the mesh + assets**: `mesh, assets = mask_to_mesh_assets(mask=..., element_size_px=...)` and enrich with `make_mesh_assets`.
3. **Instantiate configs**: `MeshDICConfig`, `BatchConfig`.
4. **Create solvers/pipelines**: `GlobalCGSolver`, `DICMeshBased`.
5. **Run the batch**: `BatchMeshBased` orchestre l'execution par frame et la propagation du warm-start.
6. **Post-process**: outputs already contain nodal displacement and Green–Lagrange strain; export NPZ/PNGs as needed.

See the tutorials in `doc/` for end-to-end examples.

## Tutorials & scripts
### PlateHole case
```bash
python doc/03_tutorial_platehole_sequence_step_by_step.py
```
Step-by-step processing of the PlateHole experiment: ROI meshing via `mask_to_mesh_assets`, sequential displacement estimation, Green–Lagrange strain extraction, node-scatter PNG exports, and aggregation of all results into a single `.npz`.

## Programmatic usage
### `d2ic` API
```python
import numpy as np
from d2ic import (
    mask_to_mesh_assets,
    MeshDICConfig,
    BatchConfig,
    DICMeshBased,
    GlobalCGSolver,
    BatchMeshBased,
)
from d2ic.mesh_assets import make_mesh_assets

ref_image = ...  # numpy array (H,W)
def_images = [...]  # list of numpy arrays
mask = ...  # binary ROI (H,W)

mesh, _ = mask_to_mesh_assets(mask=mask, element_size_px=16)
assets = make_mesh_assets(mesh, with_neighbors=True)

mesh_cfg = MeshDICConfig(max_iters=200, tol=1e-3, reg_strength=0.1)
batch_cfg = BatchConfig()

dic_mesh = DICMeshBased(mesh=mesh, solver=GlobalCGSolver(), config=mesh_cfg)

batch = BatchMeshBased(
    ref_image=ref_image,
    assets=assets,
    dic_mesh=dic_mesh,
    batch_config=batch_cfg,
)
results = batch.run(def_images)
u_all = np.stack([np.asarray(r.u_nodal) for r in results.results])
E_all = np.stack([np.asarray(r.strain) for r in results.results])
```

## Contributing
1. Fork + create a feature branch.
2. Implement your changes/tests (keep CPU/GPU parity and JIT-compatibility in mind).
3. Document the impact in your PR (performance, scripts, API updates).
4. Run `pytest` and, for DIC changes, rerun at least one tutorial pipeline.

## License
D²IC is distributed under the GPLv3 license. See `AUTHORS` for the list of contributors.
