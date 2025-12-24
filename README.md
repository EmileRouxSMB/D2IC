# D²IC

D²IC (A differentiable framework for full-field kinematic identification) is a Digital Image Correlation (DIC) engine built on top of [JAX](https://github.com/google/jax). It combines fully-jittable pipelines, a high-level Python API to process ROIs on CPU or GPU. This README summarizes how to get started, follow the workflow, and reuse the provided tutorials.

> **Note on the refactor**  
> The repository currently ships two APIs:
> - the original `D2IC.Dic` class (legacy implementation), and
> - the new `d2ic` package, a modular architecture with mask-to-mesh, batch execution, and strain utilities.
> Tutorials and scripts are being migrated progressively. New developments should target `d2ic`.

## Why D²IC?
- **Accelerated pixelwise DIC**: Gauss–Newton/CG written in `jax.numpy`, auto-differentiated gradients, identical CPU/GPU execution.
- **Native JAX preprocessing**: node neighborhoods, pixel→element mapping, and node→pixel CSR structures can run entirely in JAX (`use_jax_precompute=True`) to keep data on device.
- **Robust initialization**: ORB/SIFT detection, RANSAC filters, sparse matching strategies tuned for large motions.
- **Reproducible scripts**: step-by-step tutorials (see `doc/` directory) that export meshes, figures, and field summaries automatically.


## Repository layout
- `d2ic/`: refactored package (mask2mesh, batch runner, solvers, strain).
- `D2IC/`: legacy solver core (class `Dic`, `PixelQuad` helpers, historical utilities).
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
ROI generation scripts rely on `meshio`/`gmsh`. On Debian/Ubuntu systems, install the following packages:
```bash
sudo apt-get install -y gmsh libglu1 libxcursor-dev libxft2 libxinerama1 libfltk1.3-dev libfreetype6-dev libgl1-mesa-dev
```

## Typical workflow (new `d2ic` stack)
1. **Prepare the ROI**: binary mask (`.tif/.bmp`) under `img/<case>/roi.*`.
2. **Generate the mesh + assets**: `mesh, assets = mask_to_mesh_assets(mask=..., element_size_px=...)` and enrich with `make_mesh_assets`.
3. **Instantiate configs**: `InitMotionConfig`, `MeshDICConfig`, `BatchConfig`.
4. **Create solvers/pipelines**: `TranslationZNCCSolver`, `DICInitMotion`, `GlobalCGSolver`, `DICMeshBased`.
5. **Run the batch**: `BatchMeshBased` orchestrates per-frame initialization, solver execution, and warm-start propagation.
6. **Post-process**: outputs already contain nodal displacement and Green–Lagrange strain; export NPZ/PNGs as needed.

See the tutorials in `doc/` for end-to-end examples. The PlateHole script now relies entirely on `d2ic`.

### Legacy workflow (still available)
1. **Prepare the ROI**: binary mask (`.tif/.bmp`) under `img/<case>/roi.*`.
2. **Generate the mesh**: call `D2IC.Mask2Mesh.generate_roi_mesh` via tutorials or your own scripts.
3. **Create the `Dic` object**: `dic = Dic(mesh_path=".../mesh.msh")`.
4. **Precompute pixel data**: `dic.precompute_pixel_data(im_ref, use_jax_precompute=True)` to stay 100% JAX.
5. **Initialize displacement**: `dic.compute_feature_disp_guess` or `compute_feature_disp_guess_big_motion`.
6. **Run global DIC**: `dic.run_dic(...)` (CG with Laplace/Spring regularization) and optionally refine with `dic.run_dic_nodal`.
7. **Post-process**: `dic.compute_green_lagrange_strain_nodes` for nodal F/E, visualize via `DICPlotter`, or export fields.

## Tutorials & scripts
### PlateHole case (refactored pipeline)
```bash
python doc/03_tutorial_platehole_sequence_step_by_step.py
```
Step-by-step processing of the PlateHole experiment with the new `d2ic` batch runner: ROI meshing via `mask_to_mesh_assets`, sequential displacement estimation, Green–Lagrange strain extraction, node-scatter PNG exports, and aggregation of all results into a single `.npz`.

### ButterFly case (legacy flow)
```bash
python doc/04_tutorial_buterFly_sequence_step_by_step.py
```
Identical pipeline applied to the ButterFly DP600 sequence with double precision, GPU-friendly defaults, and tunable parameters (element size, regularization, plotted frames, etc.) declared at the top of the script.

### Legacy helper (deprecated)
Older tutorials may still import `D2IC.app_utils.run_pipeline_sequence`, which performs:
1. image loading (`skimage.io.imread`),
2. ROI meshing (`generate_roi_mesh`),
3. `Dic.precompute_pixel_data(...)` (NumPy by default or JAX if enabled),
4. sparse displacement initialization,
5. the global DIC loop `run_dic_sequence` (CG + Laplace/Spring regularization),
6. post-processing and exports via `DICPlotter`.


## Programmatic usage
### New `d2ic` stack
```python
import numpy as np
from d2ic import (
    mask_to_mesh_assets,
    InitMotionConfig,
    MeshDICConfig,
    BatchConfig,
    DICInitMotion,
    DICMeshBased,
    TranslationZNCCSolver,
    GlobalCGSolver,
    BatchMeshBased,
)
from d2ic.mesh_assets import make_mesh_assets

ref_image = ...  # numpy array (H,W)
def_images = [...]  # list of numpy arrays
mask = ...  # binary ROI (H,W)

mesh, _ = mask_to_mesh_assets(mask=mask, element_size_px=16)
assets = make_mesh_assets(mesh, with_neighbors=True)

init_cfg = InitMotionConfig()
mesh_cfg = MeshDICConfig(max_iters=200, tol=1e-3, reg_strength=0.1)
batch_cfg = BatchConfig()

dic_init = DICInitMotion(init_cfg, TranslationZNCCSolver(init_cfg))
dic_mesh = DICMeshBased(mesh=mesh, solver=GlobalCGSolver(), config=mesh_cfg)

batch = BatchMeshBased(
    ref_image=ref_image,
    assets=assets,
    dic_mesh=dic_mesh,
    batch_config=batch_cfg,
    dic_init=dic_init,
)
results = batch.run(def_images)
u_all = np.stack([np.asarray(r.u_nodal) for r in results.results])
E_all = np.stack([np.asarray(r.strain) for r in results.results])
```

### Legacy `Dic` class
```python
import jax.numpy as jnp
from skimage.io import imread
from D2IC.dic import Dic

dic = Dic(mesh_path="path/to/mesh.msh")
im_ref = imread("img/PlateHole/ohtcfrp_00.tif")
im_def = imread("img/PlateHole/ohtcfrp_10.tif")

dic.precompute_pixel_data(im_ref, use_jax_precompute=True)
disp_guess, extras = dic.compute_feature_disp_guess_big_motion(im_ref, im_def, refine=True)
disp_opt, history = dic.run_dic(
    im_ref,
    im_def,
    disp_guess=disp_guess,
    max_iter=200,
    tol=1e-3,
    reg_type="spring",
    alpha_reg=0.1,
)
F_nodes, E_nodes = dic.compute_green_lagrange_strain_nodes(disp_opt, k_ring=2, gauge_length=200.0)
```

## Contributing
1. Fork + create a feature branch.
2. Implement your changes/tests (keep CPU/GPU parity and JIT-compatibility in mind).
3. Document the impact in your PR (performance, new scripts, API updates).
4. Run `pytest` and, for DIC changes, rerun at least one tutorial pipeline.

## License
D²IC is distributed under the GPLv3 license. See `AUTHORS` for the list of contributors.
