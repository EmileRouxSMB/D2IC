# D²IC

D²IC (A differentiable framework for full-field kinematic identification) is a Digital Image Correlation (DIC) engine built on top of [JAX](https://github.com/google/jax). It combines fully-jittable pipelines, a high-level Python API to process ROIs on CPU or GPU. This README summarizes how to get started, follow the workflow, and reuse the provided tutorials.

## Why D²IC?
- **Accelerated pixelwise DIC**: Gauss–Newton/CG written in `jax.numpy`, auto-differentiated gradients, identical CPU/GPU execution.
- **Native JAX preprocessing**: node neighborhoods, pixel→element mapping, and node→pixel CSR structures can run entirely in JAX (`use_jax_precompute=True`) to keep data on device.
- **Robust initialization**: ORB/SIFT detection, RANSAC filters, sparse matching strategies tuned for large motions.
- **Reproducible scripts**: step-by-step tutorials (`doc/03_tutorial_platehole_sequence_step_by_step.py`, `doc/04_tutorial_buterFly_sequence_step_by_step.py`) that export meshes, figures, and field summaries automatically.


## Repository layout
- `D2IC/`: solver core (class `Dic`, `PixelQuad` helpers, pipeline utilities).
- `doc/`: scripted tutorials, notebooks, example outputs.
- `img/`: demo datasets (PlateHole, ButterFly, ...).

## Installation
```bash
git clone https://github.com/<your-account>/D2IC.git
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
ROI generation scripts rely on `meshio`/`gmsh`. On Debian/Ubuntu:
```bash
sudo apt-get install -y gmsh libglu1 libxcursor-dev libxft2 libxinerama1 libfltk1.3-dev libfreetype6-dev libgl1-mesa-dev
```

## Typical workflow
1. **Prepare the ROI**: binary mask (`.tif/.bmp`) under `img/<case>/roi.*`.
2. **Generate the mesh**: call `D2IC.Mask2Mesh.generate_roi_mesh` via tutorials or your own scripts.
3. **Create the `Dic` object**: `dic = Dic(mesh_path=".../mesh.msh")`.
4. **Precompute pixel data**: `dic.precompute_pixel_data(im_ref, use_jax_precompute=True)` to stay 100% JAX.
5. **Initialize displacement**: `dic.compute_feature_disp_guess` or `compute_feature_disp_guess_big_motion`.
6. **Run global DIC**: `dic.run_dic(...)` (CG with Laplace/Spring regularization) and optionally refine with `dic.run_dic_nodal`.
7. **Post-process**: `dic.compute_green_lagrange_strain_nodes` for nodal F/E, visualize via `DICPlotter`, or export fields.

## Tutorials & scripts
### PlateHole case
```bash
python doc/03_tutorial_platehole_sequence_step_by_step.py
```
Step-by-step processing of the PlateHole experiment: ROI meshing from `img/PlateHole/roi.tif`, sequential displacement estimation over the full series, PNG exports of U/ε components, and aggregation of all results into a single `.npz`.

### ButterFly case
```bash
python doc/04_tutorial_buterFly_sequence_step_by_step.py
```
Identical pipeline applied to the ButterFly DP600 sequence with double precision, GPU-friendly defaults, and tunable parameters (element size, regularization, plotted frames, etc.) declared at the top of the script.

Both tutorials rely on `D2IC.app_utils.run_pipeline_sequence`, which performs:
1. image loading (`skimage.io.imread`),
2. ROI meshing (`generate_roi_mesh`),
3. `Dic.precompute_pixel_data(...)` (NumPy by default or JAX if enabled),
4. sparse displacement initialization,
5. the global DIC loop `run_dic_sequence` (CG + Laplace/Spring regularization),
6. post-processing and exports via `DICPlotter`.

## Programmatic usage
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
Highlights:
- `history` stores `(J, ||grad||)` at each CG iteration.
- `run_dic_nodal` offers Jacobi or Gauss–Seidel-like local refinement sweeps.
- `compute_pixel_state` and `gauss_seidel_nodal_step_*` are JIT-friendly so you can assemble custom solvers.



## Contributing
1. Fork + create a feature branch.
2. Implement your changes/tests (keep CPU/GPU parity and JIT-compatibility in mind).
3. Document the impact in your PR (performance, new scripts, API updates).
4. Run `pytest` and, for DIC changes, rerun at least one tutorial pipeline.

## License
D²IC is distributed under the GPLv3 license. See `AUTHORS` for the list of contributors.
