# D²IC

Version **1.2.0** · Git tag **V1.2** · [Release notes](CHANGELOG.md)

D²IC (a differentiable framework for full-field kinematic identification) is an open-source Python package for 2D Digital Image Correlation (DIC) built on top of [JAX](https://github.com/google/jax). It implements a global finite-element DIC formulation (quadrilateral mesh on the ROI) by minimizing a pixelwise brightness-constancy objective with optional spring regularization. Because the computational core is written in JAX, the same solver kernels run on CPU or accelerator backends and expose exact gradients via automatic differentiation.

The codebase ships a single API: the `d2ic` package (mesh/asset layer, solvers, batch execution, strain post-processing).

## Why D²IC?
- **Differentiable global FE-DIC**: Gauss–Newton/CG implemented in `jax.numpy`, with exact gradients and JIT compilation.
- **Reusable mesh/pixel assets**: pixel→element mapping, shape-function weights, node→pixel adjacency (CSR-like gather) and neighbor graphs to accelerate repeated solves on sequences.
- **Initialization + sequential pipelines**: coarse translation-only initialization via ZNCC at element centers (`TranslationZNCCSolver`), optional refinement (`LocalGaussNewtonSolver`), and batch warm-start propagation (`BatchMeshBased` + propagators).
- **Strain + uncertainty**: post-processing computes mesh strains; end-to-end differentiability enables uncertainty propagation via `jax.jvp` (see `doc/11_Advanced_Use_Uncertainty_Propagation.ipynb`).
- **Reproducible examples**: scripts under `examples/` and notebooks under `doc/` reproduce validation/figures and export meshes, fields, and plots.


## Repository layout
- `src/d2ic/mesh/`: mesh data, pixel sampling and ROI meshing.
- `src/d2ic/numerics/`: JAX identification solvers.
- `src/d2ic/physics/`: photometric discrepancy and strain reconstruction.
- `src/d2ic/pipelines/`: end-to-end sequence orchestration.
- `examples/`: scripted tutorials + configs.
- `doc/`: notebooks and demonstration datasets.
- `doc/img/`: small demo datasets shipped with the repo (Sample3, PlateHole, butterFly, ...).
- `benchmarks/`: reproducible CPU/GPU performance measurements.
- `tools/`: maintenance utilities excluded from the numerical package.

## Installation
```bash
git clone https://github.com/EmileRouxSMB/D2IC.git
cd D2IC
python -m venv .venv
source .venv/bin/activate  # on Windows: .\.venv\Scripts\activate
pip install --upgrade pip
pip install -e .
```
Examples and scripts assume the package is installed (editable install recommended for local development).

### GPU support (optional)
If you want CUDA/accelerator support, install the JAX wheel matching your CUDA toolkit *before* installing D²IC:
```bash
pip install "jax[cuda13]"
```
See the [official docs](https://jax.readthedocs.io/en/latest/installation.html) for other CUDA/cuDNN pairs.

### Notebook extras (optional)
Some tutorials use extra I/O/interactive dependencies:
```bash
pip install -r requirements.txt
```

### System dependencies (meshing)
On Debian/Ubuntu systems, the `gmsh` Python wheel also requires `libGLU`:
```bash
sudo apt-get install -y libglu1-mesa
```

## Typical workflow
1. **Prepare the ROI**: reference image(s) and a binary mask (`.tif/.bmp`), see `doc/img/`.
2. **Generate the mesh + assets**: `mesh, _ = mask_to_mesh_assets(mask=..., element_size_px=...)` then `assets = make_mesh_assets(mesh, ...)`.
3. **Instantiate configs**: `MeshDICConfig`, `BatchConfig`.
4. **Create solvers/pipelines**: `GlobalCGSolver` (main solver), optionally `TranslationZNCCSolver` (coarse init) and `LocalGaussNewtonSolver` (refinement), then `DICMeshBased`.
5. **Run the batch**: `BatchMeshBased` orchestre l'execution par frame et la propagation du warm-start.
6. **Post-process**: outputs contain nodal displacement and Green–Lagrange strain; export NPZ/PNGs as needed.

See the executable examples in `examples/` and notebooks in `doc/` for end-to-end examples.

## Tutorials & scripts
### Sequential DIC tutorial (PlateHole / butterFly)
```bash
python examples/sequential_dic_platehole.py
```
Or via the CLI:
```bash
d2ic run-sequence --config examples/configs/sequential_dic.toml
```
Step-by-step processing of a sequence: ROI meshing via `mask_to_mesh_assets`, sequential displacement estimation with warm-start propagation, strain extraction, and figure/NPZ exports.

The sequence TOML accepts `paths.mask_path` for a mask stored outside the image directory. A relative `mask_path` is resolved from the TOML file directory; an absolute path is used directly. The legacy `paths.mask_filename` remains supported and is resolved inside `paths.img_dir`. If both keys are present, `mask_path` takes precedence. For example:

```toml
[paths]
img_dir = "../../images/VIS"
ref_image_name = "VIS_000.bmp"
image_pattern = "VIS_*.bmp"
mask_path = "roi/mask_final.tif"
out_dir = "runs/dic"
```

### Validation (Sample3, DIC Challenge)
```bash
python examples/validation_sample3_translation.py
```
Rigid-body translation validation on Sample3 with mean displacement tracking and uncertainty indicators.

### Advanced: uncertainty propagation (notebook)
Open `doc/11_Advanced_Use_Uncertainty_Propagation.ipynb` to reproduce the JVP-based uncertainty propagation workflow.


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
from d2ic.mesh import make_mesh_assets

ref_image = ...  # numpy array (H,W)
def_images = [...]  # list of numpy arrays
mask = ...  # binary ROI (H,W)

mesh, _ = mask_to_mesh_assets(mask=mask, element_size_px=16)
assets = make_mesh_assets(mesh, with_neighbors=True)

mesh_cfg = MeshDICConfig(max_iters=200, tol=1e-3, reg_strength=0.1)
batch_cfg = BatchConfig()

dic_mesh = DICMeshBased(solver=GlobalCGSolver(), config=mesh_cfg)

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

For a coarse first-frame initialization (translation-only ZNCC at element centers + nodal projection), see `d2ic.DICInitMotion` with `d2ic.TranslationZNCCSolver` and `d2ic.InitMotionConfig`.

## Cite us
If you use D²IC in academic work, please cite the accompanying article manuscript (not yet published):

**Plain text**
- Roux, E. *D²IC: a differentiable 2D digital image correlation framework*. Manuscript submitted to SoftwareX, 2026.

**BibTeX**
```bibtex
@unpublished{roux2025d2ic,
  title  = {{$D^2IC$}: a differentiable 2D digital image correlation framework},
  author = {Roux, Emile},
  year   = {2026},
  note   = {Manuscript submitted to SoftwareX (under review)},
}
```

## Contributing
1. Fork + create a feature branch.
2. Implement your changes/tests (keep CPU/GPU parity and JIT-compatibility in mind).
3. Document the impact in your PR (performance, scripts, API updates).
4. Run `pytest` and, for DIC changes, rerun at least one tutorial pipeline.


## License
D²IC is distributed under the GPLv3 license. See `AUTHORS` for the list of contributors.
