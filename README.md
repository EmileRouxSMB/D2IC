# D²IC

D²IC (a differentiable framework for full-field kinematic identification) is an open-source Python package for 2D Digital Image Correlation (DIC) built on top of [JAX](https://github.com/google/jax). It implements a global finite-element DIC formulation (quadrilateral mesh on the ROI) by minimizing a pixelwise brightness-constancy objective with optional spring regularization. Because the computational core is written in JAX, the same solver kernels run on CPU or accelerator backends and expose exact gradients via automatic differentiation.

The codebase ships a single API: the `d2ic` package (mesh/asset layer, solvers, batch execution, strain post-processing).

## Why D²IC?
- **Differentiable global FE-DIC**: Gauss–Newton/CG implemented in `jax.numpy`, with exact gradients and JIT compilation.
- **Reusable mesh/pixel assets**: pixel→element mapping, shape-function weights, node→pixel adjacency (CSR-like gather) and neighbor graphs to accelerate repeated solves on sequences.
- **Initialization + sequential pipelines**: coarse translation-only initialization via ZNCC at element centers (`TranslationZNCCSolver`), optional refinement (`LocalGaussNewtonSolver`), and batch warm-start propagation (`BatchMeshBased` + propagators).
- **Strain + uncertainty**: post-processing computes mesh strains; end-to-end differentiability enables uncertainty propagation via `jax.jvp` (see `doc/11_Advanced_Use_Uncertainty_Propagation.ipynb`).
- **Reproducible examples**: scripts and notebooks under `doc/` reproduce validation/figures and export meshes, fields, and plots.


## Repository layout
- `d2ic/`: package (mask2mesh, batch runner, solvers, strain).
- `doc/`: scripted tutorials + notebooks.
- `doc/img/`: small demo datasets shipped with the repo (Sample3, PlateHole, butterFly, ...).
- `scripte/`: autonomous scripts (e.g., CPU vs GPU benchmark).

## Installation
```bash
git clone https://github.com/EmileRouxSMB/D2IC.git
cd D2IC
python -m venv .venv
source .venv/bin/activate  # on Windows: .\.venv\Scripts\activate
pip install --upgrade pip
pip install -e .
```

### GPU support (optional)
If you want CUDA/accelerator support, install the JAX wheel matching your CUDA toolkit *before* installing D²IC:
```bash
pip install "jax[cuda12]"
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

See the tutorials in `doc/` for end-to-end examples.

## Tutorials & scripts
### Sequential DIC tutorial (PlateHole / butterFly)
```bash
python doc/03_0_tutorial_sequentialDIC.py
```
Step-by-step processing of a sequence: ROI meshing via `mask_to_mesh_assets`, sequential displacement estimation with warm-start propagation, strain extraction, and figure/NPZ exports.

### Validation (Sample3, DIC Challenge)
```bash
python doc/00_Validation_ImageSerie.py
```
Rigid-body translation validation on Sample3 with mean displacement tracking and uncertainty indicators.

### Advanced: uncertainty propagation (notebook)
Open `doc/11_Advanced_Use_Uncertainty_Propagation.ipynb` to reproduce the JVP-based uncertainty propagation workflow.

## Benchmark CPU vs GPU (script autonome)
To reproduce the benchmark (and regenerate `benchmark_cpu.json`, `benchmark_gpu.json`, and `benchmark_indicators.png`):
```bash
python scripte/benchmark_butterfly_cpu_vs_gpu_autonomous.py
```

Resultats extraits de `scripte/benchmark_cpu.json` et `scripte/benchmark_gpu.json` (11 frames, WSL2; GPU: NVIDIA RTX 2000 Ada Generation Laptop GPU). Temps en secondes:

| Backend | Warmup/Prep (s) | DIC / frame (s) | Total (s) |
|---|---:|---:|---:|
| CPU | 3.21 | 0.85 | 12.61 |
| GPU | 42.65 | 2.04 | 65.09 |

Commentaires (sur cette config):
- Le CPU est plus rapide : GPU/CPU = x13.3 (warmup), x2.39 (par frame), x5.16 (total).
- Le cout GPU est domine par la compilation/warmup et un probleme trop petit; sur des sequences plus longues ou des maillages plus gros, le GPU peut s'amortir.

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

For a coarse first-frame initialization (translation-only ZNCC at element centers + nodal projection), see `d2ic.DICInitMotion` with `d2ic.TranslationZNCCSolver` and `d2ic.InitMotionConfig`.

## Cite us
If you use D²IC in academic work, please cite the accompanying article manuscript (not yet published):

**Plain text**
- Roux, E. *D²IC: a differentiable 2D digital image correlation framework*. Manuscript submitted to SoftwareX, 2025.

**BibTeX**
```bibtex
@unpublished{roux2025d2ic,
  title  = {{$D^2IC$}: a differentiable 2D digital image correlation framework},
  author = {Roux, Emile},
  year   = {2025},
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
