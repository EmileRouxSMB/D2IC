# Changelog

## V1.2 — 1.2.0

Changes since `V1.1`.

### Added

- Resumable sequence processing with per-frame result storage, progress tracking,
  checkpoints, and optional legacy NPZ export.
- Explicit image lists, frame sampling, output prefixes, and configurable mask
  paths for sequence processing.
- Abaqus displacement comparison with spatial and temporal interpolation,
  coordinate conversion, and nodal error reports.
- ParaView exports for displacement, strain, images, and constitutive fields.
- Constitutive and mechanical modules, including plane-stress integration,
  Hill48 plasticity, equilibrium assembly, and alternating identification primitives.
- Adaptive mechanical Gaussian-process solver and associated examples. Its local
  relevance map describes consistency with the selected constitutive model;
  it is not a measure of violation of physical equilibrium.
- Photometric ROI erosion, shared image/mask binning, and pixel decimation.

### Changed

- Organized implementations into thematic subpackages (`core`, `mesh`,
  `numerics`, `physics`, `methods`, `batch`, `propagation`, `io`, `mechanics`,
  `constitutive`, `identification`, and `visualization`).
- Updated examples, scientific conventions, benchmark reports, and publication
  instructions.
- Removed the OpenCV dependency; image I/O now declares `tifffile` explicitly.

### Fixed

- Preserved grayscale values when reading images containing an alpha channel.

### Compatibility and validation

- Existing top-level API exports and historical flat module paths are retained
  through compatibility aliases. Prefer thematic subpackages for new imports.
- Sequence readers support legacy monolithic NPZ results alongside the new
  per-frame result store. The global NPZ remains an optional compatibility export.
- The package version is now `1.2.0`, matching the Git release tag `V1.2`;
  the metadata at `V1.1` still declared `1.0.1`.
- No tests, scientific validation runs, or distribution builds were executed
  during this release preparation. Existing benchmark reports describe their
  recorded configurations and are not new V1.2 validation results.
