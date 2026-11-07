from __future__ import annotations

from pathlib import Path

import numpy as np

from .binning import bin_image


ROI_MASK_DESCRIPTION = "D2IC_ROI_MASK_WHITE_FOREGROUND_V1"


def erode_correlation_roi(roi_mask, radius_px=0., *, image_binning=1):
    """Return the fixed photometric ROI, without changing the mesh ROI.

    Parameters are a boolean (H,W) mask at the working resolution and a
    Euclidean erosion radius in original-image pixels. Exterior and holes
    count as excluded pixels; a zero radius preserves the mask exactly.
    """
    from scipy.ndimage import distance_transform_edt

    mask = np.asarray(roi_mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError("Correlation ROI must have shape (H,W).")
    if not np.isfinite(radius_px) or radius_px < 0:
        raise ValueError("roi_erosion_px must be finite and >= 0.")
    if not np.isfinite(image_binning) or image_binning < 1:
        raise ValueError("image_binning must be >= 1.")
    if radius_px == 0:
        result = mask.copy()
    else:
        distance = distance_transform_edt(np.pad(mask, 1), sampling=image_binning)[1:-1, 1:-1]
        result = mask & (distance > radius_px)
    if not result.any():
        raise ValueError("ROI erosion leaves no pixels for correlation.")
    return result


def configure_jax_platform(preferred: str = "gpu", fallback: str = "cpu", *, verbose: bool = True) -> str:
    """
    Try to use the preferred JAX backend, falling back when unavailable.

    Parameters
    ----------
    preferred:
        Preferred backend name passed to ``jax.devices()`` (e.g. ``'gpu'`` or ``'cpu'``).
    fallback:
        Backend to use when ``preferred`` is unavailable.
    verbose:
        If True, print a one-line message describing the selected backend.

    Returns
    -------
    str
        The backend actually selected (``preferred`` or ``fallback``).
    """
    import jax

    try:
        devices = jax.devices(preferred)
    except RuntimeError:
        devices = []

    if devices:
        jax.config.update("jax_platform_name", preferred)
        backend = preferred
        msg = f"JAX backend: {preferred} ({len(devices)} device(s) detected)"
    else:
        jax.config.update("jax_platform_name", fallback)
        backend = fallback
        msg = f"JAX backend: {preferred} unavailable, falling back to {fallback}."

    if verbose:
        print(msg)

    return backend


def list_deformed_images(img_dir: Path, pattern: str, *, exclude_name: str | None = None) -> list[Path]:
    """
    List deformed image files under a directory.

    Parameters
    ----------
    img_dir:
        Directory containing images.
    pattern:
        Glob pattern (e.g. ``'*.tif'``).
    exclude_name:
        Optional filename to exclude (exact match on ``Path.name``).

    Returns
    -------
    list[Path]
        Sorted list of paths.
    """
    all_paths = sorted(img_dir.glob(pattern))
    if exclude_name is None:
        return all_paths
    return [p for p in all_paths if p.name != exclude_name]


def prepare_image(path: Path, *, binning: int = 1) -> np.ndarray:
    """
    Read an image as grayscale float32 and optionally downsample it.

    Parameters
    ----------
    path:
        Input file path.
    binning:
        Integer downsampling factor (block averaging). ``1`` disables downsampling.
    """
    if binning < 1:
        raise ValueError("binning must be >= 1.")
    img = imread_gray(path)
    if binning > 1:
        img = bin_image(img, binning)
    return img.astype(np.float32, copy=False)


def imread_gray(path: Path) -> np.ndarray:
    """
    Read an image from disk and return it as a grayscale float32 array.

    The loader tries multiple backends (imageio, tifffile, matplotlib) and
    converts RGB/RGBA images to grayscale via channel mean.
    Grayscale-alpha and palette-alpha inputs keep their first channel.
    """
    for loader in (_try_imageio, _try_tifffile, _try_matplotlib):
        arr = loader(path)
        if arr is None:
            continue
        data = np.asarray(arr)
        if data.ndim == 3:
            if data.shape[2] == 2:  # grayscale or palette index + alpha
                data = data[..., 0]
            else:
                if data.shape[2] == 4:  # drop alpha
                    data = data[..., :3]
                data = data.mean(axis=2)
        return data.astype(np.float32, copy=False)
    raise RuntimeError(f"Could not read image {path} with the available backends.")


def make_roi_mask(mask_image: np.ndarray, *, threshold: float = 0.5) -> np.ndarray:
    """
    Convert a grayscale mask image to a boolean ROI mask.

    For binary masks (two unique values), the brighter value is treated as ROI,
    with an auto-inversion when the ROI would cover nearly the full image.
    """
    data = np.asarray(mask_image)
    if data.dtype == bool:
        mask = data
        return mask
    flat = data.ravel()
    if flat.size == 0:
        return data.astype(bool)
    unique = np.unique(flat)
    if unique.size <= 2:
        dark_val = unique.min()
        bright_val = unique.max()
        mask = data == bright_val
        coverage = float(np.mean(mask))
        if coverage > 0.95 and dark_val != bright_val:
            mask = data == dark_val
        return mask
    return data > threshold


def read_roi_mask(path: Path, mask_image: np.ndarray | None = None) -> np.ndarray:
    """Read a ROI mask, honoring explicit white foreground in marked TIFFs.

    GUI-generated binary TIFFs carry :data:`ROI_MASK_DESCRIPTION` in their
    ImageDescription tag. Their nonzero pixels are always foreground, even
    when they cover more than 95% of the image. Unmarked TIFF and other legacy
    masks retain :func:`make_roi_mask`'s historical polarity heuristic.

    ``mask_image`` may supply an already decoded grayscale image to avoid a
    second image read in the sequential pipeline.
    """
    path = Path(path)
    if mask_image is None:
        mask_image = imread_gray(path)
    data = np.asarray(mask_image)
    if data.ndim != 2:
        raise ValueError(f"ROI mask must be a 2D image, got shape {data.shape}.")
    if path.suffix.lower() in {".tif", ".tiff"}:
        import tifffile

        with tifffile.TiffFile(path) as tif:
            description = tif.pages[0].description
        if description == ROI_MASK_DESCRIPTION:
            return data > 0.5
    return make_roi_mask(data)


def downsample_image(image: np.ndarray, binning: int) -> np.ndarray:
    """Compatibility alias for :func:`d2ic.io.bin_image`."""
    return bin_image(image, binning)


def _try_imageio(path: Path) -> np.ndarray | None:
    try:
        import imageio.v2 as imageio
    except Exception:  # pragma: no cover
        return None
    try:
        return imageio.imread(path)
    except Exception:  # pragma: no cover
        return None


def _try_tifffile(path: Path) -> np.ndarray | None:
    try:
        import tifffile
    except Exception:  # pragma: no cover
        return None
    try:
        return tifffile.imread(path)
    except Exception:  # pragma: no cover
        return None


def _try_matplotlib(path: Path) -> np.ndarray | None:
    try:
        import matplotlib.image as mpl_image
    except Exception:  # pragma: no cover
        return None
    try:
        return mpl_image.imread(path)
    except Exception:  # pragma: no cover
        return None
