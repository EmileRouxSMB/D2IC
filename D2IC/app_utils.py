from __future__ import annotations

from pathlib import Path

import numpy as np


def configure_jax_platform(preferred: str = "gpu", fallback: str = "cpu", *, verbose: bool = True) -> str:
    """
    Try to use the preferred JAX backend, falling back when unavailable.

    Returns the backend actually selected.
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
    """Return sorted image paths matching pattern, optionally excluding one filename."""
    all_paths = sorted(img_dir.glob(pattern))
    if exclude_name is None:
        return all_paths
    return [p for p in all_paths if p.name != exclude_name]


def prepare_image(path: Path, *, binning: int = 1) -> np.ndarray:
    """Read a grayscale image and optionally downsample it with block averaging."""
    if binning < 1:
        raise ValueError("binning must be >= 1.")
    img = imread_gray(path)
    if binning > 1:
        img = downsample_image(img, binning)
    return img.astype(np.float32, copy=False)


def imread_gray(path: Path) -> np.ndarray:
    """Read an image from disk and return it as a grayscale float32 array."""
    for loader in (_try_imageio, _try_tifffile, _try_matplotlib):
        arr = loader(path)
        if arr is None:
            continue
        data = np.asarray(arr)
        if data.ndim == 3:
            if data.shape[2] == 4:  # drop alpha
                data = data[..., :3]
            data = data.mean(axis=2)
        return data.astype(np.float32, copy=False)
    raise RuntimeError(f"Could not read image {path} with the available backends.")


def downsample_image(image: np.ndarray, binning: int) -> np.ndarray:
    """Downsample a 2D image by `binning` using block averaging."""
    if binning <= 1:
        return image
    h, w = image.shape
    new_h = h // binning
    new_w = w // binning
    if new_h == 0 or new_w == 0:
        raise ValueError("Binning factor too large for the input image size.")
    trimmed = image[: new_h * binning, : new_w * binning]
    reshaped = trimmed.reshape(new_h, binning, new_w, binning)
    return reshaped.mean(axis=(1, 3))


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

