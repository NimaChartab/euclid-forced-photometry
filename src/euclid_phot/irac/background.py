"""Fast spatial background estimation for IRAC mosaics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import sep


@dataclass(frozen=True)
class SEPBackground:
    """A SEP background map and the settings used to construct it."""

    image: np.ndarray
    rms: np.ndarray
    global_level: float
    global_rms: float
    mesh_pixels: int
    mesh_arcsec: float
    filter_size: int
    constant: bool


def estimate_sep_background(
    data: np.ndarray,
    valid_mask: np.ndarray,
    pixel_scale_arcsec: float,
    *,
    mesh_arcsec: float = 60.0,
    filter_size: int = 3,
) -> SEPBackground:
    """Estimate a robust, smoothly interpolated SEP background map.

    SEP sigma-clips object pixels inside each mesh and median-filters the mesh
    grid before interpolation.  A cutout no larger than one requested mesh is
    assigned SEP's single robust global level rather than a poorly constrained
    spatial surface.
    """
    data = np.asarray(data)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    if data.ndim != 2 or valid_mask.shape != data.shape:
        raise ValueError("data and valid_mask must be aligned 2-D arrays")
    if not np.isfinite(pixel_scale_arcsec) or pixel_scale_arcsec <= 0:
        raise ValueError("pixel_scale_arcsec must be finite and positive")
    if not np.isfinite(mesh_arcsec) or mesh_arcsec <= 0:
        raise ValueError("mesh_arcsec must be finite and positive")
    if int(filter_size) < 1:
        raise ValueError("filter_size must be a positive integer")

    finite = valid_mask & np.isfinite(data)
    if not np.any(finite):
        raise ValueError("no finite valid pixels are available for background")
    native = np.asarray(data, dtype=np.float32, order="C")
    fill = float(np.median(native[finite]))
    work = np.where(finite, native, fill).astype(np.float32, copy=False)
    mask = np.asarray(~finite, dtype=bool, order="C")
    requested_pixels = max(1, int(round(float(mesh_arcsec) / pixel_scale_arcsec)))
    mesh_pixels = min(requested_pixels, min(data.shape))
    filter_size = int(filter_size)
    background = sep.Background(
        work,
        mask=mask,
        bw=mesh_pixels,
        bh=mesh_pixels,
        fw=filter_size,
        fh=filter_size,
    )
    is_constant = min(data.shape) <= requested_pixels
    if is_constant:
        image = np.full(data.shape, background.globalback, dtype=np.float32)
        rms = np.full(data.shape, background.globalrms, dtype=np.float32)
    else:
        image = np.asarray(background.back(), dtype=np.float32)
        rms = np.asarray(background.rms(), dtype=np.float32)
    return SEPBackground(
        image=image,
        rms=rms,
        global_level=float(background.globalback),
        global_rms=float(background.globalrms),
        mesh_pixels=int(mesh_pixels),
        mesh_arcsec=float(mesh_pixels * pixel_scale_arcsec),
        filter_size=filter_size,
        constant=is_constant,
    )
