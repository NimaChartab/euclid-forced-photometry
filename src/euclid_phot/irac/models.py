"""Construct fixed-morphology Tractor priors from a MER-like catalog."""

from __future__ import annotations

import copy

import numpy as np
from tractor import NanoMaggies, PointSource

from .config import normalize_channel


def _profile_radius_arcsec(source) -> float:
    radii = []
    try:
        radii.append(float(source.getShape().re))
    except Exception:
        pass
    for name in ("shapeExp", "shapeDev"):
        try:
            radii.append(float(getattr(source, name).re))
        except Exception:
            pass
    finite = [value for value in radii if np.isfinite(value) and value > 0]
    return max(finite, default=0.0)


def clone_prior_sources(prior_sources: list, channel: str | int, *,
                        psf_halfsize_pixels: int | None = None,
                        pixel_scale_arcsec: float | None = None) -> list:
    """Clone VIS-fitted Tractor sources exactly as the WISE handoff does.

    Class, position, and all morphology parameters are preserved. Only the
    brightness object is replaced with the requested IRAC band. An optional
    ``halfsize`` prevents extended-source rendering from cropping the wide
    effective PRF, matching ``euclid_phot.wise._source_at_wise`` without
    importing the frozen reference package.
    """
    channel = normalize_channel(channel)
    sources = copy.deepcopy(list(prior_sources))
    for source in sources:
        source.brightness = NanoMaggies(**{channel: 1.0})
        if (not isinstance(source, PointSource)
                and psf_halfsize_pixels is not None
                and pixel_scale_arcsec is not None):
            re_pixels = _profile_radius_arcsec(source) / float(pixel_scale_arcsec)
            source.halfsize = int(
                int(psf_halfsize_pixels)
                + max(4, int(np.ceil(2.0 * re_pixels)))
            )
    return sources


def model_footprint_mask(catalog, cutout, prf_builder, *,
                         margin_arcsec: float = 15.0) -> np.ndarray:
    """Return the aligned mask of priors that can affect a mosaic cutout."""
    scale = float(cutout.wcs.proj_plane_pixel_scales()[0].to_value("arcsec"))
    margin = float(margin_arcsec) / scale
    ra = np.asarray(catalog["ra"], dtype=float)
    dec = np.asarray(catalog["dec"], dtype=float)
    x, y = cutout.wcs.world_to_pixel_values(ra, dec)
    height, width = cutout.shape
    near_image = (
        np.isfinite(x) & np.isfinite(y)
        & (x >= -margin) & (x <= width - 1 + margin)
        & (y >= -margin) & (y <= height - 1 + margin)
    )
    covered = np.asarray([
        any(exposure.covers(src_ra, src_dec)
            for exposure in prf_builder.exposures)
        for src_ra, src_dec in zip(ra, dec)
    ])
    return near_image & covered


def select_model_scene(catalog, prior_sources, cutout, prf_builder, *,
                       margin_arcsec: float = 15.0):
    """Apply one footprint selection to an aligned catalog/source scene."""
    if len(catalog) != len(prior_sources):
        raise ValueError("catalog and prior_sources must align one-to-one")
    keep = model_footprint_mask(
        catalog, cutout, prf_builder, margin_arcsec=margin_arcsec
    )
    return catalog[keep], [
        source for source, wanted in zip(prior_sources, keep) if wanted
    ]
