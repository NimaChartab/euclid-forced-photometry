"""Load calibrated, unpadded cutouts from public IRAC science mosaics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.wcs import WCS

@dataclass
class MosaicCutout:
    """Aligned science-mosaic arrays in microJy per mosaic pixel."""

    product: object
    data_ujy: np.ndarray
    uncertainty_ujy: np.ndarray
    invvar: np.ndarray
    coverage: np.ndarray
    quality_mask: np.ndarray
    fit_mask: np.ndarray
    wcs: WCS
    header: fits.Header
    slices_original: tuple[slice, slice]
    conversion_ujy_per_pixel_per_mjysr: float

    @property
    def channel(self) -> str:
        return self.product.product.channel

    @property
    def shape(self) -> tuple[int, int]:
        return self.data_ujy.shape


def pixel_area_sr(wcs: WCS) -> float:
    """Projected solid angle of one image pixel in steradians."""
    return abs(float(np.linalg.det(wcs.pixel_scale_matrix))) * (np.pi / 180.0) ** 2


def load_mosaic_cutout(
    product,
    ra: float,
    dec: float,
    size_arcsec: float,
    *,
    mask_bad_bits: int | None = None,
) -> MosaicCutout:
    """Read one science-mosaic cutout without padding or WCS reprojection.

    The science array remains unmasked for display. Pixel validity is carried
    only by ``fit_mask`` and ``invvar``. ``mask_bad_bits=None`` rejects every
    non-zero quality-mask value; use ``0`` to ignore the mask plane. Products
    without a mask plane (including SEIP cutouts) use the finite uncertainty
    and positive coverage planes to define valid fit pixels.
    """
    if not np.isfinite([ra, dec, size_arcsec]).all() or size_arcsec <= 0:
        raise ValueError("ra, dec, and positive size_arcsec must be finite")
    with fits.open(product.science_path, memmap=True) as hdul:
        full_shape = hdul[0].data.shape
        header = hdul[0].header.copy()
        full_wcs = WCS(header).celestial
        cutout = Cutout2D(
            hdul[0].data,
            SkyCoord(float(ra), float(dec), unit="deg"),
            (float(size_arcsec) * u.arcsec, float(size_arcsec) * u.arcsec),
            wcs=full_wcs,
            mode="trim",
            copy=False,
        )
        # Slice the memory map before copying/converting the FITS byte order.
        cutout.data = np.array(cutout.data, dtype=np.float32, copy=True)
    slices = cutout.slices_original

    def aligned(path, dtype=np.float32):
        with fits.open(path, memmap=True) as hdul:
            array = hdul[0].data
            if array.shape != full_shape:
                raise ValueError(f"{path} shape {array.shape} != science {full_shape}")
            return np.array(array[slices], dtype=dtype, copy=True)

    uncertainty_mjysr = aligned(product.uncertainty_path)
    coverage = aligned(product.coverage_path)
    quality = (aligned(product.mask_path, dtype=np.int64)
               if product.mask_path is not None
               else np.zeros(cutout.data.shape, dtype=np.int64))
    if mask_bad_bits == 0:
        quality_mask = np.zeros(quality.shape, dtype=bool)
    elif mask_bad_bits is None:
        quality_mask = quality != 0
    else:
        quality_mask = (quality & int(mask_bad_bits)) != 0

    conversion = 1.0e12 * pixel_area_sr(cutout.wcs)
    data = np.asarray(cutout.data * conversion, dtype=np.float32)
    uncertainty = np.asarray(uncertainty_mjysr * conversion, dtype=np.float32)
    detector_coverage = (
        np.isfinite(data) & np.isfinite(uncertainty) & (uncertainty > 0)
        & np.isfinite(coverage) & (coverage > 0)
    )
    fit_mask = detector_coverage & ~quality_mask
    if not np.any(fit_mask):
        raise ValueError("science-mosaic cutout has no usable pixels")
    with np.errstate(divide="ignore", invalid="ignore"):
        invvar = np.where(fit_mask, uncertainty ** -2, 0.0).astype(np.float32)
    return MosaicCutout(
        product=product,
        data_ujy=data,
        uncertainty_ujy=uncertainty,
        invvar=invvar,
        coverage=coverage,
        quality_mask=quality_mask,
        fit_mask=fit_mask,
        wcs=cutout.wcs,
        header=header,
        slices_original=slices,
        conversion_ujy_per_pixel_per_mjysr=conversion,
    )
