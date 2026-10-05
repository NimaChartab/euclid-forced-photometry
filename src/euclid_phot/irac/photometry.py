"""Tractor forced photometry on an IRAC science mosaic."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from time import perf_counter

import numpy as np
from astropy.stats import mad_std, sigma_clip
from astropy.table import Table
from tractor import ConstantSky, Image, LinearPhotoCal, Tractor
from tractor.constrained_optimizer import ConstrainedOptimizer
from tractor.wcs import ConstantFitsWcs

from .background import SEPBackground, estimate_sep_background
from .config import (
    IRAC_PRF_FLUX_CORRECTION,
    IRAC_PRF_SYSTEMATIC_FRACTION,
    UJY_PER_NMGY,
    normalize_channel,
)
from .mosaic import MosaicCutout


class AstropyWCSAdapter:
    """Adapt an astropy WCS to Tractor's astrometry.net WCS interface."""

    def __init__(self, astropy_wcs):
        self._wcs = astropy_wcs

    def radec2pixelxy(self, ra, dec):
        x, y = self._wcs.world_to_pixel_values(ra, dec)
        return True, float(x) + 1.0, float(y) + 1.0

    def pixelxy2radec(self, x, y):
        ra, dec = self._wcs.pixel_to_world_values(x - 1.0, y - 1.0)
        return float(ra), float(dec)

    def pixel_scale(self):
        scale = self._wcs.proj_plane_pixel_scales()[0]
        return float(getattr(scale, "value", scale)) * 3600.0

    def get_cd(self):
        cd = self._wcs.pixel_scale_matrix
        return [cd[0, 0], cd[0, 1], cd[1, 0], cd[1, 1]]


@dataclass
class MosaicPhotometryResult:
    """Catalog and image diagnostics from one science-mosaic fit."""

    channel: str
    catalog: Table
    cutout: MosaicCutout
    prf_builder: object
    tractor_image: Image
    tractor: Tractor
    source_model_image: np.ndarray
    background_image: np.ndarray
    background_rms_image: np.ndarray
    model_image: np.ndarray
    residual_image: np.ndarray
    chi_inflation: float
    chi_pool_kind: str
    n_chi_pool: int
    tractor_seconds: float
    source_model: str
    mission: str
    initial_flux_seconds: float
    position_fit_seconds: float
    final_flux_seconds: float
    position_iterations: int
    max_position_shift_arcsec: float
    position_shift_arcsec: np.ndarray
    position_shift_pixels: np.ndarray
    position_at_limit: np.ndarray
    background_mesh_arcsec: float
    background_global_ujy: float
    background_global_rms_ujy: float
    background_is_constant: bool


def build_tractor_image(
    cutout: MosaicCutout,
    psf: object,
    background: SEPBackground,
) -> Image:
    """Build a calibrated Tractor image after subtracting the SEP background."""
    valid = cutout.fit_mask
    science = np.asarray(cutout.data_ujy, dtype=np.float32)
    data = np.where(
        np.isfinite(science), science - background.image, 0.0
    )
    product = cutout.product.product
    product_label = getattr(product, "product_root", None)
    if product_label is None:
        product_label = f"r{getattr(product, 'aorkey', 'unknown')}"
    image = Image(
        data=np.asarray(data, dtype=np.float32),
        invvar=np.asarray(cutout.invvar, dtype=np.float32),
        psf=psf,
        wcs=ConstantFitsWcs(AstropyWCSAdapter(cutout.wcs)),
        photocal=LinearPhotoCal(UJY_PER_NMGY, band=cutout.channel),
        sky=ConstantSky(0.0),
        name=f"{cutout.channel}-{product_label}",
    )
    image.irac_astropy_wcs = cutout.wcs
    image.irac_cutout = cutout
    image.irac_background = background
    return image


def _flux_inverse_variance(fit_result, tractor: Tractor,
                           source_count: int) -> np.ndarray:
    inverse_variance = np.asarray(fit_result.IV, dtype=float)
    names = tractor.getParamNames()
    indices = [index for index, name in enumerate(names)
               if name.startswith("catalog.") and ".brightness." in name]
    if len(inverse_variance) == len(names) and len(indices) == source_count:
        return inverse_variance[indices]
    if len(inverse_variance) == source_count:
        return inverse_variance
    warnings.warn("unfamiliar Tractor variance layout; errors are NaN",
                  stacklevel=2)
    return np.zeros(source_count, dtype=float)


def _position_offsets_arcsec(sources, prior_ra, prior_dec):
    ra = np.asarray([source.getPosition().ra for source in sources], dtype=float)
    dec = np.asarray([source.getPosition().dec for source in sources], dtype=float)
    prior_ra = np.asarray(prior_ra, dtype=float)
    prior_dec = np.asarray(prior_dec, dtype=float)
    dra = (ra - prior_ra + 180.0) % 360.0 - 180.0
    dx = dra * np.cos(np.radians(prior_dec)) * 3600.0
    dy = (dec - prior_dec) * 3600.0
    return dx, dy, np.hypot(dx, dy)


def _configure_position_limits(sources, prior_ra, prior_dec, *,
                               max_shift_arcsec, derivative_step_arcsec):
    cap_deg = float(max_shift_arcsec) / 3600.0
    step_deg = float(derivative_step_arcsec) / 3600.0
    for source, ra0, dec0 in zip(sources, prior_ra, prior_dec):
        source.freezeAllParams()
        source.thawParam("pos")
        source.thawParam("brightness")
        position = source.getPosition()
        cos_dec = max(abs(float(np.cos(np.radians(dec0)))), 1.0e-6)
        position.setStepSizes([step_deg / cos_dec, step_deg])
        position.lowers = [ra0 - cap_deg / cos_dec, dec0 - cap_deg]
        position.uppers = [ra0 + cap_deg / cos_dec, dec0 + cap_deg]
        position.maxstep = [0.5 * cap_deg / cos_dec, 0.5 * cap_deg]


def _project_position_shifts(sources, prior_ra, prior_dec,
                             max_shift_arcsec):
    dx, dy, shift = _position_offsets_arcsec(sources, prior_ra, prior_dec)
    outside = np.isfinite(shift) & (shift > float(max_shift_arcsec))
    for index in np.flatnonzero(outside):
        scale = float(max_shift_arcsec) / float(shift[index])
        cos_dec = max(abs(float(np.cos(np.radians(prior_dec[index])))), 1.0e-6)
        position = sources[index].getPosition()
        position.ra = float(prior_ra[index]) + (
            dx[index] * scale / (3600.0 * cos_dec)
        )
        position.dec = float(prior_dec[index]) + dy[index] * scale / 3600.0
    return outside


def _fit_positions_and_brightness(tractor, sources, prior_ra, prior_dec, *,
                                  max_shift_arcsec, pixel_scale_arcsec,
                                  max_iterations, dlnp_min):
    _configure_position_limits(
        sources,
        prior_ra,
        prior_dec,
        max_shift_arcsec=max_shift_arcsec,
        derivative_step_arcsec=max(0.01, 0.05 * pixel_scale_arcsec),
    )
    projected = np.zeros(len(sources), dtype=bool)
    iterations = 0
    for iteration in range(int(max_iterations)):
        dlnp, _, _ = tractor.optimize()
        iterations = iteration + 1
        projected |= _project_position_shifts(
            sources, prior_ra, prior_dec, max_shift_arcsec
        )
        if not np.isfinite(dlnp) or dlnp < float(dlnp_min):
            break
    return iterations, projected


def _residual_chi_inflation(image: Image, model: np.ndarray, *,
                            minimum_pool_pixels: int = 50):
    data = np.asarray(image.getImage(), dtype=float)
    invvar = np.asarray(image.getInvvar(), dtype=float)
    valid = np.isfinite(data) & np.isfinite(model) & (invvar > 0)
    chi = np.full(data.shape, np.nan, dtype=float)
    chi[valid] = (data[valid] - model[valid]) * np.sqrt(invvar[valid])
    source_model = model - float(image.getSky().getValue())
    sigma = np.full(data.shape, np.inf, dtype=float)
    sigma[valid] = 1.0 / np.sqrt(invvar[valid])
    for multiple in (1.0, 2.0, 3.0, 5.0):
        pool = chi[valid & (np.abs(source_model) < multiple * sigma)]
        pool = pool[np.isfinite(pool)]
        if pool.size >= minimum_pool_pixels:
            scatter = float(mad_std(pool))
            return (max(1.0, scatter) if np.isfinite(scatter) else 1.0,
                    f"source-sparse (|source model|<{multiple:g}sigma)",
                    int(pool.size))
    pool = np.asarray(
        sigma_clip(chi[valid], sigma=3.0, maxiters=5).compressed(), dtype=float
    )
    scatter = float(mad_std(pool)) if pool.size else np.nan
    return (max(1.0, scatter) if np.isfinite(scatter) else 1.0,
            "all-pixels sigma-clipped fallback", int(pool.size))


def _output_catalog(prior_catalog, sources: list, channel: str,
                    flux_error_ujy: np.ndarray,
                    builder: object,
                    prior_ra: np.ndarray,
                    prior_dec: np.ndarray,
                    max_position_shift_arcsec: float,
                    pixel_scale_arcsec: float) -> Table:
    output = Table()
    if "object_id" in prior_catalog.colnames:
        output["object_id"] = prior_catalog["object_id"]
    ra = np.asarray([source.getPosition().ra for source in sources], dtype=float)
    dec = np.asarray([source.getPosition().dec for source in sources], dtype=float)
    raw_flux = np.asarray([
        source.getBrightness().getFlux(channel) * UJY_PER_NMGY
        for source in sources
    ])
    correction = IRAC_PRF_FLUX_CORRECTION[channel]
    flux = raw_flux * correction
    raw_flux_error_stat_ujy = np.asarray(flux_error_ujy, dtype=float)
    flux_error_stat_ujy = raw_flux_error_stat_ujy * correction
    raw_flux_error_ujy = np.hypot(
        raw_flux_error_stat_ujy,
        IRAC_PRF_SYSTEMATIC_FRACTION * np.abs(raw_flux),
    )
    flux_error_ujy = np.hypot(
        flux_error_stat_ujy,
        IRAC_PRF_SYSTEMATIC_FRACTION * np.abs(flux),
    )
    output["ra"] = ra
    output["dec"] = dec
    output["prior_ra"] = np.asarray(prior_ra, dtype=float)
    output["prior_dec"] = np.asarray(prior_dec, dtype=float)
    shift_dx, shift_dy, shift = _position_offsets_arcsec(
        sources, prior_ra, prior_dec
    )
    output["position_dx_arcsec"] = shift_dx
    output["position_dy_arcsec"] = shift_dy
    output["position_shift_arcsec"] = shift
    output["position_shift_pixels"] = shift / float(pixel_scale_arcsec)
    output["position_at_limit"] = (
        (max_position_shift_arcsec > 0)
        & np.isclose(shift, max_position_shift_arcsec, rtol=0.0, atol=1.0e-6)
    )
    output["source_model"] = [type(source).__name__ for source in sources]
    x, y = builder.output_wcs.world_to_pixel_values(ra, dec)
    shape = builder.output_wcs.array_shape
    if shape is None:
        shape = builder.output_wcs.pixel_shape
        shape = (shape[1], shape[0]) if shape is not None else (0, 0)
    output["in_mosaic_cutout"] = (
        (x >= -0.5) & (x < shape[1] - 0.5)
        & (y >= -0.5) & (y < shape[0] - 0.5)
    )
    if hasattr(builder, "exposure_counts"):
        exposure_count = builder.exposure_counts(x, y)
    else:
        # PRFMap readers already cache their nearest spatial grid stamp, so
        # asking for its stored contributor count remains inexpensive.
        exposure_count = np.zeros(len(sources), dtype=np.int32)
        for index, (source_x, source_y) in enumerate(zip(x, y)):
            try:
                _, exposure_count[index] = builder.render(source_x, source_y)
            except ValueError:
                exposure_count[index] = 0
    output["n_exposures"] = np.asarray(exposure_count, dtype=np.int16)
    prefix = channel.lower()
    output[f"flux_{prefix}_raw_ujy"] = raw_flux
    output[f"fluxerr_{prefix}_raw_stat_ujy"] = raw_flux_error_stat_ujy
    output[f"fluxerr_{prefix}_raw_ujy"] = raw_flux_error_ujy
    output[f"flux_{prefix}_ujy"] = flux
    output[f"fluxerr_{prefix}_stat_ujy"] = flux_error_stat_ujy
    output[f"fluxerr_{prefix}_ujy"] = flux_error_ujy
    with np.errstate(divide="ignore", invalid="ignore"):
        output[f"snr_{prefix}"] = flux / flux_error_ujy
    return output


def fit_mosaic_forced(
    cutout: MosaicCutout,
    prior_catalog: Table,
    sources: list,
    prf_builder: object,
    psf: object,
    *,
    source_model: str = "MER morphology",
    mission: str = "unknown",
    inflate_uncertainties: bool = True,
    background_mesh_arcsec: float = 60.0,
    background_filter_size: int = 3,
    max_position_shift_arcsec: float = 0.0,
    position_max_iterations: int = 15,
    position_dlnp_min: float = 0.1,
) -> MosaicPhotometryResult:
    """Fit one IRAC flux per fixed-shape prior on a mosaic.

    Positions remain fixed when ``max_position_shift_arcsec=0``. A positive
    value adds a bounded joint position/brightness stage followed by a final
    conditional flux solve.  A robust SEP background map is subtracted once
    before Tractor; the source fit cannot absorb flux into an additional sky
    parameter.
    """
    if len(prior_catalog) == 0 or len(sources) == 0:
        raise ValueError("the prior catalog contains no sources")
    if len(prior_catalog) != len(sources):
        raise ValueError("prior_catalog and sources must align one-to-one")
    if not np.isfinite(max_position_shift_arcsec) or max_position_shift_arcsec < 0:
        raise ValueError("max_position_shift_arcsec must be finite and >= 0")
    if int(position_max_iterations) < 1:
        raise ValueError("position_max_iterations must be positive")
    if not np.isfinite(position_dlnp_min) or position_dlnp_min < 0:
        raise ValueError("position_dlnp_min must be finite and >= 0")
    channel = normalize_channel(cutout.channel)
    prior_ra = np.asarray(
        [source.getPosition().ra for source in sources], dtype=float
    )
    prior_dec = np.asarray(
        [source.getPosition().dec for source in sources], dtype=float
    )
    pixel_scale_arcsec = float(
        cutout.wcs.proj_plane_pixel_scales()[0].to_value("arcsec")
    )
    background = estimate_sep_background(
        cutout.data_ujy,
        cutout.fit_mask,
        pixel_scale_arcsec,
        mesh_arcsec=background_mesh_arcsec,
        filter_size=background_filter_size,
    )
    for source in sources:
        source.freezeAllBut("brightness")
    image = build_tractor_image(cutout, psf, background)
    image.freezeAllParams()
    tractor = Tractor([image], sources, optimizer=ConstrainedOptimizer())
    started = perf_counter()
    initial_started = perf_counter()
    fit_result = tractor.optimize_forced_photometry(
        minsb=0.0,
        mindlnp=1.0,
        sky=False,
        variance=True,
        shared_params=True,
        fitstats=True,
        wantims=True,
    )
    initial_flux_seconds = perf_counter() - initial_started
    position_fit_seconds = 0.0
    final_flux_seconds = 0.0
    position_iterations = 0
    if max_position_shift_arcsec > 0:
        # Its images/statistics are superseded by the final flux solve.
        del fit_result
        image.freezeAllParams()
        position_started = perf_counter()
        position_iterations, _ = _fit_positions_and_brightness(
            tractor,
            sources,
            prior_ra,
            prior_dec,
            max_shift_arcsec=max_position_shift_arcsec,
            pixel_scale_arcsec=pixel_scale_arcsec,
            max_iterations=position_max_iterations,
            dlnp_min=position_dlnp_min,
        )
        position_fit_seconds = perf_counter() - position_started
        for source in sources:
            source.freezeAllBut("brightness")
        image.freezeAllParams()
        final_started = perf_counter()
        fit_result = tractor.optimize_forced_photometry(
            minsb=0.0,
            mindlnp=1.0,
            sky=False,
            variance=True,
            shared_params=True,
            fitstats=True,
            wantims=True,
        )
        final_flux_seconds = perf_counter() - final_started
    tractor_seconds = perf_counter() - started
    source_model_image = np.asarray(
        tractor.getModelImage(0), dtype=np.float32
    )
    model = np.asarray(
        source_model_image + background.image, dtype=np.float32
    )
    residual = np.asarray(cutout.data_ujy - model, dtype=np.float32)
    if inflate_uncertainties:
        inflation, pool_kind, pool_count = _residual_chi_inflation(
            image, source_model_image
        )
    else:
        inflation, pool_kind, pool_count = 1.0, "disabled", 0
    flux_iv = _flux_inverse_variance(fit_result, tractor, len(sources))
    with np.errstate(divide="ignore", invalid="ignore"):
        # Tractor's flux inverse variance is derived from cutout.invvar,
        # which is the public IRAC uncertainty map converted to uJy/pixel.
        # Inflate it only when the normalized residuals show extra scatter.
        flux_error = np.where(
            flux_iv > 0,
            inflation * UJY_PER_NMGY / np.sqrt(flux_iv),
            np.nan,
        )
    catalog = _output_catalog(
        prior_catalog,
        sources,
        channel,
        flux_error,
        prf_builder,
        prior_ra,
        prior_dec,
        float(max_position_shift_arcsec),
        pixel_scale_arcsec,
    )
    catalog.meta["CHANNEL"] = channel
    catalog.meta["PRFCORR"] = IRAC_PRF_FLUX_CORRECTION[channel]
    catalog.meta["PRFCREF"] = "IRAC Instrument Handbook Appendix C.1"
    catalog.meta["PRFSYS"] = IRAC_PRF_SYSTEMATIC_FRACTION
    catalog.meta["ERRMODEL"] = (
        "IRAC uncertainty-map covariance, residual inflation, and PRF systematic"
    )
    product = cutout.product.product
    if hasattr(product, "aorkey"):
        catalog.meta["AORKEY"] = product.aorkey
    catalog.meta["PRODUCT"] = str(
        getattr(product, "product_root", getattr(product, "aorkey", "unknown"))
    )
    catalog.meta["PRF"] = "effective detector-location PRF from contributor geometry"
    catalog.meta["CHIINFL"] = inflation
    catalog.meta["CHIPOOL"] = pool_kind
    catalog.meta["POSMAXAS"] = float(max_position_shift_arcsec)
    catalog.meta["POSITER"] = int(position_iterations)
    catalog.meta["BKGTYPE"] = "SEP mesh"
    catalog.meta["BKGMESH"] = float(background.mesh_arcsec)
    catalog.meta["BKGFILT"] = int(background.filter_size)
    catalog.meta["BKGLEV"] = float(background.global_level)
    catalog.meta["BKGRMS"] = float(background.global_rms)
    catalog.meta["BKGCONST"] = bool(background.constant)
    _, _, position_shift_arcsec = _position_offsets_arcsec(
        sources, prior_ra, prior_dec
    )
    position_shift_pixels = position_shift_arcsec / pixel_scale_arcsec
    position_at_limit = (
        (max_position_shift_arcsec > 0)
        & np.isclose(
            position_shift_arcsec,
            max_position_shift_arcsec,
            rtol=0.0,
            atol=1.0e-6,
        )
    )
    return MosaicPhotometryResult(
        channel=channel,
        catalog=catalog,
        cutout=cutout,
        prf_builder=prf_builder,
        tractor_image=image,
        tractor=tractor,
        source_model_image=source_model_image,
        background_image=background.image,
        background_rms_image=background.rms,
        model_image=model,
        residual_image=residual,
        chi_inflation=inflation,
        chi_pool_kind=pool_kind,
        n_chi_pool=pool_count,
        tractor_seconds=tractor_seconds,
        source_model=source_model,
        mission=mission,
        initial_flux_seconds=initial_flux_seconds,
        position_fit_seconds=position_fit_seconds,
        final_flux_seconds=final_flux_seconds,
        position_iterations=position_iterations,
        max_position_shift_arcsec=float(max_position_shift_arcsec),
        position_shift_arcsec=position_shift_arcsec,
        position_shift_pixels=position_shift_pixels,
        position_at_limit=position_at_limit,
        background_mesh_arcsec=float(background.mesh_arcsec),
        background_global_ujy=float(background.global_level),
        background_global_rms_ujy=float(background.global_rms),
        background_is_constant=bool(background.constant),
    )
