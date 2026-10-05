"""IRAC forced photometry of fixed VIS source models with an effective PRF."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import (
    DEFAULT_CALIBRATION_DIR,
    DEFAULT_COSMIC_DAWN_DIR,
    DEFAULT_DATA_ROOT,
    DEFAULT_PBCD_DIR,
    IRAC_NATIVE_PIXEL_SCALE_ARCSEC,
    normalize_channel,
)
from .models import clone_prior_sources, select_model_scene
from .photometry import MosaicPhotometryResult, fit_mosaic_forced

from .data import (
    DEFAULT_SEIP_DIR,
    MosaicInputs,
    fetch_cosmic_dawn_mosaic,
    fetch_seip_mosaic,
    fetch_sha_mosaic,
)
from .prfmap import PRFMapMosaicPSF, build_prfmap_psf


@dataclass(frozen=True)
class PreparedChannel:
    """Science image and spatial PSF grid shared by fixed/free fits."""

    inputs: MosaicInputs
    prf_builder: object
    tractor_psf: PRFMapMosaicPSF
    mission: str
    prf_backend: str

    @property
    def cutout(self):
        return self.inputs.cutout

    @property
    def local_product(self):
        return self.inputs.local_product



def fetch_irac_mosaic(
    ra: float,
    dec: float,
    channel: str | int,
    *,
    size_arcsec: float,
    mosaic_backend: str = "dawn",
    data_root: str | Path = DEFAULT_DATA_ROOT,
    force_download: bool = False,
) -> MosaicInputs:
    """Fetch an IRAC mosaic cutout into the directory layout of a run."""
    fetchers = {
        "dawn": (fetch_cosmic_dawn_mosaic, "cosmic_dawn"),
        "seip": (fetch_seip_mosaic, "seip"),
        "pbcd": (fetch_sha_mosaic, "pbcd"),
    }
    backend = str(mosaic_backend).strip().lower()
    if backend not in fetchers:
        raise ValueError("mosaic_backend must be 'dawn', 'seip', or 'pbcd'")
    fetch, subdir = fetchers[backend]
    return fetch(ra, dec, channel, size_arcsec=size_arcsec,
                 data_dir=Path(data_root) / subdir,
                 force_download=force_download)


def prepare_channel(
    ra: float,
    dec: float,
    channel: str | int,
    *,
    size_arcsec: float,
    mosaic_backend: str = "seip",
    aorkey: int | None = None,
    super_mosaic_id: str | None = None,
    pbcd_dir: str | Path = DEFAULT_PBCD_DIR,
    seip_dir: str | Path = DEFAULT_SEIP_DIR,
    cosmic_dawn_dir: str | Path = DEFAULT_COSMIC_DAWN_DIR,
    calibration_dir: str | Path = DEFAULT_CALIBRATION_DIR,
    mask_bad_bits: int | None = None,
    mission: str = "auto",
    prfmap_grid_file: str | Path | None = None,
    prfmap_prf_directory: str | Path | None = None,
    prfmap_pixel_scale_arcsec: float | None = None,
    grid_spacing_arcsec: float = 28.8,
    prf_stamp_size: int = 51,
    weighting: str = "uniform",
    pixel_integration_subsamples: int = 1,
    recenter_detector_prfs: bool = True,
    force_download: bool = False,
) -> PreparedChannel:
    """Fetch a science mosaic and attach the corrected effective-PRF grid."""
    mosaic_backend = str(mosaic_backend).strip().lower()
    if mosaic_backend == "seip":
        inputs = fetch_seip_mosaic(
            ra,
            dec,
            channel,
            size_arcsec=size_arcsec,
            super_mosaic_id=super_mosaic_id,
            data_dir=seip_dir,
            mask_bad_bits=mask_bad_bits,
            force_download=force_download,
        )
    elif mosaic_backend == "dawn":
        inputs = fetch_cosmic_dawn_mosaic(
            ra,
            dec,
            channel,
            size_arcsec=size_arcsec,
            data_dir=cosmic_dawn_dir,
            mask_bad_bits=mask_bad_bits,
            force_download=force_download,
        )
    elif mosaic_backend == "pbcd":
        inputs = fetch_sha_mosaic(
            ra,
            dec,
            channel,
            size_arcsec=size_arcsec,
            aorkey=aorkey,
            data_dir=pbcd_dir,
            mask_bad_bits=mask_bad_bits,
            force_download=force_download,
        )
    else:
        raise ValueError("mosaic_backend must be 'dawn', 'seip', or 'pbcd'")
    builder, psf, selected_mission, backend = build_prfmap_psf(
        inputs.local_product,
        inputs.cutout,
        exposures=inputs.exposures,
        mission=mission,
        calibration_dir=calibration_dir,
        force_download=force_download,
        grid_file=prfmap_grid_file,
        prf_directory=prfmap_prf_directory,
        prf_pixel_scale_arcsec=prfmap_pixel_scale_arcsec,
        grid_spacing_arcsec=grid_spacing_arcsec,
        stamp_size=prf_stamp_size,
        weighting=weighting,
        pixel_integration_subsamples=pixel_integration_subsamples,
        recenter_detector_prfs=recenter_detector_prfs,
    )
    return PreparedChannel(inputs, builder, psf, selected_mission, backend)


def fit_prepared(
    prepared: PreparedChannel,
    mer_catalog,
    prior_sources: list,
    *,
    neighbor_buffer_arcsec: float = 15.0,
    inflate_uncertainties: bool = True,
    background_mesh_arcsec: float = 60.0,
    background_filter_size: int = 3,
    max_position_shift_arcsec: float = 0.0,
    position_max_iterations: int = 15,
    position_dlnp_min: float = 0.1,
) -> MosaicPhotometryResult:
    """Fit frozen high-resolution models as a jointly blended group.

    This is the FARMER measurement-stage experiment: source classes and shapes
    come from a higher-resolution modeling stage; neighboring models are fit
    simultaneously; only brightness is free unless bounded recentering is
    explicitly requested.
    """
    cutout = prepared.cutout
    model_catalog, selected_sources = select_model_scene(
        mer_catalog,
        prior_sources,
        cutout,
        prepared.prf_builder,
        margin_arcsec=neighbor_buffer_arcsec,
    )
    pixel_scale_arcsec = float(
        cutout.wcs.proj_plane_pixel_scales()[0].to_value("arcsec")
    )
    sources = clone_prior_sources(
        selected_sources,
        cutout.channel,
        psf_halfsize_pixels=prepared.prf_builder.stamp_size // 2,
        pixel_scale_arcsec=pixel_scale_arcsec,
    )
    result = fit_mosaic_forced(
        cutout,
        model_catalog,
        sources,
        prepared.prf_builder,
        prepared.tractor_psf,
        source_model="frozen high-resolution morphology; FARMER-like",
        mission=prepared.mission,
        inflate_uncertainties=inflate_uncertainties,
        background_mesh_arcsec=background_mesh_arcsec,
        background_filter_size=background_filter_size,
        max_position_shift_arcsec=max_position_shift_arcsec,
        position_max_iterations=position_max_iterations,
        position_dlnp_min=position_dlnp_min,
    )
    result.catalog.meta["SCIENCE"] = prepared.inputs.backend
    result.catalog.meta["PRF"] = prepared.prf_backend
    result.catalog.meta["GRIDPSF"] = True
    if hasattr(prepared.prf_builder, "recenter_detector_prfs"):
        result.catalog.meta["PRFCENT"] = bool(
            prepared.prf_builder.recenter_detector_prfs
        )
        offsets = prepared.prf_builder.detector_centroid_offsets_native.values()
        radial = np.asarray([np.hypot(*offset) for offset in offsets], dtype=float)
        result.catalog.meta["PRFCOFF"] = float(np.median(radial))
    if hasattr(prepared.prf_builder, "pixel_integration_subsamples"):
        result.catalog.meta["PRFSUB"] = int(
            prepared.prf_builder.pixel_integration_subsamples
        )
    return result


def fit_irac_forced(
    sources: list,
    ra: float,
    dec: float,
    size_arcsec: float,
    channels=("IRAC1", "IRAC2"),
    *,
    mosaic_backend: str = "seip",
    data_dir: str | Path = DEFAULT_DATA_ROOT,
    neighbor_buffer_arcsec: float = 15.0,
    max_position_shift_arcsec: float | None = None,
    force_download: bool = False,
    verbose: bool = True,
    prepare_options: dict | None = None,
    fit_options: dict | None = None,
) -> dict:
    """Fit fixed VIS source models to IRAC mosaics, one channel at a time.

    Each channel's mosaic is cut to the field, its effective PRF is rebuilt
    from the detector PRFs and the contributing exposures, and the sources
    are convolved with it and fitted jointly with a background. Each
    centroid may move by up to ``max_position_shift_arcsec`` (default one
    native IRAC pixel).

    Returns ``{channel: dict}`` with ``flux_ujy``, ``flux_err_ujy``
    (statistical plus the 2% PRF systematic), ``flux_err_stat_ujy`` and
    ``position_shift_arcsec``, aligned with ``sources`` (NaN where a source
    is outside the mosaic coverage), plus the ``data``, ``model``,
    ``residual`` and ``invvar`` images in microJansky per pixel, their
    ``wcs``, the fit ``catalog`` and diagnostics.
    """
    from astropy.table import Table

    backend = str(mosaic_backend).strip().lower()
    if backend not in {"dawn", "seip", "pbcd"}:
        raise ValueError("mosaic_backend must be 'dawn', 'seip', or 'pbcd'")
    data_dir = Path(data_dir)
    options = {
        "size_arcsec": float(size_arcsec),
        "mosaic_backend": backend,
        "cosmic_dawn_dir": data_dir / "cosmic_dawn",
        "seip_dir": data_dir / "seip",
        "pbcd_dir": data_dir / "pbcd",
        "calibration_dir": data_dir / "calibration",
        "mission": "cryo" if backend == "seip" else "auto",
        "weighting": "exptime" if backend == "dawn" else "uniform",
    }
    options.update(prepare_options or {})

    n = len(sources)
    scene = Table({
        "object_id": np.arange(n),
        "ra": [float(s.getPosition().ra) for s in sources],
        "dec": [float(s.getPosition().dec) for s in sources],
    })
    results = {}
    for channel in (normalize_channel(c) for c in channels):
        if verbose:
            print(f"  {channel}: mosaic and effective PRF ({backend})")
        prepared = prepare_channel(ra, dec, channel, **options,
                                   force_download=force_download)
        shift = (IRAC_NATIVE_PIXEL_SCALE_ARCSEC[channel]
                 if max_position_shift_arcsec is None
                 else float(max_position_shift_arcsec))
        fit = fit_prepared(prepared, scene, list(sources),
                           neighbor_buffer_arcsec=neighbor_buffer_arcsec,
                           max_position_shift_arcsec=shift,
                           **(fit_options or {}))
        cat = fit.catalog
        rows = np.asarray(cat["object_id"], dtype=int)
        key = channel.lower()

        def aligned(name):
            values = np.full(n, np.nan)
            values[rows] = np.asarray(cat[name], dtype=float)
            return values

        results[channel] = {
            "flux_ujy": aligned(f"flux_{key}_ujy"),
            "flux_err_ujy": aligned(f"fluxerr_{key}_ujy"),
            "flux_err_stat_ujy": aligned(f"fluxerr_{key}_stat_ujy"),
            "position_shift_arcsec": aligned("position_shift_arcsec"),
            "data": np.asarray(fit.cutout.data_ujy, dtype=np.float32),
            "model": np.asarray(fit.model_image, dtype=np.float32),
            "residual": np.asarray(fit.residual_image, dtype=np.float32),
            "invvar": np.asarray(fit.cutout.invvar, dtype=np.float32),
            "wcs": fit.cutout.wcs,
            "catalog": cat,
            "chi_inflation": float(fit.chi_inflation),
            "prf_backend": prepared.prf_backend,
            "mosaic_backend": backend,
        }
        if verbose:
            print(f"  {channel}: {len(rows)} sources fitted, "
                  f"chi inflation {fit.chi_inflation:.2f}")
    return results
