"""Corrected IRAC mosaic photometry with a nearest-grid effective PRF."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.table import Table

from .config import (
    DEFAULT_CALIBRATION_DIR,
    DEFAULT_COSMIC_DAWN_DIR,
    DEFAULT_DATA_ROOT,
    DEFAULT_MODEL_CACHE_DIR,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_PBCD_DIR,
    DEFAULT_REFERENCE_DIR,
    IRAC_NATIVE_PIXEL_SCALE_ARCSEC,
    normalize_channel,
    validate_run_name,
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


@dataclass(frozen=True)
class IRACRun:
    """Named, restartable VIS-prior + IRAC forced-photometry run."""

    name: str
    ra: float
    dec: float
    field_size_arcsec: float
    data_root: Path
    output_dir: Path
    reference: object
    channels: dict[str, object]

    def image_set(self, band: str) -> dict:
        """Return lazy data/model/residual planes and WCS for one band."""
        band = str(band).strip().upper()
        if band == "VIS":
            return {
                "data": self.reference.image("data"),
                "model": self.reference.image("model"),
                "residual": self.reference.image("residual"),
                "invvar": self.reference.image("invvar"),
                "wcs": self.reference.wcs,
                "unit": self.reference.artifact.metadata.get(
                    "vis_bunit", "native/pixel"
                ),
            }
        if band in self.reference.wise_results:
            result = self.reference.wise_results[band]
            return {
                "data": result["data"],
                "model": result["model"],
                "residual": result["residual"],
                "invvar": result["invvar"],
                "wcs": result["wcs"],
                "unit": "nanomaggy/pixel",
            }
        channel = normalize_channel(band)
        result = self.channels[channel]
        return {
            "data": result.cutout.data_ujy,
            "model": result.model_image,
            "residual": result.residual_image,
            "invvar": result.cutout.invvar,
            "wcs": result.cutout.wcs,
            "unit": "uJy/pixel",
        }

    def to_table(self) -> Table:
        """Build one target-field catalog aligned across all fitted channels."""
        reference_catalog = self.reference.mer_cat
        reference_sources = self.reference.sources
        object_ids = np.asarray(reference_catalog["object_id"])
        prior_ra = np.asarray(
            [source.getPosition().ra for source in reference_sources], dtype=float
        )
        prior_dec = np.asarray(
            [source.getPosition().dec for source in reference_sources], dtype=float
        )
        half = self.field_size_arcsec / 2.0
        dra = ((prior_ra - self.ra + 180.0) % 360.0 - 180.0)
        dx = dra * np.cos(np.radians(self.dec)) * 3600.0
        dy = (prior_dec - self.dec) * 3600.0
        keep = (np.abs(dx) <= half) & (np.abs(dy) <= half)

        table = Table()
        table["object_id"] = object_ids[keep]
        table["prior_ra"] = prior_ra[keep]
        table["prior_dec"] = prior_dec[keep]
        table["source_model"] = np.asarray(
            [type(source).__name__ for source in reference_sources]
        )[keep]
        table["flux_vis_ujy"] = np.asarray(
            self.reference.artifact.array("flux_ujy"), dtype=float
        )[keep]
        table["fluxerr_vis_ujy"] = np.asarray(
            self.reference.artifact.array("flux_err_ujy"), dtype=float
        )[keep]

        for band, result in self.reference.wise_results.items():
            # Rows failing the VIS flux guard are NaN, as in notebook 03.
            table[f"flux_{band.lower()}_ujy"] = np.asarray(
                result["export_flux_ujy"], dtype=float)[keep]
            table[f"fluxerr_{band.lower()}_ujy"] = np.asarray(
                result["export_flux_err_ujy"], dtype=float)[keep]

        kept_ids = np.asarray(table["object_id"])
        for channel, result in self.channels.items():
            prefix = channel.lower()
            catalog = result.catalog
            lookup = {
                int(object_id): index
                for index, object_id in enumerate(catalog["object_id"])
            }
            row_index = np.asarray(
                [lookup.get(int(object_id), -1) for object_id in kept_ids], dtype=int
            )
            present = row_index >= 0

            def aligned(name, *, dtype=float, fill=np.nan):
                values = np.full(len(table), fill, dtype=dtype)
                values[present] = np.asarray(catalog[name], dtype=dtype)[
                    row_index[present]
                ]
                return values

            table[f"fit_ra_{prefix}"] = aligned("ra")
            table[f"fit_dec_{prefix}"] = aligned("dec")
            for name in (
                f"flux_{prefix}_raw_ujy",
                f"fluxerr_{prefix}_raw_stat_ujy",
                f"fluxerr_{prefix}_raw_ujy",
                f"flux_{prefix}_ujy",
                f"fluxerr_{prefix}_stat_ujy",
                f"fluxerr_{prefix}_ujy",
                f"snr_{prefix}",
                "position_dx_arcsec",
                "position_dy_arcsec",
                "position_shift_arcsec",
            ):
                output_name = name if name.startswith(("flux", "snr")) else (
                    f"{name}_{prefix}"
                )
                table[output_name] = aligned(name)
            table[f"n_exposures_{prefix}"] = aligned(
                "n_exposures", dtype=np.int16, fill=0
            )
            table[f"covered_{prefix}"] = present & aligned(
                "in_mosaic_cutout", dtype=bool, fill=False
            )

        table.meta["RUN_NAME"] = self.name
        table.meta["TARGET_RA"] = self.ra
        table.meta["TARGET_DEC"] = self.dec
        table.meta["FIELD_AS"] = self.field_size_arcsec
        return table

    def cutout(self, band: str, ra: float, dec: float,
               size_arcsec: float) -> dict:
        """Data, uncertainty, model, residual and invvar around one position."""
        from astropy.coordinates import SkyCoord
        from astropy.nddata import Cutout2D
        from astropy.nddata.utils import NoOverlapError
        import astropy.units as u

        if float(size_arcsec) <= 0:
            raise ValueError("size_arcsec must be positive")
        images = self.image_set(band)
        position = SkyCoord(float(ra) * u.deg, float(dec) * u.deg)
        product = {"wcs": None, "unit": images.get("unit", "")}
        for kind in ("data", "model", "residual", "invvar"):
            try:
                cut = Cutout2D(
                    np.asarray(images[kind], dtype=float), position,
                    float(size_arcsec) * u.arcsec, wcs=images["wcs"],
                    mode="partial", fill_value=0.0 if kind == "invvar" else np.nan,
                    copy=True,
                )
            except NoOverlapError as error:
                raise ValueError(
                    f"({ra}, {dec}) is outside the fitted {band} field"
                ) from error
            product[kind] = np.asarray(cut.data, dtype=float)
            product["wcs"] = product["wcs"] or cut.wcs
        invvar = product["invvar"]
        product["uncertainty"] = np.where(
            invvar > 0, 1.0 / np.sqrt(np.where(invvar > 0, invvar, 1.0)), np.nan)
        return product

    def save_cutout(self, band: str, ra: float, dec: float,
                    size_arcsec: float, *, overwrite: bool = False) -> Path:
        """Write one band's cutout as FITS below ``output_dir/cutouts``.

        Extensions, in order: data (primary), uncertainty, model, residual,
        invvar. Each carries the cutout WCS and unit.
        """
        product = self.cutout(band, ra, dec, size_arcsec)
        tag = (f"ra{float(ra):.6f}_dec{float(dec):+.6f}_{float(size_arcsec):g}as"
               .replace("+", "p").replace("-", "m").replace(".", "p"))
        path = self.output_dir / "cutouts" / f"{self.name}_{band.lower()}_{tag}.fits"
        path.parent.mkdir(parents=True, exist_ok=True)
        common = product["wcs"].to_header(relax=True)
        common.update(RUNNAME=self.name, BAND=str(band), RA_CTR=float(ra),
                      DEC_CTR=float(dec), SIZE_AS=float(size_arcsec))
        unit = product["unit"]
        hdus = []
        for name in ("data", "uncertainty", "model", "residual", "invvar"):
            header = common.copy()
            header["EXTNAME"] = name.upper()
            if unit:
                header["BUNIT"] = f"1/({unit})^2" if name == "invvar" else unit
            data = np.asarray(product[name], dtype=np.float32)
            hdus.append(fits.PrimaryHDU(data, header=header) if not hdus
                        else fits.ImageHDU(data, header=header))
        fits.HDUList(hdus).writeto(path, overwrite=overwrite)
        return path

    def save_catalog(self, filename: str = "irac_forced_photometry.fits",
                     *, overwrite: bool = False) -> Path:
        """Write the aligned catalog below ``examples/output/irac/<run_name>/``."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / filename
        self.to_table().write(path, overwrite=overwrite)
        return path

    def save_images(self, filename: str = "irac_image_products.fits",
                    *, overwrite: bool = False) -> Path:
        """Write VIS/IRAC data, model, and residual planes to one FITS file."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / filename
        hdus = [fits.PrimaryHDU()]
        hdus[0].header["RUNNAME"] = self.name
        for band in ("VIS", *self.reference.wise_results, *self.channels):
            images = self.image_set(band)
            for kind in ("data", "model", "residual"):
                header = images["wcs"].to_header(relax=True)
                header["BUNIT"] = images["unit"]
                hdus.append(fits.ImageHDU(
                    np.asarray(images[kind], dtype=np.float32),
                    header=header,
                    name=f"{band}_{kind}".upper(),
                ))
        fits.HDUList(hdus).writeto(path, overwrite=overwrite)
        return path


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


def run_irac_photometry(
    run_name: str,
    ra: float,
    dec: float,
    size_arcsec: float,
    *,
    channels=("IRAC1", "IRAC2", "IRAC3", "IRAC4"),
    wise_bands=(),
    mosaic_backend: str = "dawn",
    data_root: str | Path = DEFAULT_DATA_ROOT,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    image_padding_arcsec: float = 10.0,
    reference_buffer_arcsec: float = 15.0,
    reference_batch_size_arcsec: float = 300.0,
    n_workers: int = 1,
    force_download: bool = False,
    force_refit: bool = False,
    prepare_options: dict | None = None,
    fit_options: dict | None = None,
) -> IRACRun:
    """Run or reload the complete named VIS-prior IRAC pipeline.

    Missing inputs are downloaded beneath ``data_root``. Fit checkpoints use
    ``run_name`` as their stable namespace; with ``force_refit=False`` a
    validated existing result is returned without fitting again. Requested
    catalog/image exports are written separately through :class:`IRACRun` to
    ``output_root/run_name``. ``wise_bands=("W1", "W2")`` also fits the same
    VIS models to unWISE, as in notebook 03.
    """
    from .cached import cached_channel, cached_reference

    run_name = validate_run_name(run_name)
    channels = tuple(normalize_channel(channel) for channel in channels)
    if not channels:
        raise ValueError("channels must contain at least one IRAC channel")
    if len(set(channels)) != len(channels):
        raise ValueError("channels must not contain duplicates")
    values = np.asarray([
        ra, dec, size_arcsec, image_padding_arcsec,
        reference_buffer_arcsec, reference_batch_size_arcsec,
    ], dtype=float)
    if not np.isfinite(values).all() or size_arcsec <= 0:
        raise ValueError("coordinates and sizes must be finite; size must be positive")
    if image_padding_arcsec < 0 or reference_buffer_arcsec < 0:
        raise ValueError("image and reference padding must be non-negative")
    if reference_batch_size_arcsec <= 0:
        raise ValueError("reference_batch_size_arcsec must be positive")

    data_root = Path(data_root).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    data_root.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)
    model_cache = data_root / DEFAULT_MODEL_CACHE_DIR.name / run_name
    reference_data = data_root / DEFAULT_REFERENCE_DIR.name / run_name
    output_dir = output_root / run_name
    irac_size = float(size_arcsec + 2.0 * image_padding_arcsec)
    reference_size = float(irac_size + 2.0 * reference_buffer_arcsec)

    reference = cached_reference(
        ra,
        dec,
        reference_size,
        cache_dir=model_cache,
        data_dir=reference_data,
        wise_bands=tuple(wise_bands),
        n_workers=n_workers,
        field_size_arcsec=reference_size,
        batch_size_arcsec=reference_batch_size_arcsec,
        tile_halo_arcsec=reference_buffer_arcsec,
        force_refit=force_refit,
        force_download=force_download,
    )

    backend = str(mosaic_backend).strip().lower()
    if backend not in {"dawn", "seip", "pbcd"}:
        raise ValueError("mosaic_backend must be 'dawn', 'seip', or 'pbcd'")
    common_prepare = {
        "size_arcsec": irac_size,
        "mosaic_backend": backend,
        "cosmic_dawn_dir": data_root / "cosmic_dawn",
        "seip_dir": data_root / "seip",
        "pbcd_dir": data_root / "pbcd",
        "calibration_dir": data_root / "calibration",
        "mission": "cryo" if backend == "seip" else "auto",
        "weighting": "exptime" if backend == "dawn" else "uniform",
        "grid_spacing_arcsec": 28.8,
        "prf_stamp_size": 51,
        "pixel_integration_subsamples": 1,
        "recenter_detector_prfs": True,
    }
    common_prepare.update(prepare_options or {})
    common_fit = {
        "neighbor_buffer_arcsec": reference_buffer_arcsec,
        "inflate_uncertainties": True,
        "background_mesh_arcsec": 60.0,
        "background_filter_size": 3,
    }
    common_fit.update(fit_options or {})

    results = {}
    for channel in channels:
        channel_fit = dict(common_fit)
        # Permit a bounded recentering of exactly one native detector pixel.
        channel_fit.setdefault(
            "max_position_shift_arcsec",
            IRAC_NATIVE_PIXEL_SCALE_ARCSEC[channel],
        )
        results[channel] = cached_channel(
            ra,
            dec,
            channel,
            reference,
            cache_dir=model_cache,
            prepare_options=common_prepare,
            fit_options=channel_fit,
            force_refit=force_refit,
            force_download=force_download,
        )

    return IRACRun(
        name=run_name,
        ra=float(ra),
        dec=float(dec),
        field_size_arcsec=float(size_arcsec),
        data_root=data_root,
        output_dir=output_dir,
        reference=reference,
        channels=results,
    )
