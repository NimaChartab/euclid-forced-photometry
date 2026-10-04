"""External catalogue checks for the corrected IRAC pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
from astropy import units as u

from .netutils import download_file


DAWN_PL_CATALOG_URLS = {
    "EDFF": "https://dawn.calet.org/pl/DAWN_EDFF_PL_Aug11.fits",
    "EDFN": "https://dawn.calet.org/pl/DAWN_EDFN_PL_Aug11.fits",
}

_DAWN_COMPARISON_CHANNELS = ("IRAC1", "IRAC2")
_DAWN_MATCH_RADIUS_ARCSEC = 0.6
_DAWN_ISOLATION_ARCSEC = 6.0
_DAWN_MIN_SNR = 3.0


def dawn_comparison_defaults() -> dict[str, object]:
    """Return the standard settings used for public DAWN comparisons."""
    return {
        "channels": _DAWN_COMPARISON_CHANNELS,
        "match_radius_arcsec": _DAWN_MATCH_RADIUS_ARCSEC,
        "isolation_arcsec": _DAWN_ISOLATION_ARCSEC,
        "minimum_snr": _DAWN_MIN_SNR,
    }


def fetch_dawn_pl_catalog(
    field: str,
    cache_dir: str | Path,
    *,
    force: bool = False,
) -> Path:
    """Download one public DAWN PL catalogue once and return its local path."""
    field = str(field).strip().upper().replace("-", "")
    if field not in DAWN_PL_CATALOG_URLS:
        raise ValueError("field must be 'EDFF' or 'EDFN'")
    url = DAWN_PL_CATALOG_URLS[field]
    path = Path(cache_dir) / url.rsplit("/", 1)[-1]
    return download_file(url, path, force=force, timeout=1800.0)


def _column_lookup(names) -> dict[str, str]:
    return {str(name).upper(): str(name) for name in names}


def read_dawn_region(
    path: str | Path,
    ra: float,
    dec: float,
    *,
    radius_arcsec: float,
) -> Table:
    """Read only a small sky region from a memory-mapped DAWN PL FITS table."""
    if radius_arcsec <= 0:
        raise ValueError("radius_arcsec must be positive")
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    required = {"ALPHA_J2000", "DELTA_J2000"}
    with fits.open(path, memmap=True) as hdul:
        hdu = next(
            (
                item for item in hdul
                if getattr(item, "data", None) is not None
                and getattr(item.data, "names", None) is not None
                and required <= set(_column_lookup(item.data.names))
            ),
            None,
        )
        if hdu is None:
            raise ValueError("DAWN FITS table has no ALPHA_J2000/DELTA_J2000 columns")
        data = hdu.data
        names = _column_lookup(data.names)
        catalog_ra = np.asarray(data[names["ALPHA_J2000"]], dtype=float)
        catalog_dec = np.asarray(data[names["DELTA_J2000"]], dtype=float)
        cos_dec = max(np.cos(np.radians(float(dec))), 1.0e-6)
        radius_deg = float(radius_arcsec) / 3600.0
        dra = ((catalog_ra - float(ra) + 180.0) % 360.0) - 180.0
        box = (
            np.isfinite(catalog_ra) & np.isfinite(catalog_dec)
            & (np.abs(dra) <= radius_deg / cos_dec)
            & (np.abs(catalog_dec - float(dec)) <= radius_deg)
        )
        indices = np.flatnonzero(box)
        if indices.size:
            center = SkyCoord(float(ra)*u.deg, float(dec)*u.deg)
            positions = SkyCoord(catalog_ra[indices]*u.deg,
                                 catalog_dec[indices]*u.deg)
            indices = indices[positions.separation(center).arcsec <= radius_arcsec]

        output = Table()
        standardized = {
            "ID": "dawn_id",
            "ALPHA_J2000": "dawn_ra",
            "DELTA_J2000": "dawn_dec",
            "IRAC_CH1_FLUX": "dawn_flux_IRAC1_ujy",
            "IRAC_CH1_FLUXERR": "dawn_flux_err_IRAC1_ujy",
            "IRAC_CH2_FLUX": "dawn_flux_IRAC2_ujy",
            "IRAC_CH2_FLUXERR": "dawn_flux_err_IRAC2_ujy",
            "IRAC_CH1_VALID": "dawn_valid_IRAC1",
            "IRAC_CH2_VALID": "dawn_valid_IRAC2",
            "MODEL_FLAG": "dawn_model_flag",
            "SOLUTION_MODEL": "dawn_solution_model",
            "FULL_DEPTH_PL": "dawn_full_depth",
        }
        for original, renamed in standardized.items():
            if original in names:
                output[renamed] = np.asarray(data[names[original]][indices])
    output.meta["SOURCE"] = "DAWN survey PL v1.0"
    output.meta["RADIUS"] = float(radius_arcsec)
    return output


def match_dawn_irac(
    photometry: Table,
    dawn: Table,
    *,
    max_separation_arcsec: float = _DAWN_MATCH_RADIUS_ARCSEC,
) -> Table:
    """Nearest-neighbour match a pipeline comparison table to DAWN IRAC."""
    if max_separation_arcsec <= 0:
        raise ValueError("max_separation_arcsec must be positive")
    if not len(dawn):
        result = photometry[:0].copy()
        result["dawn_match_sep_arcsec"] = np.asarray([], dtype=float)
        return result
    for name in ("ra", "dec"):
        if name not in photometry.colnames:
            raise ValueError(f"photometry is missing {name}")
    ours = SkyCoord(
        np.asarray(photometry["ra"], float)*u.deg,
        np.asarray(photometry["dec"], float)*u.deg,
    )
    external = SkyCoord(
        np.asarray(dawn["dawn_ra"], float)*u.deg,
        np.asarray(dawn["dawn_dec"], float)*u.deg,
    )
    index, separation, _ = ours.match_to_catalog_sky(external)
    matched = separation.arcsec <= float(max_separation_arcsec)
    result = photometry[matched].copy()
    external_index = np.asarray(index[matched], dtype=int)
    result["dawn_match_sep_arcsec"] = separation.arcsec[matched]
    for name in dawn.colnames:
        result[name] = dawn[name][external_index]
    return result


def compare_dawn(
    photometry: Table,
    dawn: Table,
    *,
    channels=None,
    match_radius_arcsec: float | None = None,
    isolation_arcsec: float | None = None,
    minimum_snr: float | None = None,
) -> dict[str, Table]:
    """Match the pipeline catalog to DAWN and flag a clean comparison sample.

    For each channel the returned table holds the matches, the distance to
    the nearest other prior source (``nearest_source_arcsec``), the flux
    ratio ``<channel>_over_dawn`` and ``comparison_selected``: both S/N at
    least ``minimum_snr``, finite positive errors, DAWN valid, and no prior
    neighbour within ``isolation_arcsec``.
    """
    defaults = dawn_comparison_defaults()
    channels = defaults["channels"] if channels is None else tuple(channels)
    radius = (defaults["match_radius_arcsec"] if match_radius_arcsec is None
              else float(match_radius_arcsec))
    isolation_min = (defaults["isolation_arcsec"] if isolation_arcsec is None
                     else float(isolation_arcsec))
    snr_min = defaults["minimum_snr"] if minimum_snr is None else float(minimum_snr)

    positions = SkyCoord(np.asarray(photometry["ra"], float)*u.deg,
                         np.asarray(photometry["dec"], float)*u.deg)
    if len(positions) > 1:
        _, nearest, _ = positions.match_to_catalog_sky(positions, nthneighbor=2)
        nearest = nearest.arcsec
    else:
        nearest = np.full(len(positions), np.inf)
    isolation = dict(zip(np.asarray(photometry["object_id"]).tolist(), nearest))

    matches_all = match_dawn_irac(photometry, dawn,
                                  max_separation_arcsec=radius)
    comparisons = {}
    for channel in channels:
        band = str(channel).upper()
        if f"dawn_flux_{band}_ujy" not in matches_all.colnames:
            raise ValueError(f"the public DAWN PL catalog has no {channel} "
                             "photometry; use IRAC1 and/or IRAC2")
        matches = matches_all.copy()
        matches["nearest_source_arcsec"] = [
            isolation[object_id] for object_id in matches["object_id"].tolist()]
        ours = np.asarray(matches[f"flux_{band}_ujy"], float)
        ours_err = np.asarray(matches[f"flux_err_{band}_ujy"], float)
        dawn_flux = np.asarray(matches[f"dawn_flux_{band}_ujy"], float)
        dawn_err = np.asarray(matches[f"dawn_flux_err_{band}_ujy"], float)
        with np.errstate(divide="ignore", invalid="ignore"):
            selected = (
                np.isfinite(ours) & np.isfinite(ours_err)
                & np.isfinite(dawn_flux) & np.isfinite(dawn_err)
                & (ours_err > 0) & (dawn_err > 0)
                & (ours / ours_err >= snr_min) & (dawn_flux / dawn_err >= snr_min)
                & (np.asarray(matches["nearest_source_arcsec"]) >= isolation_min)
            )
            matches[f"{band}_over_dawn"] = ours / dawn_flux
        if f"dawn_valid_{band}" in matches.colnames:
            selected &= np.asarray(matches[f"dawn_valid_{band}"], bool)
        matches["comparison_selected"] = selected
        comparisons[channel] = matches
    return comparisons


def select_dawn_disagreements(
    comparison: Table,
    channel: str,
    *,
    minimum_snr: float = 5.0,
    top_n: int | None = 20,
) -> Table:
    """Rank the clean comparison sample by ``|log10(pipeline / DAWN)|``.

    Starts from ``comparison_selected`` and additionally requires positive
    fluxes and S/N of at least ``minimum_snr`` in both catalogs.
    """
    band = str(channel).upper()
    flux = np.asarray(comparison[f"flux_{band}_ujy"], float)
    err = np.asarray(comparison[f"flux_err_{band}_ujy"], float)
    dawn_flux = np.asarray(comparison[f"dawn_flux_{band}_ujy"], float)
    dawn_err = np.asarray(comparison[f"dawn_flux_err_{band}_ujy"], float)
    with np.errstate(divide="ignore", invalid="ignore"):
        snr, dawn_snr = flux / err, dawn_flux / dawn_err
        ratio = flux / dawn_flux
        keep = (np.asarray(comparison["comparison_selected"], bool)
                & (flux > 0) & (dawn_flux > 0)
                & (snr >= minimum_snr) & (dawn_snr >= minimum_snr))
        selected = comparison[keep].copy()
        selected["snr"] = snr[keep]
        selected["dawn_snr"] = dawn_snr[keep]
        selected["over_dawn"] = ratio[keep]
        selected["disagreement_dex"] = np.abs(np.log10(ratio[keep]))
        selected["difference_sigma"] = ((flux - dawn_flux)
                                        / np.hypot(err, dawn_err))[keep]
    selected.sort("disagreement_dex", reverse=True)
    if top_n is not None:
        selected = selected[:int(top_n)]
    return selected[[
        "object_id", "ra", "dec", "model",
        f"flux_{band}_ujy", f"flux_err_{band}_ujy",
        f"dawn_flux_{band}_ujy", f"dawn_flux_err_{band}_ujy",
        "snr", "dawn_snr", "over_dawn", "disagreement_dex",
        "difference_sigma", "nearest_source_arcsec", "dawn_match_sep_arcsec",
    ]]


WISE_IRAC_PAIRS = {"W1": "IRAC1", "W2": "IRAC2"}


def compare_wise_irac(
    photometry: Table,
    *,
    pairs=None,
    minimum_snr: float = 5.0,
    isolation_arcsec: float | None = None,
    flux_fraction: float = 1.0 / 3.0,
) -> dict[str, Table]:
    """Compare unWISE and IRAC fluxes fitted with the same VIS models.

    ``comparison_selected`` requires S/N of at least ``minimum_snr`` in both
    bands and no VIS neighbour brighter than ``flux_fraction`` of the source
    within ``isolation_arcsec`` (default two WISE FWHM), the isolation
    criterion of notebook 03.
    """
    from ..wise import _WISE_FWHM_ARCSEC, select_isolated_sources

    if isolation_arcsec is None:
        isolation_arcsec = 2.0 * _WISE_FWHM_ARCSEC

    pairs = WISE_IRAC_PAIRS if pairs is None else dict(pairs)
    isolated = select_isolated_sources(
        np.asarray(photometry["ra"], float),
        np.asarray(photometry["dec"], float),
        np.asarray(photometry["flux_VIS_ujy"], float),
        radius_arcsec=isolation_arcsec, flux_fraction=flux_fraction,
    )
    comparisons = {}
    for wise_band, channel in pairs.items():
        w, c = wise_band.upper(), channel.upper()
        wise_flux = np.asarray(photometry[f"flux_{w}_ujy"], float)
        wise_err = np.asarray(photometry[f"flux_err_{w}_ujy"], float)
        irac_flux = np.asarray(photometry[f"flux_{c}_ujy"], float)
        irac_err = np.asarray(photometry[f"flux_err_{c}_ujy"], float)
        with np.errstate(divide="ignore", invalid="ignore"):
            selected = (
                isolated & (wise_err > 0) & (irac_err > 0)
                & (wise_flux / wise_err >= minimum_snr)
                & (irac_flux / irac_err >= minimum_snr)
            )
            ratio = wise_flux / irac_flux
        table = photometry["object_id", "ra", "dec", "model",
                           f"flux_{w}_ujy", f"flux_err_{w}_ujy",
                           f"flux_{c}_ujy", f"flux_err_{c}_ujy"].copy()
        table["isolated"] = isolated
        table[f"{w}_over_{c}"] = ratio
        table["comparison_selected"] = selected
        comparisons[wise_band] = table
    return comparisons
