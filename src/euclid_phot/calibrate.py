"""Empirical flux-error calibration for the Euclid (VIS / NISP) bands.

MER mosaics are resampled coadds with correlated pixels (per-pixel chi std
~0.65 on the demo field), so the formal inverse-variance errors from the
flux fit are miscalibrated at PSF scale. This module is the Euclid analogue
of the WISE chi-inflation (wise.py): fit ``PointSource`` fluxes at N
source-free positions in one simultaneous flux-only solve and take
``mad_std(flux / err)`` as the inflation factor, clamped to >= 1. Exact for
point sources; a lower bound for extended sources.
"""
from __future__ import annotations

import warnings

import numpy as np

_UJY_PER_NMGY = 3.631


def draw_empty_positions(cutout, mer_cat=None, *,
                         n_positions: int = 200,
                         avoid_radius_arcsec: float = 1.5,
                         edge_margin_pix: int = 20,
                         rng=None,
                         max_batches: int = 200):
    """Random source-free sky positions inside a cutout.

    Positions are drawn uniformly in pixel space, then rejected if they fall
    on an unusable pixel (``rms <= 0``, non-finite data, or a coadd-fatal FLG
    bit when the cutout carries a flag plane), within
    ``avoid_radius_arcsec`` of any catalog source, or within
    ``avoid_radius_arcsec`` of an already-accepted position (so the
    simultaneous point-source fits stay effectively independent).

    Parameters
    ----------
    cutout : Cutout
        Provides ``data``, ``rms``, ``wcs`` and (optionally) ``flag``.
    mer_cat : astropy.table.Table or None
        Known sources to avoid; needs ``ra`` / ``dec`` columns. ``None``
        skips the source-avoidance test.
    n_positions : int
        Target number of positions. Fewer may be returned on a small or
        crowded cutout (with a warning).
    avoid_radius_arcsec : float
        Exclusion radius around catalog sources and accepted positions.
    edge_margin_pix : int
        Keep positions at least this many pixels from the cutout edge (so
        the PSF stamp fits inside the image). Capped at a quarter of the
        shorter side on very small cutouts.
    rng : numpy Generator, int seed, or None
        Randomness source (``np.random.default_rng(rng)``).

    Returns
    -------
    (ra, dec) : ndarray, ndarray
        Accepted positions in degrees.
    """
    from .images import _invvar_from_rms

    rng = np.random.default_rng(rng)
    data = np.asarray(cutout.data)
    H, W = data.shape
    margin = min(int(edge_margin_pix), max(1, min(H, W) // 4))

    if cutout.rms is not None:
        usable = _invvar_from_rms(cutout.rms, data) > 0
    else:
        usable = np.isfinite(data)
    flag = getattr(cutout, "flag", None)
    if flag is not None:
        # Same default bad-bit policy as build_tractor_image.
        from .config import MER_VIS_BAD_BITS
        usable = usable & ((np.asarray(flag) & int(MER_VIS_BAD_BITS)) == 0)

    cat_ra = cat_dec = None
    if mer_cat is not None and len(mer_cat) > 0:
        cat_ra = np.asarray(mer_cat["ra"], dtype=float)
        cat_dec = np.asarray(mer_cat["dec"], dtype=float)

    avoid_deg = float(avoid_radius_arcsec) / 3600.0
    acc_ra: list[float] = []
    acc_dec: list[float] = []

    for _ in range(max_batches):
        n_left = n_positions - len(acc_ra)
        if n_left <= 0:
            break
        n_draw = max(4 * n_left, 64)
        ix = rng.integers(margin, W - margin, size=n_draw)
        iy = rng.integers(margin, H - margin, size=n_draw)
        ok = usable[iy, ix]
        if not ok.any():
            continue
        ra, dec = cutout.wcs.pixel_to_world_values(ix[ok], iy[ok])
        ra = np.atleast_1d(np.asarray(ra, float))
        dec = np.atleast_1d(np.asarray(dec, float))
        cosd = np.cos(np.radians(dec))
        if cat_ra is not None:
            keep = np.empty(len(ra), dtype=bool)
            for k in range(len(ra)):
                d2 = ((cat_ra - ra[k]) * cosd[k]) ** 2 + (cat_dec - dec[k]) ** 2
                keep[k] = d2.min() > avoid_deg ** 2
            ra, dec, cosd = ra[keep], dec[keep], cosd[keep]
        for k in range(len(ra)):
            if len(acc_ra) >= n_positions:
                break
            if acc_ra:
                ar = np.asarray(acc_ra)
                ad = np.asarray(acc_dec)
                d2 = ((ar - ra[k]) * cosd[k]) ** 2 + (ad - dec[k]) ** 2
                if d2.min() <= avoid_deg ** 2:
                    continue
            acc_ra.append(float(ra[k]))
            acc_dec.append(float(dec[k]))

    if len(acc_ra) < n_positions:
        warnings.warn(
            f"draw_empty_positions: only {len(acc_ra)} of the requested "
            f"{n_positions} source-free positions fit in this cutout "
            f"(crowded or small field); the calibration proceeds with what "
            f"was found.", stacklevel=2)
    return np.asarray(acc_ra), np.asarray(acc_dec)


def measure_error_inflation(cutout, psf_stamp, mer_cat=None, *,
                            band: str = "VIS",
                            n_positions: int = 200,
                            avoid_radius_arcsec: float = 1.5,
                            mag_zero: float | None = None,
                            invvar: np.ndarray | None = None,
                            min_positions: int = 20,
                            rng=None) -> dict:
    """Measure the PSF-scale error-inflation factor for one band.

    Fits ``PointSource`` fluxes at source-free positions (all in one
    simultaneous flux-only solve, exactly like the science fit) and compares
    the scatter of the recovered fluxes to their formal errors.

    Parameters
    ----------
    cutout, psf_stamp
        The band's cutout and (field-average) PSF stamp, as used by the
        science fit.
    mer_cat : Table or None
        Known sources to avoid when drawing positions.
    invvar : ndarray, optional
        Override the inverse-variance map (e.g. to reproduce a bright-star
        pixel mask applied to the science fit).
    min_positions : int
        Below this many usable empty positions the calibration is skipped
        (``inflation = 1.0``, method ``"insufficient-empty-positions"``).

    Returns
    -------
    dict
        ``inflation`` (>= 1.0), ``n_positions``, ``chi_mad`` (the raw
        mad_std of flux/err before clamping), ``empirical_sigma_ujy``,
        ``formal_err_med_ujy``, ``method``.
    """
    from astropy.stats import mad_std

    from .fit import fit_forced_photometry
    from .images import build_tractor_image
    from .models import build_sources_from_coords

    ra, dec = draw_empty_positions(
        cutout, mer_cat, n_positions=n_positions,
        avoid_radius_arcsec=avoid_radius_arcsec, rng=rng)
    base = {
        "band": band,
        "n_positions": int(len(ra)),
        "method": "empty-position point-source",
    }
    if len(ra) < min_positions:
        warnings.warn(
            f"error calibration for {band}: only {len(ra)} empty positions "
            f"(< {min_positions}); skipping (inflation = 1).", stacklevel=2)
        return {**base, "inflation": 1.0, "chi_mad": np.nan,
                "empirical_sigma_ujy": np.nan, "formal_err_med_ujy": np.nan,
                "method": "insufficient-empty-positions"}

    sources = build_sources_from_coords(ra, dec, band=band, model="point")
    tim = build_tractor_image(cutout, psf_stamp, mag_zero=mag_zero,
                              invvar=invvar)
    # flag_unphysical=False: pure-noise fluxes are negative half the time
    # by construction.
    _, _, flux_err_ujy = fit_forced_photometry(
        tim, sources, band=band, flag_unphysical=False, return_errors=True)
    flux_ujy = np.array(
        [s.brightness.getFlux(band) for s in sources]) * _UJY_PER_NMGY

    ok = (np.isfinite(flux_ujy) & np.isfinite(flux_err_ujy)
          & (flux_err_ujy > 0))
    if ok.sum() < min_positions:
        warnings.warn(
            f"error calibration for {band}: only {int(ok.sum())} positions "
            f"with a finite formal error; skipping (inflation = 1).",
            stacklevel=2)
        return {**base, "inflation": 1.0, "chi_mad": np.nan,
                "empirical_sigma_ujy": np.nan, "formal_err_med_ujy": np.nan,
                "method": "insufficient-empty-positions"}

    chi = flux_ujy[ok] / flux_err_ujy[ok]
    chi_mad = float(mad_std(chi))
    inflation = max(1.0, chi_mad) if np.isfinite(chi_mad) else 1.0
    return {
        **base,
        "n_positions": int(ok.sum()),
        "inflation": float(inflation),
        "chi_mad": chi_mad,
        "empirical_sigma_ujy": float(mad_std(flux_ujy[ok])),
        "formal_err_med_ujy": float(np.median(flux_err_ujy[ok])),
        # Per-position samples for viz.show_error_calibration.
        "chi": chi,
        "flux_ujy": flux_ujy[ok],
        "flux_err_ujy": flux_err_ujy[ok],
        "ra": ra[ok] if len(ra) == ok.size else ra,
        "dec": dec[ok] if len(dec) == ok.size else dec,
    }


def calibrate_result_errors(result, *,
                            bands=None,
                            n_positions: int = 200,
                            avoid_radius_arcsec: float = 1.5,
                            rng=0,
                            verbose: bool = False) -> dict:
    """Calibrate the Euclid-band flux errors of a ``ForcedPhotometryResult``.

    For every Euclid band in the result, measures the empty-position
    inflation factor and multiplies ``result.flux_errs_ujy[band]`` in place.
    WISE bands are skipped; their errors already carry the empirical
    chi-inflation from :func:`euclid_phot.wise.fit_wise_forced`.

    The prior band reuses the science fit's bright-star pixel mask
    (``result.prior_pixel_mask``) so the calibration sees the same
    inverse-variance map as the fluxes it calibrates.

    ``rng`` defaults to the fixed seed 0 so repeated runs of the same field
    are bit-identical.

    Returns the per-band calibration dict, which is also stored on
    ``result.error_calibration``.
    """
    euclid_present = [b for b in ("VIS", "Y", "J", "H")
                      if b in result.flux_errs_ujy and b in result.cutouts
                      and b in result.psf_stamps]
    if bands is not None:
        euclid_present = [b for b in bands if b in euclid_present]

    rng = np.random.default_rng(rng)
    prior_band = (result.prior or {}).get("band")
    calib = getattr(result, "error_calibration", None)
    if calib is None:
        calib = {}

    for band in euclid_present:
        invvar = None
        if band == prior_band and result.prior_pixel_mask is not None:
            from .images import _invvar_from_rms
            cut = result.cutouts[band]
            invvar = np.where(result.prior_pixel_mask, 0.0,
                              _invvar_from_rms(cut.rms, cut.data))
        info = measure_error_inflation(
            result.cutouts[band], result.psf_stamps[band], result.mer_cat,
            band=band, n_positions=n_positions,
            avoid_radius_arcsec=avoid_radius_arcsec, invvar=invvar, rng=rng)
        result.flux_errs_ujy[band] = (
            np.asarray(result.flux_errs_ujy[band], dtype=float)
            * info["inflation"])
        calib[band] = info
        if verbose:
            print(f"  {band}: error inflation x{info['inflation']:.2f} "
                  f"({info['method']}, n={info['n_positions']})")

    try:
        result.error_calibration = calib
    except AttributeError:
        pass
    return calib


def pull_vs_mer(result, *, band: str = "VIS") -> dict:
    """Pull distribution of Tractor vs MER fluxes (diagnostic only).

    Computes ``(flux_tractor - flux_mer) / sqrt(err_tractor^2 + err_mer^2)``
    per source. Tractor and MER share pixels, so the pull is biased narrow;
    it is a consistency check, not an error calibration.

    Returns ``{"pull", "pull_std", "n", "note"}``; ``pull`` is aligned with
    the sources that had finite values in both catalogs (``mask`` gives
    their indices into the result arrays).
    """
    from astropy.stats import mad_std

    col_map = {
        "VIS": ("flux_vis_sersic", "fluxerr_vis_sersic"),
        "Y": ("flux_y_sersic", "fluxerr_y_sersic"),
        "J": ("flux_j_sersic", "fluxerr_j_sersic"),
        "H": ("flux_h_sersic", "fluxerr_h_sersic"),
    }
    if band not in col_map:
        raise ValueError(f"pull_vs_mer supports Euclid bands only, got {band!r}")
    fcol, ecol = col_map[band]
    mer = result.mer_cat
    if mer is None or fcol not in getattr(mer, "colnames", []):
        raise ValueError(
            f"result.mer_cat has no {fcol!r} column (user-coords run?); "
            "the MER pull diagnostic needs a real MER catalog.")

    def _filled(col):
        a = mer[col]
        return np.asarray(a.filled(np.nan) if hasattr(a, "filled") else a,
                          dtype=float)

    f_mer, e_mer = _filled(fcol), _filled(ecol)
    f_t = np.asarray(result.fluxes_ujy[band], dtype=float)
    e_t = np.asarray(result.flux_errs_ujy[band], dtype=float)
    ok = (np.isfinite(f_t) & np.isfinite(e_t) & (e_t > 0)
          & np.isfinite(f_mer) & np.isfinite(e_mer) & (e_mer > 0))
    pull = (f_t[ok] - f_mer[ok]) / np.sqrt(e_t[ok] ** 2 + e_mer[ok] ** 2)
    return {
        "pull": pull,
        "mask": np.where(ok)[0],
        "pull_std": float(mad_std(pull)) if pull.size else np.nan,
        "n": int(pull.size),
        "note": ("diagnostic only: Tractor and MER share pixels, so their "
                 "errors are correlated and the pull is biased narrow; do "
                 "not use as a calibration source."),
    }
