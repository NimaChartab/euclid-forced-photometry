"""Assemble the per-object science catalog from a forced-photometry run.

Joins the per-band flux/error arrays into a single
:class:`astropy.table.Table` with one row per object and unit-carrying
columns. Fluxes are microJansky on the AB system,

    m_AB = 23.9 - 2.5 * log10(flux_uJy)
    sigma_m = (2.5 / ln 10) * (sigma_flux / flux)

matching the Euclid MER convention. The per-band-family error definition is
recorded in the table metadata.
"""
from __future__ import annotations

import numpy as np

_AB_ZP_UJY = 23.9
_BAND_ORDER = ("VIS", "Y", "J", "H", "W1", "W2")


def _model_re_arcsec(sources):
    """Fitted effective radius (arcsec) per Tractor source; 0 for point
    sources. ``FixedCompositeGalaxy`` takes the larger of its two
    component radii."""
    if sources is None:
        return None
    radii = np.zeros(len(sources))
    for i, src in enumerate(sources):
        shapes = [getattr(src, "shape", None)]
        if shapes[0] is None:
            shapes = [getattr(src, "shapeExp", None),
                      getattr(src, "shapeDev", None)]
        vals = []
        for sh in shapes:
            try:
                vals.append(float(sh.re))
            except (AttributeError, TypeError):
                continue
        radii[i] = max(vals) if vals else 0.0
    return radii


def flux_to_ab_mag(flux_ujy):
    """AB magnitude from flux in microJansky. Non-positive flux -> nan."""
    f = np.asarray(flux_ujy, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        m = _AB_ZP_UJY - 2.5 * np.log10(f)
    return np.where(np.isfinite(f) & (f > 0), m, np.nan)


def flux_err_to_mag_err(flux_ujy, flux_err_ujy):
    """1-sigma magnitude error from flux and flux error (both microJansky)."""
    f = np.asarray(flux_ujy, dtype=float)
    e = np.asarray(flux_err_ujy, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        m_err = (2.5 / np.log(10.0)) * (e / f)
    return np.where(np.isfinite(f) & (f > 0) & np.isfinite(e), m_err, np.nan)


def build_catalog(result, *, origin: str | None = None,
                  ebv="auto", with_flags: bool = True, depth: bool = True):
    """Build the per-object catalog Table from a ``ForcedPhotometryResult``.

    Parameters
    ----------
    result : ForcedPhotometryResult
    origin : str, optional
        Origin tag written to the ``origin`` column. Defaults to the
        source-list mode (``mer`` / ``free`` / ``coords``).
    ebv : 'auto', float, array, or None
        Galactic E(B-V) for the extinction columns (``ebv``,
        ``a_<band>_mag``, ``mag_<band>_ab_extcorr``; flux columns stay
        observed-frame). ``'auto'`` takes the MER per-source ``gal_ebv``
        when the run carried a real MER catalog, else one field value
        from the IRSA dust service (cached under the run's ``data_dir``);
        if neither is available the columns are skipped with a warning.
        ``None`` skips them silently.
    with_flags : bool
        Add the quality flags: ``blended`` / ``n_neighbors`` /
        ``nearest_arcsec`` (:func:`euclid_phot.flags.blend_flags`) and,
        when the source list is star-classified, ``bright_star`` /
        ``near_bright_star`` / ``masked`` / ``edge`` / ``reliable``
        (:func:`euclid_phot.flags.flag_sources`).
    depth : bool
        Record the per-band 5-sigma point-source depth
        (``meta['depth_5sigma_ab']``), computed from the median reported
        flux error, i.e. from the calibrated errors when the run used
        ``calibrate_errors=True``.

    Returns
    -------
    astropy.table.Table
        One row per source, aligned with ``result.sources``. Columns:
        ``object_id, ra, dec, origin, model, is_star, flux_quality`` and, for
        each band that was measured, ``flux_<band>_ujy``, ``flux_err_<band>_ujy``,
        ``mag_<band>_ab``, ``mag_err_<band>_ab``. Units are attached to columns
        and the error definition per band family is recorded in ``table.meta``.

        ``is_star`` is True where the fitted model is a ``PointSource``
        (catalog stars and galaxies that collapsed to a point source); it is
        not a copy of the MER ``is_star`` classification.
    """
    import astropy.units as u
    from astropy.table import Table

    sources = result.sources
    n = len(sources)

    ra = np.array([float(s.getPosition().ra) for s in sources])
    dec = np.array([float(s.getPosition().dec) for s in sources])

    mer = result.mer_cat
    if mer is not None and "object_id" in getattr(mer, "colnames", []):
        object_id = np.asarray(mer["object_id"])
    else:
        object_id = np.arange(n, dtype=np.int64)

    model = (list(result.chosen_models) if result.chosen_models is not None
             else [type(s).__name__ for s in sources])
    is_star = np.array([m == "PointSource" for m in model])

    if origin is None:
        origin = (result.prior or {}).get("objects", "mer")

    quality = (result.flux_quality if result.flux_quality is not None
               else np.ones(n, dtype=bool))

    tab = Table()
    tab["object_id"] = object_id
    tab["ra"] = ra * u.deg
    tab["dec"] = dec * u.deg
    tab["origin"] = np.array([origin] * n)
    tab["model"] = np.array(model)
    tab["is_star"] = np.asarray(is_star, dtype=bool)
    tab["flux_quality"] = np.asarray(quality, dtype=bool)

    bands = [b for b in _BAND_ORDER if b in result.fluxes_ujy]
    bands += [b for b in result.fluxes_ujy if b not in bands]
    for b in bands:
        flux = np.asarray(result.fluxes_ujy[b], dtype=float)
        err = np.asarray(
            result.flux_errs_ujy.get(b, np.full(n, np.nan)), dtype=float)
        if b in result.flux_errs_ujy:
            # nan error = the band fit never constrained this source;
            # export no measurement rather than the seed value.
            flux = np.where(np.isfinite(err), flux, np.nan)
        tab[f"flux_{b}_ujy"] = flux * u.uJy
        tab[f"flux_err_{b}_ujy"] = err * u.uJy
        tab[f"mag_{b}_ab"] = flux_to_ab_mag(flux) * u.mag
        tab[f"mag_err_{b}_ab"] = flux_err_to_mag_err(flux, err) * u.mag

    tab.meta["flux_unit"] = "microJansky (AB)"
    tab.meta["mag_zeropoint"] = f"AB, m = {_AB_ZP_UJY} - 2.5*log10(flux_uJy)"
    calib = getattr(result, "error_calibration", None) or {}
    error_origin = {}
    if any(b in result.fluxes_ujy for b in ("VIS", "Y", "J", "H")):
        if calib:
            error_origin["VIS/Y/J/H"] = (
                "formal Tractor inverse-variance x empirical PSF-scale "
                "inflation (empty-position point-source calibration; "
                "factors in meta['error_inflation']. Exact for point "
                "sources; a lower bound for extended sources, which "
                "integrate more correlated coadd noise)")
        else:
            error_origin["VIS/Y/J/H"] = (
                "formal Tractor inverse-variance (per-pixel noise only; "
                "not calibrated for correlated coadd noise; rerun with "
                "calibrate_errors=True)")
    if any(b in result.fluxes_ujy for b in ("W1", "W2")):
        error_origin["W1/W2"] = (
            "formal unWISE inverse-variance x empirical chi-inflation "
            "(confusion + coadd-correlated noise; calibrated against "
            "Schlafly+2019 dflux)")
    tab.meta["error_provenance"] = error_origin
    if calib:
        tab.meta["error_inflation"] = {
            b: round(float(d["inflation"]), 4) for b, d in calib.items()}
        tab.meta["error_inflation_method"] = {
            b: d.get("method", "") for b, d in calib.items()}

    if with_flags and n > 0:
        from .flags import blend_flags, flag_sources
        # First band present in canonical order is the prior band.
        if bands:
            bf = blend_flags(
                ra, dec, np.asarray(result.fluxes_ujy[bands[0]], float),
                profile_re_arcsec=_model_re_arcsec(
                    getattr(result, "sources", None)))
            for c in bf.colnames:
                tab[c] = bf[c]
        if bands:
            prior_cut = (result.cutouts or {}).get(bands[0])
            fs = flag_sources(
                ra, dec, np.asarray(result.fluxes_ujy[bands[0]], float),
                is_star,
                wcs=getattr(prior_cut, "wcs", None),
                flag_plane=getattr(prior_cut, "flag", None),
                shape=getattr(prior_cut, "shape", None))
            for c in fs.colnames:
                tab[c] = fs[c]
        # det_quality_flag bits 7/8: source inside the MER VIS/NIR
        # bright-star polygon masks.
        mer_cat = getattr(result, "mer_cat", None)
        if (mer_cat is not None and hasattr(mer_cat, "colnames")
                and "det_quality_flag" in mer_cat.colnames
                and len(mer_cat) == n):
            dq = np.ma.filled(
                np.ma.asarray(mer_cat["det_quality_flag"], dtype=np.int64), 0)
            tab["det_quality_flag"] = dq
            tab["mer_bright_star_mask"] = ((dq >> 7) & 1).astype(bool) \
                | ((dq >> 8) & 1).astype(bool)

    if ebv is not None and n > 0:
        from .extinction import add_extinction_columns, ebv_for_result
        if isinstance(ebv, str) and ebv == "auto":
            ebv_val, ebv_source = ebv_for_result(
                result, data_dir=getattr(result, "data_dir", None))
            if ebv_val is None:
                import warnings
                warnings.warn(
                    f"no E(B-V) available ({ebv_source}); skipping the "
                    "extinction columns. Pass ebv= explicitly to add them.",
                    stacklevel=2)
            else:
                add_extinction_columns(tab, ebv_val, ebv_source=ebv_source)
        else:
            add_extinction_columns(tab, ebv)

    if depth:
        depths = {}
        for b in bands:
            err = np.asarray(result.flux_errs_ujy.get(b, ()), dtype=float)
            err = err[np.isfinite(err) & (err > 0)]
            if err.size:
                depths[b] = round(float(flux_to_ab_mag(
                    5.0 * np.median(err))), 3)
        if depths:
            tab.meta["depth_5sigma_ab"] = depths
            tab.meta["depth_definition"] = (
                "AB mag of a source whose flux is 5x the median reported "
                "flux error (point-source forced-photometry depth; uses "
                "the calibrated errors when error calibration ran)")

    from datetime import datetime, timezone

    from . import __version__
    tab.meta["euclid_phot_version"] = __version__
    tab.meta["created_utc"] = datetime.now(timezone.utc).isoformat(
        timespec="seconds")
    tab.meta["target"] = result.target
    tab.meta["prior"] = result.prior
    return tab
