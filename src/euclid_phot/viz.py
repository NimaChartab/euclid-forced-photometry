"""Small matplotlib helpers used by the notebooks."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.nddata import Cutout2D
from astropy.stats import sigma_clipped_stats
import astropy.units as u


def plot_workflow(ax=None, *, save_path=None):
    """Draw the forced-photometry pipeline as a labeled flow diagram.

    Five phases, left to right: archive inputs, source-model
    construction, the prior fit on VIS, forced photometry on
    the target bands, and the multi-band catalog. Pass
    ``save_path`` to write a PNG.
    """
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    if ax is None:
        _, ax = plt.subplots(figsize=(14, 6.2))
    ax.set_xlim(0, 14); ax.set_ylim(0, 6.2); ax.axis("off")

    def box(x, y, w, h, title, body, fc):
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.14",
            linewidth=1.1, edgecolor="0.25", facecolor=fc))
        ax.text(x + w / 2, y + h - 0.26, title, ha="center", va="center",
                fontsize=11, fontweight="bold")
        if body:
            ax.text(x + w / 2, y + (h - 0.42) / 2, body, ha="center",
                    va="center", fontsize=9.6, linespacing=1.45)

    def arrow(x0, y0, x1, y1):
        ax.add_patch(FancyArrowPatch(
            (x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=13,
            linewidth=1.2, color="0.3", shrinkA=2, shrinkB=2))

    def header(xc, text):
        ax.text(xc, 5.85, text, ha="center", va="center", fontsize=11.5,
                color="0.35", fontweight="bold")

    c_in, c_mod, c_fit = "#dce7f3", "#e6e0f0", "#eef0f2"
    c_tgt, c_out = "#e0eedd", "#f7ecd5"

    header(1.45, "Archive inputs")
    box(0.25, 3.95, 2.4, 1.25, "MER catalog",
        "positions, Sersic shapes\n(IRSA TAP)", c_in)
    box(0.25, 2.30, 2.4, 1.25, "Image cutouts",
        "VIS + Y/J/H science + RMS\n(IRSA/S3)", c_in)
    box(0.25, 0.65, 2.4, 1.25, "PSF stamps",
        "CATALOG-PSF at sources;\nGRID-PSF on a 12\" grid", c_in)

    header(4.35, "Source models")
    box(3.25, 1.85, 2.2, 2.5, "Two choices",
        "MER positions\n\n"
        "catalog models or\nVIS model-selection tree", c_mod)

    header(7.15, "Prior fit (VIS)")
    box(6.05, 1.55, 2.2, 3.0, "VIS fit",
        "profiles and positions\nfrom the selected mode;\n"
        "joint source fluxes;\nmasked pixels excluded", c_fit)

    header(9.95, "Forced photometry")
    box(8.85, 3.20, 2.2, 1.35, "NISP Y/J/H",
        "shapes frozen at VIS;\nflux per band", c_tgt)
    box(8.85, 1.55, 2.2, 1.35, "unWISE W1/W2",
        "shapes frozen at VIS;\nflux + joint sky offset", c_tgt)

    header(12.4, "Catalog")
    box(11.65, 1.55, 2.1, 3.0, "Per-object table",
        "fluxes + AB magnitudes;\nestimated errors;\n"
        "foreground corrections;\nquality + blend flags;\n"
        "error summaries", c_out)

    for y in (4.55, 2.90, 1.25):
        arrow(2.65, y, 3.25, 3.10)
    arrow(5.45, 3.10, 6.05, 3.85)
    arrow(8.25, 2.55, 8.85, 3.70)
    arrow(8.25, 2.20, 8.85, 2.20)
    arrow(11.05, 3.90, 11.65, 3.40)
    arrow(11.05, 2.20, 11.65, 2.60)
    if save_path is not None:
        ax.figure.savefig(save_path, dpi=130, bbox_inches="tight")
    return ax


UJY_PER_PIXEL = "µJy/pixel"


def ujy_per_unit(cutout=None, *, band=None, mag_zero=None):
    """Microjansky per image unit, for displaying images in µJy/pixel.

    MER images: ``10**(0.4 * (23.9 - MAGZERO))`` from the cutout header
    (or ``mag_zero``). unWISE (``band="W1"``/``"W2"``): Vega nanomaggies,
    converted with the Vega-to-AB offsets of :mod:`euclid_phot.wise`.
    """
    if band in ("W1", "W2"):
        from .wise import _UJY_PER_NMGY, _VEGA_OFFSET
        return _UJY_PER_NMGY * 10 ** (-_VEGA_OFFSET[band] / 2.5)
    if mag_zero is None:
        header = getattr(cutout, "header", None)
        if header is None or "MAGZERO" not in header:
            raise ValueError("no MAGZERO; pass mag_zero")
        mag_zero = float(header["MAGZERO"])
    return 10 ** (0.4 * (23.9 - float(mag_zero)))


def add_colorbar(ax, image=None, label=None):
    """Attach a colorbar of the same height to ``ax`` for its first image."""
    image = ax.images[0] if image is None else image
    cbar = ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
    if label:
        cbar.set_label(label)
    return cbar


def show_cutout(cutout, mer_cat=None,
                *,
                ax=None,
                vmin_pct: float = 1.0,
                vmax_pct: float = 99.0,
                source_marker_color: str | None = None,
                title: str | None = None,
                colorbar: bool = True,
                unit: str | None = None):
    """Linear-stretch grayscale cutout, with optional MER source overlay.

    Images with a ``MAGZERO`` header keyword are shown in µJy/pixel;
    ``colorbar`` adds a colorbar labelled with the unit.

    ``source_marker_color`` is auto-set to cyan for stars, lime for galaxies
    when ``mer_cat`` carries the ``is_star`` column; pass an explicit color
    to override.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    data = np.asarray(cutout.data if hasattr(cutout, "data") else cutout)
    header = getattr(cutout, "header", None)
    if unit is None and header is not None and "MAGZERO" in header:
        data = data * ujy_per_unit(cutout)
        unit = UJY_PER_PIXEL
    finite = data[np.isfinite(data) & (data != 0)]
    vmin, vmax = np.nanpercentile(finite, [vmin_pct, vmax_pct])
    image = ax.imshow(data, origin="lower", cmap="gray_r", vmin=vmin, vmax=vmax)
    if colorbar:
        if unit is None and header is not None:
            unit = header.get("BUNIT")
        add_colorbar(ax, image, unit or "image units")

    if mer_cat is not None and hasattr(cutout, "wcs"):
        has_isstar = "is_star" in mer_cat.colnames if hasattr(mer_cat, "colnames") else False
        for row in mer_cat:
            px, py = cutout.wcs.world_to_pixel_values(row["ra"], row["dec"])
            if source_marker_color is not None:
                color = source_marker_color
            elif has_isstar:
                color = "cyan" if bool(row["is_star"]) else "lime"
            else:
                color = "lime"
            ax.plot(px, py, "o", ms=8, mfc="none", mec=color, mew=0.8)

    if title is not None:
        ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([])
    return ax


def show_residual(data, model, footprint=None,
                  *,
                  ax=None, vlim_sigma: float = 5.0, title: str | None = None,
                  invvar=None, colorbar: bool = True, unit: str | None = None,
                  scale: float = 1.0):
    """Plot ``data - model`` against a stated noise reference.

    With ``invvar=None`` the color limits are
    +/-vlim_sigma x sigma_clipped_std(residual); if the noise model is
    violated by a factor k, the displayed range is also k wider than the
    formal floor. With ``invvar`` the limits are
    +/-vlim_sigma x 1/sqrt(median(invvar)), the formal per-pixel noise,
    which is the scale to use when judging consistency with the noise model.
    ``scale`` multiplies the displayed residual, e.g. ``ujy_per_unit(cutout)``
    with ``unit="µJy/pixel"``; the returned ``info`` stays in input units.

    Returns ``(ax, info)`` where ``info`` is a dict with the displayed
    ``sigma``, the empirical ``residual_std`` (sigma-clipped), the
    measured ``chi_std`` and ``chi_mad`` when ``invvar`` is given, and
    the fraction of pixels with ``|chi| > 3``.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    residual = np.asarray(data) - np.asarray(model)

    if footprint is not None and (~footprint).sum() > 100:
        _, _, emp_sigma = sigma_clipped_stats(residual[~footprint], sigma=3.0)
    else:
        _, _, emp_sigma = sigma_clipped_stats(residual, sigma=3.0)

    info: dict = {"residual_std": float(emp_sigma)}

    if invvar is not None:
        invvar = np.asarray(invvar)
        good = invvar > 0
        if good.any():
            # 1/sqrt(median(invvar)), not median(1/sqrt(invvar)): low-invvar
            # fill pixels at a coverage edge drag the latter high.
            formal_sigma = float(1.0 / np.sqrt(np.median(invvar[good])))
        else:
            formal_sigma = float(emp_sigma)
        info["formal_sigma"] = formal_sigma
        info["sigma_used"] = formal_sigma
        with np.errstate(invalid="ignore"):
            chi = np.where(good, residual * np.sqrt(invvar), np.nan)
        finite = chi[np.isfinite(chi)]
        if finite.size:
            info["chi_std"] = float(np.std(finite))
            from astropy.stats import mad_std as _mad
            info["chi_mad"] = float(_mad(finite))
            info["pct_chi_gt_3"] = float(100.0 * (np.abs(finite) > 3).mean())
        vlim = vlim_sigma * formal_sigma
    else:
        info["sigma_used"] = float(emp_sigma)
        vlim = vlim_sigma * float(emp_sigma)

    shown = residual
    cmap = plt.get_cmap("RdBu_r")
    if invvar is not None:
        # Zero-weight pixels render gray: the fit never saw them.
        shown = np.where(np.asarray(invvar) > 0, residual, np.nan)
        cmap = cmap.copy()
        cmap.set_bad("0.75")
    image = ax.imshow(shown * scale, origin="lower", cmap=cmap,
                      vmin=-vlim * scale, vmax=vlim * scale)
    if colorbar:
        add_colorbar(ax, image, f"data − model ({unit})" if unit else "data − model")
    if title is not None:
        ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([])
    return ax, info


def show_chi_map(data, model, invvar, *,
                 ax=None, vlim: float = 5.0, title: str | None = None,
                 colorbar: bool = True):
    """Plot ``chi = (data - model) * sqrt(invvar)`` on a +/-vlim sigma scale.

    A perfect fit against a correct invvar model produces N(0,1) chi;
    pixels beyond |chi| ~ 3 mark a model deficiency.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    residual = np.asarray(data) - np.asarray(model)
    invvar = np.asarray(invvar)
    with np.errstate(invalid="ignore"):
        chi = np.where(invvar > 0, residual * np.sqrt(invvar), np.nan)
    # Zero-weight pixels have no chi; show them gray.
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("0.75")
    image = ax.imshow(chi, origin="lower", cmap=cmap, vmin=-vlim, vmax=vlim)
    if colorbar:
        add_colorbar(ax, image, "(data − model) / σ_pixel")
    if title is not None:
        ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([])
    finite = chi[np.isfinite(chi)]
    from astropy.stats import mad_std
    info = {"chi_std": float(np.std(finite)) if finite.size else float("nan"),
            # Robust width, insensitive to bright-source cores.
            "chi_mad": float(mad_std(finite)) if finite.size else float("nan"),
            "pct_chi_gt_3": float(100.0 * (np.abs(finite) > 3).mean())
            if finite.size else float("nan")}
    return ax, info


def show_psf_grid(grid: dict, *, cutout=None, mer_cat=None,
                  n_examples: int = 3, log_floor: float = 1e-5):
    """Spatial layout and variation of a PSF-stamp extraction.

    The left panel maps every stamp position, colored by FWHM, with the
    cutout footprint and (optionally) the MER sources overlaid; it shows
    whether the product covers the field and how much the PSF varies
    across it. The remaining panels show the stamps at the 5th
    percentile, median, and 95th percentile FWHM on a logarithmic
    stretch. Applied to a
    GRID-PSF extraction the map traces the regular grid; applied to a
    CATALOG-PSF extraction it traces the source distribution.

    ``grid`` is the dict returned by
    :func:`euclid_phot.psf.extract_grid_psf` (or
    :func:`euclid_phot.psf.extract_catalog_psf`; the two share a schema).
    Returns ``(fig, info)`` where ``info`` holds the FWHM statistics and
    the median spacing between neighboring stamps in arcsec.
    """
    ra = np.asarray(grid["ra"], float)
    dec = np.asarray(grid["dec"], float)
    fwhm = np.asarray(grid["fwhm"], float)
    stamps = np.asarray(grid["stamps"], float)
    finite = np.isfinite(fwhm)

    # Median nearest-neighbor separation, in arcsec, on the tangent plane.
    cosd = np.cos(np.deg2rad(np.median(dec)))
    dx = (ra[:, None] - ra[None, :]) * cosd
    dy = dec[:, None] - dec[None, :]
    dist = np.hypot(dx, dy) * 3600.0
    np.fill_diagonal(dist, np.inf)
    spacing = float(np.median(dist.min(axis=1))) if len(ra) > 1 else np.nan

    order = np.flatnonzero(finite)[np.argsort(fwhm[finite])]
    if len(order):
        qi = [int(round(q * (len(order) - 1))) for q in (0.05, 0.50, 0.95)]
        picks = [order[i] for i in qi][:n_examples]
    else:
        picks = []
    labels = ["5th pct FWHM", "median FWHM", "95th pct FWHM"][:len(picks)]

    fig = plt.figure(figsize=(5.4 + 2.5 * len(picks), 4.2))
    gs = fig.add_gridspec(1, 1 + len(picks),
                          width_ratios=[2.0] + [1.0] * len(picks),
                          wspace=0.35)
    axm = fig.add_subplot(gs[0])
    sc = axm.scatter(ra, dec, c=fwhm, s=26, cmap="viridis", zorder=2,
                     alpha=0.85, label="PSF stamps")
    if mer_cat is not None:
        axm.scatter(np.asarray(mer_cat["ra"], float),
                    np.asarray(mer_cat["dec"], float),
                    s=2, c="k", marker=".", zorder=3,
                    label="MER sources")
    if cutout is not None and hasattr(cutout, "wcs"):
        H, W = cutout.shape
        cx = np.array([-0.5, W - 0.5, W - 0.5, -0.5, -0.5])
        cy = np.array([-0.5, -0.5, H - 0.5, H - 0.5, -0.5])
        cra, cdec = cutout.wcs.pixel_to_world_values(cx, cy)
        axm.plot(cra, cdec, "k--", lw=1.0, zorder=3, label="cutout")
    axm.invert_xaxis()
    axm.set_xlabel("RA (deg)")
    axm.set_ylabel("Dec (deg)")
    axm.legend(fontsize=11, loc="upper right")
    cb = fig.colorbar(sc, ax=axm, fraction=0.046, pad=0.04)
    cb.ax.set_title("FWHM\n(arcsec)", fontsize=12)

    for k, (i, lab) in enumerate(zip(picks, labels, strict=True)):
        ax = fig.add_subplot(gs[1 + k])
        norm = stamps[i] / stamps[i].max()
        ax.imshow(np.log10(np.maximum(norm, log_floor)), origin="lower",
                  cmap="magma", vmin=np.log10(log_floor), vmax=0)
        ax.set_title(f"{lab}\n{fwhm[i]:.3f}\"", fontsize=12)
        ax.set_xticks([]); ax.set_yticks([])

    info = {"n_stamps": int(len(ra)),
            "median_spacing_arcsec": spacing,
            "fwhm_min": float(np.nanmin(fwhm)) if finite.any() else np.nan,
            "fwhm_median": float(np.nanmedian(fwhm)) if finite.any() else np.nan,
            "fwhm_max": float(np.nanmax(fwhm)) if finite.any() else np.nan}
    return fig, info


_WAVELENGTHS_UM = {"VIS": 0.71, "Y": 1.08, "J": 1.37, "H": 1.77,
                   "W1": 3.368, "W2": 4.618}


def show_error_calibration(calib: dict, *, ax=None):
    """Histogram of the empty-position flux/error ratios behind a calibration.

    ``calib`` is one band's dict from
    :func:`euclid_phot.calibrate.measure_error_inflation` (or one entry of
    ``result.error_calibration``). If the formal errors were correct, the
    normalized fluxes (flux/err at source-free positions) would follow
    N(0, 1); the measured width is the inflation factor. Both Gaussians are
    overplotted so the miscalibration is visible at a glance.
    """
    chi = np.asarray(calib.get("chi", []), dtype=float)
    if chi.size == 0:
        raise ValueError(
            "calib carries no per-position samples ('chi'); pass the dict "
            "returned by measure_error_inflation / calibrate_result_errors.")
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    k = float(calib.get("chi_mad", np.nan))
    lim = max(4.0, 3.5 * (k if np.isfinite(k) else 1.0))
    bins = np.linspace(-lim, lim, 41)
    ax.hist(chi, bins=bins, density=True, histtype="stepfilled",
            alpha=0.45, color="C0",
            label=f"empty positions (n={chi.size})")
    x = np.linspace(-lim, lim, 400)
    norm = 1.0 / np.sqrt(2 * np.pi)
    ax.plot(x, norm * np.exp(-x**2 / 2.0), "k--", lw=1.2,
            label="N(0,1): formal errors correct")
    if np.isfinite(k) and k > 0:
        ax.plot(x, (norm / k) * np.exp(-x**2 / (2 * k**2)), "C3-", lw=1.5,
                label=f"N(0,{k:.2f}): measured")
    band = calib.get("band", "?")
    ax.set_xlabel("flux / formal error at source-free positions")
    ax.set_ylabel("density")
    ax.set_title(f"{band}: error inflation x{calib.get('inflation', np.nan):.2f}"
                 f" ({calib.get('method', '')})", fontsize=13)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    return ax


def show_dmag_vs_mag(ref_ujy, flux_ujy, *,
                     err_ujy=None, ref_err_ujy=None,
                     sel=None, flagged=None,
                     ax=None, gridsize: int = 45,
                     target_per_bin: int = 80, max_bins: int = 15,
                     min_per_bin: int = 5, ylim: float = 1.0,
                     colorbar: bool = True,
                     xlabel: str = "reference AB magnitude",
                     ylabel: str = "m_Tractor - m_ref (mag)",
                     title: str | None = None):
    """Magnitude difference vs magnitude against a reference catalog.

    A log-density hexbin of ``-2.5 log10(flux/ref)`` against the reference
    AB magnitude, with the running median, the +/- NMAD band, and (when the
    two pipelines' errors are given) an independent-error reference scale,
    all in equal-population magnitude bins. Shared data can correlate the
    measurements; agreement with this curve does not establish noise-only
    scatter or calibrated uncertainties.

    Parameters
    ----------
    ref_ujy, flux_ujy : ndarray
        Reference and measured fluxes (microJansky), aligned.
    err_ujy, ref_err_ujy : ndarray, optional
        The two pipelines' 1-sigma flux errors; both are needed for the
        independent-error reference curve.
    sel : ndarray of bool, optional
        Points entering the hexbin and the binned statistics. Defaults to
        every position where both fluxes are finite and positive.
    flagged : ndarray of bool, optional
        Overplotted as open gray circles (label ``'flagged'``) and kept out
        of the binned statistics.
    ylim : float
        Half-height of the panel in magnitudes.
    colorbar : bool
        Attach a colorbar for the hexbin density (sources per bin, log
        scale). Default True.

    Returns
    -------
    (ax, info) : info holds the binned curves -- ``centers``, ``median``,
        ``nmad``, ``expected`` (NaN when errors were not given) -- and
        ``n_bins``.
    """
    from astropy.stats import mad_std

    ref = np.asarray(ref_ujy, dtype=float)
    flux = np.asarray(flux_ujy, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        mag = 23.9 - 2.5 * np.log10(np.where(ref > 0, ref, np.nan))
        dmag = -2.5 * np.log10(np.where(flux > 0, flux, np.nan)
                               / np.where(ref > 0, ref, np.nan))
        pred = None
        if err_ujy is not None and ref_err_ujy is not None:
            pred = 1.0857 * np.sqrt(
                (np.asarray(err_ujy, dtype=float) / flux) ** 2
                + (np.asarray(ref_err_ujy, dtype=float) / ref) ** 2)

    ok = np.isfinite(mag) & np.isfinite(dmag)
    sel = ok if sel is None else (np.asarray(sel, dtype=bool) & ok)
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.5, 6.5))
    if not sel.any():
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
        return ax, {"centers": np.array([]), "median": np.array([]),
                    "nmad": np.array([]), "expected": np.array([]),
                    "n_bins": 0}

    x0 = float(np.floor(np.nanmin(mag[sel])))
    x1 = float(np.ceil(np.nanmax(mag[sel])))
    hb = ax.hexbin(mag[sel], dmag[sel], gridsize=gridsize, cmap="YlGnBu",
                   bins="log", mincnt=1, extent=(x0, x1, -ylim, ylim),
                   linewidths=0.2)
    if colorbar:
        cb = ax.figure.colorbar(hb, ax=ax, pad=0.02)
        cb.set_label("sources per bin")
    if flagged is not None:
        fl = np.asarray(flagged, dtype=bool) & ok
        ax.scatter(mag[fl], dmag[fl], s=24, facecolors="none",
                   edgecolors="0.5", label="flagged")

    # Running median, NMAD, and expected scatter in equal-population bins.
    mvals, dvals = mag[sel], dmag[sel]
    pvals = pred[sel] if pred is not None else None
    nbins = max(4, min(max_bins, mvals.size // target_per_bin))
    edges = np.quantile(mvals, np.linspace(0, 1, nbins + 1))
    ctr, med, nmad, prd = [], [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        inb = (mvals >= lo) & (mvals <= hi)
        if inb.sum() >= min_per_bin:
            ctr.append(float(np.median(mvals[inb])))
            med.append(float(np.median(dvals[inb])))
            nmad.append(float(mad_std(dvals[inb])))
            prd.append(float(np.nanmedian(pvals[inb]))
                       if pvals is not None else float("nan"))
    ctr, med, nmad, prd = (np.array(ctr), np.array(med),
                           np.array(nmad), np.array(prd))
    ax.plot(ctr, med, "k-", lw=1.6, label="median")
    ax.fill_between(ctr, med - nmad, med + nmad, color="k", alpha=0.15,
                    label="+/- NMAD")
    if pvals is not None and np.isfinite(prd).any():
        ax.plot(ctr, prd, "r:", lw=1.5, label="independent-error reference")
        ax.plot(ctr, -prd, "r:", lw=1.5)
    ax.axhline(0, color="r", ls="--", lw=1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_ylim(-ylim, ylim)
    if title is not None:
        ax.set_title(title)
    return ax, {"centers": ctr, "median": med, "nmad": nmad,
                "expected": prd, "n_bins": int(len(ctr))}


def _sky_cutout(image, wcs, ra, dec, size_arcsec):
    cut = Cutout2D(np.asarray(image), SkyCoord(ra * u.deg, dec * u.deg),
                   size_arcsec * u.arcsec, wcs=wcs, mode="partial",
                   fill_value=np.nan, copy=True)
    return np.asarray(cut.data, dtype=float), cut.wcs


def _offset_frame(wcs, shape, ra, dec):
    """Pixel -> (RA, Dec) offset in arcsec for a cutout, and its extent.

    Offsets come from the cutout WCS, so markers and pixels always agree.
    The RA axis points east, which is to the left on a north-up image.
    """
    ny, nx = shape
    cx, cy = (nx - 1) / 2.0, (ny - 1) / 2.0
    scale = wcs.proj_plane_pixel_scales()[0].to_value(u.arcsec)
    ra0, _ = wcs.pixel_to_world_values(cx, cy)
    ra1, _ = wcs.pixel_to_world_values(cx + 1, cy)
    sx = 1.0 if ((ra1 - ra0 + 180.0) % 360.0 - 180.0) > 0 else -1.0

    def to_offsets(src_ra, src_dec):
        px, py = wcs.world_to_pixel_values(src_ra, src_dec)
        return sx * (px - cx) * scale, (py - cy) * scale

    extent = (sx * (-0.5 - cx) * scale, sx * (nx - 0.5 - cx) * scale,
              (-0.5 - cy) * scale, (ny - 0.5 - cy) * scale)
    return to_offsets, extent


def _positive_vmax(image, percentile=99.5):
    values = np.asarray(image)
    values = values[np.isfinite(values) & (values > 0)]
    vmax = float(np.percentile(values, percentile)) if values.size else 1.0
    return vmax if np.isfinite(vmax) and vmax > 0 else 1.0


def show_images(images, *, title=None, units=None):
    """Show a row of observed images, given as ``{label: 2-D array}``.

    Each panel has its own colorbar; ``units`` maps labels to unit strings.
    """
    units = units or {}
    fig, axes = plt.subplots(1, len(images), figsize=(4 * len(images), 4.6),
                             squeeze=False)
    for ax, (label, image) in zip(axes[0], images.items()):
        lo, hi = np.nanpercentile(image, [2.0, 99.5])
        shown = ax.imshow(image, origin="lower", cmap="gray_r", vmin=lo, vmax=hi)
        add_colorbar(ax, shown, units.get(label))
        ax.set_title(label, fontsize=16)
        ax.set_axis_off()
    if title:
        fig.suptitle(title, fontsize=17)
    fig.tight_layout()
    return fig, axes


def show_fit(result, bands=("VIS",), *, ra=None, dec=None, size_arcsec=50.0,
             vmax_percentile=99.5, show_sources=True):
    """Data, model and residual rows for the given bands of a fit.

    ``result`` is a :class:`ForcedPhotometryResult`; ``bands`` may hold the
    prior band and any fitted WISE or IRAC band. Data and model share one
    scale per band; the residual uses the same absolute scale, symmetric
    about zero. Circles mark the fitted sources, placed through each
    cutout's WCS, with east to the left.
    """
    ra = result.target[0] if ra is None else float(ra)
    dec = result.target[1] if dec is None else float(dec)
    bands = (bands,) if isinstance(bands, str) else tuple(bands)
    src_ra = np.array([float(s.getPosition().ra) for s in result.sources])
    src_dec = np.array([float(s.getPosition().dec) for s in result.sources])

    fig, axes = plt.subplots(3, len(bands), figsize=(4.4 * len(bands), 12.5),
                             squeeze=False)
    for col, band in enumerate(bands):
        images = result.image_set(band)
        panels = {}
        for kind in ("data", "model", "residual"):
            panels[kind], cut_wcs = _sky_cutout(images[kind], images["wcs"],
                                                ra, dec, size_arcsec)
        to_offsets, extent = _offset_frame(cut_wcs, panels["data"].shape, ra, dec)
        dx, dy = to_offsets(src_ra, src_dec)
        inside = ((dx - extent[0]) * (dx - extent[1]) <= 0) & \
                 ((dy - extent[2]) * (dy - extent[3]) <= 0)
        vmax = _positive_vmax(panels["data"], vmax_percentile)
        for row, (kind, label) in enumerate((("data", "data"),
                                             ("model", "model"),
                                             ("residual", "residual"))):
            ax = axes[row, col]
            residual = kind == "residual"
            shown = ax.imshow(panels[kind], origin="lower", extent=extent,
                              cmap="RdBu_r" if residual else "gray_r",
                              vmin=-vmax if residual else 0.0, vmax=vmax)
            if show_sources:
                ax.scatter(dx[inside], dy[inside], s=18, facecolors="none",
                           edgecolors="tab:blue", linewidths=0.45)
            ax.set_title(f"{band} {label}", fontsize=14)
            if row == 2:
                ax.set_xlabel("RA offset (arcsec)")
            if col == 0:
                ax.set_ylabel("Dec offset (arcsec)")
            cbar = fig.colorbar(shown, ax=ax, fraction=0.046, pad=0.04)
            if images.get("unit"):
                cbar.set_label(images["unit"])
    fig.tight_layout()
    return fig, axes


def show_dawn_comparison(comparisons):
    """Pipeline versus DAWN flux for the selected sample of each channel."""
    fig, axes = plt.subplots(1, len(comparisons),
                             figsize=(5.6 * len(comparisons), 5.0), squeeze=False)
    for ax, (channel, matches) in zip(axes[0], comparisons.items()):
        band = channel.upper()
        sel = np.asarray(matches["comparison_selected"], bool)
        ours = np.asarray(matches[f"flux_{band}_ujy"], float)[sel]
        ours_err = np.asarray(matches[f"flux_err_{band}_ujy"], float)[sel]
        dawn = np.asarray(matches[f"dawn_flux_{band}_ujy"], float)[sel]
        dawn_err = np.asarray(matches[f"dawn_flux_err_{band}_ujy"], float)[sel]
        ax.errorbar(dawn, ours, xerr=dawn_err, yerr=ours_err, fmt="o", ms=3.5,
                    alpha=0.45, color="tab:blue", ecolor="0.7",
                    elinewidth=0.5, capsize=0)
        both = np.r_[dawn, ours]
        both = both[np.isfinite(both) & (both > 0)]
        if both.size:
            lo, hi = np.percentile(both, [1.0, 99.0])
            ax.plot([lo, hi], [lo, hi], "k--", lw=1)
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(f"DAWN {channel} flux (µJy)")
        ax.set_ylabel(f"This work {channel} flux (µJy)")
        ratio = np.asarray(matches[f"{band}_over_dawn"], float)[sel]
        median = float(np.nanmedian(ratio)) if ratio.size else np.nan
        ax.set_title(f"{channel}: N={sel.sum()}, median ratio {median:.3f}",
                     fontsize=14)
        ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig, axes


def show_wise_irac_comparison(comparisons, *, pairs=None, minimum_snr=5.0):
    """unWISE versus IRAC flux, and their ratio, for each band pair.

    Every source with positive fluxes is shown. Blue sources are selected
    on IRAC alone: IRAC S/N of at least ``minimum_snr`` and IRAC flux at
    least ``minimum_snr`` times the WISE error, so WISE could measure them
    whatever flux it assigned. Selecting on WISE S/N instead would keep
    faint sources only when WISE gave them extra flux. Orange rings mark
    VIS-isolated sources; medians use the blue sample.
    """
    from .irac.catalogs import WISE_IRAC_PAIRS

    pairs = WISE_IRAC_PAIRS if pairs is None else dict(pairs)
    fig, axes = plt.subplots(2, len(comparisons),
                             figsize=(5.8 * len(comparisons), 9.4),
                             squeeze=False, sharex="col",
                             gridspec_kw={"height_ratios": [2.2, 1]})
    for col, (wise_band, table) in enumerate(comparisons.items()):
        channel = pairs[wise_band]
        w, c = wise_band.upper(), channel.upper()
        wise = np.asarray(table[f"flux_{w}_ujy"], float)
        wise_err = np.asarray(table[f"flux_err_{w}_ujy"], float)
        irac = np.asarray(table[f"flux_{c}_ujy"], float)
        irac_err = np.asarray(table[f"flux_err_{c}_ujy"], float)
        isolated = np.asarray(table["isolated"], bool)
        with np.errstate(divide="ignore", invalid="ignore"):
            shown = (wise > 0) & (irac > 0)
            bright = shown & (irac / irac_err >= minimum_snr) & \
                (irac >= minimum_snr * wise_err)
            ratio = wise / irac
        faint = shown & ~bright
        n_hidden = int((np.isfinite(wise) & np.isfinite(irac) & ~shown).sum())
        med_all = float(np.median(ratio[bright])) if bright.any() else np.nan
        iso = bright & isolated
        med_iso = float(np.median(ratio[iso])) if iso.any() else np.nan

        for row, y in ((0, wise), (1, ratio)):
            ax = axes[row, col]
            ax.scatter(irac[faint], y[faint], s=6, color="0.75", alpha=0.6,
                       label=f"other ({faint.sum()})")
            ax.scatter(irac[bright], y[bright], s=10, color="tab:blue",
                       alpha=0.6, label=f"IRAC-selected ({bright.sum()})")
            ax.scatter(irac[iso], y[iso], s=50, facecolors="none",
                       edgecolors="tab:orange", linewidths=1.2,
                       label=f"IRAC-selected, isolated ({iso.sum()})")
            ax.set_xscale("log")
            ax.grid(alpha=0.2)

        ax = axes[0, col]
        both = np.r_[irac[shown], wise[shown]]
        lo, hi = np.percentile(both, [0.5, 99.9]) if both.size else (0.1, 100)
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_yscale("log")
        ax.set_ylabel(f"{wise_band} flux (µJy)")
        ax.set_title(f"{wise_band} vs {channel}", fontsize=14)
        ax.legend(loc="upper left", fontsize=10)

        ax = axes[1, col]
        ax.axhline(1.0, color="k", ls="--", lw=1)
        ax.axhline(med_all, color="tab:blue", lw=1.2,
                   label=f"median, IRAC-selected: {med_all:.3f}")
        ax.axhline(med_iso, color="tab:orange", lw=1.2,
                   label=f"median, isolated: {med_iso:.3f}")
        ax.set_yscale("log")
        ax.set_ylim(0.1, 10)
        ax.set_xlabel(f"{channel} flux (µJy)")
        ax.set_ylabel(f"{wise_band} / {channel}")
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[-2:], labels[-2:], loc="upper right", fontsize=9)
        if n_hidden:
            ax.text(0.02, 0.04, f"{n_hidden} with a non-positive flux not shown",
                    transform=ax.transAxes, fontsize=9, color="0.4")
    fig.tight_layout()
    return fig, axes


def show_unwise_comparison(comparisons):
    """Our WISE fluxes against the Schlafly et al. (2019) catalog.

    ``comparisons`` is the output of :func:`euclid_phot.compare_unwise_2019`.
    Grey: matched sources with a bright VIS neighbour; open circles:
    VIS-isolated with low S/N; filled: VIS-isolated above the S/N cut.
    """
    fig, axes = plt.subplots(1, len(comparisons),
                             figsize=(6.5 * len(comparisons), 6), squeeze=False)
    for ax, (band, t) in zip(axes[0], comparisons.items()):
        ref = np.asarray(t["catalog_flux_ujy"], float)
        flux = np.asarray(t["flux_ujy"], float)
        err = np.asarray(t["flux_err_ujy"], float)
        sel = np.asarray(t["selected"], bool)
        iso = np.asarray(t["isolated"], bool)
        hi = np.asarray(t["clean"], bool)
        low = sel & iso & ~hi
        crowded = sel & ~iso
        snr = t.meta.get("minimum_snr", 5.0)
        ax.scatter(ref[crowded], flux[crowded], color="lightgray", s=18,
                   label=f"VIS neighbour (n={crowded.sum()})")
        ax.scatter(ref[low], flux[low], facecolors="none", edgecolors="C0", s=36,
                   label=f"VIS-isolated, S/N≤{snr:g} (n={low.sum()})")
        ax.errorbar(ref[hi], flux[hi], yerr=err[hi], fmt="o", color="C0", ms=5,
                    capsize=2, label=f"VIS-isolated, S/N>{snr:g} (n={hi.sum()})")
        if sel.any():
            limits = [ref[sel].min() * 0.5, ref[sel].max() * 2]
            ax.plot(limits, limits, "r--", label="1:1")
        ax.set(xscale="log", yscale="log", title=band,
               xlabel="Schlafly flux (µJy, catalog convention)",
               ylabel="Tractor flux (µJy, 19×19-pixel PSF convention)")
        ax.legend(fontsize=11)
    fig.tight_layout()
    return fig, axes

