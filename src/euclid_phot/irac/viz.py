"""Matplotlib helpers used by the IRAC notebook."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.nddata import Cutout2D
import astropy.units as u


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


def show_inputs(images, *, title=None):
    """Show a row of observed images, given as ``{label: 2-D array}``."""
    fig, axes = plt.subplots(1, len(images), figsize=(4 * len(images), 4.6),
                             squeeze=False)
    for ax, (label, image) in zip(axes[0], images.items()):
        lo, hi = np.nanpercentile(image, [2.0, 99.5])
        ax.imshow(image, origin="lower", cmap="gray_r", vmin=lo, vmax=hi)
        ax.set_title(label, fontsize=16)
        ax.set_axis_off()
    if title:
        fig.suptitle(title, fontsize=17)
    fig.tight_layout()
    return fig, axes


def show_fit(run, channels=("IRAC1",), *, ra=None, dec=None, size_arcsec=50.0,
             vmax_percentile=99.5, show_sources=True):
    """Data, model and residual rows for VIS and the given IRAC channels.

    Data and model share one scale per band; the residual uses the same
    absolute scale, symmetric about zero. Circles mark the prior sources.
    """
    ra = run.ra if ra is None else float(ra)
    dec = run.dec if dec is None else float(dec)
    channels = (channels,) if isinstance(channels, str) else tuple(channels)
    bands = ("VIS", *(str(c).upper() for c in channels))

    table = run.to_table()
    src_ra = np.asarray(table["prior_ra"], float)
    src_dec = np.asarray(table["prior_dec"], float)

    fig, axes = plt.subplots(3, len(bands), figsize=(4.4 * len(bands), 12.5),
                             squeeze=False)
    for col, band in enumerate(bands):
        images = run.image_set(band)
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
        key = channel.lower()
        sel = np.asarray(matches["comparison_selected"], bool)
        ours = np.asarray(matches[f"flux_{key}_ujy"], float)[sel]
        ours_err = np.asarray(matches[f"fluxerr_{key}_ujy"], float)[sel]
        dawn = np.asarray(matches[f"dawn_{key}_flux_ujy"], float)[sel]
        dawn_err = np.asarray(matches[f"dawn_{key}_fluxerr_ujy"], float)[sel]
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
        ratio = np.asarray(matches[f"{key}_over_dawn"], float)[sel]
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
    from .catalogs import WISE_IRAC_PAIRS

    pairs = WISE_IRAC_PAIRS if pairs is None else dict(pairs)
    fig, axes = plt.subplots(2, len(comparisons),
                             figsize=(5.8 * len(comparisons), 9.4),
                             squeeze=False, sharex="col",
                             gridspec_kw={"height_ratios": [2.2, 1]})
    for col, (wise_band, table) in enumerate(comparisons.items()):
        channel = pairs[wise_band]
        w, c = wise_band.lower(), channel.lower()
        wise = np.asarray(table[f"flux_{w}_ujy"], float)
        wise_err = np.asarray(table[f"fluxerr_{w}_ujy"], float)
        irac = np.asarray(table[f"flux_{c}_ujy"], float)
        irac_err = np.asarray(table[f"fluxerr_{c}_ujy"], float)
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
