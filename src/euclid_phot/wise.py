"""unWISE cutouts and forced photometry on W1 / W2.

Data flow:

1. ``fetch_unwise_cutouts``: download (and cache) the unwise.me tarball,
   extract per-band img/invvar FITS.
2. ``get_wise_psf``: construct the spatially-correct unWISE PSF via the
   ``unwise_psf`` library.
3. ``fit_wise_forced``: forced photometry at the Euclid source
                                     positions on a smaller "fit" cutout that
                                     contains the sources plus a PSF-wing
                                     buffer, with a jointly-fit constant sky,
                                     an empirical chi-based error inflation,
                                     and outside-prior unWISE-catalog point
                                     sources in the buffer ring.

Method notes. Lang, Hogg & Schlegel (2016) is the canonical reference for
WISE forced photometry from a higher-resolution prior. Differences here:
(i) one joint scalar sky is fit (small cutouts retain a residual pedestal);
(ii) the VIS model classes and shapes are kept, frozen, with flux the only
free parameter, as for NISP (``source_models="point"`` restores the
Lang+2016 point-source reduction; the Euclid profiles are unresolved at
unWISE resolution, so the two agree closely); (iii) the flux errors carry
an empirical chi inflation (the
chi scatter on source-sparse pixels, validated against the Schlafly+2019
``dflux``), where Lang+2016 report formal errors and legacypipe floors the
per-pixel variance and adds a source-Poisson term.
"""
from __future__ import annotations

import sys
import tarfile
import urllib.request
from copy import deepcopy as _deepcopy
from pathlib import Path

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.stats import mad_std
from astropy.wcs import WCS
from astroquery.ipac.irsa import Irsa
from tractor import (
    ConstantSky,
    Image,
    LinearPhotoCal,
    NanoMaggies,
    PointSource,
    RaDecPos,
    Tractor,
)
from tractor.galaxy import disable_galaxy_cache
from tractor.psf import PixelizedPSF
from tractor.wcs import ConstantFitsWcs

from .config import DEFAULT_WISE_CACHE_DIR, UNWISE_PIXEL_SCALE, WISE_COADD_VERSION
from .images import AstropyWCSAdapter

_UJY_PER_NMGY = 3.631
_VEGA_OFFSET = {"W1": 2.699, "W2": 3.339}
_WISE_FWHM_ARCSEC = 6.94


def vega_mag_to_ujy(mag_vega, band: str):
    """Convert a WISE Vega magnitude to AB microJansky.

    Accepts scalar or array input. The Vega -> AB offsets are
    ``W1 = 2.699``, ``W2 = 3.339`` (Lang 2014 / Schlafly+2019, the
    convention used by unWISE and CatWISE2020).

    m_AB = m_Vega + Vega_offset[band]; f_uJy = 3.631 * 10**((22.5 - m_AB) / 2.5).
    """
    m_ab = np.asarray(mag_vega, dtype=float) + _VEGA_OFFSET[band]
    return _UJY_PER_NMGY * 10.0 ** ((22.5 - m_ab) / 2.5)


def query_catwise2020(ra: float, dec: float, radius_arcsec: float):
    """Pull the CatWISE2020 catalog inside ``radius_arcsec`` of (ra, dec).

    Returns an astropy Table with columns ``ra``, ``dec``, ``ra_pm``,
    ``dec_pm`` (proper-motion-corrected positions), ``w1mpro_pm`` /
    ``w2mpro_pm`` (Vega magnitudes from the proper-motion-aware fit),
    their errors, and derived ``w1_ujy`` / ``w2_ujy`` AB-microJansky
    columns ready to compare against ``fit_wise_forced``.

    Catalog: Marocco et al. 2021 ApJS 253 8. The IRSA table is
    ``catwise_2020``. The Vega -> AB offsets used here (W1=2.699,
    W2=3.339; Lang 2014 / Schlafly+2019) match the unWISE convention;
    they differ from Wright+2010 AllWISE by ~0.02 mag and are the right
    choice when comparing to forced photometry on unWISE coadds.
    """
    import math
    for name, val in (("ra", ra), ("dec", dec), ("radius_arcsec", radius_arcsec)):
        if not math.isfinite(val):
            raise ValueError(f"{name} must be finite, got {val!r}")
    if radius_arcsec <= 0:
        raise ValueError(f"radius_arcsec must be positive, got {radius_arcsec}")

    radius_deg = radius_arcsec / 3600.0
    adql = f"""
    SELECT ra, dec, ra_pm, dec_pm,
           w1mpro_pm, w1sigmpro_pm,
           w2mpro_pm, w2sigmpro_pm,
           w1nm, w2nm
    FROM catwise_2020
    WHERE CONTAINS(POINT('J2000', ra, dec),
                   CIRCLE('J2000', {float(ra)}, {float(dec)}, {radius_deg}))=1
    """
    from .netutils import retry
    cat = retry(lambda: Irsa.query_tap(query=adql).to_table(),
                what="IRSA TAP CatWISE query")

    # np.asarray on a MaskedColumn substitutes the ~1e20 fill value;
    # force masked -> NaN.
    def _col_to_nan(col):
        if hasattr(col, "filled"):
            return np.asarray(col.filled(np.nan), dtype=float)
        return np.asarray(col, dtype=float)

    w1mag = _col_to_nan(cat["w1mpro_pm"])
    w2mag = _col_to_nan(cat["w2mpro_pm"])
    cat["w1_ujy"] = np.where(np.isfinite(w1mag),
                              vega_mag_to_ujy(w1mag, "W1"), np.nan)
    cat["w2_ujy"] = np.where(np.isfinite(w2mag),
                              vega_mag_to_ujy(w2mag, "W2"), np.nan)
    return cat


def query_unwise_2019(ra: float, dec: float, radius_arcsec: float):
    """Pull the unWISE catalog (Schlafly et al. 2019) inside a sky circle.

    The natural comparison reference for ``fit_wise_forced``: Schlafly+2019
    used simultaneous PSF fitting (crowdsource), the same algorithm class
    as Tractor, though on earlier-epoch coadds than neo7.

    The IRSA TAP table is ``unwise_2019``. Fluxes are stored in Vega
    nanomaggies (``flux_1`` for W1, ``flux_2`` for W2) and converted to AB
    microJansky via the Lang 2014 / Schlafly+2019 Vega offsets (W1=2.699,
    W2=3.339) and 3.631 microJy = 1 AB nanomaggy. Source-quality cut:
    ``primary_1 = 1`` (the canonical deduplicated detection).

    Returns
    -------
    astropy.table.Table with columns ``ra``, ``dec``, ``flux_1``,
    ``dflux_1``, ``flux_2``, ``dflux_2``, ``primary_1``, ``primary_2``,
    ``flags_unwise_1``, ``flags_unwise_2``, plus derived ``w1_ujy``,
    ``w1_err_ujy``, ``w2_ujy``, ``w2_err_ujy`` in AB microJansky.
    """
    import math
    for name, val in (("ra", ra), ("dec", dec), ("radius_arcsec", radius_arcsec)):
        if not math.isfinite(val):
            raise ValueError(f"{name} must be finite, got {val!r}")
    if radius_arcsec <= 0:
        raise ValueError(f"radius_arcsec must be positive, got {radius_arcsec}")

    radius_deg = radius_arcsec / 3600.0
    adql = f"""
    SELECT ra, dec,
           flux_1, dflux_1, flux_2, dflux_2,
           primary_1, primary_2,
           flags_unwise_1, flags_unwise_2
    FROM unwise_2019
    WHERE CONTAINS(POINT('J2000', ra, dec),
                   CIRCLE('J2000', {float(ra)}, {float(dec)}, {radius_deg}))=1
      AND primary_1 = 1
    """
    from .netutils import retry
    cat = retry(lambda: Irsa.query_tap(query=adql).to_table(),
                what="IRSA TAP unWISE query")

    def _col(col):
        return np.asarray(
            col.filled(np.nan) if hasattr(col, "filled") else col,
            dtype=float)

    f1 = _col(cat["flux_1"])
    f2 = _col(cat["flux_2"])
    df1 = _col(cat["dflux_1"])
    df2 = _col(cat["dflux_2"])
    # Vega nMgy -> AB nMgy: f_AB = f_Vega * 10^(-VegaOffset/2.5).
    # AB nMgy -> AB microJansky: 3.631.
    s1 = 10 ** (-_VEGA_OFFSET["W1"] / 2.5) * _UJY_PER_NMGY
    s2 = 10 ** (-_VEGA_OFFSET["W2"] / 2.5) * _UJY_PER_NMGY
    cat["w1_ujy"] = f1 * s1
    cat["w2_ujy"] = f2 * s2
    cat["w1_err_ujy"] = df1 * s1
    cat["w2_err_ujy"] = df2 * s2
    return cat


def select_isolated_sources(ra, dec, flux,
                            *,
                            radius_arcsec: float | None = None,
                            flux_fraction: float = 1.0 / 3.0):
    """Bright-neighbor isolation cut for clean external-comparison samples.

    Returns a boolean mask, True where a source has no neighbor within
    ``radius_arcsec`` whose flux exceeds ``flux_fraction`` of its own: at
    the 6.94" WISE FWHM a bright neighbor within ~2 FWHM blends into the
    target and corrupts both the forced and the catalog flux. (Lang+2016
    instead compare only unique positional matches within 4".)

    Parameters
    ----------
    ra, dec : array-like (deg)
        Source positions (e.g. the Euclid/MER prior positions).
    flux : array-like
        Per-source flux used for the brightness-ratio test (any consistent
        unit; VIS flux is the natural choice since it defines the prior).
    radius_arcsec : float, optional
        Neighbor search radius. Defaults to ``2 * 6.94" = 13.88"`` (two
        WISE FWHM).
    flux_fraction : float
        A neighbor disqualifies the target if its flux exceeds this fraction
        of the target flux. Default 1/3.

    Returns
    -------
    ndarray of bool, shape (N,)
    """
    if radius_arcsec is None:
        radius_arcsec = 2.0 * _WISE_FWHM_ARCSEC
    ra = np.asarray(ra, dtype=float)
    dec = np.asarray(dec, dtype=float)
    flux = np.asarray(flux, dtype=float)
    n = len(ra)
    coords = SkyCoord(ra, dec, unit="deg")
    isolated = np.ones(n, dtype=bool)
    for i in range(n):
        sep = coords[i].separation(coords).arcsec
        # Neighbors within the radius, excluding self (sep == 0).
        near = (sep > 0) & (sep <= radius_arcsec)
        if not near.any():
            continue
        ti = flux[i]
        if not np.isfinite(ti) or ti <= 0:
            isolated[i] = False
            continue
        if np.any(flux[near] > flux_fraction * ti):
            isolated[i] = False
    return isolated


def _unwise_cutout_url(ra: float, dec: float, size_pix: int,
                       version: str) -> str:
    """Build the unwise.me/cutout_fits request URL.

    Must be https: unwise.me answers plain http with a 308, which urllib
    does not follow on Python < 3.11. ``bands=12`` requests W1 + W2.
    """
    # cutout_fits serves only img/invvar/n/std; file_msk=on is silently
    # ignored. The bitmask is fetched per tile by _fetch_unwise_mask_tile.
    return (
        f"https://unwise.me/cutout_fits?version={version}"
        f"&ra={ra}&dec={dec}&size={size_pix}"
        f"&bands=12&file_img_m=on&file_invvar_m=on"
    )


def _fetch_unwise_mask_tile(coadd_id: str, version: str,
                            ref_wcs: WCS, ref_shape, cache_dir: Path):
    """Fetch the unWISE 31-bit bitmask for ``coadd_id`` and resample it onto the
    cutout grid (``ref_wcs`` / ``ref_shape``).

    Pulls the per-tile ``-msk.fits.gz`` from the coadd archive (~90 kB
    gzipped, a static Meisner 2018 product shared across NEO epochs) and
    reprojects it nearest-neighbor. Returns an int32 array shaped like the
    cutout, or None if the tile is unavailable (masking is then skipped).
    """
    from reproject import reproject_interp

    tile3 = coadd_id[:3]
    url = (f"https://unwise.me/data/{version}/unwise-coadds/fulldepth/"
           f"{tile3}/{coadd_id}/unwise-{coadd_id}-msk.fits.gz")
    local = Path(cache_dir) / f"unwise-{coadd_id}-msk.fits.gz"
    try:
        if not local.exists():
            tmp = local.with_suffix(local.suffix + ".part")
            from .netutils import retry
            retry(lambda: urllib.request.urlretrieve(url, tmp),
                  what="unwise.me mask download")
            with tmp.open("rb") as f:
                if f.read(2) != b"\x1f\x8b":   # not gzip -> an HTML error page
                    tmp.unlink(missing_ok=True)
                    return None
            tmp.replace(local)
        with fits.open(local) as hdul:
            mdata = hdul[0].data.astype(np.float64)
            mwcs = WCS(hdul[0].header)
        arr, footprint = reproject_interp(
            (mdata, mwcs), ref_wcs, shape_out=tuple(ref_shape),
            order="nearest-neighbor")
        out = np.where((footprint > 0) & np.isfinite(arr),
                       np.rint(arr), 0).astype(np.int32)
        return out
    except Exception:
        return None


# unWISE bitmask groups (Meisner 2018, 31 bits). The default drops spike,
# halo, and ghost pixels; the broad near-bright-star bits (0-3) and latent
# bits (13-20) flag large, mostly recoverable areas and stay unset.
def _UNWISE_BIT(*bits):
    return sum(1 << b for b in bits)



UNWISE_SPIKE_BITS = _UNWISE_BIT(27, 28, 29, 30)
UNWISE_HALO_BITS = _UNWISE_BIT(23, 24)
UNWISE_GHOST_BITS = _UNWISE_BIT(11, 12, 25, 26)
UNWISE_BAD_BITS = UNWISE_SPIKE_BITS | UNWISE_HALO_BITS | UNWISE_GHOST_BITS


def _closest_coadd_id(member_names, ra: float, dec: float) -> str:
    """Pick the unWISE coadd tile closest to (ra, dec) from tarball members.

    A cutout_fits request near a tile boundary returns files from every
    overlapping tile; the fit runs on one tile's pixel grid, so the tile
    whose center is nearest the target is used (the name encodes the
    center: '2709p666' = RA 270.9, Dec +66.6). Pixels beyond that tile's
    edge enter the fit with zero weight.
    """
    ids: list[str] = []
    for n in member_names:
        cid = n.split("/")[-1].split("-")[1]
        if cid not in ids:
            ids.append(cid)
    if len(ids) == 1:
        return ids[0]

    cosd = float(np.cos(np.radians(dec)))

    def _dist2(cid):
        ra_c = int(cid[:4]) / 10.0
        dec_c = int(cid[5:8]) / 10.0 * (1.0 if cid[4] == "p" else -1.0)
        d_ra = ((ra_c - ra + 540.0) % 360.0) - 180.0
        return (d_ra * cosd) ** 2 + (dec_c - dec) ** 2

    ids.sort(key=_dist2)
    import warnings
    warnings.warn(
        f"unWISE cutout spans {len(ids)} coadd tiles ({', '.join(ids)}); "
        f"fitting on the closest, {ids[0]}. Sources beyond that tile's edge "
        "fall on zero-weight pixels and come back with NaN errors; recenter "
        "the target or shrink the cutout to keep the field on one tile.",
        stacklevel=3)
    return ids[0]


def fetch_unwise_cutouts(ra: float, dec: float, size_arcsec: float,
                         *,
                         data_dir: str | Path = DEFAULT_WISE_CACHE_DIR,
                         version: str = WISE_COADD_VERSION,
                         bands=("W1", "W2"),
                         show_download_path: bool = False) -> dict:
    """Download (or cache) unWISE coadds for a small field.

    Returns
    -------
    dict[band] -> dict with keys ``data``, ``ivar``, ``wcs``, ``header``,
    plus ``coadd_id`` and ``size_arcsec`` at the top level.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    size_pix = int(round(size_arcsec / UNWISE_PIXEL_SCALE))

    tarball = data_dir / f"unwise_{ra:.4f}_{dec:.4f}_{int(round(size_arcsec))}.tar.gz"
    if not tarball.exists() or show_download_path:
        url = _unwise_cutout_url(ra, dec, size_pix, version)
        # unwise.me can return an HTML 200 on out-of-coverage targets;
        # download to .part and validate the gzip magic.
        tmp = tarball.with_suffix(tarball.suffix + ".part")
        from .netutils import retry
        retry(lambda: urllib.request.urlretrieve(url, tmp),
              what="unwise.me cutout download")
        with tmp.open("rb") as f:
            if f.read(2) != b"\x1f\x8b":
                tmp.unlink(missing_ok=True)
                raise RuntimeError(
                    f"unwise.me did not return a tarball for ra={ra}, dec={dec}, "
                    f"size={size_pix}px (likely outside coverage)")
        tmp.replace(tarball)

    extract_dir = data_dir / tarball.stem
    extract_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tarball) as tar:
        names = tar.getnames()
        # tarfile filter="data" requires Python 3.12.
        if sys.version_info >= (3, 12):
            tar.extractall(extract_dir, filter="data")
        else:
            tar.extractall(extract_dir)
    fits_members = [n for n in names if n.endswith((".fits", ".fits.gz"))]
    if not fits_members:
        raise RuntimeError(f"no FITS members in unwise tarball {tarball}")
    coadd_id = _closest_coadd_id(fits_members, ra, dec)

    # The bundled demo ships a pre-aligned bitmask inside the tarball;
    # macOS AppleDouble "._" sidecars are not valid FITS.
    msk = None
    msk_members = [n for n in names
                   if "-msk" in n.split("/")[-1]
                   and not n.split("/")[-1].startswith("._")
                   and n.endswith((".fits", ".fits.gz"))]
    if msk_members:
        with fits.open(extract_dir / msk_members[0].split("/")[-1]) as hdul:
            msk = np.nan_to_num(hdul[0].data, nan=0).astype(np.int32)

    out: dict = {"coadd_id": coadd_id, "size_arcsec": size_arcsec}
    ref_wcs = ref_shape = None
    for band_num, band in [(1, "W1"), (2, "W2")]:
        if band not in bands:
            continue
        img_path = extract_dir / f"unwise-{coadd_id}-w{band_num}-img-m.fits"
        iv_path = extract_dir / f"unwise-{coadd_id}-w{band_num}-invvar-m.fits.gz"
        with fits.open(img_path) as hdul:
            data = hdul[0].data.astype(np.float32)
            header = hdul[0].header.copy()
            wcs = WCS(hdul[0].header)
        with fits.open(iv_path) as hdul:
            ivar = hdul[0].data.astype(np.float32)
        if ref_wcs is None:
            ref_wcs, ref_shape = wcs, data.shape
        out[band] = {"data": data, "ivar": ivar, "wcs": wcs, "header": header}

    # No bitmask in the tarball: fetch the per-tile mask; a miss leaves
    # mask=None and masking is skipped.
    if msk is None and ref_wcs is not None:
        msk = _fetch_unwise_mask_tile(coadd_id, version, ref_wcs, ref_shape, data_dir)
    for band in list(out):
        if band in ("W1", "W2"):
            out[band]["mask"] = msk
    return out


def get_wise_psf(band_number: int, coadd_id: str, *, sidelen: int = 151,
                 modelname: str | None = None) -> np.ndarray:
    """Return a unit-flux-normalized unWISE PSF stamp.

    ``unwise_psf.get_unwise_psf`` has a Python-3 slice-indexing bug when
    ``sidelen`` is passed; we trim ourselves to an odd stamp.

    ``modelname`` defaults to the model matched to ``WISE_COADD_VERSION``
    (``"neo7_unwisecat"``), which beats the version-agnostic default on the
    demo field: median(Tractor/Schlafly+2019) on bright isolated sources
    moves from 0.975 to 0.990 in W1 and 0.85 to 0.95 in W2. The per-frame
    rotation path overshoots W1 by ~4%, and legacypipe's W1 fluxrescale
    (x1.04) pushes the ratio away from unity; neither is used. The residual
    few-percent W1/W2 scale uncertainty is inherent to the unWISE PSF model
    near the NEP.
    """
    try:
        from unwise_psf import unwise_psf as up
    except ImportError as exc:
        raise ImportError(
            "The unWISE W1/W2 step needs the `unwise_psf` PSF model, which the "
            "upstream repo does not package for pip. Install it manually:\n"
            "    git clone https://github.com/legacysurvey/unwise_psf\n"
            "    export PYTHONPATH=\"$PWD/unwise_psf/py:$PYTHONPATH\"\n"
            "(plus `pip install fitsio 'setuptools<81'`; unwise_psf reads its "
            "model files with fitsio and imports the deprecated pkg_resources "
            "API). VIS + NISP run without it."
        ) from exc
    if modelname is None:
        modelname = f"{WISE_COADD_VERSION}_unwisecat"
    import warnings
    fallback = False
    # unwise_psf hits the deprecated pkg_resources API when resolving a
    # model file; suppress only that warning.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*pkg_resources.*")
        try:
            full = up.get_unwise_psf(band_number, coadd_id, modelname=modelname)
        except OSError:
            # *_unwisecat models exist only for some versions/bands.
            fallback = True
            full = up.get_unwise_psf(band_number, coadd_id)
    if fallback:
        warnings.warn(
            f"unwise_psf has no '{modelname}' model for W{band_number}; "
            "falling back to the default PSF model.", stacklevel=2)
    h = full.shape[0] // 2
    half = sidelen // 2
    stamp = full[h - half:h + half + 1, h - half:h + half + 1].astype(np.float32)
    stamp /= stamp.sum()
    return stamp


# ---------------------------------------------------------------------------
# Forced photometry
# ---------------------------------------------------------------------------

def _source_at_wise(vis_src, band: str, *, point: bool = False,
                    psf_halfsize_pix: int | None = None):
    """Clone a VIS-fitted source for the unWISE fit in ``band``.

    The VIS model class and shape are kept (positions and shapes frozen;
    flux is the only free parameter), so the same light profile is
    measured in every band. ``point=True`` collapses the source to a
    PointSource instead, the Lang+2016 (section 3.2) reduction; every
    Euclid source is unresolved at the ~6.9 arcsec WISE PSF (~30x a
    typical Euclid R_eff), so the two choices agree closely.

    Profile sources carry an explicit ``halfsize`` covering the PSF
    stamp: Tractor sizes a galaxy's FFT patch from this hint, and the
    auto-sized patch of a sub-pixel galaxy crops the 151x151 PSF,
    losing ~2-3 percent of its wing flux.
    """
    if point or isinstance(vis_src, PointSource):
        pos = vis_src.getPosition().copy()
        ps = PointSource(pos, NanoMaggies(**{band: 1.0}))
        # At 2.75"/pix and 6.94" FWHM a free centroid wanders into a
        # neighbor and biases the flux by tens of percent.
        ps.freezeParam("pos")
        return ps
    cloned = _deepcopy(vis_src)
    cloned.brightness = NanoMaggies(**{band: 1.0})
    if psf_halfsize_pix is not None:
        try:
            re_pix = float(cloned.getShape().re) / UNWISE_PIXEL_SCALE
        except Exception:
            re_pix = 0.0
        cloned.halfsize = int(psf_halfsize_pix
                              + max(4, int(np.ceil(2.0 * re_pix))))
    return cloned


def _supplement_sources(ra: float, dec: float, prior_radius_arcsec: float,
                        existing_coords: SkyCoord,
                        *,
                        merge_sep_arcsec: float = 3.0,
                        min_radius_arcsec: float = 0.0):
    """Pull unWISE-catalog sources in the PSF-wing buffer outside the
    Euclid prior, excluding any within ``merge_sep_arcsec`` of an existing
    Euclid source and any inside ``min_radius_arcsec`` of the field center.

    Supplements inside the MER footprint can absorb W1/W2 flux that
    belongs to a MER prior (a detection 4-10 arcsec from a MER galaxy may
    be a sub-threshold counterpart or a confused blend), so they are used
    only in the buffer ring.
    """
    prior_radius_deg = prior_radius_arcsec / 3600.0
    # Admit a primary detection in either band: the same list feeds both
    # fits, and a W2-primary source sub-threshold in W1 must still absorb
    # W2 wing flux.
    adql = f"""
    SELECT ra, dec, flux_1, flux_2
    FROM unwise_2019
    WHERE CONTAINS(POINT('J2000',ra,dec),
                   CIRCLE('J2000',{ra},{dec},{prior_radius_deg}))=1
      AND (primary_1 = 1 OR primary_2 = 1)
    """
    from .netutils import retry
    uw_cat = retry(lambda: Irsa.query_tap(query=adql).to_table(),
                   what="IRSA TAP unWISE isolation query")
    if len(uw_cat) == 0:
        return uw_cat, np.zeros(0, dtype=bool)
    # TAP row order is non-deterministic, and a near-degenerate LSQR solve
    # is sensitive to column order at the ~1% level; sort for bit-identical
    # reruns.
    uw_cat.sort(["ra", "dec"])
    uw_coord = SkyCoord(uw_cat["ra"], uw_cat["dec"], unit="deg")
    center = SkyCoord(ra, dec, unit="deg")
    keep = np.ones(len(uw_cat), dtype=bool)
    if min_radius_arcsec > 0:
        sep_center = uw_coord.separation(center).arcsec
        keep &= sep_center >= min_radius_arcsec
    if len(existing_coords) > 0:
        for k in range(len(uw_cat)):
            if not keep[k]:
                continue
            if uw_coord[k].separation(existing_coords).arcsec.min() < merge_sep_arcsec:
                keep[k] = False
    return uw_cat, keep


def fit_wise_forced(sources, wise_cutouts: dict, *,
                    ra: float, dec: float,
                    cutout_size_arcsec: float,
                    psf_stamps: dict | None = None,
                    bands=("W1", "W2"),
                    source_models: str = "prior",
                    supplement_with_unwise_catalog: bool = True,
                    inflate_uncertainties: bool = True,
                    mask_bad_bits: int | None = UNWISE_BAD_BITS) -> dict:
    """Forced photometry on W1/W2 at the Euclid positions, with a jointly-fit
    constant sky.

    Parameters
    ----------
    sources : list of VIS-fitted Tractor sources.
    wise_cutouts : output of ``fetch_unwise_cutouts``.
    ra, dec, cutout_size_arcsec : geometry of the Euclid field.
    psf_stamps : optional dict[band] -> stamp; built via ``get_wise_psf`` if missing.
    source_models : {'prior', 'point'}
        ``'prior'`` keeps each VIS-fitted model class and shape, frozen,
        with flux the only free parameter, the same treatment as the NISP
        bands. ``'point'`` collapses every source to a PointSource
        (Lang et al. 2016, section 3.2); at the ~6.9 arcsec WISE PSF the
        Euclid profiles are unresolved, so the two agree closely.
    supplement_with_unwise_catalog : whether to add outside-prior point sources.
    inflate_uncertainties : whether to scale the formal flux errors by the
                            chi scatter on source-sparse pixels (see the
                            module docstring).

    Returns
    -------
    dict with per-band entries ``flux_ujy``, ``flux_err_ujy``, ``sky``,
    ``chi_inflation`` (one value per band) and the residuals for diagnostics.

    Notes
    -----
    The chi-inflated ``flux_err_ujy`` reproduces the Schlafly+2019
    statistical errors in W1 to ~10% (median sigma ratio 1.1) on the demo
    field, but the W2 errors come out ~1.8x smaller than Schlafly's: the
    sky-floor chi inflation underestimates the W2 confusion noise. Treat W2
    uncertainties as a lower bound.
    """
    if source_models not in ("prior", "point"):
        raise ValueError(
            f"source_models = {source_models!r}; must be 'prior' or 'point'.")
    disable_galaxy_cache()
    coadd_id = wise_cutouts["coadd_id"]
    if psf_stamps is None:
        psf_stamps = {
            "W1": get_wise_psf(1, coadd_id),
            "W2": get_wise_psf(2, coadd_id),
        }

    # Pad by 5x WISE FWHM per edge so edge sources keep their PSF wings
    # inside the fit region; below ~3x FWHM the lost wing flux is
    # mis-attributed to in-cutout neighbors.
    fit_size_arcsec = float(cutout_size_arcsec + 10 * _WISE_FWHM_ARCSEC)
    fit_size_pix = int(round(fit_size_arcsec / UNWISE_PIXEL_SCALE))
    prior_radius_arcsec = fit_size_arcsec / 2.0 + 4 * _WISE_FWHM_ARCSEC

    src_coord = SkyCoord(
        [s.getPosition().ra for s in sources],
        [s.getPosition().dec for s in sources], unit="deg")

    suppl_cat = suppl_mask = None
    if supplement_with_unwise_catalog:
        # The MER footprint is a square, so the supplement exclusion radius
        # is the half-diagonal plus one WISE FWHM; a half-side radius would
        # leave the corner wedges exposed.
        min_radius = cutout_size_arcsec / np.sqrt(2.0) + _WISE_FWHM_ARCSEC
        try:
            suppl_cat, suppl_mask = _supplement_sources(
                ra, dec, prior_radius_arcsec, src_coord,
                min_radius_arcsec=min_radius)
        except Exception as exc:
            # Supplements only absorb buffer-ring wing flux; a transient
            # IRSA outage must not kill the W1/W2 measurement.
            import warnings
            warnings.warn(
                f"unwise_2019 supplement query failed ({exc!r}); fitting "
                "without buffer-ring supplements. Edge-source W1/W2 fluxes "
                "may absorb a little neighboring wing flux.", stacklevel=2)
            suppl_cat = suppl_mask = None

    results: dict = {}
    n_euclid = len(sources)

    for band in bands:
        if band not in wise_cutouts:
            continue
        full = wise_cutouts[band]
        cx, cy = full["wcs"].world_to_pixel_values(ra, dec)
        co_d = Cutout2D(full["data"], (float(cx), float(cy)),
                        fit_size_pix, wcs=full["wcs"],
                        mode="partial", fill_value=0.0)
        co_iv = Cutout2D(full["ivar"], (float(cx), float(cy)),
                         fit_size_pix, wcs=full["wcs"],
                         mode="partial", fill_value=0.0)
        data = co_d.data.astype(np.float32)
        ivar = co_iv.data.astype(np.float32)
        wcs_fit = co_d.wcs

        # Zero the invvar on bright-star artifact pixels; skipped when the
        # cached tarball predates the mask.
        if mask_bad_bits and full.get("mask") is not None:
            co_m = Cutout2D(full["mask"], (float(cx), float(cy)),
                            fit_size_pix, wcs=full["wcs"],
                            mode="partial", fill_value=0)
            bad = (co_m.data.astype(np.int32) & int(mask_bad_bits)) != 0
            ivar = np.where(bad, 0.0, ivar).astype(np.float32)

        scale = 10 ** (_VEGA_OFFSET[band] / 2.5)
        tim = Image(
            data=data, invvar=ivar,
            psf=PixelizedPSF(psf_stamps[band]),
            wcs=ConstantFitsWcs(AstropyWCSAdapter(wcs_fit)),
            photocal=LinearPhotoCal(scale, band=band),
            sky=ConstantSky(0.0), name=f"unWISE-{band}",
        )

        psf_half = int(psf_stamps[band].shape[0]) // 2
        band_sources = []
        for s in sources:
            cp = _source_at_wise(s, band,
                                 point=(source_models == "point"),
                                 psf_halfsize_pix=psf_half)
            cp.freezeAllBut("brightness")
            band_sources.append(cp)

        if suppl_cat is not None and suppl_mask is not None:
            cat_col = "flux_1" if band == "W1" else "flux_2"
            for k in range(len(suppl_cat)):
                if not suppl_mask[k]:
                    continue
                val = suppl_cat[cat_col][k]
                if hasattr(val, "mask") and np.ma.is_masked(val):
                    init_ab = 1.0
                elif not np.isfinite(val):
                    init_ab = 1.0
                else:
                    init_ab = float(val) * 10 ** (-_VEGA_OFFSET[band] / 2.5)
                ps = PointSource(
                    RaDecPos(float(suppl_cat["ra"][k]),
                             float(suppl_cat["dec"][k])),
                    NanoMaggies(**{band: init_ab}))
                ps.freezeAllBut("brightness")
                band_sources.append(ps)

        tr = Tractor([tim], band_sources)
        tim.freezeAllParams()
        tim.thawParam("sky")
        R = tr.optimize_forced_photometry(
            minsb=0.0, mindlnp=1.0, sky=True, variance=True,
            shared_params=False, fitstats=True, wantims=True,
        )

        n_sky = tim.numberOfParams()
        sky_value = float(tim.getSky().getValue())
        # R.IV ordering is version-dependent: [sky_iv, *src_iv] or source
        # IVs only. Accept both; raise loudly on a third layout.
        n_src = len(band_sources)
        if len(R.IV) == n_sky + n_src:
            flux_iv_ab = np.array(R.IV[n_sky:], dtype=float)
        elif len(R.IV) == n_src:
            flux_iv_ab = np.array(R.IV, dtype=float)
        else:
            raise RuntimeError(
                f"unexpected R.IV length {len(R.IV)} "
                f"(expected {n_src} or {n_sky + n_src}); "
                f"Tractor's parameter ordering has changed")
        fluxes_ab = np.array([s.brightness.getFlux(band) for s in band_sources])

        _, mod, _, chi, _ = R.ims1[0]
        with np.errstate(all="ignore"):
            sig_pix = np.where(ivar > 0, 1.0 / np.sqrt(ivar), np.inf)
        # Chi-inflation pool: pixels where the joint model (sources + sky)
        # is within mult*sigma of zero, relaxing mult until >= min_pool
        # pixels; crowded fields fall back to a sigma-clipped MAD. Pooling
        # on |mod - sky| instead doubles the factor (W1 sigma ratio vs
        # Schlafly+2019: ~1.1 here, ~2.1 sky-subtracted).
        min_pool = 50
        chi_pool = None
        chi_pool_kind = None
        for mult in (1.0, 2.0, 3.0, 5.0):
            mask = (np.abs(mod) < mult * sig_pix) & (ivar > 0)
            if mask.sum() >= min_pool:
                chi_pool = chi[mask]
                chi_pool_kind = f"source-sparse (|mod|<{mult:g}sigma)"
                break
        if chi_pool is None:
            from astropy.stats import sigma_clip
            valid = chi[ivar > 0]
            clipped = sigma_clip(valid, sigma=3.0, maxiters=5)
            chi_pool = np.asarray(clipped.compressed())
            chi_pool_kind = "all-pixels sigma-clipped (crowded-fallback)"
        if not inflate_uncertainties or chi_pool.size == 0:
            chi_infl = 1.0
        else:
            chi_mad = float(mad_std(chi_pool))
            chi_infl = max(1.0, chi_mad) if np.isfinite(chi_mad) else 1.0
        with np.errstate(divide="ignore", invalid="ignore"):
            # NaN for an unconstrained source, matching the VIS/NISP
            # sentinel.
            flux_err_ab = np.where(flux_iv_ab > 0,
                                   chi_infl / np.sqrt(flux_iv_ab), np.nan)

        # Only the first n_euclid entries align with the input list;
        # supplements are not exported.
        flux_ujy = fluxes_ab[:n_euclid] * _UJY_PER_NMGY
        flux_err_ujy = flux_err_ab[:n_euclid] * _UJY_PER_NMGY

        results[band] = {
            "flux_ujy": flux_ujy,
            "flux_err_ujy": flux_err_ujy,
            "sky": sky_value,
            "chi_inflation": chi_infl,
            "chi_pool_kind": chi_pool_kind,
            "n_chi_pool": int(chi_pool.size),
            "residual": data - mod,
            "model": mod,
            "data": data,
            "invvar": ivar,
        }
    return results
