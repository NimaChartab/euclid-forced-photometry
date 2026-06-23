"""Cutout discovery and fetch for Euclid Q1 MER mosaics.

Two cache layers: a small cutout-FITS cache at
``data_dir/<band>_<ra>_<dec>_<size>.fits`` (read directly when present), and
a fallback to S3 lazy partial reads (``fits.open(..., use_fsspec=True)`` plus
``hdu.section``) whose result is written back to the cutout cache. Full
mosaic tiles are never persisted.

Public: ``Cutout``, ``discover_mer_mosaics``, ``fetch_cutout``,
``trim_catalog_to_cutout``.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.wcs import WCS
from astroquery.ipac.irsa import Irsa
from reproject import reproject_interp
from reproject.mosaicking import reproject_and_coadd

from .config import (
    DEFAULT_CUTOUT_DIR,
    MER_COLLECTION,
    MER_TILE_HALF_DEG,
    SIA_SEARCH_PAD_DEG,
)


@dataclass
class Cutout:
    """One band's cutout: science array, RMS (when paired), WCS, header.

    ``flag`` is the optional MER FLG (quality) plane, an integer bitmask
    aligned with ``data`` (non-zero where a pixel is saturated, a cosmic ray,
    a bad pixel, or otherwise invalid). It is only populated when
    ``fetch_cutout(..., with_flag=True)`` is used; ``build_tractor_image``
    consumes it to drop flagged pixels from the fit.
    """
    band: str
    data: np.ndarray
    rms: np.ndarray | None
    wcs: WCS
    header: fits.Header
    flag: np.ndarray | None = None

    @property
    def shape(self):
        return self.data.shape

    @property
    def pixel_scale_arcsec(self) -> float:
        return float(abs(self.wcs.pixel_scale_matrix[0, 0]) * 3600.0)


# ---------------------------------------------------------------------------
# Tile discovery
# ---------------------------------------------------------------------------

def _classify_ptype(fname: str) -> str | None:
    if "BGSUB-MOSAIC" in fname: return "science"
    if "RMS"          in fname: return "rms"
    if "GRID-PSF"     in fname: return "psf_grid"
    if "CATALOG-PSF"  in fname: return "psf_catalog"
    if "FLAG"         in fname: return "flag"
    if "BGMOD"        in fname: return "bgmodel"
    return None


def _separation_arcsec(t, target):
    if t["s_ra"] is None or t["s_dec"] is None:
        return float("inf")
    return (SkyCoord(t["s_ra"], t["s_dec"], unit="deg")
            .separation(target).arcsec)


def _closest_tile(tiles, target):
    with_coords = [t for t in tiles if t["s_ra"] is not None and t["s_dec"] is not None]
    if not with_coords:
        return tiles[0]
    return min(with_coords, key=lambda t: _separation_arcsec(t, target))


def discover_mer_mosaics(ra: float, dec: float, half_size_deg: float,
                         *, bands=("VIS", "Y", "J", "H"), verbose=False):
    """SIA query returning dict[band][ptype] with closest tile and ``tiles`` list."""
    from .netutils import retry
    target = SkyCoord(ra=ra, dec=dec, unit="deg")
    search_radius_deg = MER_TILE_HALF_DEG + half_size_deg + SIA_SEARCH_PAD_DEG
    sia_results = retry(
        lambda: Irsa.query_sia(
            pos=(target, search_radius_deg * u.deg),
            collection=MER_COLLECTION,
        ),
        what="IRSA SIA discovery")

    raw = []
    for row in sia_results:
        band = row["energy_bandpassname"]
        if band not in bands:
            continue
        url = str(row["access_url"])
        fname = url.split("/")[-1]
        ptype = _classify_ptype(fname)
        if ptype is None:
            continue
        cloud_meta = json.loads(row["cloud_access"])
        s3_path = f"{cloud_meta['aws']['bucket_name']}/{cloud_meta['aws']['key']}"
        m = re.search(r"TILE(\d+)", fname)
        tile_id = m.group(1) if m else fname
        try:
            s_ra = float(row["s_ra"]); s_dec = float(row["s_dec"])
        except Exception:
            s_ra = s_dec = None
        raw.append(dict(band=band, ptype=ptype, tile_id=tile_id,
                        s3=s3_path, url=url, s_ra=s_ra, s_dec=s_dec,
                        fname=fname))

    cosd = float(np.cos(np.radians(dec)))
    thresh_deg = MER_TILE_HALF_DEG + half_size_deg + 0.005
    keep = []
    for r in raw:
        if r["s_ra"] is None or r["s_dec"] is None:
            keep.append(r); continue
        # Wrap the RA difference into [-180, 180].
        d_ra = ((r["s_ra"] - ra + 540.0) % 360.0) - 180.0
        if (abs(d_ra) * cosd <= thresh_deg
                and abs(r["s_dec"] - dec) <= thresh_deg):
            keep.append(r)
    raw = keep

    by_bp = defaultdict(list)
    seen = set()
    for r in raw:
        key = (r["band"], r["ptype"], r["tile_id"])
        if key in seen:
            continue
        seen.add(key)
        by_bp[(r["band"], r["ptype"])].append(r)

    products = {}
    for (band, ptype), tiles in by_bp.items():
        products.setdefault(band, {})
        closest = _closest_tile(tiles, target)
        products[band][ptype] = {**closest, "tiles": tiles}

    if verbose:
        print(f"SIA: {len(raw)} matching products; "
              f"bands available: {', '.join(sorted(products))}")
    return products


# ---------------------------------------------------------------------------
# Cutout fetch: S3 lazy + bundled FITS cache
# ---------------------------------------------------------------------------

def _open_mosaic(s3_path: str, mosaic_cache_dir: Path | None):
    """Open a MER mosaic, preferring the local mosaic cache; with
    ``mosaic_cache_dir=None``, read straight from S3 and persist nothing."""
    fname = s3_path.split("/")[-1]
    if mosaic_cache_dir is not None:
        local_path = Path(mosaic_cache_dir) / fname
        if local_path.exists():
            try:
                with local_path.open("rb") as _f:
                    if _f.read(6) != b"SIMPLE":
                        raise OSError("not a FITS file")
                return fits.open(str(local_path), memmap=True), f"local ({fname[:40]})"
            except OSError:
                pass
    from .netutils import S3_FSSPEC_KWARGS, retry
    return (
        retry(lambda: fits.open(f"s3://{s3_path}", use_fsspec=True,
                                fsspec_kwargs=S3_FSSPEC_KWARGS),
              what=f"S3 open {fname[:40]}"),
        f"s3-lazy ({fname[:40]})",
    )


def _build_output_wcs(ra, dec, size_arcsec, pixel_scale_deg, ctype):
    size_pix = int(round(size_arcsec / 3600.0 / pixel_scale_deg))
    out_wcs = WCS(naxis=2)
    out_wcs.wcs.crval = [ra, dec]
    out_wcs.wcs.crpix = [size_pix / 2.0 + 0.5, size_pix / 2.0 + 0.5]
    out_wcs.wcs.cdelt = [-pixel_scale_deg, pixel_scale_deg]
    out_wcs.wcs.ctype = list(ctype)
    return out_wcs, size_pix


def _read_tile_partial(tile, ra, dec, size_arcsec, mosaic_cache_dir):
    hdul_ctx, _ = _open_mosaic(tile["s3"], mosaic_cache_dir)
    with hdul_ctx as hdul:
        for hdu in hdul:
            if not hasattr(hdu, "shape") or hdu.shape is None:
                continue
            if len(hdu.shape) < 2 or hdu.header.get("NAXIS", 0) < 2:
                continue
            wcs = WCS(hdu.header)
            cx, cy = wcs.world_to_pixel_values(ra, dec)
            ps_arcsec = abs(wcs.pixel_scale_matrix[0, 0]) * 3600.0
            size_pix = int(round(size_arcsec / ps_arcsec))
            margin = size_pix
            ix, iy = int(round(float(cx))), int(round(float(cy)))
            y0 = max(0, iy - margin)
            y1 = min(hdu.shape[0], iy + margin + 1)
            x0 = max(0, ix - margin)
            x1 = min(hdu.shape[1], ix + margin + 1)
            if x0 >= x1 or y0 >= y1:
                return None
            sub = np.array(hdu.section[y0:y1, x0:x1], dtype=np.float64)
            cut_wcs = wcs.deepcopy()
            cut_wcs.wcs.crpix = [wcs.wcs.crpix[0] - x0,
                                 wcs.wcs.crpix[1] - y0]
            return sub, cut_wcs, hdu.header
    return None


def _single_tile_cutout(tile, ra, dec, size_arcsec, mosaic_cache_dir):
    result = _read_tile_partial(tile, ra, dec, size_arcsec, mosaic_cache_dir)
    if result is None:
        raise ValueError(f"tile {tile.get('tile_id')} has no overlap with cutout")
    data, wcs, header = result
    cx, cy = wcs.world_to_pixel_values(ra, dec)
    ps_arcsec = abs(wcs.pixel_scale_matrix[0, 0]) * 3600.0
    size_pix = int(round(size_arcsec / ps_arcsec))
    co = Cutout2D(data, (float(cx), float(cy)), size_pix,
                  wcs=wcs, mode="partial", fill_value=0.0)
    return co.data.astype(np.float64), co.wcs, header


def _stitched_cutout(tiles, ra, dec, size_arcsec, mosaic_cache_dir):
    # Mean coadd. For the RMS layer this under-weights (never over-weights)
    # tile-overlap strips.
    inputs = []
    sample_ps_deg = sample_ctype = sample_header = None
    for t in tiles:
        result = _read_tile_partial(t, ra, dec, size_arcsec, mosaic_cache_dir)
        if result is None:
            continue
        data, wcs, header = result
        inputs.append((data, wcs))
        if sample_ps_deg is None:
            sample_ps_deg = abs(wcs.pixel_scale_matrix[0, 0])
            sample_ctype = list(wcs.wcs.ctype)
            sample_header = header
    if not inputs:
        raise ValueError("no tiles overlap the cutout")
    out_wcs, size_pix = _build_output_wcs(
        ra, dec, size_arcsec, sample_ps_deg, sample_ctype)
    out_data, _ = reproject_and_coadd(
        inputs, out_wcs, shape_out=(size_pix, size_pix),
        reproject_function=reproject_interp,
        match_background=False, combine_function="mean",
    )
    return np.nan_to_num(out_data, nan=0.0).astype(np.float64), out_wcs, sample_header


def _stitched_flag_cutout(tiles, ra, dec, size_arcsec, mosaic_cache_dir):
    """Stitch a multi-tile FLG cutout: nearest-neighbor resample plus
    bitwise OR (bilinear blending would corrupt the bitmask)."""
    inputs = []
    sample_ps_deg = sample_ctype = sample_header = None
    for t in tiles:
        result = _read_tile_partial(t, ra, dec, size_arcsec, mosaic_cache_dir)
        if result is None:
            continue
        data, wcs, header = result
        inputs.append((data, wcs))
        if sample_ps_deg is None:
            sample_ps_deg = abs(wcs.pixel_scale_matrix[0, 0])
            sample_ctype = list(wcs.wcs.ctype)
            sample_header = header
    if not inputs:
        raise ValueError("no tiles overlap the cutout")
    out_wcs, size_pix = _build_output_wcs(
        ra, dec, size_arcsec, sample_ps_deg, sample_ctype)
    out_bits = np.zeros((size_pix, size_pix), dtype=np.int32)
    for data, wcs in inputs:
        arr, footprint = reproject_interp(
            (data.astype(np.float64), wcs), out_wcs,
            shape_out=(size_pix, size_pix), order="nearest-neighbor")
        covered = (footprint > 0) & np.isfinite(arr)
        out_bits[covered] |= np.rint(arr[covered]).astype(np.int32)
    return out_bits, out_wcs, sample_header


def _cutout_cache_path(band, ra, dec, size_arcsec, data_dir, ptype="science"):
    # The sparse FLG bitmask shrinks ~5x under gzip; science/RMS images
    # do not compress.
    ext = "fits.gz" if ptype == "flag" else "fits"
    fname = f"{band.lower()}_{ptype}_{ra:.4f}_{dec:.4f}_{int(round(size_arcsec))}.{ext}"
    return Path(data_dir) / fname


def _write_cutout_fits(path: Path, data: np.ndarray, wcs: WCS, header: fits.Header):
    path.parent.mkdir(parents=True, exist_ok=True)
    hdr = wcs.to_header()
    for key in ("MAGZERO", "MAGZP", "TELESCOP", "INSTRUME", "FILTER",
                "BUNIT", "EXPTIME", "TIMESYS", "DATE-OBS"):
        if key in header:
            hdr[key] = header[key]
    # Atomic write.
    tmp = path.with_suffix(path.suffix + ".tmp")
    fits.PrimaryHDU(data=data.astype(np.float32), header=hdr).writeto(
        tmp, overwrite=True)
    tmp.replace(path)


def _read_cutout_fits(path: Path) -> tuple[np.ndarray, WCS, fits.Header]:
    with fits.open(path) as hdul:
        data = hdul[0].data.astype(np.float64)
        wcs = WCS(hdul[0].header)
        header = hdul[0].header.copy()
    return data, wcs, header


def _write_mask_fits(path: Path, mask: np.ndarray, wcs: WCS, header: fits.Header):
    """Cache an integer flag plane as int32 (a float32 cast would corrupt
    the high bits). astropy gzips only when the filename ends in ``.gz``,
    so the temp name preserves the suffix."""
    path.parent.mkdir(parents=True, exist_ok=True)
    hdr = wcs.to_header()
    tmp = path.with_name(path.stem + ".tmp" + path.suffix)
    fits.PrimaryHDU(data=mask.astype(np.int32), header=hdr).writeto(tmp, overwrite=True)
    tmp.replace(path)


def _read_mask_fits(path: Path) -> np.ndarray:
    with fits.open(path) as hdul:
        return np.nan_to_num(hdul[0].data, nan=0).astype(np.int32)


def fetch_cutout(band: str, ra: float, dec: float, size_arcsec: float,
                 *,
                 products: dict | None = None,
                 data_dir: str | Path = DEFAULT_CUTOUT_DIR,
                 mosaic_cache_dir: str | Path | None = None,
                 show_download_path: bool = False,
                 with_rms: bool = True,
                 with_flag: bool = False) -> Cutout:
    """Fetch a single-band cutout, using the bundled FITS cache when possible.

    Parameters
    ----------
    band : {'VIS','Y','J','H'}
    ra, dec : float
    size_arcsec : float
    products : optional, output of ``discover_mer_mosaics``. If None and a
        live download is required, this calls ``discover_mer_mosaics`` itself.
    data_dir : Path
        Where to look for / write the small post-cutout FITS cache.
    mosaic_cache_dir : Path or None
        If set, also consult this directory for full-mosaic files (read-only).
    show_download_path : bool
        If True, ignore the bundled cutout cache and re-fetch live from S3.
    with_rms : bool
        Whether to also fetch the matching RMS map.

    Returns
    -------
    Cutout
    """
    data_dir = Path(data_dir)
    sci_cache = _cutout_cache_path(band, ra, dec, size_arcsec, data_dir, "science")
    rms_cache = _cutout_cache_path(band, ra, dec, size_arcsec, data_dir, "rms")
    flag_cache = _cutout_cache_path(band, ra, dec, size_arcsec, data_dir, "flag")

    have_sci = sci_cache.exists()
    have_rms = rms_cache.exists() if with_rms else True
    have_flag = flag_cache.exists() if with_flag else True

    if have_sci and have_rms and have_flag and not show_download_path:
        data, wcs, header = _read_cutout_fits(sci_cache)
        rms = None
        if with_rms:
            rms, _, _ = _read_cutout_fits(rms_cache)
        flag = _read_mask_fits(flag_cache) if with_flag else None
        return Cutout(band=band, data=data, rms=rms, wcs=wcs, header=header,
                      flag=flag)

    if products is None:
        half_size_deg = size_arcsec / 2.0 / 3600.0
        products = discover_mer_mosaics(ra, dec, half_size_deg)

    if band not in products or "science" not in products[band]:
        raise ValueError(f"no science tile available for band {band!r}")

    sci_tiles = products[band]["science"].get("tiles") or [products[band]["science"]]
    data, wcs, header = (
        _single_tile_cutout(sci_tiles[0], ra, dec, size_arcsec, mosaic_cache_dir)
        if len(sci_tiles) == 1
        else _stitched_cutout(sci_tiles, ra, dec, size_arcsec, mosaic_cache_dir)
    )
    _write_cutout_fits(sci_cache, data, wcs, header)

    rms = None
    if with_rms and "rms" in products[band]:
        rms_tiles = products[band]["rms"].get("tiles") or [products[band]["rms"]]
        rms, _, _ = (
            _single_tile_cutout(rms_tiles[0], ra, dec, size_arcsec, mosaic_cache_dir)
            if len(rms_tiles) == 1
            else _stitched_cutout(rms_tiles, ra, dec, size_arcsec, mosaic_cache_dir)
        )
        _write_cutout_fits(rms_cache, rms, wcs, header)

    flag = None
    if with_flag:
        if "flag" in products.get(band, {}):
            flag_tiles = (products[band]["flag"].get("tiles")
                          or [products[band]["flag"]])
            if len(flag_tiles) == 1:
                # Cutout2D copies the pixel grid exactly; bits preserved.
                fdata, _, _ = _single_tile_cutout(
                    flag_tiles[0], ra, dec, size_arcsec, mosaic_cache_dir)
                flag = np.rint(fdata).astype(np.int32)
            else:
                flag, _, _ = _stitched_flag_cutout(
                    flag_tiles, ra, dec, size_arcsec, mosaic_cache_dir)
            _write_mask_fits(flag_cache, flag, wcs, header)
        else:
            import warnings
            warnings.warn(
                f"with_flag=True but no FLG tile is available for band {band!r}; "
                "returning a cutout with flag=None (no pixel masking).",
                stacklevel=2)

    return Cutout(band=band, data=data, rms=rms, wcs=wcs, header=header, flag=flag)


# ---------------------------------------------------------------------------
# Catalog trimming
# ---------------------------------------------------------------------------

def trim_catalog_to_cutout(mer_cat, wcs: WCS, shape, *, edge_margin_pix: int = 1):
    """Drop MER rows whose pixel positions fall off the cutout.

    Boundary rows stay in the source list (their light is in the pixels);
    their own photometry is flagged downstream
    (:func:`euclid_phot.flags.flag_sources`, column ``edge``).

    Returns
    -------
    astropy.table.Table  (possibly shorter than the input)
    """
    H, W = shape
    keep = np.ones(len(mer_cat), dtype=bool)
    for i, row in enumerate(mer_cat):
        px, py = wcs.world_to_pixel_values(row["ra"], row["dec"])
        if not (edge_margin_pix <= px < W - edge_margin_pix
                and edge_margin_pix <= py < H - edge_margin_pix):
            keep[i] = False
    return mer_cat[keep]
