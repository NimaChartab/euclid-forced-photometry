"""Discover and cache public Spitzer/IRAC PBCD mosaic products."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astroquery.ipac.irsa import Irsa

from .config import DEFAULT_PBCD_DIR, normalize_channel
from .netutils import download_file, retry


SHA_COLLECTION = "spitzer_sha"
_PBCD_PATH_RE = re.compile(r"/r(?P<aor>\d+)/ch(?P<channel>[1-4])/pbcd/")
_PRODUCT_RE = re.compile(
    r"^(?P<root>SPITZER_I(?P<channel>[1-4])_(?P<aor>\d+)_.*)_"
    r"[AE]\d+_(?P<kind>maic|munc|mcov|mmsk|refptg|irsa)\."
    r"(?P<extension>fits|tbl)$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class PBCDProductSet:
    """URLs for one public PBCD mosaic and its reconstruction metadata."""

    channel: str
    aorkey: int
    product_root: str
    science_url: str
    uncertainty_url: str
    coverage_url: str
    refptg_url: str
    mask_url: str | None = None
    source_table_url: str | None = None
    center_ra: float = np.nan
    center_dec: float = np.nan
    fov_deg: float = np.nan
    exposure_time: float = np.nan


@dataclass(frozen=True)
class PBCDLocalSet:
    """Local cached paths corresponding to :class:`PBCDProductSet`."""

    product: PBCDProductSet
    science_path: Path
    uncertainty_path: Path
    coverage_path: Path
    refptg_path: Path
    mask_path: Path | None = None
    source_table_path: Path | None = None


def _file_name(url: str) -> str:
    return url.rstrip("/").rsplit("/", 1)[-1]


def discover_pbcd(
    ra: float,
    dec: float,
    channel: str | int,
    *,
    search_radius_arcsec: float = 1.0,
    aorkey: int | None = None,
    selection: str = "best_aor",
) -> list[PBCDProductSet]:
    """Discover complete PBCD mosaics that cover a coordinate.

    ``best_aor`` chooses the complete mosaic whose archive footprint center is
    nearest the target. This avoids an arbitrary edge-of-mosaic choice, but is
    not a depth estimate. Pass an explicit ``aorkey`` for a reproducible
    science run, or choose ``all`` to inspect every overlapping PBCD mosaic.
    """
    channel = normalize_channel(channel)
    if selection not in {"best_aor", "all"}:
        raise ValueError("selection must be 'best_aor' or 'all'")
    if not np.isfinite([ra, dec, search_radius_arcsec]).all() \
            or search_radius_arcsec <= 0:
        raise ValueError("ra, dec, and positive search_radius_arcsec must be finite")

    rows = retry(
        lambda: Irsa.query_sia(
            pos=(SkyCoord(float(ra), float(dec), unit="deg"),
                 float(search_radius_arcsec) * u.arcsec),
            collection=SHA_COLLECTION,
        ),
        what="IRSA Spitzer PBCD discovery",
    )
    urls = sorted({str(value).strip() for value in rows["access_url"]})
    metadata = {}
    for row in rows:
        url = str(row["access_url"]).strip()

        def number(name):
            if name not in rows.colnames or np.ma.is_masked(row[name]):
                return np.nan
            try:
                return float(row[name])
            except (TypeError, ValueError):
                return np.nan

        metadata[url] = {
            "center_ra": number("s_ra"),
            "center_dec": number("s_dec"),
            "fov_deg": number("s_fov"),
            "exposure_time": number("t_exptime"),
        }
    channel_number = int(channel[-1])
    by_root: dict[tuple[int, str], dict[str, str]] = {}
    for url in urls:
        path_match = _PBCD_PATH_RE.search(url)
        if path_match is None or int(path_match.group("channel")) != channel_number:
            continue
        this_aor = int(path_match.group("aor"))
        if aorkey is not None and this_aor != int(aorkey):
            continue
        match = _PRODUCT_RE.match(_file_name(url))
        if match is not None:
            root = match.group("root")
            by_root.setdefault((this_aor, root), {})[
                match.group("kind").lower()
            ] = url

    products: list[PBCDProductSet] = []
    for (this_aor, root), found in sorted(by_root.items()):
        if not {"maic", "munc", "mcov", "refptg"}.issubset(found):
            continue
        science_metadata = metadata.get(found["maic"], {})
        products.append(PBCDProductSet(
            channel=channel,
            aorkey=this_aor,
            product_root=root,
            science_url=found["maic"],
            uncertainty_url=found["munc"],
            coverage_url=found["mcov"],
            mask_url=found.get("mmsk"),
            refptg_url=found["refptg"],
            source_table_url=found.get("irsa"),
            **science_metadata,
        ))

    if not products:
        qualifier = f" in AOR {aorkey}" if aorkey is not None else ""
        raise ValueError(
            f"SHA: no complete {channel} PBCD mosaic{qualifier} at "
            f"({ra:.6f}, {dec:.6f})"
        )
    products.sort(key=lambda product: (
        SkyCoord(product.center_ra, product.center_dec, unit="deg").separation(
            SkyCoord(float(ra), float(dec), unit="deg")
        ).deg
        if np.isfinite(product.center_ra) and np.isfinite(product.center_dec)
        else np.inf,
        product.aorkey,
        product.product_root,
    ))
    if aorkey is None and selection == "best_aor":
        products = products[:1]
    return products


def fetch_pbcd(products: list[PBCDProductSet], *,
               data_dir: str | Path = DEFAULT_PBCD_DIR,
               force_download: bool = False) -> list[PBCDLocalSet]:
    """Download PBCD files into a deterministic channel/AOR cache."""
    if not products:
        raise ValueError("products must contain at least one PBCD product")
    root = Path(data_dir)
    local_sets: list[PBCDLocalSet] = []
    for product in products:
        folder = root / product.channel.lower() / f"r{product.aorkey}"

        def fetch(url: str | None) -> Path | None:
            if url is None:
                return None
            return download_file(url, folder / _file_name(url),
                                 force=force_download)

        local_sets.append(PBCDLocalSet(
            product=product,
            science_path=fetch(product.science_url),
            uncertainty_path=fetch(product.uncertainty_url),
            coverage_path=fetch(product.coverage_url),
            refptg_path=fetch(product.refptg_url),
            mask_path=fetch(product.mask_url),
            source_table_path=fetch(product.source_table_url),
        ))
    manifest = root / products[0].channel.lower() / "pbcd_manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    temporary = manifest.with_suffix(".json.tmp")
    temporary.write_text(json.dumps([asdict(item) for item in products],
                                    indent=2, sort_keys=True) + "\n")
    temporary.replace(manifest)
    return local_sets
