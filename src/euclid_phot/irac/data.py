"""Science-mosaic acquisition for the corrected IRAC pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
import re
from urllib.parse import urlencode
from urllib.request import urlopen
from xml.etree import ElementTree

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from astroquery.ipac.irsa import Irsa

from .archive import PBCDLocalSet, discover_pbcd, fetch_pbcd
from .config import (
    DEFAULT_COSMIC_DAWN_DIR,
    DEFAULT_DATA_ROOT,
    DEFAULT_PBCD_DIR,
    IRAC_NATIVE_PIXEL_SCALE_ARCSEC,
    normalize_channel,
)
from .mosaic import MosaicCutout, load_mosaic_cutout
from .netutils import (
    download_file,
    ensure_https_ca_bundle,
    https_ssl_context,
    retry,
)
from .prf import ExposureGeometry, parse_refptg


DEFAULT_SEIP_DIR = DEFAULT_DATA_ROOT / "seip"
SEIP_COLLECTION = "spitzer_seip"
SEIP_CUTOUT_URL = "https://irsa.ipac.caltech.edu/cgi-bin/Cutouts/nph-cutouts"
COSMIC_DAWN_ROOT = "https://irsa.ipac.caltech.edu/data/SPITZER/Cosmic_Dawn"
_SEIP_NAME = re.compile(
    r"(?P<root>\d+\.\d+-\d+)\.IRAC\.(?P<channel>[1-4])\."
    r"(?P<kind>mosaic|unc|cov)\.fits$",
    re.IGNORECASE,
)
_COSMIC_DAWN_NAME = re.compile(
    r"CDS_(?P<field>[A-Z0-9]+)_ch(?P<channel>[1-4])_"
    r"ima_(?P<version>v\d+(?:lin|drz)?)\.fits$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class SEIPProductSet:
    """Remote files for one channel of one SEIP Super Mosaic tile."""

    channel: str
    super_mosaic_id: str
    product_root: str
    science_url: str
    uncertainty_url: str
    coverage_url: str
    dcelist_url: str


@dataclass(frozen=True)
class SEIPLocalSet:
    """Cached cutouts and contributor list for a SEIP Super Mosaic."""

    product: SEIPProductSet
    science_path: Path
    uncertainty_path: Path
    coverage_path: Path
    dcelist_path: Path
    mask_path: Path | None = None


@dataclass(frozen=True)
class CosmicDawnProductSet:
    """Public Moneti et al. mosaic products for one field and channel."""

    channel: str
    field: str
    product_root: str
    version: str
    science_url: str
    uncertainty_url: str
    coverage_url: str
    exposure_time_url: str
    info_url: str


@dataclass(frozen=True)
class CosmicDawnLocalSet:
    """Cached, aligned cutouts from a Cosmic Dawn mosaic."""

    product: CosmicDawnProductSet
    science_path: Path
    uncertainty_path: Path
    coverage_path: Path
    info_path: Path
    exposure_time_path: Path | None = None
    mask_path: Path | None = None


@dataclass(frozen=True)
class MosaicInputs:
    """A cached science mosaic, aligned cutout, and contributing exposures."""

    local_product: PBCDLocalSet | SEIPLocalSet | CosmicDawnLocalSet
    cutout: MosaicCutout
    exposures: tuple[ExposureGeometry, ...]
    backend: str


def fetch_sha_mosaic(
    ra: float,
    dec: float,
    channel: str | int,
    *,
    size_arcsec: float,
    aorkey: int | None = None,
    data_dir: str | Path = DEFAULT_PBCD_DIR,
    mask_bad_bits: int | None = None,
    force_download: bool = False,
) -> MosaicInputs:
    """Fetch one public SHA AOR-level PBCD mosaic."""
    products = discover_pbcd(ra, dec, channel, aorkey=aorkey)
    local = fetch_pbcd(
        products,
        data_dir=data_dir,
        force_download=force_download,
    )[0]
    cutout = load_mosaic_cutout(
        local,
        ra,
        dec,
        size_arcsec,
        mask_bad_bits=mask_bad_bits,
    )
    return MosaicInputs(
        local_product=local,
        cutout=cutout,
        exposures=tuple(parse_refptg(local.refptg_path)),
        backend="SHA PBCD mosaic (single AOR)",
    )


def _request_cosmic_dawn_science_cutout(
    ra: float,
    dec: float,
    channel: str,
    size_arcsec: float,
) -> tuple[str, str, str]:
    """Return ``(cutout_url, field, version)`` from the public IRSA service."""
    if not 1.0 <= float(size_arcsec) <= 600.0:
        raise ValueError("Cosmic Dawn cutout size must be between 1 and 600 arcsec")
    query = urlencode({
        "mission": "Cosmic_Dawn",
        "locstr": f"{float(ra):.10f} {float(dec):.10f} eq",
        "sizeX": f"{float(size_arcsec):.6f}",
        "units": "arcsec",
        "mode": "PI",
        "min_size": "1",
        "max_size": "600",
        "ntable_cutouts": "1",
        "cutouttbl1": "irac",
    })
    ensure_https_ca_bundle()

    def request():
        with urlopen(f"{SEIP_CUTOUT_URL}?{query}", timeout=180.0) as response:
            return response.read()

    xml = retry(request, attempts=3, delay_seconds=2.0,
                what="IRSA Cosmic Dawn cutout request")
    root = ElementTree.fromstring(xml)
    if root.attrib.get("status", "").lower() != "ok":
        raise RuntimeError("Cosmic Dawn cutout service did not return status=ok")
    matches = []
    for node in root.findall(".//cutouts/fits"):
        if not node.text:
            continue
        url = node.text.strip()
        match = _COSMIC_DAWN_NAME.search(url)
        if match is not None and int(match.group("channel")) == int(channel[-1]):
            matches.append((url, match.group("field").upper(),
                            match.group("version").lower()))
    if not matches:
        raise ValueError(
            f"Cosmic Dawn: no {channel} mosaic at ({ra:.6f}, {dec:.6f})"
        )
    fields = {item[1] for item in matches}
    if len(fields) != 1:
        raise RuntimeError(f"ambiguous Cosmic Dawn fields at target: {sorted(fields)}")
    return matches[0]


def _write_remote_aligned_cutout(
    remote_url: str,
    science_path: Path,
    output_path: Path,
    *,
    force: bool,
) -> Path:
    """Read only the byte ranges needed to align an ancillary remote FITS."""
    if output_path.is_file() and not force:
        return output_path
    ensure_https_ca_bundle()
    with fits.open(science_path, memmap=True) as science_hdul:
        science_header = science_hdul[0].header.copy()
        science_shape = science_hdul[0].shape
    science_wcs = WCS(science_header).celestial
    fsspec_kwargs = {"block_size": 4 * 1024 * 1024}
    if remote_url.lower().startswith("https://"):
        fsspec_kwargs["ssl"] = https_ssl_context()
    with fits.open(
        remote_url,
        use_fsspec=True,
        fsspec_kwargs=fsspec_kwargs,
    ) as remote_hdul:
        remote_hdu = remote_hdul[0]
        remote_header = remote_hdu.header.copy()
        remote_wcs = WCS(remote_header).celestial
        if not np.allclose(
            remote_wcs.pixel_scale_matrix,
            science_wcs.pixel_scale_matrix,
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise ValueError("Cosmic Dawn ancillary and science WCS grids differ")
        x0 = int(round(remote_wcs.wcs.crpix[0] - science_wcs.wcs.crpix[0]))
        y0 = int(round(remote_wcs.wcs.crpix[1] - science_wcs.wcs.crpix[1]))
        ny, nx = science_shape
        slices = (slice(y0, y0 + ny), slice(x0, x0 + nx))
        if x0 < 0 or y0 < 0 or x0 + nx > remote_hdu.shape[1] \
                or y0 + ny > remote_hdu.shape[0]:
            raise ValueError("Cosmic Dawn ancillary cutout lies outside full mosaic")
        data = np.asarray(remote_hdu.section[slices])
        cutout_wcs = remote_wcs.slice(slices)
    if data.shape != science_shape:
        raise ValueError(
            f"Cosmic Dawn ancillary shape {data.shape} != science {science_shape}"
        )
    remote_header.update(cutout_wcs.to_header(relax=True))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".tmp")
    try:
        fits.writeto(temporary, data, remote_header, overwrite=True)
        temporary.replace(output_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return output_path


def parse_cosmic_dawn_info(
    path: str | Path,
    channel: str | int,
    cutout: MosaicCutout,
) -> list[ExposureGeometry]:
    """Build local detector geometry from a Moneti contributor table."""
    channel = normalize_channel(channel)
    table = Table.read(path, format="ascii.ipac")
    names = {name.upper(): name for name in table.colnames}
    required = ("MJD", "RA", "DEC", "PA", "EXPTIME")
    if not all(name in names for name in required):
        missing = [name for name in required if name not in names]
        raise ValueError(f"Cosmic Dawn info table is missing {missing}")
    mjd = np.asarray(table[names["MJD"]], dtype=float)
    ra = np.asarray(table[names["RA"]], dtype=float)
    dec = np.asarray(table[names["DEC"]], dtype=float)
    pa = np.asarray(table[names["PA"]], dtype=float)
    exptime = np.asarray(table[names["EXPTIME"]], dtype=float)
    center = SkyCoord(
        *cutout.wcs.pixel_to_world_values(
            (cutout.shape[1] - 1) / 2.0,
            (cutout.shape[0] - 1) / 2.0,
        ),
        unit="deg",
    )
    corners = cutout.wcs.pixel_to_world(
        np.asarray([0, cutout.shape[1] - 1, 0, cutout.shape[1] - 1]),
        np.asarray([0, 0, cutout.shape[0] - 1, cutout.shape[0] - 1]),
    )
    cutout_radius = float(np.max(center.separation(corners).deg))
    detector_radius = (
        np.sqrt(2.0) * 128.0 * IRAC_NATIVE_PIXEL_SCALE_ARCSEC[channel] / 3600.0
    )
    positions = SkyCoord(ra, dec, unit="deg")
    nearby = (
        np.isfinite(mjd) & np.isfinite(exptime) & (exptime > 0)
        & np.isfinite(ra) & np.isfinite(dec) & np.isfinite(pa)
        & (positions.separation(center).deg <= cutout_radius + detector_radius)
    )
    scale = IRAC_NATIVE_PIXEL_SCALE_ARCSEC[channel] / 3600.0
    exposures = []
    for index in np.flatnonzero(nearby):
        radians = np.radians(float(pa[index]))
        wcs = WCS(naxis=2)
        wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        wcs.wcs.cunit = ["deg", "deg"]
        wcs.wcs.crpix = [128.5, 128.5]
        wcs.wcs.crval = [float(ra[index]), float(dec[index])]
        wcs.wcs.cd = np.asarray([
            [-scale * np.cos(radians), scale * np.sin(radians)],
            [scale * np.sin(radians), scale * np.cos(radians)],
        ])
        wcs.array_shape = (256, 256)
        exposure_mission = "warm" if mjd[index] >= 54966.0 else "cryo"
        if channel in {"IRAC3", "IRAC4"}:
            exposure_mission = "cryo"
        exposures.append(ExposureGeometry(
            filename=f"MJD{mjd[index]:.7f}",
            wcs=wcs,
            exposure_time=float(exptime[index]),
            mjd=float(mjd[index]),
            mission=exposure_mission,
        ))
    if not exposures:
        raise ValueError("no Cosmic Dawn contributing exposure covers the cutout")
    return exposures


def fetch_cosmic_dawn_mosaic(
    ra: float,
    dec: float,
    channel: str | int,
    *,
    size_arcsec: float,
    data_dir: str | Path = DEFAULT_COSMIC_DAWN_DIR,
    mask_bad_bits: int | None = None,
    force_download: bool = False,
) -> MosaicInputs:
    """Fetch aligned public Cosmic Dawn/SLS mosaic cutouts without full files."""
    channel = normalize_channel(channel)
    cache_dir = Path(data_dir) / channel.lower() / _cache_key(
        ra, dec, size_arcsec
    )
    manifest_path = cache_dir / "manifest.json"
    if manifest_path.is_file() and not force_download:
        manifest = json.loads(manifest_path.read_text())
        local = CosmicDawnLocalSet(
            product=CosmicDawnProductSet(**manifest["product"]),
            science_path=Path(manifest["science_path"]),
            uncertainty_path=Path(manifest["uncertainty_path"]),
            coverage_path=Path(manifest["coverage_path"]),
            info_path=Path(manifest["info_path"]),
        )
        if all(path.is_file() for path in (
            local.science_path, local.uncertainty_path,
            local.coverage_path, local.info_path,
        )):
            cutout = load_mosaic_cutout(
                local, ra, dec, size_arcsec, mask_bad_bits=mask_bad_bits
            )
            return MosaicInputs(
                local, cutout,
                tuple(parse_cosmic_dawn_info(local.info_path, channel, cutout)),
                "Cosmic Dawn/SLS mosaic",
            )

    cutout_url, field, version = _request_cosmic_dawn_science_cutout(
        ra, dec, channel, size_arcsec
    )
    number = int(channel[-1])
    image_root = f"{COSMIC_DAWN_ROOT}/images/{field}"
    filename_root = f"CDS_{field}_ch{number}"
    info_version_match = re.match(r"v\d+", version)
    if info_version_match is None:
        raise ValueError(f"unrecognized Cosmic Dawn version {version!r}")
    info_version = info_version_match.group(0)
    product = CosmicDawnProductSet(
        channel=channel,
        field=field,
        product_root=f"{filename_root}_{version}",
        version=version,
        science_url=f"{image_root}/{filename_root}_ima_{version}.fits",
        uncertainty_url=f"{image_root}/{filename_root}_unc_{version}.fits",
        coverage_url=f"{image_root}/{filename_root}_cov_{version}.fits",
        exposure_time_url=f"{image_root}/{filename_root}_tim_{version}.fits",
        info_url=(
            f"{COSMIC_DAWN_ROOT}/tables/"
            f"{filename_root}_info_{info_version}.tbl"
        ),
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    science_path = download_file(
        cutout_url, cache_dir / "science.fits", force=force_download
    )
    local = CosmicDawnLocalSet(
        product=product,
        science_path=science_path,
        uncertainty_path=_write_remote_aligned_cutout(
            product.uncertainty_url, science_path,
            cache_dir / "uncertainty.fits", force=force_download,
        ),
        coverage_path=_write_remote_aligned_cutout(
            product.coverage_url, science_path,
            cache_dir / "coverage.fits", force=force_download,
        ),
        info_path=download_file(
            product.info_url, cache_dir / "contributors.tbl",
            force=force_download,
        ),
    )
    manifest = {
        "product": asdict(product),
        "science_path": str(local.science_path.resolve()),
        "uncertainty_path": str(local.uncertainty_path.resolve()),
        "coverage_path": str(local.coverage_path.resolve()),
        "info_path": str(local.info_path.resolve()),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    cutout = load_mosaic_cutout(
        local, ra, dec, size_arcsec, mask_bad_bits=mask_bad_bits
    )
    exposures = tuple(parse_cosmic_dawn_info(local.info_path, channel, cutout))
    return MosaicInputs(local, cutout, exposures, "Cosmic Dawn/SLS mosaic")


def _seip_science_products(ra: float, dec: float, channel: str):
    """Return original full-mosaic URLs covering one coordinate."""
    rows = retry(
        lambda: Irsa.query_sia(
            pos=(SkyCoord(float(ra), float(dec), unit="deg"), 1.0 * u.arcsec),
            collection=SEIP_COLLECTION,
            maxrec=10000,
        ),
        what="IRSA SEIP discovery",
    )
    suffix = f".IRAC.{channel[-1]}.mosaic.fits"
    products = {}
    for row in rows:
        if str(row["energy_bandpassname"]).upper() != channel:
            continue
        url = str(row["access_url"]).strip()
        if not url.endswith(suffix) or ".median_mosaic.fits" in url:
            continue
        root = url.rsplit("/", 1)[-1].removesuffix(suffix)
        if root:
            products[root] = url
    if not products:
        raise ValueError(
            f"SEIP: no {channel} Super Mosaic at ({ra:.6f}, {dec:.6f})"
        )
    return products


def _request_seip_cutouts(ra: float, dec: float, size_arcsec: float):
    """Request science/uncertainty/coverage cutouts and return their URLs."""
    if not 18.0 <= float(size_arcsec) <= 1800.0:
        raise ValueError("SEIP cutout size must be between 18 and 1800 arcsec")
    query = urlencode({
        "mission": "SEIP",
        "locstr": f"{float(ra):.10f} {float(dec):.10f} eq",
        "sizeX": f"{float(size_arcsec):.6f}",
        "units": "arcsec",
        "mode": "PI",
        "min_size": "18",
        "max_size": "1800",
        "ntable_cutouts": "2",
        "cutouttbl1": "science",
        "cutouttbl2": "ancillary",
    })
    ensure_https_ca_bundle()

    def request():
        with urlopen(f"{SEIP_CUTOUT_URL}?{query}", timeout=180.0) as response:
            return response.read()

    xml = retry(request, attempts=3, delay_seconds=2.0,
                what="IRSA SEIP cutout request")
    root = ElementTree.fromstring(xml)
    if root.attrib.get("status", "").lower() != "ok":
        raise RuntimeError("SEIP cutout service did not return status=ok")
    urls = [node.text.strip() for node in root.findall(".//cutouts/fits")
            if node.text and node.text.strip()]
    if not urls:
        raise RuntimeError("SEIP cutout service returned no FITS URLs")
    return urls


def _group_cutout_urls(urls, channel: str):
    grouped: dict[str, dict[str, str]] = {}
    for url in urls:
        match = _SEIP_NAME.search(url)
        if match is None or int(match.group("channel")) != int(channel[-1]):
            continue
        grouped.setdefault(match.group("root"), {})[
            match.group("kind").lower()
        ] = url
    return {
        root: files for root, files in grouped.items()
        if {"mosaic", "unc", "cov"}.issubset(files)
    }


def _cache_key(ra: float, dec: float, size_arcsec: float) -> str:
    text = f"{float(ra):.8f}:{float(dec):.8f}:{float(size_arcsec):.3f}"
    return sha256(text.encode("ascii")).hexdigest()[:12]


def _select_deepest_cutout(grouped, cache_dir: Path, *, force: bool):
    """Select the overlapping tile with the largest median cutout coverage."""
    ranked = []
    for root, urls in grouped.items():
        path = download_file(
            urls["cov"], cache_dir / f"candidate_{root}.cov.fits", force=force
        )
        coverage = np.asarray(fits.getdata(path, memmap=True), dtype=float)
        finite = coverage[np.isfinite(coverage) & (coverage > 0)]
        depth = float(np.median(finite)) if finite.size else -np.inf
        ranked.append((depth, root))
    if not ranked:
        raise ValueError("SEIP returned no complete Super Mosaic cutout")
    return max(ranked)[1]


def parse_seip_dcelist(
    path: str | Path,
    channel: str | int,
    cutout: MosaicCutout,
) -> list[ExposureGeometry]:
    """Build local CBCD WCSes from a SEIP contributor ``dcelist``.

    The list supplies pointing centers and position angles but not the full
    CBCD distortion. This is the linear geometry used here to select, rotate,
    and weight the local contributing detector PRFs.
    """
    channel = normalize_channel(channel)
    values = np.loadtxt(path, comments="#", usecols=(0, 4, 5, 6, 7), ndmin=2)
    dce, exptime, ra, dec, pa = values.T
    center = SkyCoord(
        *cutout.wcs.pixel_to_world_values(
            (cutout.shape[1] - 1) / 2.0,
            (cutout.shape[0] - 1) / 2.0,
        ),
        unit="deg",
    )
    corners = cutout.wcs.pixel_to_world(
        np.asarray([0, cutout.shape[1] - 1, 0, cutout.shape[1] - 1]),
        np.asarray([0, 0, cutout.shape[0] - 1, cutout.shape[0] - 1]),
    )
    cutout_radius = float(np.max(center.separation(corners).deg))
    detector_radius = (
        np.sqrt(2.0) * 128.0 * IRAC_NATIVE_PIXEL_SCALE_ARCSEC[channel] / 3600.0
    )
    positions = SkyCoord(ra, dec, unit="deg")
    nearby = (
        np.isfinite(exptime) & (exptime > 0)
        & np.isfinite(ra) & np.isfinite(dec) & np.isfinite(pa)
        & (positions.separation(center).deg <= cutout_radius + detector_radius)
    )
    scale = IRAC_NATIVE_PIXEL_SCALE_ARCSEC[channel] / 3600.0
    exposures = []
    for dce_id, seconds, pointing_ra, pointing_dec, angle in zip(
        dce[nearby], exptime[nearby], ra[nearby], dec[nearby], pa[nearby]
    ):
        radians = np.radians(float(angle))
        wcs = WCS(naxis=2)
        wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        wcs.wcs.cunit = ["deg", "deg"]
        wcs.wcs.crpix = [128.5, 128.5]
        wcs.wcs.crval = [float(pointing_ra), float(pointing_dec)]
        wcs.wcs.cd = np.asarray([
            [-scale * np.cos(radians), scale * np.sin(radians)],
            [scale * np.sin(radians), scale * np.cos(radians)],
        ])
        wcs.array_shape = (256, 256)
        exposures.append(ExposureGeometry(
            filename=f"DCE{int(dce_id)}",
            wcs=wcs,
            exposure_time=float(seconds),
        ))
    if not exposures:
        raise ValueError("no SEIP contributing exposure covers the cutout")
    return exposures


def fetch_seip_mosaic(
    ra: float,
    dec: float,
    channel: str | int,
    *,
    size_arcsec: float,
    super_mosaic_id: str | None = None,
    data_dir: str | Path = DEFAULT_SEIP_DIR,
    mask_bad_bits: int | None = None,
    force_download: bool = False,
) -> MosaicInputs:
    """Fetch the deepest overlapping SEIP Super Mosaic cutout."""
    channel = normalize_channel(channel)
    cache_dir = Path(data_dir) / channel.lower() / _cache_key(
        ra, dec, size_arcsec
    )
    manifest_path = cache_dir / "manifest.json"
    if manifest_path.is_file() and not force_download:
        manifest = json.loads(manifest_path.read_text())
        local = SEIPLocalSet(
            product=SEIPProductSet(**manifest["product"]),
            science_path=Path(manifest["science_path"]),
            uncertainty_path=Path(manifest["uncertainty_path"]),
            coverage_path=Path(manifest["coverage_path"]),
            dcelist_path=Path(manifest["dcelist_path"]),
        )
        if all(path.is_file() for path in (
            local.science_path, local.uncertainty_path,
            local.coverage_path, local.dcelist_path,
        )):
            cutout = load_mosaic_cutout(
                local, ra, dec, size_arcsec, mask_bad_bits=mask_bad_bits
            )
            return MosaicInputs(
                local, cutout,
                tuple(parse_seip_dcelist(local.dcelist_path, channel, cutout)),
                "SEIP Super Mosaic",
            )

    originals = _seip_science_products(ra, dec, channel)
    grouped = _group_cutout_urls(
        _request_seip_cutouts(ra, dec, size_arcsec), channel
    )
    grouped = {root: files for root, files in grouped.items()
               if root in originals}
    if super_mosaic_id is not None:
        wanted = str(super_mosaic_id)
        grouped = {
            root: files for root, files in grouped.items() if wanted in root
        }
        if not grouped:
            raise ValueError(f"SEIP Super Mosaic {wanted!r} does not cover target")
    cache_dir.mkdir(parents=True, exist_ok=True)
    root = _select_deepest_cutout(grouped, cache_dir, force=force_download)
    urls = grouped[root]
    original_science = originals[root]
    dcelist_url = original_science.removesuffix("mosaic.fits") + "dcelist.txt"
    product = SEIPProductSet(
        channel=channel,
        super_mosaic_id=root.split(".", 1)[-1],
        product_root=root,
        science_url=original_science,
        uncertainty_url=urls["unc"],
        coverage_url=urls["cov"],
        dcelist_url=dcelist_url,
    )
    local = SEIPLocalSet(
        product=product,
        science_path=download_file(
            urls["mosaic"], cache_dir / "science.fits", force=force_download
        ),
        uncertainty_path=download_file(
            urls["unc"], cache_dir / "uncertainty.fits", force=force_download
        ),
        coverage_path=download_file(
            urls["cov"], cache_dir / "coverage.fits", force=force_download
        ),
        dcelist_path=download_file(
            dcelist_url, cache_dir / "dcelist.txt", force=force_download
        ),
    )
    manifest = {
        "product": asdict(product),
        "science_path": str(local.science_path.resolve()),
        "uncertainty_path": str(local.uncertainty_path.resolve()),
        "coverage_path": str(local.coverage_path.resolve()),
        "dcelist_path": str(local.dcelist_path.resolve()),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    cutout = load_mosaic_cutout(
        local, ra, dec, size_arcsec, mask_bad_bits=mask_bad_bits
    )
    exposures = tuple(parse_seip_dcelist(local.dcelist_path, channel, cutout))
    return MosaicInputs(local, cutout, exposures, "SEIP Super Mosaic")
