"""Reconstruct a position-dependent effective PRF for an IRAC PBCD mosaic.

The reconstruction combines the official 5x5 detector PRF grid with the
refined pointing geometry in the PBCD ``refptg.tbl``. It is evaluated directly
on the mosaic pixel grid and normalized to unit integrated flux.
"""

from __future__ import annotations

import shutil
import tarfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from scipy.ndimage import map_coordinates

from .config import DEFAULT_CALIBRATION_DIR, normalize_channel
from .netutils import download_file


CRYO_PRF_URL = (
    "https://irsa.ipac.caltech.edu/data/SPITZER/docs/irac/"
    "calibrationfiles/psfprf/070131_prfs.tar"
)
WARM_PRF_URL = (
    "https://irsa.ipac.caltech.edu/data/SPITZER/docs/irac/"
    "calibrationfiles/psfprf/140915_warm_prfs.tgz"
)
_ARRAY_GRID = np.asarray([25, 77, 129, 181, 233], dtype=float)
_CRYO_END_MJD = 54966.0


def mission_from_header(channel: str | int, header) -> str:
    """Select the cryogenic or warm PRF set from a mosaic header."""
    channel = normalize_channel(channel)
    if channel in {"IRAC3", "IRAC4"}:
        return "cryo"
    mjd = float(header.get("MJD_OBS", header.get("MJD-OBS", np.nan)))
    if np.isfinite(mjd):
        return "warm" if mjd >= _CRYO_END_MJD else "cryo"
    date = str(header.get("DATE_OBS", header.get("DATE-OBS", "")))
    return "warm" if date >= "2009-05-15" else "cryo"


def _safe_extract_prfs(archive: Path, destination: Path, prefix: str) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:*") as bundle:
        for member in bundle.getmembers():
            name = Path(member.name).name
            if not member.isfile() or not name.startswith(prefix) \
                    or not name.endswith(".fits"):
                continue
            source = bundle.extractfile(member)
            if source is None:
                continue
            with source, (destination / name).open("wb") as output:
                shutil.copyfileobj(source, output)


def fetch_prf_files(channel: str | int, mission: str, *,
                    data_dir: str | Path = DEFAULT_CALIBRATION_DIR,
                    force_download: bool = False) -> list[Path]:
    """Cache the 25 official oversampled detector PRFs for one channel."""
    channel = normalize_channel(channel)
    number = int(channel[-1])
    mission = str(mission).lower()
    if mission not in {"cryo", "warm"}:
        raise ValueError("mission must be 'cryo' or 'warm'")
    if mission == "warm" and number not in {1, 2}:
        raise ValueError("warm-mission PRFs only exist for IRAC1 and IRAC2")
    root = Path(data_dir)
    folder = root / "prf" / mission
    prefix = f"IRAC{number}_" if mission == "cryo" else f"IRACPC{number}_"
    files = sorted(folder.glob(f"{prefix}col*_row*.fits"))
    if len(files) == 25 and not force_download:
        return files
    url = CRYO_PRF_URL if mission == "cryo" else WARM_PRF_URL
    archive = download_file(
        url,
        root / "downloads" / url.rsplit("/", 1)[-1],
        force=force_download,
    )
    _safe_extract_prfs(archive, folder, prefix)
    files = sorted(folder.glob(f"{prefix}col*_row*.fits"))
    if len(files) != 25:
        raise RuntimeError(f"expected 25 {mission} {channel} PRFs; found {len(files)}")
    return files


class DetectorPRFLibrary:
    """Official oversampled PRFs indexed by native detector position."""

    def __init__(self, files: list[str | Path]):
        if len(files) != 25:
            raise ValueError("a detector PRF library requires all 25 grid files")
        self.tables: dict[tuple[int, int], np.ndarray] = {}
        self.channel: int | None = None
        for file in files:
            with fits.open(file, memmap=True) as hdul:
                data = np.asarray(hdul[0].data, dtype=np.float32)
                header = hdul[0].header
            channel = int(header["CHAN"])
            if self.channel is None:
                self.channel = channel
            elif channel != self.channel:
                raise ValueError("PRF files contain more than one IRAC channel")
            col = int(str(header["PRFCOL"]).strip())
            row = int(str(header["PRFROW"]).strip())
            if data.shape[0] < 126 or data.shape[1] < 126:
                raise ValueError(f"unexpected PRF array {data.shape} in {file}")
            table = np.asarray(data[1:126, 1:126], dtype=np.float32)
            norm = float(np.sum(table[2::5, 2::5]))
            if not np.isfinite(norm) or norm <= 0:
                raise ValueError(f"invalid PRF normalization in {file}")
            self.tables[(col, row)] = table / norm
        if len(self.tables) != 25:
            raise ValueError("PRF grid locations are incomplete or duplicated")

    @staticmethod
    def _nearest_grid(pixel: float) -> int:
        # Calibration filenames use one-based detector coordinates.
        return int(_ARRAY_GRID[np.argmin(np.abs(_ARRAY_GRID - (pixel + 1.0)))])

    def sample(self, dx: np.ndarray, dy: np.ndarray, *,
               detector_x: float, detector_y: float) -> np.ndarray:
        """Sample a local PRF at detector offsets ``dx, dy`` in native pixels."""
        key = (self._nearest_grid(detector_x), self._nearest_grid(detector_y))
        table = self.tables[key]
        x = 62.0 + 5.0 * np.asarray(dx, dtype=float)
        y = 62.0 + 5.0 * np.asarray(dy, dtype=float)
        return map_coordinates(
            table, [y, x], order=1, mode="constant", cval=0.0,
        )


@dataclass(frozen=True)
class ExposureGeometry:
    """Refined WCS and weight for one CBCD contributing to a PBCD mosaic."""

    filename: str
    wcs: WCS
    exposure_time: float
    mjd: float = np.nan
    mission: str | None = None
    detector_shape: tuple[int, int] = (256, 256)

    def detector_position(self, ra: float, dec: float) -> tuple[float, float]:
        x, y = self.wcs.world_to_pixel_values(float(ra), float(dec))
        return float(x), float(y)

    def covers(self, ra: float, dec: float, *, margin: float = 0.0) -> bool:
        x, y = self.detector_position(ra, dec)
        height, width = self.detector_shape
        return (
            np.isfinite(x) and np.isfinite(y)
            and -0.5 + margin <= x <= width - 0.5 - margin
            and -0.5 + margin <= y <= height - 0.5 - margin
        )


def _column(table: Table, name: str):
    lookup = {column.upper(): column for column in table.colnames}
    try:
        return table[lookup[name.upper()]]
    except KeyError as error:
        raise ValueError(f"refptg table is missing required column {name}") from error


def parse_refptg(path: str | Path, *,
                 detector_crpix: tuple[float, float] = (128.5, 128.5),
                 detector_shape: tuple[int, int] = (256, 256)) -> list[ExposureGeometry]:
    """Parse a PBCD refined-pointing table into linear TAN exposure WCSes.

    The PBCD table supplies refined CRVAL and CD values but not the detector
    reference pixel; full-array IRAC CBCDs use ``(128.5, 128.5)``.
    """
    table = Table.read(path, format="ascii.ipac")
    required = {
        name: _column(table, name)
        for name in (
            "Filename", "EXPTIME", "RARFND", "DECRFND",
            "CD11RFND", "CD12RFND", "CD21RFND", "CD22RFND",
        )
    }
    exposures: list[ExposureGeometry] = []
    for index in range(len(table)):
        values = [float(required[name][index]) for name in (
            "EXPTIME", "RARFND", "DECRFND", "CD11RFND", "CD12RFND",
            "CD21RFND", "CD22RFND",
        )]
        if not np.isfinite(values).all() or values[0] <= 0:
            continue
        wcs = WCS(naxis=2)
        wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        wcs.wcs.cunit = ["deg", "deg"]
        wcs.wcs.crpix = np.asarray(detector_crpix, dtype=float)
        wcs.wcs.crval = values[1:3]
        wcs.wcs.cd = np.asarray([[values[3], values[4]],
                                 [values[5], values[6]]])
        wcs.array_shape = detector_shape
        exposures.append(ExposureGeometry(
            filename=str(required["Filename"][index]).strip(),
            wcs=wcs,
            exposure_time=values[0],
            detector_shape=detector_shape,
        ))
    if not exposures:
        raise ValueError(f"no valid exposure geometry in {path}")
    return exposures
