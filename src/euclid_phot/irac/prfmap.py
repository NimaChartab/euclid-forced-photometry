"""PRFMap-style spatial PSF grids for Tractor.

The preferred path consumes the FITS grid produced by the public PRFMap
package, matching how The Farmer uses PRFMap output.  When those field-specific
products are unavailable, ``PRFMapGeneratedGrid`` follows PRFMap's published
nearest-detector-PRF, rotate, stack, and resample procedure using the PBCD
``refptg.tbl`` and the official IPAC calibration PRFs.
"""

# The PRF reconstruction methodology adopted here follows the public Cosmic
# Dawn PRFMap package: https://github.com/cosmic-dawn/prfmap

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from scipy.ndimage import map_coordinates, rotate, shift
from scipy.spatial import cKDTree
from tractor.psf import PixelizedPSF

from .config import IRAC_NATIVE_PIXEL_SCALE_ARCSEC, normalize_channel
from .prf import (
    DetectorPRFLibrary,
    ExposureGeometry,
    fetch_prf_files,
    mission_from_header,
    parse_refptg,
)


def _position_angle_deg(wcs: WCS) -> float:
    """Position angle of image +Y, east of north."""
    matrix = np.asarray(wcs.celestial.pixel_scale_matrix, dtype=float)
    return float(np.degrees(np.arctan2(matrix[0, 1], matrix[1, 1])))


def _angle_difference_deg(first: float, second: float) -> float:
    return float((first - second + 180.0) % 360.0 - 180.0)


def _normalize_stamp(stamp: np.ndarray) -> np.ndarray:
    stamp = np.asarray(stamp, dtype=np.float64)
    norm = float(np.sum(stamp[np.isfinite(stamp)]))
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("PRFMap stamp has non-positive normalization")
    return np.asarray(np.where(np.isfinite(stamp), stamp / norm, 0.0),
                      dtype=np.float32)


def _centered_size(array: np.ndarray, size: int) -> np.ndarray:
    """Center-crop or zero-pad a square array to an odd output size."""
    array = np.asarray(array, dtype=float)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError("PRF stamps must be square 2-D arrays")
    if size < 5 or size % 2 == 0:
        raise ValueError("stamp_size must be an odd integer >= 5")
    if array.shape[0] > size:
        start = (array.shape[0] - size) // 2
        array = array[start:start + size, start:start + size]
    elif array.shape[0] < size:
        before = (size - array.shape[0]) // 2
        after = size - array.shape[0] - before
        array = np.pad(array, ((before, after), (before, after)))
    return array


def _resample_centered(array: np.ndarray, scale_factor: float,
                       output_size: int) -> np.ndarray:
    """Point-sample an array on an odd grid with coincident geometric centers."""
    if not np.isfinite(scale_factor) or scale_factor <= 0:
        raise ValueError("scale_factor must be finite and positive")
    if output_size < 5 or output_size % 2 == 0:
        raise ValueError("output_size must be an odd integer >= 5")
    array = np.asarray(array, dtype=np.float64)
    input_center_y = (array.shape[0] - 1) / 2.0
    input_center_x = (array.shape[1] - 1) / 2.0
    output_center = (output_size - 1) / 2.0
    output = np.arange(output_size, dtype=np.float64) - output_center
    grid_x, grid_y = np.meshgrid(
        input_center_x + output / scale_factor,
        input_center_y + output / scale_factor,
    )
    return map_coordinates(
        array, [grid_y, grid_x], order=3, mode="constant", cval=0.0,
    )


def _integrated_resample_centered(
    array: np.ndarray,
    scale_factor: float,
    output_size: int,
    *,
    subsamples: int = 5,
) -> np.ndarray:
    """Integrate an oversampled PRF over each output mosaic pixel.

    ``scale_factor`` is input-pixel size divided by output-pixel size.  The
    legacy point sampler evaluated only the output pixel center.  Here a
    regular subpixel grid samples the full pixel footprint; normalization is
    applied after resampling, so the mean and integral differ only by a common
    factor.
    """
    if int(subsamples) < 1:
        raise ValueError("subsamples must be a positive integer")
    if int(subsamples) == 1:
        return _resample_centered(array, scale_factor, output_size)
    array = np.asarray(array, dtype=np.float64)
    input_center_y = (array.shape[0] - 1) / 2.0
    input_center_x = (array.shape[1] - 1) / 2.0
    output_center = (output_size - 1) / 2.0
    output = np.arange(output_size, dtype=np.float64) - output_center
    subpixel = (np.arange(int(subsamples)) + 0.5) / int(subsamples) - 0.5
    integrated = np.zeros((output_size, output_size), dtype=np.float64)
    for dy in subpixel:
        for dx in subpixel:
            grid_x, grid_y = np.meshgrid(
                input_center_x + (output + dx) / scale_factor,
                input_center_y + (output + dy) / scale_factor,
            )
            integrated += map_coordinates(
                array,
                [grid_y, grid_x],
                order=3,
                mode="constant",
                cval=0.0,
            )
    return integrated / float(int(subsamples) ** 2)


def _first_moment_centroid(array: np.ndarray) -> tuple[float, float]:
    """Return the finite first-moment centroid as ``(x, y)`` pixels."""
    array = np.asarray(array, dtype=np.float64)
    finite = np.where(np.isfinite(array), array, 0.0)
    norm = float(finite.sum())
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("detector PRF has non-positive normalization")
    yy, xx = np.indices(finite.shape, dtype=np.float64)
    return float(np.sum(finite * xx) / norm), float(np.sum(finite * yy) / norm)


def _recenter_prf(array: np.ndarray) -> tuple[np.ndarray, tuple[float, float]]:
    """Center a detector PRF by its first moment before rotation/stacking."""
    array = np.asarray(array, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError("detector PRF must be a 2-D array")
    centroid_x, centroid_y = _first_moment_centroid(array)
    target_x = (array.shape[1] - 1) / 2.0
    target_y = (array.shape[0] - 1) / 2.0
    offset = (centroid_x - target_x, centroid_y - target_y)
    centered = shift(
        array,
        shift=(-offset[1], -offset[0]),
        order=3,
        mode="constant",
        cval=0.0,
        prefilter=True,
    )
    return centered, offset


class PRFMapGeneratedGrid:
    """Lazy fixed grid reconstructed with the public PRFMap algorithm.

    This fallback differs intentionally from ``euclid_phot.irac``: PRFs are evaluated
    only at fixed grid nodes, detector PRFs are explicitly rotated into the
    mosaic pixel frame, and the nearest node is used for each source.  Uniform
    frame weighting is the PRFMap default; exposure-time weighting is optional.
    """

    backend = "generated PRFMap-compatible grid"

    def __init__(
        self,
        exposures: list[ExposureGeometry],
        detector_prfs: DetectorPRFLibrary | Mapping[str, DetectorPRFLibrary],
        output_wcs: WCS,
        *,
        channel: str | int,
        grid_spacing_arcsec: float = 28.8,
        stamp_size: int = 51,
        weighting: str = "uniform",
        oversampling: int = 5,
        pixel_integration_subsamples: int = 1,
        recenter_detector_prfs: bool = True,
    ):
        if not exposures:
            raise ValueError("at least one exposure is required")
        if grid_spacing_arcsec <= 0:
            raise ValueError("grid_spacing_arcsec must be positive")
        if weighting not in {"uniform", "exptime"}:
            raise ValueError("weighting must be 'uniform' or 'exptime'")
        if oversampling < 1:
            raise ValueError("oversampling must be positive")
        if pixel_integration_subsamples < 1:
            raise ValueError("pixel_integration_subsamples must be positive")
        self.exposures = list(exposures)
        if isinstance(detector_prfs, Mapping):
            self.detector_prfs_by_mission = {
                str(name).lower(): library
                for name, library in detector_prfs.items()
            }
            if not self.detector_prfs_by_mission:
                raise ValueError("detector_prfs mapping cannot be empty")
        else:
            self.detector_prfs_by_mission = {"default": detector_prfs}
        self.detector_prfs = detector_prfs
        self.output_wcs = output_wcs.celestial
        self.output_wcs.array_shape = output_wcs.array_shape
        self.channel = normalize_channel(channel)
        self.stamp_size = int(stamp_size)
        self.weighting = weighting
        self.oversampling = int(oversampling)
        self.pixel_integration_subsamples = int(pixel_integration_subsamples)
        self.recenter_detector_prfs = bool(recenter_detector_prfs)
        self.detector_centroid_offsets_native = {}
        self._detector_tables = {}
        for library_name, library in self.detector_prfs_by_mission.items():
            for key, table in library.tables.items():
                centered, offset = _recenter_prf(table)
                table_key = (library_name, *key)
                self.detector_centroid_offsets_native[table_key] = (
                    offset[0] / self.oversampling,
                    offset[1] / self.oversampling,
                )
                self._detector_tables[table_key] = (
                    centered if self.recenter_detector_prfs else table
                )
        centroid_label = (
            "centroid-recentered" if self.recenter_detector_prfs
            else "native optical-axis centroid"
        )
        sampling_label = (
            "PRFMap-compatible point resampling"
            if self.pixel_integration_subsamples == 1
            else (
                f"{self.pixel_integration_subsamples}x"
                f"{self.pixel_integration_subsamples} mosaic-pixel integration"
            )
        )
        self.backend = (
            f"generated PRFMap-compatible grid; {centroid_label}; "
            f"{sampling_label}; PRF libraries "
            f"{'+'.join(sorted(self.detector_prfs_by_mission))}"
        )
        mosaic_scale = float(
            np.sqrt(abs(np.linalg.det(self.output_wcs.pixel_scale_matrix)))
            * 3600.0
        )
        self.grid_spacing_pixels = float(grid_spacing_arcsec) / mosaic_scale
        self.mosaic_pixel_scale_arcsec = mosaic_scale
        self._mosaic_pa = _position_angle_deg(self.output_wcs)
        self._cache: dict[tuple[int, int], tuple[np.ndarray, int]] = {}

    def _node(self, px: float, py: float) -> tuple[float, float, tuple[int, int]]:
        ix = int(np.rint(float(px) / self.grid_spacing_pixels))
        iy = int(np.rint(float(py) / self.grid_spacing_pixels))
        return (ix * self.grid_spacing_pixels, iy * self.grid_spacing_pixels,
                (ix, iy))

    def _render_node(self, px: float, py: float) -> tuple[np.ndarray, int]:
        ra, dec = self.output_wcs.pixel_to_world_values(px, py)
        combined = None
        total_weight = 0.0
        frame_count = 0
        detector_scales = []
        for exposure in self.exposures:
            if not exposure.covers(ra, dec):
                continue
            if "default" in self.detector_prfs_by_mission:
                library_name = "default"
            else:
                library_name = str(exposure.mission or "").lower()
                if library_name not in self.detector_prfs_by_mission:
                    raise ValueError(
                        f"no {library_name or 'unspecified'} PRF library for "
                        f"exposure {exposure.filename}"
                    )
            library = self.detector_prfs_by_mission[library_name]
            detector_x, detector_y = exposure.detector_position(ra, dec)
            key = (
                library._nearest_grid(detector_x),
                library._nearest_grid(detector_y),
            )
            table = self._detector_tables[(library_name, *key)]
            relative_pa = _angle_difference_deg(
                _position_angle_deg(exposure.wcs), self._mosaic_pa
            )
            # This is PRFMap's convention: scipy rotates an image by -PA.
            rotated = rotate(
                table,
                -relative_pa,
                reshape=False,
                order=3,
                mode="constant",
                cval=0.0,
            )
            weight = (
                1.0 if self.weighting == "uniform"
                else float(exposure.exposure_time)
            )
            if combined is None:
                combined = np.zeros_like(rotated, dtype=np.float64)
            combined += weight * rotated
            total_weight += weight
            frame_count += 1
            detector_scales.append(float(
                np.sqrt(abs(np.linalg.det(
                    exposure.wcs.celestial.pixel_scale_matrix
                ))) * 3600.0
            ))
        if frame_count == 0:
            raise ValueError(
                f"no refptg exposure covers PRFMap node ({px:.2f}, {py:.2f})"
            )
        combined /= total_weight
        input_scale = float(np.median(detector_scales)) / self.oversampling
        factor = input_scale / self.mosaic_pixel_scale_arcsec
        stamp = _integrated_resample_centered(
            combined,
            factor,
            self.stamp_size,
            subsamples=self.pixel_integration_subsamples,
        )
        return _normalize_stamp(stamp), frame_count

    def render(self, px: float, py: float) -> tuple[np.ndarray, int]:
        """Return the nearest fixed-grid PRFMap stamp."""
        node_x, node_y, key = self._node(px, py)
        if key not in self._cache:
            try:
                self._cache[key] = self._render_node(node_x, node_y)
            except ValueError:
                # A quantized node can fall just outside a rotated PBCD edge.
                # Preserve nearest-grid behavior where possible, but do not
                # fail a source that is covered at its exact position.
                self._cache[key] = self._render_node(float(px), float(py))
        stamp, count = self._cache[key]
        return stamp.copy(), count


class PRFMapFileGrid:
    """Nearest-grid reader for public PRF grids.

    Genuine PRFMap products identify stamps with ``ID_GRIDPT`` and use the
    ``mosaic_gpNNNNNN.fits`` naming convention.  DAWN's released IRAC grid is
    a FITS table with the same sky coordinates plus an explicit ``filename``
    column.  Supporting both here lets a controlled fit change only the PRF.
    """

    backend = "PRFMap FITS grid"

    def __init__(
        self,
        grid_file: str | Path,
        prf_directory: str | Path,
        exposures: list[ExposureGeometry],
        output_wcs: WCS,
        *,
        channel: str | int,
        stamp_size: int = 51,
        prf_pixel_scale_arcsec: float | None = None,
    ):
        self.grid_file = Path(grid_file)
        self.prf_directory = Path(prf_directory)
        if not self.grid_file.is_file():
            raise FileNotFoundError(self.grid_file)
        if not self.prf_directory.is_dir():
            raise FileNotFoundError(self.prf_directory)
        self.exposures = list(exposures)
        self.output_wcs = output_wcs.celestial
        self.output_wcs.array_shape = output_wcs.array_shape
        self.channel = normalize_channel(channel)
        self.stamp_size = int(stamp_size)
        self.prf_pixel_scale_arcsec = prf_pixel_scale_arcsec
        # Public PRFMap grids are text; DAWN distributes its grid as a FITS
        # binary table.  Generic ``.txt`` files are not auto-identified by
        # Astropy, so retain the permissive ASCII reader for that case.
        is_fits = any(
            str(self.grid_file).lower().endswith(suffix)
            for suffix in (".fits", ".fit", ".fits.gz", ".fit.gz")
        )
        table = Table.read(
            self.grid_file, format="fits" if is_fits else "ascii"
        )
        names = {name.upper(): name for name in table.colnames}
        id_name = names.get("ID_GRIDPT")
        if id_name is None:
            raise ValueError("PRFMap grid needs an ID_GRIDPT column")
        if "RA" in names and ("DEC" in names or "DEC_CEN" in names):
            dec_name = names.get("DEC", names.get("DEC_CEN"))
            x, y = self.output_wcs.world_to_pixel_values(
                np.asarray(table[names["RA"]], float),
                np.asarray(table[dec_name], float),
            )
        elif "X" in names and "Y" in names:
            # PRFMap grid coordinates are FITS/IRAF one-based pixels.
            x = np.asarray(table[names["X"]], float) - 1.0
            y = np.asarray(table[names["Y"]], float) - 1.0
        else:
            raise ValueError("PRFMap grid needs RA/Dec or X/Y columns")
        finite = np.isfinite(x) & np.isfinite(y)
        self._ids = np.asarray(table[id_name], int)[finite]
        filename_name = names.get("FILENAME")
        self._filenames = (
            np.asarray(table[filename_name]).astype(str)[finite]
            if filename_name is not None else None
        )
        self._xy = np.column_stack((x[finite], y[finite]))
        if not len(self._ids):
            raise ValueError("PRFMap grid contains no finite nodes")
        self._tree = cKDTree(self._xy)
        self._cache: dict[int, tuple[np.ndarray, int]] = {}
        self._loaded_paths: set[Path] = set()

    @property
    def loaded_paths(self) -> tuple[Path, ...]:
        """PRF files used by this fit, for precise cache invalidation."""
        return tuple(sorted(self._loaded_paths))

    def _path(self, grid_id: int, filename: str | None = None) -> Path:
        if filename:
            relative = Path(str(filename).strip())
            candidates = (
                self.prf_directory / relative,
                self.prf_directory / relative.name,
                self.grid_file.parent / relative,
            )
            for candidate in candidates:
                if candidate.is_file():
                    return candidate
            raise FileNotFoundError(candidates[0])
        base = self.prf_directory / f"mosaic_gp{int(grid_id):06d}.fits"
        if base.is_file():
            return base
        compressed = base.with_suffix(".fits.gz")
        if compressed.is_file():
            return compressed
        raise FileNotFoundError(base)

    def _load(self, grid_id: int, filename: str | None = None) -> tuple[np.ndarray, int]:
        path = self._path(grid_id, filename)
        data, header = fits.getdata(path, header=True,
                                    memmap=True)
        self._loaded_paths.add(path.resolve())
        data = np.asarray(data, dtype=float)
        input_scale = self.prf_pixel_scale_arcsec
        if input_scale is None and header.get("SAMP_NEW"):
            input_scale = (
                IRAC_NATIVE_PIXEL_SCALE_ARCSEC[self.channel]
                / float(header["SAMP_NEW"])
            )
        output_scale = float(
            np.sqrt(abs(np.linalg.det(self.output_wcs.pixel_scale_matrix)))
            * 3600.0
        )
        if input_scale is not None and not np.isclose(input_scale, output_scale,
                                                       rtol=1.0e-3):
            data = _resample_centered(
                data, float(input_scale) / output_scale, self.stamp_size
            )
        else:
            data = _centered_size(data, self.stamp_size)
        return _normalize_stamp(data), int(header.get("NFRAMES", 0))

    def render(self, px: float, py: float) -> tuple[np.ndarray, int]:
        _, index = self._tree.query([float(px), float(py)], k=1)
        index = int(index)
        grid_id = int(self._ids[index])
        if grid_id not in self._cache:
            filename = (
                None if self._filenames is None else self._filenames[index]
            )
            self._cache[grid_id] = self._load(grid_id, filename)
        stamp, count = self._cache[grid_id]
        return stamp.copy(), count


class PRFMapMosaicPSF(PixelizedPSF):
    """A Tractor PSF that selects the nearest PRFMap stamp per source."""

    def __init__(self, builder, reference_pixel: tuple[float, float]):
        self.builder = builder
        stamp, _ = builder.render(*reference_pixel)
        super().__init__(stamp)

    def _local(self, px: float, py: float) -> PixelizedPSF:
        stamp, _ = self.builder.render(float(px), float(py))
        return PixelizedPSF(stamp)

    def getPointSourcePatch(self, px, py, minval=0.0, modelMask=None,
                            radius=None, derivs=False, **kwargs):
        patch = self._local(px, py).getPointSourcePatch(
            px, py, minval=minval, modelMask=modelMask, radius=radius,
            derivs=False, **kwargs,
        )
        if not derivs or patch is None:
            return patch
        epsilon = 0.01
        patch_dx = (
            self.getPointSourcePatch(
                px + epsilon, py, minval=minval, modelMask=modelMask,
                radius=radius, **kwargs,
            )
            - self.getPointSourcePatch(
                px - epsilon, py, minval=minval, modelMask=modelMask,
                radius=radius, **kwargs,
            )
        ) * (0.5 / epsilon)
        patch_dy = (
            self.getPointSourcePatch(
                px, py + epsilon, minval=minval, modelMask=modelMask,
                radius=radius, **kwargs,
            )
            - self.getPointSourcePatch(
                px, py - epsilon, minval=minval, modelMask=modelMask,
                radius=radius, **kwargs,
            )
        ) * (0.5 / epsilon)
        return patch, patch_dx, patch_dy

    def getFourierTransform(self, px, py, radius):
        return self._local(px, py).getFourierTransform(px, py, radius)


def build_prfmap_psf(
    local_product,
    mosaic_cutout,
    *,
    exposures: list[ExposureGeometry] | tuple[ExposureGeometry, ...] | None = None,
    mission: str = "auto",
    calibration_dir,
    force_download: bool = False,
    grid_file: str | Path | None = None,
    prf_directory: str | Path | None = None,
    prf_pixel_scale_arcsec: float | None = None,
    grid_spacing_arcsec: float = 28.8,
    stamp_size: int = 51,
    weighting: str = "uniform",
    pixel_integration_subsamples: int = 1,
    recenter_detector_prfs: bool = True,
):
    """Build a genuine or fallback PRFMap grid and its Tractor adapter."""
    if (grid_file is None) != (prf_directory is None):
        raise ValueError("grid_file and prf_directory must be supplied together")
    channel = normalize_channel(local_product.product.channel)
    exposures = (list(exposures) if exposures is not None
                 else parse_refptg(local_product.refptg_path))
    exposure_missions = {
        str(exposure.mission).lower()
        for exposure in exposures
        if exposure.mission is not None
    }
    if mission == "auto" and exposure_missions:
        selected_missions = sorted(exposure_missions)
    else:
        selected_missions = [
            mission_from_header(channel, mosaic_cutout.header)
            if mission == "auto" else str(mission).lower()
        ]
    selected_mission = "+".join(selected_missions)
    if grid_file is not None:
        builder = PRFMapFileGrid(
            grid_file,
            prf_directory,
            exposures,
            mosaic_cutout.wcs,
            channel=channel,
            stamp_size=stamp_size,
            prf_pixel_scale_arcsec=prf_pixel_scale_arcsec,
        )
    else:
        libraries = {
            this_mission: DetectorPRFLibrary(fetch_prf_files(
                channel,
                this_mission,
                data_dir=calibration_dir,
                force_download=force_download,
            ))
            for this_mission in selected_missions
        }
        detector_prfs = (
            next(iter(libraries.values()))
            if len(libraries) == 1 else libraries
        )
        builder = PRFMapGeneratedGrid(
            exposures,
            detector_prfs,
            mosaic_cutout.wcs,
            channel=channel,
            grid_spacing_arcsec=grid_spacing_arcsec,
            stamp_size=stamp_size,
            weighting=weighting,
            pixel_integration_subsamples=pixel_integration_subsamples,
            recenter_detector_prfs=recenter_detector_prfs,
        )
    reference = ((mosaic_cutout.shape[1] - 1) / 2.0,
                 (mosaic_cutout.shape[0] - 1) / 2.0)
    return (builder, PRFMapMosaicPSF(builder, reference), selected_mission,
            builder.backend)
