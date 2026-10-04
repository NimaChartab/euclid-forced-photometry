"""Immutable, versioned fit products with memory-mapped image access."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS

from .source_io import read_sources, write_sources

SCHEMA_VERSION = 1


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))


def input_records(paths):
    """Local input identities; added unrelated files do not invalidate a run."""
    records = []
    for path in sorted({Path(p).resolve() for p in paths if Path(p).is_file()}):
        stat = path.stat()
        records.append({"path": str(path), "size": stat.st_size,
                        "mtime_ns": stat.st_mtime_ns})
    return records


def inputs_unchanged(records):
    for record in records:
        try:
            stat = Path(record["path"]).stat()
        except OSError:
            return False
        if (stat.st_size, stat.st_mtime_ns) != (record["size"], record["mtime_ns"]):
            return False
    return True


class ArtifactWriter:
    def __init__(self, path):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.arrays = {}

    def array(self, name, value):
        value = np.asarray(value)
        if value.dtype.hasobject:
            raise ValueError("object arrays cannot be checkpointed")
        np.save(self.path / f"{name}.npy", value, allow_pickle=False)
        self.arrays[name] = {"shape": list(value.shape), "dtype": str(value.dtype)}

    def wcs(self, wcs, shape):
        # A parent mosaic header has a different CRPIX from a trimmed cutout.
        header = wcs.to_header(relax=True)
        header["NAXIS"] = 2
        header["NAXIS1"], header["NAXIS2"] = int(shape[1]), int(shape[0])
        header.totextfile(self.path / "wcs.hdr", overwrite=True)

    def finish(self, metadata, inputs=()):
        files = {p.name: p.stat().st_size for p in self.path.iterdir()
                 if p.is_file() and p.name != "manifest.json"}
        write_json(self.path / "manifest.json", {
            "schema": SCHEMA_VERSION, "arrays": self.arrays,
            "files": files, "metadata": metadata, "inputs": input_records(inputs),
        })


class Artifact:
    """Small handle. No image plane is read until explicitly requested."""
    def __init__(self, path):
        self.path = Path(path).resolve()
        self.manifest = json.loads((self.path / "manifest.json").read_text())
        if self.manifest["schema"] != SCHEMA_VERSION:
            raise ValueError("unsupported artifact schema")
        self.metadata = self.manifest["metadata"]
        self.cache_hit = False

    def validate(self, *, check_inputs=True):
        try:
            for name, size in self.manifest["files"].items():
                if (self.path / name).stat().st_size != size:
                    return False
            for name, spec in self.manifest["arrays"].items():
                array = self.array(name)
                if list(array.shape) != spec["shape"] or str(array.dtype) != spec["dtype"]:
                    return False
                del array
        except (OSError, ValueError):
            return False
        return not check_inputs or inputs_unchanged(self.manifest["inputs"])

    def array(self, name):
        return np.load(self.path / f"{name}.npy", mmap_mode="r", allow_pickle=False)

    def has_array(self, name):
        """Return whether this artifact contains a named memory-mapped array."""
        return name in self.manifest["arrays"]

    @property
    def catalog(self):
        return Table.read(self.path / "catalog.fits")

    @property
    def sources(self):
        return read_sources(self.path / "sources.json")

    @property
    def wcs(self):
        header = fits.Header.fromtextfile(self.path / "wcs.hdr")
        wcs = WCS(header).celestial
        wcs.array_shape = (header["NAXIS2"], header["NAXIS1"])
        return wcs


class StoredCutout:
    def __init__(self, artifact):
        self.artifact = artifact

    @property
    def wcs(self):
        return self.artifact.wcs

    @property
    def shape(self):
        return tuple(self.artifact.manifest["arrays"]["data_ujy"]["shape"])

    @property
    def channel(self):
        return self.artifact.metadata["channel"]

    def __getattr__(self, name):
        if name in ("data_ujy", "uncertainty_ujy", "invvar", "fit_mask",
                    "quality_mask", "coverage"):
            return self.artifact.array(name)
        raise AttributeError(name)


class StoredFit(Artifact):
    """Catalog/images for inspection; live Tractor/PRF graphs are not retained."""
    @property
    def cutout(self):
        return StoredCutout(self)

    def __getattr__(self, name):
        if name in self.manifest["arrays"]:
            return self.array(name)
        if name in self.metadata:
            return self.metadata[name]
        if name in ("position_shift_arcsec", "position_shift_pixels", "position_at_limit"):
            return np.asarray(self.catalog[name])
        raise AttributeError(name)


class StoredWise(Artifact):
    def __getitem__(self, name):
        if name == "wcs":
            return self.wcs
        if name in self.manifest["arrays"]:
            return self.array(name)
        return self.metadata[name]


def save_fit(path, result, inputs=()):
    writer = ArtifactWriter(path)
    result.catalog.write(writer.path / "catalog.fits", overwrite=True)
    write_sources(writer.path / "sources.json", list(result.tractor.catalog))
    for name in ("data_ujy", "uncertainty_ujy", "invvar", "coverage",
                 "quality_mask", "fit_mask"):
        writer.array(name, getattr(result.cutout, name))
    for name in ("source_model_image", "background_image", "background_rms_image",
                 "model_image", "residual_image"):
        writer.array(name, getattr(result, name))
    writer.wcs(result.cutout.wcs, result.cutout.shape)
    metadata = {name: getattr(result, name) for name in (
        "channel", "chi_inflation", "chi_pool_kind", "n_chi_pool", "tractor_seconds",
        "source_model", "mission", "initial_flux_seconds", "position_fit_seconds",
        "final_flux_seconds", "position_iterations", "max_position_shift_arcsec",
        "background_mesh_arcsec", "background_global_ujy", "background_global_rms_ujy",
        "background_is_constant",
    )}
    # Convert NumPy scalar diagnostics to JSON-native values.
    metadata = {key: value.item() if isinstance(value, np.generic) else value
                for key, value in metadata.items()}
    writer.finish(metadata, inputs)
