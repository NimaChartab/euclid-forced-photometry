"""Restartable, process-isolated stages used by notebook 04.

Cache entries are immutable generations. Only a successfully validated worker
result replaces the current pointer; an interrupted refit leaves the previous
generation usable. Images are exposed as read-only NumPy memory maps.
"""
from __future__ import annotations

import fcntl
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time
import sys
import uuid

from .netutils import ensure_https_ca_bundle
from .storage import Artifact, StoredFit, StoredWise, write_json


_WORKER_PACKAGES = (
    "numpy", "scipy", "astropy", "sep", "tractor", "unwise_psf",
    "fitsio", "reproject", "astroquery",
)


def _worker_python():
    """Return the isolated fitting interpreter.

    Jupyter kernels often come from a broad Conda environment whose binary
    dependencies differ from this checkout. Prefer the repository virtual
    environment when present; allow an explicit override for other installs.
    """
    configured = os.environ.get("IRAC_PHOT_WORKER_PYTHON")
    if configured:
        candidate = Path(configured).expanduser()
        if not candidate.is_file():
            raise FileNotFoundError(
                f"IRAC_PHOT_WORKER_PYTHON does not exist: {candidate}"
            )
        # Keep virtual-environment symlinks intact. Resolving them to the base
        # interpreter loses the adjacent pyvenv.cfg and its site-packages.
        return str(candidate.absolute())
    repository = Path(__file__).resolve().parents[3]
    for relative in (Path(".venv/bin/python"), Path(".venv/Scripts/python.exe")):
        candidate = repository / relative
        if candidate.is_file():
            return str(candidate.absolute())
    return str(Path(sys.executable).absolute())


@lru_cache(maxsize=None)
def _runtime_versions(executable):
    """Read package versions from the interpreter that performs the fit."""
    probe = (
        "import importlib.metadata as m,json,sys;"
        f"names={_WORKER_PACKAGES!r};"
        "versions={n:(m.version(n) if n in {d.metadata['Name'] for d in "
        "m.distributions()} else 'unregistered') for n in names};"
        "versions['python']=sys.version;print(json.dumps(versions))"
    )
    result = subprocess.run(
        [executable, "-c", probe], capture_output=True, text=True, check=False,
    )
    if result.returncode:
        raise RuntimeError(
            f"Cannot inspect fitting interpreter {executable}: "
            f"{result.stderr.strip()}"
        )
    return json.loads(result.stdout)


def scene_geometry(field_size_arcsec, *, display_size_arcsec=50.0,
                   neighbor_buffer_arcsec=15.0, reference_buffer_arcsec=20.0):
    """Square science side length in arcsec, with image/prior padding.

    Any finite positive size is accepted (e.g. 341, 400, 412.5). This does
    not tile the VIS fit; its peak memory still grows with the field area.
    """
    values = [float(v) for v in (field_size_arcsec, display_size_arcsec,
                                neighbor_buffer_arcsec, reference_buffer_arcsec)]
    if not all(math.isfinite(v) and v >= 0 for v in values) or values[0] <= 0:
        raise ValueError("field size must be positive; padding must be finite and nonnegative")
    field, display, neighbor, reference = values
    scene = field + max(math.sqrt(2.0) * display, 2.0 * neighbor)
    return {"field_size_arcsec": field, "irac_size_arcsec": scene,
            "reference_size_arcsec": scene + 2.0 * max(reference, neighbor)}


def reference_tile_layout(size_arcsec, *, field_size_arcsec=None,
                          batch_size_arcsec=200.0, halo_arcsec=20.0):
    """Partition a square reference field into overlapping bounded fits.

    Each returned tile owns a non-overlapping core. Its fitted cutout includes
    ``halo_arcsec`` on every side so sources used by the merged catalog are
    never measured directly on an internal tile boundary.
    """
    size = float(size_arcsec)
    field = size if field_size_arcsec is None else float(field_size_arcsec)
    batch = float(batch_size_arcsec)
    halo = float(halo_arcsec)
    if not all(math.isfinite(v) for v in (size, field, batch, halo)):
        raise ValueError("reference tile sizes must be finite")
    if size <= 0 or field <= 0 or batch <= 0 or halo < 0:
        raise ValueError(
            "reference, field, and batch sizes must be positive, and the "
            "tile halo must be nonnegative"
        )
    n = int(math.ceil(field / batch))
    if n == 1:
        return [{
            "index": 0, "ix": 0, "iy": 0, "nx": 1, "ny": 1,
            "x_arcsec": 0.0, "y_arcsec": 0.0,
            "core_size_arcsec": size, "fit_size_arcsec": size,
        }]
    core = size / n
    tiles = []
    for iy in range(n):
        for ix in range(n):
            tiles.append({
                "index": iy * n + ix, "ix": ix, "iy": iy,
                "nx": n, "ny": n,
                "x_arcsec": -size / 2.0 + (ix + 0.5) * core,
                "y_arcsec": -size / 2.0 + (iy + 0.5) * core,
                "core_size_arcsec": core,
                "fit_size_arcsec": core + 2.0 * halo,
            })
    return tiles


def _plain(value):
    if isinstance(value, Path):
        return str(value.expanduser().resolve())
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    return value


def _digest(value):
    return hashlib.sha256(json.dumps(_plain(value), sort_keys=True,
                                    allow_nan=False).encode()).hexdigest()[:24]


def _implementation(kind):
    src = Path(__file__).resolve().parents[2]
    irac = src / "euclid_phot" / "irac"
    files = [irac / name for name in
             ("cached.py", "cache_worker.py", "storage.py", "source_io.py")]
    if kind in ("reference", "wise"):
        # The VIS prior depends on euclid_phot but not on the IRAC modules.
        files.extend(path for path in sorted((src / "euclid_phot").rglob("*.py"))
                     if irac not in path.parents)
    else:
        # Plotting helpers do not affect the fit.
        files.extend(path for path in sorted(irac.rglob("*.py"))
                     if path.name != "viz.py")
    code = hashlib.sha256()
    for path in sorted(set(files)):
        code.update(str(path.relative_to(src)).encode())
        code.update(path.read_bytes())
    worker_python = _worker_python()
    return {"code": code.hexdigest(), "worker_python": worker_python,
            "versions": _runtime_versions(worker_python)}


def _current(stage, cls):
    try:
        pointer = json.loads((stage / "current.json").read_text())
        artifact = cls(stage / pointer["generation"])
        if artifact.validate():
            artifact.cache_hit = True
            return artifact
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return None


def _run_worker(request, log):
    ensure_https_ca_bundle()
    env = os.environ.copy()
    src = str(Path(__file__).resolve().parents[2])
    env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONUNBUFFERED"] = "1"
    with log.open("w") as stream:
        process = subprocess.Popen([_worker_python(), "-m", "euclid_phot.irac.cache_worker",
                                    str(request)], stdout=stream,
                                   stderr=subprocess.STDOUT, env=env)
        try:
            code = process.wait()
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise
    if code:
        with log.open("rb") as stream:
            stream.seek(max(0, log.stat().st_size - 4000))
            tail = stream.read().decode(errors="replace")
        raise RuntimeError(f"Fitting worker exited {code}. Log: {log}\n{tail}")


def _ensure(kind, settings, *, cache_dir, force_refit=False, force_download=False):
    settings = _plain(settings)
    identity = {"kind": kind, "settings": settings, "implementation": _implementation(kind)}
    stage = Path(cache_dir).expanduser().resolve() / kind / _digest(identity)
    stage.mkdir(parents=True, exist_ok=True)
    cls = {"reference": Artifact, "wise": StoredWise, "irac": StoredFit}[kind]
    label = {"reference": "VIS models"}.get(
        kind, str(settings.get("channel", settings.get("band", kind))))
    # A kernel crash releases flock automatically; simultaneous notebooks do
    # not duplicate a fit or read an unfinished generation.
    with (stage / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        existing = _current(stage, cls)
        if existing is not None and not (force_refit or force_download):
            print(f"{label}: reloaded from cache", flush=True)
            return existing
        generation = "run-" + uuid.uuid4().hex
        destination = stage / generation
        request = stage / f"{generation}.request.json"
        log = stage / f"{generation}.log"
        write_json(request, {**identity, "output": str(destination),
                             "force_download": bool(force_download)})
        print(f"{label}: fitting (log: {log.name})", flush=True)
        start = time.monotonic()
        _run_worker(request, log)
        print(f"{label}: done in {time.monotonic() - start:.0f} s", flush=True)
        artifact = cls(destination)
        if not artifact.validate():
            raise RuntimeError(f"Incomplete or changed fit products: {destination}; see {log}")
        pending = stage / f"{generation}.pointer.json"
        write_json(pending, {"generation": generation})
        pending.replace(stage / "current.json")
        return artifact


class CachedReference:
    def __init__(self, artifact, wise_results):
        self.artifact = artifact
        self.path = artifact.path
        self.cache_hit = artifact.cache_hit
        self.wise_results = wise_results

    @property
    def mer_cat(self):
        return self.artifact.catalog

    @property
    def sources(self):
        return self.artifact.sources

    @property
    def flux_quality(self):
        return self.artifact.array("flux_quality")

    @property
    def fluxes_ujy(self):
        return {"VIS": self.artifact.array("flux_ujy"),
                **{band: result["export_flux_ujy"] for band, result in self.wise_results.items()}}

    @property
    def flux_errs_ujy(self):
        return {"VIS": self.artifact.array("flux_err_ujy"),
                **{band: result["export_flux_err_ujy"] for band, result in self.wise_results.items()}}

    @property
    def has_images(self):
        """Whether this checkpoint includes display-ready VIS image planes."""
        return self.artifact.has_array("vis_data")

    @property
    def wcs(self):
        if not self.has_images:
            raise ValueError(
                "This tiled VIS checkpoint has no merged image. Use a reference "
                "batch size at least as large as the reference field to save "
                "VIS data/model/residual planes."
            )
        return self.artifact.wcs

    def image(self, kind):
        """Return one lazy VIS plane: ``data``, ``model``, or ``residual``."""
        if kind not in {"data", "model", "residual", "invvar"}:
            raise ValueError("kind must be data, model, residual, or invvar")
        if not self.has_images:
            _ = self.wcs  # raise the detailed message above
        return self.artifact.array(f"vis_{kind}")


def cached_reference(ra, dec, size_arcsec, *, cache_dir, data_dir,
                     wise_bands=(), n_workers=1, force_refit=False,
                     force_download=False, field_size_arcsec=None,
                     batch_size_arcsec=200.0, tile_halo_arcsec=20.0):
    scene_geometry(size_arcsec)  # Reject invalid sizes before any worker starts.
    field_size = (float(size_arcsec) if field_size_arcsec is None
                  else float(field_size_arcsec))
    reference_tile_layout(
        size_arcsec, field_size_arcsec=field_size,
        batch_size_arcsec=batch_size_arcsec, halo_arcsec=tile_halo_arcsec,
    )
    settings = {"ra": float(ra), "dec": float(dec), "size_arcsec": float(size_arcsec),
                "data_dir": Path(data_dir), "n_workers": int(n_workers),
                "field_size_arcsec": field_size,
                "batch_size_arcsec": float(batch_size_arcsec),
                "tile_halo_arcsec": float(tile_halo_arcsec)}
    artifact = _ensure("reference", settings, cache_dir=cache_dir,
                       force_refit=force_refit, force_download=force_download)
    wise = {}
    for band in wise_bands:
        if band not in ("W1", "W2"):
            raise ValueError(f"unsupported WISE band: {band}")
        wise[band] = _ensure("wise", {**settings, "band": band,
                             "reference_path": artifact.path}, cache_dir=cache_dir,
                             force_refit=force_refit, force_download=force_download)
    return CachedReference(artifact, wise)


def cached_channel(ra, dec, channel, reference, *, cache_dir,
                   prepare_options, fit_options=None, force_refit=False,
                   force_download=False):
    """Load a completed channel or fit just this channel in a fresh process."""
    if not reference.artifact.validate():
        raise ValueError("Reference checkpoint is stale; call cached_reference again")
    options = dict(prepare_options)
    if "force_download" in options:
        raise ValueError("pass force_download to cached_channel, not prepare_options")
    scene_geometry(options["size_arcsec"])
    settings = {"ra": float(ra), "dec": float(dec), "channel": channel,
                "reference_path": reference.path, "prepare_options": options,
                "fit_options": fit_options or {}}
    return _ensure("irac", settings, cache_dir=cache_dir,
                   force_refit=force_refit, force_download=force_download)
