"""One disposable fitting process. Invoke through euclid_phot.irac.cached."""
from __future__ import annotations

import json
from pathlib import Path
import resource
import subprocess
import sys
from time import perf_counter

import numpy as np

from .cached import _digest, reference_tile_layout
from .netutils import ensure_https_ca_bundle
from .source_io import write_sources
from .storage import Artifact, ArtifactWriter, save_fit, write_json


def _raw_directory(settings):
    # The frozen downloader rounds size/coordinates in its filenames. A
    # wrapper namespace preserves exact arbitrary-size identities without
    # editing the reference implementation.
    geometry = {k: settings[k] for k in (
        "ra", "dec", "size_arcsec", "field_size_arcsec",
        "batch_size_arcsec", "tile_halo_arcsec",
    ) if k in settings}
    return Path(settings["data_dir"]) / "checkpoint-inputs" / _digest(geometry)


def _dependency_files(reference):
    return [reference.path / name for name in reference.manifest["files"]] + [
        reference.path / "manifest.json",
        *[record["path"] for record in reference.manifest["inputs"]],
    ]


def _write_reference(output, catalog, sources, quality, flux, flux_error,
                     *, metadata, inputs, images=None, wcs=None):
    writer = ArtifactWriter(output)
    catalog.write(writer.path / "catalog.fits", overwrite=True)
    write_sources(writer.path / "sources.json", sources)
    writer.array("flux_quality", quality)
    writer.array("flux_ujy", flux)
    writer.array("flux_err_ujy", flux_error)
    if images is not None:
        for name, value in images.items():
            writer.array(f"vis_{name}", value)
        writer.wcs(wcs, images["data"].shape)
    writer.finish(metadata, inputs)


def _reference_images(result):
    """Render compact VIS data/model/residual planes for notebook display."""
    import importlib
    from tractor import Tractor
    build_tractor_image = importlib.import_module(
        "euclid_phot.images"
    ).build_tractor_image

    cutout = result.cutouts["VIS"]
    # With per-source PSFs the pipeline keeps the samples in psf_data and
    # leaves psf_stamps empty; render the model the same way it was fitted.
    image = build_tractor_image(
        cutout,
        result.psf_stamps.get("VIS"),
        psf_data=None if "VIS" in result.psf_stamps else result.psf_data["VIS"],
        pixel_mask=result.prior_pixel_mask,
    )
    model = np.asarray(Tractor([image], result.sources).getModelImage(0), np.float32)
    data = np.asarray(image.getImage(), np.float32)
    return {
        "data": data,
        "model": model,
        "residual": np.asarray(data - model, np.float32),
        "invvar": np.asarray(image.getInvvar(), np.float32),
    }, cutout.wcs


def _run_frozen_reference(settings, *, mer_catalog=None, cutouts=None):
    import importlib
    run_forced_photometry = importlib.import_module(
        "euclid_phot"
    ).run_forced_photometry
    options = dict(
        prior={"band": "VIS", "objects": "mer", "model_selection": "tree",
               "free_shapes": False, "refine_positions": False},
        target_bands={"euclid": (), "wise": ()},
        data_dir=settings["data_dir"],
        force_download=settings.get("force_download", False),
        n_workers=settings["n_workers"], persource_psf=True,
        calibrate_errors=False, verbose=True,
    )
    if mer_catalog is not None:
        options["mer_catalog"] = mer_catalog
    if cutouts is not None:
        options["cutouts"] = cutouts
    return run_forced_photometry(
        settings["ra"], settings["dec"], settings["size_arcsec"],
        **options,
    )


def _parent_vis_cutout(settings):
    """Slice a previously cached larger VIS cutout for one tile."""
    science_path = settings.get("parent_science_path")
    rms_path = settings.get("parent_rms_path")
    if not science_path or not rms_path:
        return None
    import importlib
    from astropy import units as u
    from astropy.coordinates import SkyCoord
    from astropy.io import fits
    from astropy.nddata import Cutout2D
    from astropy.wcs import WCS
    Cutout = importlib.import_module("euclid_phot.cutouts").Cutout
    position = SkyCoord(settings["ra"], settings["dec"], unit="deg")
    size = float(settings["size_arcsec"]) * u.arcsec

    def read(path, *, keep_header=False):
        with fits.open(path, memmap=True) as hdul:
            header = hdul[0].header.copy()
            cutout = Cutout2D(
                hdul[0].data, position, (size, size),
                wcs=WCS(header).celestial, mode="partial", fill_value=0.0,
                copy=True,
            )
            array = np.asarray(cutout.data, dtype=np.float64)
        return (array, cutout.wcs, header) if keep_header else array

    data, wcs, header = read(science_path, keep_header=True)
    rms = read(rms_path)
    return Cutout("VIS", data, rms, wcs, header)


def _seed_tile_psf_cache(settings):
    """Filter a cached parent CATALOG-PSF file into the tile cache key."""
    parent = settings.get("parent_psf_path")
    if not parent or settings.get("force_download", False):
        return
    from astropy.coordinates import SkyCoord
    parent = Path(parent)
    radius = max(60.0, float(settings["size_arcsec"]) * 0.75)
    destination = Path(settings["data_dir"]) / "psf" / (
        f"psf_stamps_vis_{settings['ra']:.4f}_{settings['dec']:.4f}_"
        f"r{int(round(radius))}.npz"
    )
    if destination.is_file():
        return
    with np.load(parent) as values:
        ra = np.asarray(values["ra"], float)
        dec = np.asarray(values["dec"], float)
        center = SkyCoord(settings["ra"], settings["dec"], unit="deg")
        positions = SkyCoord(ra, dec, unit="deg")
        keep = positions.separation(center).arcsec <= radius
        if not np.any(keep):
            raise RuntimeError(f"parent PSF cache has no stamps for tile: {parent}")
        payload = {
            "stamps": values["stamps"][keep], "ra": values["ra"][keep],
            "dec": values["dec"][keep], "x": values["x"][keep],
            "y": values["y"][keep], "fwhm": values["fwhm"][keep],
            "stmpsize": values["stmpsize"],
        }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.stem + ".tmp.npz")
    np.savez(temporary, **payload)
    temporary.replace(destination)


def _reference_tile(settings, output, force_download):
    from astropy.table import Table
    catalog_path = Path(settings["mer_catalog_path"])
    owner_path = Path(settings["owner_ids_path"])
    catalog = Table.read(catalog_path)
    run_settings = {
        "ra": settings["ra"], "dec": settings["dec"],
        "size_arcsec": settings["size_arcsec"],
        "data_dir": settings["data_dir"], "n_workers": settings["n_workers"],
        "force_download": force_download,
    }
    run_settings.update(
        parent_science_path=settings.get("parent_science_path"),
        parent_rms_path=settings.get("parent_rms_path"),
        parent_psf_path=settings.get("parent_psf_path"),
    )
    cutout = _parent_vis_cutout(run_settings)
    _seed_tile_psf_cache(run_settings)
    result = _run_frozen_reference(
        run_settings, mer_catalog=catalog,
        cutouts={"VIS": cutout} if cutout is not None else None,
    )
    owner_ids = np.load(owner_path, allow_pickle=False)
    lookup = {int(value): index for index, value in enumerate(
        np.asarray(result.mer_cat["object_id"]))}
    missing = [int(value) for value in owner_ids if int(value) not in lookup]
    if missing:
        raise RuntimeError(
            f"reference tile lost {len(missing)} owned source(s); first IDs: "
            f"{missing[:5]}"
        )
    indices = np.asarray([lookup[int(value)] for value in owner_ids], dtype=int)
    raw_dir = Path(settings["data_dir"])
    inputs = [catalog_path, owner_path, *raw_dir.rglob("*")]
    _write_reference(
        output, result.mer_cat[indices],
        [result.sources[index] for index in indices],
        np.asarray(result.flux_quality)[indices],
        np.asarray(result.fluxes_ujy["VIS"])[indices],
        np.asarray(result.flux_errs_ujy["VIS"])[indices],
        metadata={"n_sources": len(indices), "tile": settings["tile"]},
        inputs=inputs,
    )


def _load_global_mer(settings, raw_dir, force_download):
    import importlib
    from astropy.table import Table
    halo = float(settings["tile_halo_arcsec"])
    expanded_size = float(settings["size_arcsec"]) + 2.0 * halo
    path = raw_dir / (
        f"mer_catalog_expanded_{settings['ra']:.4f}_{settings['dec']:.4f}_"
        f"{expanded_size:.3f}.fits"
    )
    if path.is_file() and not force_download:
        return Table.read(path), path
    query = importlib.import_module("euclid_phot.catalog").query_mer_catalog
    catalog = query(
        settings["ra"], settings["dec"], expanded_size / 7200.0,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp.fits")
    catalog.write(temporary, overwrite=True)
    temporary.replace(path)
    return catalog, path


def _cached_parent_reference_files(settings):
    """Find a prior full-field cache large enough to supply every tile."""
    root = Path(settings["data_dir"]) / "checkpoint-inputs"
    rounded = int(round(float(settings["size_arcsec"])))
    science_candidates = []
    pattern = (
        f"vis_science_{settings['ra']:.4f}_{settings['dec']:.4f}_*.fits"
    )
    for path in root.glob(f"*/cutouts/{pattern}"):
        try:
            size = int(path.stem.rsplit("_", 1)[-1])
        except ValueError:
            continue
        rms = path.with_name(path.name.replace("vis_science_", "vis_rms_"))
        if size >= rounded and rms.is_file():
            science_candidates.append((size, path, rms))
    if not science_candidates:
        return {}
    _, science, rms = min(science_candidates, key=lambda item: item[0])
    psf_candidates = []
    psf_pattern = f"psf_stamps_vis_{settings['ra']:.4f}_{settings['dec']:.4f}_r*.npz"
    for path in root.glob(f"*/psf/{psf_pattern}"):
        try:
            radius = int(path.stem.rsplit("_r", 1)[-1])
        except ValueError:
            continue
        psf_candidates.append((radius, path))
    output = {"parent_science_path": str(science),
              "parent_rms_path": str(rms)}
    if psf_candidates:
        output["parent_psf_path"] = str(
            max(psf_candidates, key=lambda item: item[0])[1]
        )
    return output


def _reference(settings, output, force_download, implementation=None):
    raw_dir = _raw_directory(settings)
    layout = reference_tile_layout(
        settings["size_arcsec"],
        field_size_arcsec=settings["field_size_arcsec"],
        batch_size_arcsec=settings["batch_size_arcsec"],
        halo_arcsec=settings["tile_halo_arcsec"],
    )
    if len(layout) == 1:
        run_settings = dict(settings, data_dir=raw_dir,
                            force_download=force_download)
        result = _run_frozen_reference(run_settings)
        images, wcs = _reference_images(result)
        inputs = [p for p in raw_dir.rglob("*")
                  if "wise" not in p.relative_to(raw_dir).parts]
        _write_reference(
            output, result.mer_cat, result.sources, result.flux_quality,
            result.fluxes_ujy["VIS"], result.flux_errs_ujy["VIS"],
            metadata={"n_sources": len(result.sources), "n_tiles": 1,
                      "settings": settings,
                      "vis_model_psf": "field-average VIS PSF",
                      "vis_bunit": str(
                          result.cutouts["VIS"].header.get("BUNIT", "native/pixel")
                      )},
            inputs=inputs, images=images, wcs=wcs,
        )
        return

    from astropy import units as u
    from astropy.coordinates import SkyCoord
    catalog, catalog_path = _load_global_mer(settings, raw_dir, force_download)
    center = SkyCoord(settings["ra"], settings["dec"], unit="deg")
    frame = center.skyoffset_frame()
    positions = SkyCoord(
        np.asarray(catalog["ra"], float), np.asarray(catalog["dec"], float),
        unit="deg",
    ).transform_to(frame)
    x = positions.lon.wrap_at(180.0 * u.deg).to_value(u.arcsec)
    y = positions.lat.to_value(u.arcsec)
    half = float(settings["size_arcsec"]) / 2.0
    central = (np.abs(x) <= half) & (np.abs(y) <= half)
    central_catalog = catalog[central]
    central_x = x[central]
    central_y = y[central]
    n = layout[0]["nx"]
    core = layout[0]["core_size_arcsec"]
    owner_x = np.clip(np.floor((central_x + half) / core).astype(int), 0, n - 1)
    owner_y = np.clip(np.floor((central_y + half) / core).astype(int), 0, n - 1)
    tile_artifacts = []
    implementation_code = (implementation or {}).get("code", "unknown")
    parent_files = _cached_parent_reference_files(settings)
    if parent_files:
        print(
            f"Using cached parent VIS cutout: {parent_files['parent_science_path']}",
            flush=True,
        )

    for tile in layout:
        owns = (owner_x == tile["ix"]) & (owner_y == tile["iy"])
        owner_ids = np.asarray(central_catalog["object_id"])[owns]
        if not len(owner_ids):
            continue
        fit_half = tile["fit_size_arcsec"] / 2.0
        included = ((np.abs(x - tile["x_arcsec"]) <= fit_half)
                    & (np.abs(y - tile["y_arcsec"]) <= fit_half))
        tile_catalog = catalog[included]
        tile_center = SkyCoord(
            lon=tile["x_arcsec"] * u.arcsec,
            lat=tile["y_arcsec"] * u.arcsec,
            frame=frame,
        ).icrs
        tile_identity = {
            "tile": tile, "ra": float(tile_center.ra.deg),
            "dec": float(tile_center.dec.deg),
            "owner_ids": [int(value) for value in owner_ids],
            "implementation": implementation_code,
        }
        stage = raw_dir / "reference-tiles" / _digest(tile_identity)
        artifact_path = stage / "artifact"
        try:
            artifact = Artifact(artifact_path)
            reusable = artifact.validate() and not force_download
        except (OSError, ValueError, KeyError, TypeError):
            reusable = False
        if reusable:
            print(
                f"[tile {tile['index'] + 1}/{len(layout)}] reuse "
                f"{len(owner_ids)} owned sources",
                flush=True,
            )
            tile_artifacts.append(artifact)
            continue

        stage.mkdir(parents=True, exist_ok=True)
        catalog_input = stage / "mer_catalog.fits"
        owner_input = stage / "owner_ids.npy"
        tile_catalog.write(catalog_input, overwrite=True)
        np.save(owner_input, owner_ids, allow_pickle=False)
        tile_data = stage / "inputs"
        tile_settings = {
            "ra": float(tile_center.ra.deg),
            "dec": float(tile_center.dec.deg),
            "size_arcsec": float(tile["fit_size_arcsec"]),
            "data_dir": str(tile_data),
            "n_workers": int(settings["n_workers"]),
            "mer_catalog_path": str(catalog_input),
            "owner_ids_path": str(owner_input),
            "tile": tile,
            **parent_files,
        }
        request_path = stage / "request.json"
        write_json(request_path, {
            "kind": "reference_tile", "settings": tile_settings,
            "output": str(artifact_path),
            "force_download": bool(force_download),
            "implementation": implementation or {},
        })
        print(
            f"[tile {tile['index'] + 1}/{len(layout)}] fit "
            f"{len(owner_ids)} owned / {len(tile_catalog)} modeled sources; "
            f"side={tile['fit_size_arcsec']:.1f}\"",
            flush=True,
        )
        completed = subprocess.run(
            [sys.executable, "-m", "euclid_phot.irac.cache_worker", str(request_path)],
            check=False,
        )
        if completed.returncode:
            raise RuntimeError(
                f"reference tile {tile['index'] + 1} exited "
                f"{completed.returncode}; completed tiles remain cached"
            )
        artifact = Artifact(artifact_path)
        if not artifact.validate():
            raise RuntimeError(f"reference tile is incomplete: {artifact_path}")
        tile_artifacts.append(artifact)

    source_by_id = {}
    quality_by_id = {}
    flux_by_id = {}
    error_by_id = {}
    inputs = [catalog_path]
    for artifact in tile_artifacts:
        tile_catalog = artifact.catalog
        tile_sources = artifact.sources
        quality = artifact.array("flux_quality")
        flux = artifact.array("flux_ujy")
        error = artifact.array("flux_err_ujy")
        for index, value in enumerate(tile_catalog["object_id"]):
            object_id = int(value)
            if object_id in source_by_id:
                raise RuntimeError(f"duplicate reference tile owner: {object_id}")
            source_by_id[object_id] = tile_sources[index]
            quality_by_id[object_id] = bool(quality[index])
            flux_by_id[object_id] = float(flux[index])
            error_by_id[object_id] = float(error[index])
        inputs.extend(_dependency_files(artifact))

    wanted = [int(value) for value in central_catalog["object_id"]]
    missing = [value for value in wanted if value not in source_by_id]
    if missing:
        raise RuntimeError(
            f"tiled reference merge lost {len(missing)} source(s); first IDs: "
            f"{missing[:5]}"
        )
    _write_reference(
        output, central_catalog, [source_by_id[value] for value in wanted],
        np.asarray([quality_by_id[value] for value in wanted], dtype=bool),
        np.asarray([flux_by_id[value] for value in wanted], dtype=float),
        np.asarray([error_by_id[value] for value in wanted], dtype=float),
        metadata={"n_sources": len(wanted), "n_tiles": len(tile_artifacts),
                  "tile_layout": layout, "settings": settings},
        inputs=inputs,
    )


def _wise(settings, output, force_download):
    import importlib
    wise = importlib.import_module("euclid_phot.wise")
    fetch_unwise_cutouts = wise.fetch_unwise_cutouts
    fit_wise_forced = wise.fit_wise_forced
    get_wise_psf = wise.get_wise_psf
    reference = Artifact(settings["reference_path"])
    if not reference.validate():
        raise ValueError("Reference changed before WISE worker started")
    band = settings["band"]
    raw_dir = _raw_directory(settings) / "wise"
    cutouts = fetch_unwise_cutouts(
        settings["ra"], settings["dec"], max(180.0, settings["size_arcsec"] + 120.0),
        data_dir=raw_dir, force_download=force_download,
    )
    psf = get_wise_psf(int(band[-1]), cutouts["coadd_id"],
                       ra=settings["ra"], dec=settings["dec"])
    result = fit_wise_forced(
        reference.sources, cutouts, ra=settings["ra"], dec=settings["dec"],
        cutout_size_arcsec=settings["size_arcsec"], psf_stamps={band: psf}, bands=(band,),
        catalog_cache_dir=raw_dir / "catalogs",
    )[band]
    writer = ArtifactWriter(output)
    for name in ("flux_ujy", "flux_err_ujy", "data", "invvar", "model", "residual"):
        writer.array(name, result[name])
    for name in ("flux_ujy", "flux_err_ujy"):
        writer.array("export_" + name, np.where(reference.array("flux_quality"),
                                               result[name], np.nan))
    writer.wcs(result["wcs"], result["data"].shape)
    writer.finish({name: result[name] for name in
                   ("sky", "chi_inflation", "chi_pool_kind", "n_chi_pool")},
                  [*raw_dir.rglob("*"), *_dependency_files(reference)])


def _irac(settings, output, force_download):
    from .config import DEFAULT_CALIBRATION_DIR
    from .pipeline import fit_prepared, prepare_channel
    reference = Artifact(settings["reference_path"])
    if not reference.validate():
        raise ValueError("Reference changed before IRAC worker started")
    options = settings["prepare_options"]
    prepared = prepare_channel(settings["ra"], settings["dec"], settings["channel"],
                               **options, force_download=force_download)
    result = fit_prepared(prepared, reference.catalog, reference.sources,
                          **settings["fit_options"])
    inputs = _dependency_files(reference)
    inputs.extend(value for key, value in vars(prepared.local_product).items()
                  if key.endswith("_path") and value is not None)
    inputs.append(Path(prepared.local_product.science_path).parent / "manifest.json")
    calibration = Path(options.get("calibration_dir", DEFAULT_CALIBRATION_DIR))
    # Only this channel's calibration files; adding IRAC3/4 must not
    # invalidate completed IRAC1/2 measurements.
    inputs.extend(calibration.rglob(f"{prepared.cutout.channel}_*.fits"))
    if options.get("prfmap_grid_file"):
        inputs.append(options["prfmap_grid_file"])
        # Large released grids can contain tens of thousands of stamps.  The
        # builder records the subset selected for this scene, which is both
        # sufficient for invalidation and much cheaper than recording every
        # file in the parent directory.
        inputs.extend(getattr(prepared.prf_builder, "loaded_paths", ()))
    save_fit(output, result, inputs)


def run(request):
    started = perf_counter()
    ensure_https_ca_bundle()
    print(f"Worker Python: {sys.executable}", flush=True)
    settings = request["settings"]
    output = Path(request["output"])
    kind = request["kind"]
    if kind == "reference":
        _reference(
            settings, output, request["force_download"],
            implementation=request["implementation"],
        )
    else:
        {"reference_tile": _reference_tile, "wise": _wise, "irac": _irac}[kind](
            settings, output, request["force_download"])
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    manifest["metadata"].update(
        stage_seconds=perf_counter() - started,
        peak_rss_mib=peak / (1024**2 if sys.platform == "darwin" else 1024),
        settings=settings, implementation=request["implementation"],
    )
    write_json(manifest_path, manifest)
    print(f"Saved {request['kind']} checkpoint: {output}", flush=True)


if __name__ == "__main__":
    run(json.loads(Path(sys.argv[1]).read_text()))
