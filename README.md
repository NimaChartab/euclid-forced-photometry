# euclid_phot: multi-band forced photometry for Euclid Q1

Forced photometry across Euclid Q1 (VIS + NISP Y/J/H) and unWISE
W1/W2, packaged as a Python engine plus reference Jupyter notebooks
for analyzing Euclid and ancillary survey data.


## Notebooks

1. **`notebooks/01_multiband_forced_photometry.ipynb`**: the full
   pipeline on a 200 arcsec field. MER catalog and cutouts, bright-star
   masking, model selection, the VIS prior fit, propagation to NISP and
   unWISE, residual diagnostics, error calibration, and the final
   multi-band catalog.
2. **`notebooks/02_model_selection.ipynb`**: the chi-squared decision
   tree in isolation, traced tier by tier on a 50 arcsec field.
3. **`notebooks/03_injection_recovery.ipynb`**: validation. Sources of
   known flux are injected into the real pixels and recovered by the
   unmodified pipeline (flux bias below a few percent, pull standard
   deviation near unity).

Runtime on the bundled data, Apple M3 Pro with eight worker threads:
~25 minutes for notebook 01, ~1 minute for 02, a few minutes for 03.
The model-selection tree dominates notebook 01; with the MER prior
instead it finishes in a few minutes.

## Install

Two dependencies are not on PyPI and are installed from GitHub in
steps 2-3 below.

```bash
git clone https://github.com/nchartab/euclid-forced-photometry
cd euclid-forced-photometry

# 1. The package and its PyPI dependencies.
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]              # [dev] adds jupyter, nbconvert, nbclient, ruff

# 2. Tractor (dstndstn/tractor). --no-build-isolation is needed because
#    tractor's setup.py imports numpy at build time; this also builds the
#    _mp_fourier C extension used for the PSF FFT.
pip install cython
pip install --no-build-isolation "git+https://github.com/dstndstn/tractor.git"

# 3. astrometry.net's Python utilities (tractor's optimizer imports
#    astrometry.util at fit time). Its setup.py defers to `make`, so pip
#    cannot install it; a source clone on the import path works since the
#    modules tractor needs are pure Python.
mkdir -p .anet
git clone --depth 1 https://github.com/dstndstn/astrometry.net.git .anet/astrometry
python -c "import site, pathlib; pathlib.Path(site.getsitepackages()[0], 'astrometry_net.pth').write_text(str(pathlib.Path('.anet').resolve()))"

# 4. Demo data.
python -m euclid_phot.examples_data   # downloads ~149 MB of demo cutouts
jupyter lab notebooks/
```

Conda users: create an environment first, then run steps 1-4 inside it.
The conda environment replaces the `venv`, so skip the two `python -m venv
.venv` / `source .venv/bin/activate` lines in step 1 and start at `pip
install -e .[dev]`.

```bash
conda create -n euclid python=3.11
conda activate euclid
pip install -e .[dev]              # step 1, without the venv lines
# ... steps 2-4 from the block above ...
```

### WISE photometry (optional)

The VIS + NISP workflow needs nothing beyond the commands above. The
unWISE W1/W2 section additionally needs the `unwise_psf` PSF model,
which is not pip-installable, plus `fitsio` (its FITS reader) and
`setuptools<81` (it imports the deprecated `pkg_resources` API):

```bash
git clone --depth 1 https://github.com/legacysurvey/unwise_psf .unwise_psf
python -c "import site, pathlib; pathlib.Path(site.getsitepackages()[0], 'unwise_psf.pth').write_text(str(pathlib.Path('.unwise_psf/py').resolve()))"
pip install fitsio "setuptools<81"
```

The `.pth` file makes the package visible to Jupyter kernels as well,
unlike an exported `PYTHONPATH`. To run without `unwise_psf`, set
`TARGET_BANDS["wise"] = ()` in notebook 01, or pass
`target_bands={"wise": ()}` to the driver.

The bundled data fetch is a ~149 MB download (~166 MB extracted;
50 arcsec and 200 arcsec cutouts of the same demo field).

## Library

The notebooks import a single package, `euclid_phot`, that exposes both
individual pipeline steps and a one-line wrapper:

```python
import euclid_phot as ep

# 1. MER catalog.
mer = ep.query_mer_catalog(269.48, 67.30, half_size_deg=200/3600/2)

# 2. Cutouts.
vis = ep.fetch_cutout("VIS", 269.48, 67.30, 200)
nisp_cutouts = {b: ep.fetch_cutout(b, 269.48, 67.30, 200) for b in ("Y", "J", "H")}

# 3. PSFs (VIS for the prior fit; one stamp per NISP band for propagation).
psf = ep.extract_catalog_psf("VIS", 269.48, 67.30, radius_arcsec=150)
psf_stamp_vis, psf_fwhm_vis = ep.psf_summary(psf)
psf_stamps_nisp = {}
for b in ("Y", "J", "H"):
    psf_data = ep.extract_catalog_psf(b, 269.48, 67.30, radius_arcsec=150)
    psf_stamps_nisp[b], _ = ep.psf_summary(psf_data)

# 4. Prior fit on VIS (positions + Sersic shape + brightness).
tim_vis = ep.build_tractor_image(vis, psf_stamp_vis)
sources = ep.build_sources_from_mer(mer, band="VIS", psf_fwhm_arcsec=psf_fwhm_vis)
tractor_vis, quality = ep.fit_forced_photometry(tim_vis, sources, band="VIS")
quality, _ = ep.fit_free_shapes(tractor_vis, tim_vis, sources, quality, band="VIS")

# 5. Propagate to NISP Y/J/H with shapes frozen at the VIS fit; only flux
#    is free. Returns {band: {"flux_ujy": ndarray, "flux_err_ujy": ndarray}}.
nisp = ep.fit_nisp_forced(
    sources, nisp_cutouts, psf_stamps_nisp, bands=("Y", "J", "H"))
flux_Y_ujy, flux_Y_err_ujy = nisp["Y"]["flux_ujy"], nisp["Y"]["flux_err_ujy"]

# 6. Propagate to unWISE W1/W2. fit_wise_forced keeps the VIS models frozen
#    (flux-only, as for NISP), fits a per-band sky offset jointly with the
#    source brightnesses, and reports the chi-MAD inflation factor k on
#    source-sparse pixels. source_models="point" collapses every source to
#    a PointSource instead (Lang et al. 2016, section 3.2).
wise_cutouts = ep.fetch_unwise_cutouts(269.48, 67.30, size_arcsec=320)
wise_results = ep.fit_wise_forced(
    sources, wise_cutouts,
    ra=269.48, dec=67.30, cutout_size_arcsec=200,
    bands=("W1", "W2"))

# 7. Cross-check W1/W2 against the Schlafly et al. 2019 unWISE catalog.
uwcat = ep.query_unwise_2019(269.48, 67.30, radius_arcsec=120)

# The same pipeline through the one-call driver:
result = ep.run_forced_photometry(
    269.48, 67.30, 200.0,
    prior={
        "band": "VIS",              # image that defines positions + shapes
        "objects": "mer",           # "mer" = MER catalog, "coords" = user_coords
        "model_selection": "tree",  # see "Source models" below
        "free_shapes": True,        # refine shapes after the flux fit
    },
    target_bands={
        "euclid": ("Y", "J", "H"),  # NISP bands to propagate to
        "wise":   ("W1", "W2"),     # also fit unWISE; () to skip
    },
    n_workers=4,
)

# Per-band flux + error arrays, assembled into a per-object catalog
# (flux, 1-sigma error, AB magnitude per band, model class, quality flag):
catalog = result.to_table()
catalog.write("photometry.ecsv", overwrite=True)

# Forced photometry at your own positions (no MER catalog required). Only
# the supplied sources are modeled, so use isolated targets or pass shapes.
result = ep.run_forced_photometry(
    269.48, 67.30, 200.0,
    prior={"objects": "coords"},
    user_coords=[(269.41, 67.30), (269.55, 67.28)],   # (ra, dec) in degrees
    target_bands={"euclid": ("Y", "J", "H"), "wise": ()})
```

**Source models.** `model_selection` controls how each source's model
class is chosen. `"prior"` (the default) asserts the catalog or user
class: MER mode builds `tractor.sersic.SersicGalaxy` with the catalog's
continuous Sersic index, clipped to [0.4, 6.0], just inside Tractor's
valid Sersic range of [0.29, 6.3].
`"tree"` lets a chi-squared ladder pick each source's class per blob;
the source list stays 1:1 with the input, and detection/ladder
thresholds can be tuned through `prior["detect"]` and
`prior["selector"]`. A literal class name (`"point"`, `"sersic"`,
`"exp"`, `"dev"`) forces that model for every source, seeded from
catalog shapes. With `free_shapes=False` only brightness is fit. On the
tree path `free_shapes` is ignored, since shapes are already fit per
blob.

**Catalog.** `result.to_table()` returns one row per source: flux,
1-sigma error, and AB magnitude per band; the model class; blend flags
(`blended` / `n_neighbors` / `nearest_arcsec`); quality flags
(`bright_star` / `near_bright_star` / `masked` / `edge` / `reliable`,
plus MER's `det_quality_flag` where present); extinction-corrected
magnitudes from the MER per-source E(B-V) (Gordon et al. 2023
coefficients); and per-band 5-sigma depths in the metadata.

**Errors.** Calibrated empirically per band. VIS/NISP formal errors
carry a PSF-scale inflation measured from point-source fits at
source-free positions (the resampled coadds have correlated pixel
noise); WISE errors carry a chi-inflation validated against Schlafly
et al. (2019). `calibrate_errors=False` skips this.

**Pixel masking.** `with_flag=True` drops the MER coadd-fatal FLG
pixels. `mask_bright_stars=True` masks the MER STARSIGNAL star
footprints (halos and diffraction spikes) so their light cannot bias
neighbors; the covered stars are themselves not measured (`nan` in the
catalog). A geometric fallback exists for data without the FLG plane.

**PSF products.** CATALOG-PSF (one stamp per MER source) or GRID-PSF
(the model on a regular ~12 arcsec grid; auto-selected for
user-supplied positions). Override with `psf_product`.

## Command line

The console script wraps the driver for single cutouts:

```bash
euclid-phot run --ra 269.48 --dec 67.30 --size 50 --out catalog.ecsv
euclid-phot fetch-data
```

## Live download mode

By default the notebooks read the bundled cutouts, so there are no
network calls after the one-time `examples_data` fetch. To re-run the
full IRSA TAP + S3 + unwise.me download path live, set
`SHOW_DOWNLOAD_PATH = True` in the first cell of the notebook. Expect
three to five minutes per notebook for network transfer and unWISE
coadd assembly. The result is byte-identical to the bundle for the demo
target.

## Multicore

`n_workers` threads the cutout fetch, the per-band NISP fits, the
per-blob tree fits, and the per-source PSF flux refit (the same kwarg
exists on `fit_nisp_forced` and `run_model_selection`). The fetch
always benefits. The fit speedup depends on whether your Tractor
build's compiled FFT releases the GIL, which varies between builds.
