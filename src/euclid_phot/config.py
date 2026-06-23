"""Default constants used across the package."""
from __future__ import annotations

from pathlib import Path

# IRSA table names use British spelling ("catalogue");
# euclid_q1_mer_catalog is an unknown table on IRSA.
MER_TABLE = "euclid_q1_mer_catalogue"
MER_MORPHOLOGY_TABLE = "euclid_q1_mer_morphology"
MER_COLLECTION = "euclid_DpdMerBksMosaic"

# S3 bucket holding Q1 MER products
S3_BUCKET = "nasa-irsa-euclid-q1"

# Nominal pixel scales (arcsec / pix). Used for sanity checks and seeding only;
# the actual scale is always read from the WCS at runtime.
VIS_PIXEL_SCALE = 0.10
NISP_PIXEL_SCALE = 0.30
UNWISE_PIXEL_SCALE = 2.75

# Nominal PSF HWHM (arcsec). Used when no per-source PSF is available.
VIS_PSF_HWHM = 0.08
NISP_PSF_HWHM = 0.25
UNWISE_PSF_HWHM = 3.0

# Default search radius for IRSA SIA tile discovery. Padded so a 50″ cutout
# still finds every overlapping MER tile (each tile is ~32′ across).
MER_TILE_HALF_DEG = 32.0 / 60.0 / 2.0  # tile half-side in degrees
SIA_SEARCH_PAD_DEG = 0.05

# AB magnitude zero point used by tractor.LinearPhotoCal.
AB_MAG_ZP = 22.5

# MER mosaic FLG plane: OR-combined per-frame VIS quality bits (McCracken
# et al. 2025, Q1 VIS processing, Appendix B.1). STARSIGNAL (18) and
# OBJECTS (24) mark detected-source footprints, not defects.
MER_VIS_FLAG_BITS = {
    0: "INVALID", 1: "HOT", 2: "COLD", 3: "SAT", 4: "COSMIC", 5: "GHOST",
    7: "BAD_COLUMN", 8: "BAD_CLUSTER", 9: "CR_REGION", 12: "OVRCOL",
    15: "CHARINJ", 17: "SATXTALKGHOST", 18: "STARSIGNAL", 21: "ADCMAX",
    22: "NO_DATA", 24: "OBJECTS",
}

# Coadd-unusable bits only. Per-frame defect bits (HOT, COSMIC, ...) are OR'd
# from any input frame, but the coadd used the clean frames there; COSMIC-only
# pixels match clean sky (|data|/rms median 0.41 vs 0.42, ~16% of the field).
MER_VIS_BAD_BITS = (1 << 0) | (1 << 3) | (1 << 22)  # INVALID | SAT | NO_DATA

# STARSIGNAL: bright-star footprints (halos, diffraction spikes). Default
# bright-star pixel mask; improves chi std 0.77 -> 0.57 on the demo field
# at the cost of the masked stars' own photometry.
MER_VIS_STARSIGNAL = 1 << 18

# R_lambda = A_lambda / E(B-V). Euclid: Gordon et al. (2023) MW curve,
# A_lambda/A_V = 0.678/0.366/0.261/0.160 (Hunt et al. 2025, arXiv:2405.13499)
# times R_V = 3.1. WISE: Yuan, Liu & Xiang (2013), 0.19 (W1), 0.12 (W2).
EXTINCTION_COEFF = {
    "VIS": 3.1 * 0.678,   # 2.102
    "Y":   3.1 * 0.366,   # 1.135
    "J":   3.1 * 0.261,   # 0.809
    "H":   3.1 * 0.160,   # 0.496
    "W1":  0.19,
    "W2":  0.12,
}

# neo7 (Meisner et al. 2021) is the newest coadd release with a matched
# unwise_psf model; unwise.me serves through neo11.
WISE_COADD_VERSION = "neo7"

DEFAULT_DATA_DIR = Path("examples/data")
DEFAULT_CUTOUT_DIR = DEFAULT_DATA_DIR / "cutouts"
DEFAULT_PSF_DIR = DEFAULT_DATA_DIR / "psf"
DEFAULT_WISE_CACHE_DIR = DEFAULT_DATA_DIR / "wise"

# Pinned to the data-bundle release tag, which moves only when the bundled
# files change. Override with EUCLID_PHOT_DATA_URL.
TUTORIAL_DATA_URL = (
    "https://github.com/nchartab/euclid-forced-photometry/"
    "releases/download/v0.4.1-data/tutorial-data-v0.4.1.tar.gz"
)
TUTORIAL_DATA_SHA256: str | None = (
    "a8c4235d34423e741902abd42fadf85b9de2286ee8422278d50528b4049cd92a"
)

# Demo target used by the notebooks.
DEMO_TARGET_RA = 269.48
DEMO_TARGET_DEC = 67.30
DEMO_CUTOUT_SIZE_ARCSEC = 50.0
