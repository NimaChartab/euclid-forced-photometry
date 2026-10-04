"""Configuration shared by the standalone IRAC mosaic pipeline."""

from __future__ import annotations

from pathlib import Path
import re


REPO_ROOT = Path(__file__).resolve().parents[3]
# All downloaded inputs and restartable fit checkpoints live below one root.
# User-facing exports are kept separate under the parallel examples/output
# tree so the data directory remains a disposable cache.
DEFAULT_DATA_ROOT = REPO_ROOT / "examples" / "data" / "irac"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "examples" / "output" / "irac"
DEFAULT_PBCD_DIR = DEFAULT_DATA_ROOT / "pbcd"
DEFAULT_COSMIC_DAWN_DIR = DEFAULT_DATA_ROOT / "cosmic_dawn"
DEFAULT_CALIBRATION_DIR = DEFAULT_DATA_ROOT / "calibration"
DEFAULT_MODEL_CACHE_DIR = DEFAULT_DATA_ROOT / "model_cache"
DEFAULT_REFERENCE_DIR = DEFAULT_DATA_ROOT / "vis_prior"
DEFAULT_CATALOG_DIR = DEFAULT_DATA_ROOT / "catalogs"

_RUN_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")

UJY_PER_NMGY = 3.631
IRAC_CHANNELS = tuple(f"IRAC{i}" for i in range(1, 5))
IRAC_NATIVE_PIXEL_SCALE_ARCSEC = {
    "IRAC1": 1.221,
    "IRAC2": 1.213,
    "IRAC3": 1.222,
    "IRAC4": 1.220,
}
IRAC_PSF_FWHM_ARCSEC = {
    "IRAC1": 1.66,
    "IRAC2": 1.72,
    "IRAC3": 1.88,
    "IRAC4": 1.98,
}

# IRAC Instrument Handbook Appendix C.1 infinite-aperture corrections for
# fluxes measured from the unit-normalized core PRFs.  Channels 1 and 2 are
# measured from the PRFs; the Handbook directs channels 3 and 4 to the
# corresponding Table 4.8 corrections because PRF-based estimates are not
# available there.
IRAC_PRF_FLUX_CORRECTION = {
    "IRAC1": 0.943,
    "IRAC2": 0.929,
    "IRAC3": 0.772,
    "IRAC4": 0.737,
}

# The reconstructed-PRF validation has a 2% robust flux scatter.  Treat this
# as a multiplicative systematic and add it in quadrature to the statistical
# uncertainty reported by Tractor.
IRAC_PRF_SYSTEMATIC_FRACTION = 0.02


def validate_run_name(name: str) -> str:
    """Return a safe, user-chosen run name for cache/output directories."""
    name = str(name).strip()
    if not _RUN_NAME.fullmatch(name):
        raise ValueError(
            "run_name must be 1-80 characters, start with a letter or digit, "
            "and contain only letters, digits, '.', '_', or '-'"
        )
    return name


def normalize_channel(channel: str | int) -> str:
    """Return a canonical ``IRAC1`` ... ``IRAC4`` channel name."""
    text = str(channel).strip().upper().replace("CHANNEL", "CH")
    if text.isdigit():
        text = f"IRAC{text}"
    elif text.startswith("CH"):
        text = "IRAC" + text[2:]
    if text not in IRAC_CHANNELS:
        raise ValueError(f"channel must be one of {IRAC_CHANNELS}; got {channel!r}")
    return text
