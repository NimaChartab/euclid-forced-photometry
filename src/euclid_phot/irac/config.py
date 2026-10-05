"""IRAC constants and default directories."""

from __future__ import annotations

from pathlib import Path


# Same convention as euclid_phot.config: relative to the working directory.
# run_forced_photometry passes data_dir / "irac" explicitly.
DEFAULT_DATA_ROOT = Path("examples/data") / "irac"
DEFAULT_PBCD_DIR = DEFAULT_DATA_ROOT / "pbcd"
DEFAULT_COSMIC_DAWN_DIR = DEFAULT_DATA_ROOT / "cosmic_dawn"
DEFAULT_CALIBRATION_DIR = DEFAULT_DATA_ROOT / "calibration"

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
