"""Spitzer/IRAC forced photometry with fixed VIS source models.

``run_forced_photometry(..., target_bands={"irac": ("IRAC1", ...)})`` calls
:func:`fit_irac_forced`, which cuts each IRAC mosaic to the field, rebuilds
its spatially varying effective PRF from the detector PRFs and the
contributing exposures, and fits the VIS models to the IRAC pixels.
Notebook 03 shows the workflow.
"""

from .archive import PBCDLocalSet, PBCDProductSet, discover_pbcd, fetch_pbcd
from .background import SEPBackground, estimate_sep_background
from .catalogs import (
    DAWN_PL_CATALOG_URLS,
    WISE_IRAC_PAIRS,
    compare_dawn,
    compare_wise_irac,
    dawn_comparison_defaults,
    fetch_dawn_pl_catalog,
    match_dawn_irac,
    read_dawn_region,
    select_dawn_disagreements,
)
from .config import (
    IRAC_CHANNELS,
    IRAC_NATIVE_PIXEL_SCALE_ARCSEC,
    IRAC_PRF_FLUX_CORRECTION,
    IRAC_PRF_SYSTEMATIC_FRACTION,
    IRAC_PSF_FWHM_ARCSEC,
    normalize_channel,
)
from .data import (
    CosmicDawnLocalSet,
    CosmicDawnProductSet,
    MosaicInputs,
    SEIPLocalSet,
    SEIPProductSet,
    fetch_cosmic_dawn_mosaic,
    fetch_seip_mosaic,
    fetch_sha_mosaic,
    parse_cosmic_dawn_info,
    parse_seip_dcelist,
)
from .models import clone_prior_sources, model_footprint_mask, select_model_scene
from .mosaic import MosaicCutout, load_mosaic_cutout
from .photometry import MosaicPhotometryResult, fit_mosaic_forced
from .pipeline import (
    PreparedChannel,
    fetch_irac_mosaic,
    fit_irac_forced,
    fit_prepared,
    prepare_channel,
)
from .prf import DetectorPRFLibrary, ExposureGeometry, parse_refptg
from .prfmap import (
    PRFMapFileGrid,
    PRFMapGeneratedGrid,
    PRFMapMosaicPSF,
    build_prfmap_psf,
)

__all__ = [
    "CosmicDawnLocalSet", "CosmicDawnProductSet", "DAWN_PL_CATALOG_URLS",
    "DetectorPRFLibrary", "ExposureGeometry", "IRAC_CHANNELS",
    "IRAC_NATIVE_PIXEL_SCALE_ARCSEC", "IRAC_PRF_FLUX_CORRECTION",
    "IRAC_PRF_SYSTEMATIC_FRACTION", "IRAC_PSF_FWHM_ARCSEC", "MosaicCutout",
    "MosaicInputs", "MosaicPhotometryResult", "PBCDLocalSet", "PBCDProductSet",
    "PRFMapFileGrid", "PRFMapGeneratedGrid", "PRFMapMosaicPSF",
    "PreparedChannel", "SEIPLocalSet", "SEIPProductSet", "SEPBackground",
    "WISE_IRAC_PAIRS", "build_prfmap_psf", "clone_prior_sources",
    "compare_dawn", "compare_wise_irac", "dawn_comparison_defaults",
    "discover_pbcd", "estimate_sep_background", "fetch_cosmic_dawn_mosaic",
    "fetch_dawn_pl_catalog", "fetch_irac_mosaic", "fetch_pbcd",
    "fetch_seip_mosaic", "fetch_sha_mosaic", "fit_irac_forced",
    "fit_mosaic_forced", "fit_prepared", "load_mosaic_cutout",
    "match_dawn_irac", "model_footprint_mask", "normalize_channel",
    "parse_cosmic_dawn_info", "parse_refptg", "parse_seip_dcelist",
    "prepare_channel", "read_dawn_region", "select_dawn_disagreements",
    "select_model_scene",
]
