"""Spitzer/IRAC forced photometry with a Euclid VIS morphology prior.

``run_irac_photometry`` fits the VIS scene with ``euclid_phot``, builds a
spatially varying effective PRF for each IRAC mosaic, and fits the frozen
VIS models to the IRAC pixels. Each stage is cached under a named run, so a
repeated call reloads the validated result. Notebook 04 shows the workflow.
"""

from .archive import (
    PBCDLocalSet,
    PBCDProductSet,
    discover_pbcd,
    fetch_pbcd,
)
from .catalogs import (
    DAWN_PL_CATALOG_URLS,
    WISE_IRAC_PAIRS,
    compare_dawn,
    compare_wise_irac,
    select_dawn_disagreements,
    dawn_comparison_defaults,
    fetch_dawn_pl_catalog,
    match_dawn_irac,
    read_dawn_region,
)
from .background import SEPBackground, estimate_sep_background
from .data import (
    DEFAULT_COSMIC_DAWN_DIR,
    DEFAULT_SEIP_DIR,
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
from .models import (
    clone_prior_sources,
    model_footprint_mask,
    select_model_scene,
)
from .mosaic import MosaicCutout, load_mosaic_cutout
from .photometry import MosaicPhotometryResult, fit_mosaic_forced
from .pipeline import (
    IRACRun,
    PreparedChannel,
    fetch_irac_mosaic,
    fit_prepared,
    prepare_channel,
    run_irac_photometry,
)
from .prf import (
    DetectorPRFLibrary,
    ExposureGeometry,
    parse_refptg,
)
from .prfmap import (
    PRFMapFileGrid,
    PRFMapGeneratedGrid,
    PRFMapMosaicPSF,
    build_prfmap_psf,
)
from .cached import (
    cached_channel,
    cached_reference,
    reference_tile_layout,
    scene_geometry,
)
from .config import (
    DEFAULT_CATALOG_DIR,
    DEFAULT_DATA_ROOT,
    DEFAULT_MODEL_CACHE_DIR,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_REFERENCE_DIR,
    IRAC_PRF_FLUX_CORRECTION,
    IRAC_PRF_SYSTEMATIC_FRACTION,
    validate_run_name,
)
from .storage import Artifact, StoredFit, StoredWise

from . import viz

__all__ = [
    "compare_dawn",
    "compare_wise_irac",
    "WISE_IRAC_PAIRS",
    "fetch_irac_mosaic",
    "select_dawn_disagreements",
    "viz",
    "cached_channel",
    "cached_reference",
    "reference_tile_layout",
    "scene_geometry",
    "StoredFit",
    "StoredWise",
    "Artifact",
    "DEFAULT_CATALOG_DIR",
    "DEFAULT_DATA_ROOT",
    "DEFAULT_MODEL_CACHE_DIR",
    "DEFAULT_OUTPUT_ROOT",
    "DEFAULT_REFERENCE_DIR",
    "IRAC_PRF_FLUX_CORRECTION",
    "IRAC_PRF_SYSTEMATIC_FRACTION",
    "IRACRun",
    "DetectorPRFLibrary",
    "DAWN_PL_CATALOG_URLS",
    "DEFAULT_COSMIC_DAWN_DIR",
    "DEFAULT_SEIP_DIR",
    "ExposureGeometry",
    "MosaicCutout",
    "MosaicPhotometryResult",
    "MosaicInputs",
    "PBCDLocalSet",
    "PBCDProductSet",
    "PRFMapFileGrid",
    "PRFMapGeneratedGrid",
    "PRFMapMosaicPSF",
    "PreparedChannel",
    "CosmicDawnLocalSet",
    "CosmicDawnProductSet",
    "SEIPLocalSet",
    "SEIPProductSet",
    "SEPBackground",
    "build_prfmap_psf",
    "clone_prior_sources",
    "discover_pbcd",
    "dawn_comparison_defaults",
    "fetch_pbcd",
    "fetch_dawn_pl_catalog",
    "fetch_cosmic_dawn_mosaic",
    "fetch_seip_mosaic",
    "fetch_sha_mosaic",
    "fit_mosaic_forced",
    "fit_prepared",
    "load_mosaic_cutout",
    "model_footprint_mask",
    "parse_refptg",
    "parse_cosmic_dawn_info",
    "parse_seip_dcelist",
    "prepare_channel",
    "run_irac_photometry",
    "estimate_sep_background",
    "match_dawn_irac",
    "read_dawn_region",
    "select_model_scene",
    "validate_run_name",
]
