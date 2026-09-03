# -*- coding: utf-8 -*-
"""
hydroflow — a distributed, physics-based hydrological + hydrodynamic model.

Implements Variable Source Area (VSA) saturation-excess runoff, Green-Ampt
infiltration-excess and impervious urban shedding (the Pradhan & Ogden 2010
"One-Parameter Model"), feeding explicit grid-based kinematic / diffusive-wave /
Muskingum–Cunge channel routing.  Optional Google Earth Engine forcing (IMERG
rainfall, SERVES deficit, SoilGrids, LULC/LCZ).

Subpackages
-----------
core   : the science — DEM preprocessing, precipitation, runoff generation
         (VSA / Green-Ampt / impervious), kinematic/diffusive-wave routing.
         Pure NumPy/SciPy/rasterio; no QGIS, no Qt.
gee    : Google Earth Engine integrations (IMERG rainfall, SERVES deficit,
         SoilGrids, LULC/LCZ).  Optional; requires earthengine-api.
utils  : shared helpers (CPU/GPU backend selection).
cli    : the ``hydroflow`` command-line interface (config-file driven runs).

Quick start
-----------
    from hydroflow import Config, run_pipeline

    cfg = Config(DEM_PATH="dem.tif", OUTPUT_DIR="results/")
    cfg.update_output_paths()
    results = run_pipeline(cfg, stages=("process_dem", "routing"))

``Config`` is the canonical configuration object; ``OpmConfig`` is an alias.
"""

__version__ = "0.1.0"

from .config import Config, OpmConfig
from .pipeline import run_pipeline, DEFAULT_STAGES

__all__ = [
    "Config",
    "OpmConfig",
    "run_pipeline",
    "DEFAULT_STAGES",
    "__version__",
]
