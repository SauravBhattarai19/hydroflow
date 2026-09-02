# -*- coding: utf-8 -*-
"""
hydroflow.core.runoff — rainfall → effective-runoff generation.

Modules
-------
engine : RunoffEngine — dispatches by cfg.RUNOFF_SOURCE
         ('none' | 'coefficient' | 'raster' | 'scs_cn' | 'vsa_opm').
vsa    : VsaOpmMixin — the VSA sandbox (Pradhan & Ogden 2010), Green-Ampt
         infiltration-excess and impervious shedding mechanics.
soil   : OPM soil-parameter resolution (SD_max, phi, Rawls suction table).
gpu    : RunoffEngineGPU — CuPy device-array variant (import explicitly;
         kept out of this namespace so CPU-only installs never touch CuPy).
"""

from .engine import RunoffEngine
from .soil import (
    OPM_SD_MIN,
    OPM_Q_MIN,
    resolve_sd_params,
    resolve_zone_divides,
    per_zone_sd_from_raster,
    usda_psi_m,
)

__all__ = [
    "RunoffEngine", "OPM_SD_MIN", "OPM_Q_MIN",
    "resolve_sd_params", "resolve_zone_divides", "per_zone_sd_from_raster",
    "usda_psi_m",
]
