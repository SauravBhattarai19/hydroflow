# -*- coding: utf-8 -*-
"""
hydroflow.core.routing — the flood-routing engine.

Modules
-------
terrain    : D8 tables, raster loading, slopes, topological order,
             downstream map, Strahler orders (static grid derivation).
hydraulics : per-step Manning / diffusive-wave discharge kernels and the
             volume flux limiter (backend-agnostic via the xp argument).
surface    : Manning's n, land-cover lookups, impervious fraction and
             confined-channel geometry fields.
router     : initialise_grid + run_time_loop (kinematic or diffusive wave,
             adaptive CFL time-stepping).
reporting  : hydrograph CSV, VSA partition series, mass-balance ledger.
gpu        : CuPy-vectorized variants of the terrain kernels.

The flat namespace below mirrors the historical `routing_utils` module, so
`from hydroflow.core import routing as ru` exposes the same API.
"""

from .terrain import (
    D8_MOVE,
    D8_DIAGONAL,
    load_rasters,
    compute_slope_grid,
    topological_order,
    build_downstream_map,
    compute_strahler_order,
)
from .hydraulics import (
    mannings_velocity,
    cell_discharge,
    mannings_discharge,
    diffusive_wave_discharge,
    normal_depth,
    muskingum_cunge_step,
    flux_limiter,
    build_rainfall_array,
)
from .surface import (
    resolve_mannings_n,
    resolve_lulc_field,
    resolve_impervious_fraction,
    build_channel_geometry,
)
from .router import initialise_grid, run_time_loop, main
from .reporting import save_hydrograph, write_partition_series, append_mass_balance_csv

__all__ = [
    "D8_MOVE", "D8_DIAGONAL", "load_rasters", "compute_slope_grid",
    "topological_order", "build_downstream_map", "compute_strahler_order",
    "mannings_velocity", "cell_discharge", "mannings_discharge",
    "diffusive_wave_discharge", "normal_depth", "muskingum_cunge_step",
    "flux_limiter", "build_rainfall_array",
    "resolve_mannings_n", "resolve_lulc_field", "resolve_impervious_fraction",
    "build_channel_geometry", "initialise_grid", "run_time_loop", "main",
    "save_hydrograph", "write_partition_series", "append_mass_balance_csv",
]
