"""
gpu.py
====================
GPU-accelerated variants of the routing utility functions.

Re-exports the terrain/hydraulics functions that need no change, and overrides:
  - compute_slope_grid   → fully vectorized (no Python for-loop over cells)
  - build_downstream_map → fully vectorized (no Python for-loop over cells)
  - flux_limiter         → uses xp.minimum / xp.maximum (CuPy-compatible)

mannings_velocity and cell_discharge use only Python arithmetic operators
(**, *, /) which CuPy arrays support natively; they are re-exported as-is.

load_rasters and topological_order are CPU-only initialisation routines
(they use rasterio and run once); re-exported unchanged.

Usage
-----
    from hydroflow.core.routing import gpu as ru
    # All callers use the same API as routing_utils.
"""

import numpy as np
import cupy as cp

# Re-export CPU-safe functions unchanged
from .terrain import (
    D8_MOVE,
    D8_DIAGONAL,
    load_rasters,
    topological_order,
)
from .hydraulics import (
    mannings_velocity,
    cell_discharge,
)

# ── Vectorized D8 lookup tables ───────────────────────────────────────────────
# Direction codes in pysheds D8 convention:
#   64 128  1
#   32   *  2
#   16   8  4
_D8_CODES   = np.array([64, 128,  1,  2,  4,  8, 16, 32], dtype=np.int32)
_D8_DR      = np.array([-1,  -1,  0,  1,  1,  1,  0, -1], dtype=np.int32)
_D8_DC      = np.array([ 0,   1,  1,  1,  0, -1, -1, -1], dtype=np.int32)
_D8_IS_DIAG = np.array([ 0,   1,  0,  1,  0,  1,  0,  1], dtype=np.bool_)

# 256-entry LUT: fdir code → index into _D8_* arrays  (-1 = unknown/invalid)
_D8_CODE_TO_IDX = np.full(256, -1, dtype=np.int32)
for _i, _code in enumerate(_D8_CODES):
    _D8_CODE_TO_IDX[_code] = _i
del _i, _code   # clean up module namespace


# ── Vectorized slope grid ─────────────────────────────────────────────────────

def compute_slope_grid(dem, fdir, ws_mask, cell_size, min_slope, nodata_dem=None):
    """
    Vectorized drop-in replacement for routing_utils.compute_slope_grid.

    Eliminates the Python for-loop over active cells; uses NumPy fancy
    indexing throughout.  Produces bit-identical results to the original.

    Same signature:
        compute_slope_grid(dem, fdir, ws_mask, cell_size, min_slope,
                           nodata_dem=None) → slope (2-D float64 array)
    """
    nrows, ncols = dem.shape

    # ── Build nodata mask for the entire DEM ─────────────────────────────────
    _nodata = float(nodata_dem) if nodata_dem is not None else None
    nodata_mask = np.isnan(dem).copy()
    if _nodata is not None and not np.isnan(_nodata):
        nodata_mask |= (dem == _nodata)

    rows, cols = np.where(ws_mask)
    if len(rows) == 0:
        return np.zeros_like(dem, dtype=np.float64)

    # ── Vectorized D8 neighbor offsets ───────────────────────────────────────
    d_codes  = fdir[rows, cols].astype(np.int32)
    dir_idx  = _D8_CODE_TO_IDX[d_codes]            # -1 for invalid codes
    safe_idx = np.maximum(dir_idx, 0)              # clip -1 → 0 for safe indexing

    dr   = _D8_DR[safe_idx]
    dc   = _D8_DC[safe_idx]
    dist = np.where(
        _D8_IS_DIAG[safe_idx],
        cell_size * (2.0 ** 0.5),
        float(cell_size),
    )
    valid_dir = dir_idx >= 0

    # ── Forward slope: cell → downstream ─────────────────────────────────────
    rn = rows + dr
    cn = cols + dc

    in_bounds_fwd = (rn >= 0) & (rn < nrows) & (cn >= 0) & (cn < ncols)
    rn_s = np.clip(rn, 0, nrows - 1)
    cn_s = np.clip(cn, 0, ncols - 1)

    # Forward is usable when: direction valid, in-bounds, downstream not nodata
    use_fwd  = valid_dir & in_bounds_fwd & ~nodata_mask[rn_s, cn_s]
    dz_fwd   = dem[rows, cols] - dem[rn_s, cn_s]
    slope_fwd = np.where(use_fwd, np.maximum(dz_fwd / dist, min_slope), min_slope)

    # ── Backward slope fallback: upstream → cell (boundary handling) ──────────
    rb = rows - dr
    cb = cols - dc

    in_bounds_back = (rb >= 0) & (rb < nrows) & (cb >= 0) & (cb < ncols)
    rb_s = np.clip(rb, 0, nrows - 1)
    cb_s = np.clip(cb, 0, ncols - 1)

    # Backward usable when: forward not valid, direction valid, upstream ok
    use_back   = (~use_fwd) & valid_dir & in_bounds_back & ~nodata_mask[rb_s, cb_s]
    dz_back    = dem[rb_s, cb_s] - dem[rows, cols]
    slope_back = np.where(use_back, np.maximum(dz_back / dist, min_slope), min_slope)

    # ── Assemble output ───────────────────────────────────────────────────────
    slope_vals = np.where(use_fwd, slope_fwd,
                 np.where(use_back, slope_back, min_slope))

    slope = np.zeros_like(dem, dtype=np.float64)
    slope[rows, cols] = slope_vals
    return slope


# ── Vectorized downstream map ─────────────────────────────────────────────────

def build_downstream_map(sorted_rows, sorted_cols, fdir, ws_mask, nrows, ncols):
    """
    Vectorized drop-in replacement for routing_utils.build_downstream_map.

    Same signature:
        build_downstream_map(sorted_rows, sorted_cols, fdir, ws_mask,
                             nrows, ncols) → ds_idx (1-D int64 array)
    """
    n_cells = len(sorted_rows)

    # Flat-index → position-in-sorted-list lookup
    flat_to_pos = np.full(nrows * ncols, -1, dtype=np.int64)
    flat_ids    = sorted_rows * ncols + sorted_cols
    flat_to_pos[flat_ids] = np.arange(n_cells, dtype=np.int64)

    # ── Vectorized D8 downstream neighbour ───────────────────────────────────
    d_codes  = fdir[sorted_rows, sorted_cols].astype(np.int32)
    dir_idx  = _D8_CODE_TO_IDX[d_codes]
    safe_idx = np.maximum(dir_idx, 0)

    dr = _D8_DR[safe_idx]
    dc = _D8_DC[safe_idx]
    rn = sorted_rows + dr
    cn = sorted_cols + dc

    in_bounds = (rn >= 0) & (rn < nrows) & (cn >= 0) & (cn < ncols)
    rn_s = np.clip(rn, 0, nrows - 1)
    cn_s = np.clip(cn, 0, ncols - 1)

    # Check ws_mask only for in-bounds neighbours (avoids OOB index)
    fwd_ok = (dir_idx >= 0) & in_bounds
    in_ws  = np.zeros(n_cells, dtype=bool)
    in_ws[fwd_ok] = ws_mask[rn_s[fwd_ok], cn_s[fwd_ok]]

    # Flat index of downstream cell; -1 where there is none
    ds_flat = np.where(fwd_ok & in_ws,
                       (rn_s * ncols + cn_s).astype(np.int64),
                       np.int64(-1))

    # Map flat index → sorted-list position
    ds_idx = np.where(
        ds_flat >= 0,
        flat_to_pos[np.maximum(ds_flat, np.int64(0))],
        np.int64(-1),
    )
    return ds_idx


# ── GPU-aware flux limiter ─────────────────────────────────────────────────────

def flux_limiter(Q_out, volume, dt, xp=cp):
    """
    Volume-conservative CFL limiter.

    Q_out_limited = min(Q_out, max(volume, 0) / dt)

    xp defaults to cupy so this module always uses GPU ops when imported.
    Pass xp=numpy to run on CPU (e.g., for unit tests).
    """
    return xp.minimum(Q_out, xp.maximum(volume, 0.0) / dt)
