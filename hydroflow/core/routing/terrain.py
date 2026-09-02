# -*- coding: utf-8 -*-
"""
terrain.py — static grid derivation from the watershed rasters.

D8 direction tables, raster loading, slope grid, topological (upstream→
downstream) cell ordering, the downstream-index map and Strahler stream
orders.  Pure NumPy; the GPU-vectorized variants live in routing.gpu.
"""

import numpy as np
import rasterio



# ---------------------------------------------------------------------------
# D8 directional look-up (pysheds convention)
# Encoding:  64 128  1
#            32   *  2
#            16   8  4
# ---------------------------------------------------------------------------
# Maps encoded value -> (row_delta, col_delta)
D8_MOVE = {
    64:  (-1,  0),   # N
    128: (-1,  1),   # NE
    1:   ( 0,  1),   # E
    2:   ( 1,  1),   # SE
    4:   ( 1,  0),   # S
    8:   ( 1, -1),   # SW
    16:  ( 0, -1),   # W
    32:  (-1, -1),   # NW
}

# Diagonal directions (distance = cell_size * sqrt(2))
D8_DIAGONAL = {128, 2, 8, 32}


# ---------------------------------------------------------------------------
# 1.  Raster loading
# ---------------------------------------------------------------------------

def load_rasters(cfg):
    """
    Load all four input rasters and return numpy arrays plus spatial metadata.

    Parameters
    ----------
    cfg : module
        The imported config module.

    Returns
    -------
    dem        : 2-D float64 array  – clipped elevation (m)
    fdir       : 2-D int array      – D8 flow direction
    faccum     : 2-D float array    – clipped flow accumulation
    ws_mask    : 2-D bool array     – True where cell is inside the watershed
    transform  : affine.Affine      – rasterio transform of the rasters
    nodata_dem : float              – nodata value from the DEM raster
    """
    with rasterio.open(cfg.ROUTING_DEM_PATH) as src:
        dem        = src.read(1).astype(np.float64)
        nodata_dem = src.nodata
        transform  = src.transform
        # Cell size: use config override if set, otherwise derive from transform.
        # transform.a = pixel width (x), transform.e = pixel height (negative).
        # For square pixels in a projected CRS both magnitudes are equal.
        if getattr(cfg, 'CELL_SIZE', None) is not None:
            cell_size = float(cfg.CELL_SIZE)
        else:
            cell_size = float(abs(transform.a))

    with rasterio.open(cfg.ROUTING_FLOW_DIR_PATH) as src:
        fdir = src.read(1)

    with rasterio.open(cfg.ROUTING_FLOW_ACCUM_PATH) as src:
        faccum     = src.read(1).astype(np.float64)
        nodata_fa  = src.nodata

    with rasterio.open(cfg.ROUTING_WATERSHED_MASK_PATH) as src:
        ws_raw = src.read(1)

    # Active cells: inside watershed AND dem has valid data
    ws_mask = (ws_raw > 0)
    if nodata_dem is not None:
        ws_mask &= (dem != nodata_dem)
    if nodata_fa is not None:
        ws_mask &= (faccum != nodata_fa)

    print(f"  Rasters loaded  |  shape={dem.shape}  |  active cells={ws_mask.sum():,}"
          f"  |  cell_size={cell_size:.2f} m")
    return dem, fdir, faccum, ws_mask, transform, nodata_dem, cell_size


# ---------------------------------------------------------------------------
# 2.  Slope calculation
# ---------------------------------------------------------------------------

def compute_slope_grid(dem, fdir, ws_mask, cell_size, min_slope, nodata_dem=None):
    """
    For every active cell compute slope along its D8 flow direction.

        S0 = max( (elev_current - elev_downstream) / distance , min_slope )

    Diagonal neighbours use distance = cell_size * sqrt(2).

    Boundary handling
    -----------------
    Cells whose downstream neighbour is out-of-bounds *or* carries a nodata
    value are treated as watershed-boundary cells.  Their slope is estimated
    from the *backward* gradient (upstream cell → current cell), which is the
    best available proxy for the local channel slope.  Two common failure modes
    are guarded against:

    1. Out-of-bounds downstream (raster edge) — handled explicitly.
    2. In-bounds but nodata downstream — the value may be NaN or the DEM's
       sentinel (e.g. -9999, -32768).  Python's built-in max() silently returns
       min_slope when the first argument is NaN (max(nan, x) == x), so this
       case must be detected *before* attempting arithmetic.

    Parameters
    ----------
    dem       : 2-D float64 array
    fdir      : 2-D int array  (D8 encoding)
    ws_mask   : 2-D bool array
    cell_size : float  [m]
    min_slope : float  [m/m]  – floor applied everywhere
    nodata_dem: float or None – nodata sentinel in *dem*

    Returns
    -------
    slope : 2-D float64 array  (same shape as dem, zero for inactive cells)
    """
    nrows, ncols = dem.shape
    slope = np.zeros_like(dem, dtype=np.float64)

    # Pre-compute nodata sentinel as float64 for reliable comparison
    _nodata = float(nodata_dem) if nodata_dem is not None else None

    def _is_nodata(val):
        """True if val is NaN or equals the DEM nodata sentinel."""
        if np.isnan(val):
            return True
        if _nodata is not None and not np.isnan(_nodata) and val == _nodata:
            return True
        return False

    rows, cols = np.where(ws_mask)

    for r, c in zip(rows, cols):
        d = int(fdir[r, c])
        if d not in D8_MOVE:
            slope[r, c] = min_slope
            continue

        dr, dc = D8_MOVE[d]
        rn, cn = r + dr, c + dc
        dist   = cell_size * (2 ** 0.5 if d in D8_DIAGONAL else 1.0)

        # ── Case 1: valid downstream cell (in-bounds, not nodata) ───────────
        if (0 <= rn < nrows and 0 <= cn < ncols and not _is_nodata(dem[rn, cn])):
            dz = dem[r, c] - dem[rn, cn]
            slope[r, c] = max(dz / dist, min_slope)
            continue

        # ── Case 2: boundary cell (downstream out-of-bounds or nodata) ──────
        # Use the backward gradient (upstream cell → this cell) as proxy for
        # the local channel slope.
        rb, cb = r - dr, c - dc
        if (0 <= rb < nrows and 0 <= cb < ncols and not _is_nodata(dem[rb, cb])):
            dz_back = dem[rb, cb] - dem[r, c]
            slope[r, c] = max(dz_back / dist, min_slope)
            continue

        slope[r, c] = min_slope

    return slope


# ---------------------------------------------------------------------------
# 3.  Topological ordering
# ---------------------------------------------------------------------------

def topological_order(faccum, fdir, ws_mask):
    """
    Return the (row, col) indices of all active watershed cells sorted from
    upstream → downstream (ascending flow accumulation).

    Using flow accumulation as a proxy for topological rank is valid because
    a downstream cell always has a higher accumulation than any of its upstream
    contributors.

    Returns
    -------
    sorted_rows : 1-D int array
    sorted_cols : 1-D int array
    outlet_rc   : (int, int) – row/col of the outlet (highest accumulation)
    """
    rows, cols = np.where(ws_mask)
    accum_vals = faccum[rows, cols]

    # Sort ascending: upstream first
    order       = np.argsort(accum_vals)
    sorted_rows = rows[order]
    sorted_cols = cols[order]

    # Outlet = last cell (maximum accumulation)
    outlet_rc = (int(sorted_rows[-1]), int(sorted_cols[-1]))
    print(f"  Outlet cell     |  row={outlet_rc[0]}  col={outlet_rc[1]}"
          f"  accum={faccum[outlet_rc]:.0f}")

    return sorted_rows, sorted_cols, outlet_rc


# ---------------------------------------------------------------------------
# 4.  Downstream neighbour lookup (vectorised)
# ---------------------------------------------------------------------------

def build_downstream_map(sorted_rows, sorted_cols, fdir, ws_mask, nrows, ncols):
    """
    For each active cell (indexed in topological order) find the flat index
    of its downstream neighbour (or -1 if the neighbour is out-of-bounds or
    outside the watershed).

    Returns
    -------
    ds_idx : 1-D int array  (same length as sorted_rows)
        Flat index into sorted_rows/sorted_cols of the downstream cell,
        or -1 for the outlet / cells that drain off-mask.
    """
    n_cells = len(sorted_rows)

    # Build a (nrows*ncols) -> position-in-sorted-list lookup
    flat_to_pos = np.full(nrows * ncols, -1, dtype=np.int64)
    flat_ids    = sorted_rows * ncols + sorted_cols
    flat_to_pos[flat_ids] = np.arange(n_cells, dtype=np.int64)

    ds_idx = np.full(n_cells, -1, dtype=np.int64)

    for i, (r, c) in enumerate(zip(sorted_rows, sorted_cols)):
        d = int(fdir[r, c])
        if d not in D8_MOVE:
            continue
        dr, dc = D8_MOVE[d]
        rn, cn = r + dr, c + dc
        if 0 <= rn < nrows and 0 <= cn < ncols and ws_mask[rn, cn]:
            ds_flat     = rn * ncols + cn
            ds_idx[i]   = flat_to_pos[ds_flat]

    return ds_idx


# ---------------------------------------------------------------------------
# 9.  Strahler stream order
# ---------------------------------------------------------------------------

def compute_strahler_order(ds_idx, n_cells):
    """
    Compute Strahler stream order for each cell.

    Cells must be in topological order (upstream first).  Single O(n) pass:
    headwater cells (no upstream) get order 1; when two streams of the same
    order merge, the result is order + 1; otherwise the higher order continues.

    Parameters
    ----------
    ds_idx  : (n_cells,) int array — downstream neighbour index (-1 = outlet)
    n_cells : int

    Returns
    -------
    order : (n_cells,) int array — Strahler order per cell
    """
    order     = np.ones(n_cells, dtype=np.int32)
    max_order = np.zeros(n_cells, dtype=np.int32)
    max_count = np.zeros(n_cells, dtype=np.int32)

    for i in range(n_cells):
        if max_order[i] == 0:
            order[i] = 1
        elif max_count[i] >= 2:
            order[i] = max_order[i] + 1
        else:
            order[i] = max_order[i]

        ds = int(ds_idx[i])
        if ds >= 0:
            if order[i] > max_order[ds]:
                max_order[ds] = order[i]
                max_count[ds] = 1
            elif order[i] == max_order[ds]:
                max_count[ds] += 1

    return order
