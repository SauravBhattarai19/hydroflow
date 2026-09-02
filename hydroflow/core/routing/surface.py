# -*- coding: utf-8 -*-
"""
surface.py — spatial parameter fields on the routing grid.

Manning's n (scalar / LULC / LCZ / raster, with per-Strahler-order channel
override), land-cover lookups, impervious fraction, and the confined-channel
geometry (mask, width, storage area).
"""

import numpy as np

from .terrain import compute_strahler_order
from ..io_utils import align_raster_to_dem




# ---------------------------------------------------------------------------
# 8.  Spatially variable Manning's n
# ---------------------------------------------------------------------------

def resolve_mannings_n(cfg, grid_data):
    """
    Build a per-cell Manning's n array from config.

    Supports three sources (``MANNINGS_N_SOURCE``):
      scalar  – uniform value from ``MANNINGS_N``
      lulc    – LULC class codes remapped via ``LULC_LOOKUP_CSV``
      raster  – pre-computed Manning's n GeoTIFF

    In all modes, cells whose flow accumulation exceeds a threshold
    are overridden with ``MANNINGS_N_CHANNEL`` (if not None).

    Returns 1-D float64 array of shape ``(n_cells,)``.
    """
    import os
    import pandas as pd

    source     = getattr(cfg, 'MANNINGS_N_SOURCE', 'scalar').lower()
    n_fallback = float(cfg.MANNINGS_N)
    s_rows     = grid_data['s_rows']
    s_cols     = grid_data['s_cols']
    n_cells    = len(s_rows)

    # ── Source: scalar ────────────────────────────────────────────────────
    if source == 'scalar':
        n_1d = np.full(n_cells, n_fallback, dtype=np.float64)

    # ── Source: pre-computed raster ───────────────────────────────────────
    elif source == 'raster':
        path = getattr(cfg, 'MANNINGS_N_RASTER_PATH', None)
        if path is None or not os.path.isfile(path):
            raise FileNotFoundError(
                f"MANNINGS_N_SOURCE='raster' but MANNINGS_N_RASTER_PATH "
                f"not found: {path}"
            )
        dem_path = cfg.ROUTING_DEM_PATH
        n_2d = align_raster_to_dem(path, dem_path, resampling='bilinear')
        n_1d = n_2d[s_rows, s_cols].astype(np.float64)
        bad = (n_1d <= 0) | ~np.isfinite(n_1d)
        if bad.any():
            n_1d[bad] = n_fallback

    # ── Source: LULC class codes → lookup CSV ────────────────────────────
    elif source == 'lulc':
        lulc_path = getattr(cfg, 'MANNINGS_N_LULC_PATH', 'gee')
        dem_path  = cfg.ROUTING_DEM_PATH

        if str(lulc_path).lower() == 'gee':
            try:
                from ...gee.serves_gee import download_lulc_raster
            except ImportError:
                print("  [WARN] earthengine-api not installed; "
                      f"using scalar n={n_fallback}")
                return np.full(n_cells, n_fallback, dtype=np.float64)

            output_dir = getattr(cfg, 'OUTPUT_DIR', 'output/')
            cached = os.path.join(output_dir, 'lulc_mannings.tif')
            result = download_lulc_raster(
                dem_path=dem_path,
                watershed_geojson_path=getattr(
                    cfg, 'OPM_WATERSHED_GEOJSON', 'output/watershed.geojson'),
                output_path=cached,
                project=getattr(cfg, 'GEE_PROJECT', None),
            )
            if result is None:
                print("  [WARN] GEE LULC download failed; "
                      f"using scalar n={n_fallback}")
                return np.full(n_cells, n_fallback, dtype=np.float64)
            lulc_2d = align_raster_to_dem(cached, dem_path,
                                          resampling='nearest')
        else:
            if not os.path.isfile(lulc_path):
                raise FileNotFoundError(
                    f"MANNINGS_N_LULC_PATH not found: {lulc_path}"
                )
            lulc_2d = align_raster_to_dem(lulc_path, dem_path,
                                          resampling='nearest')

        csv_path = getattr(cfg, 'LULC_LOOKUP_CSV', 'lulc_lookup.csv')
        lut = pd.read_csv(csv_path)
        code_to_n = dict(zip(lut['class_code'].astype(int),
                             lut['mannings_n'].astype(float)))
        lulc_1d = lulc_2d[s_rows, s_cols]
        n_1d = np.full(n_cells, n_fallback, dtype=np.float64)
        for code, nval in code_to_n.items():
            n_1d[lulc_1d == code] = nval

    # ── Source: WUDAPT Local Climate Zones ───────────────────────────────
    elif source == 'lcz':
        try:
            from ...gee.serves_gee import download_lcz_raster
        except ImportError:
            print("  [WARN] earthengine-api not installed; "
                  f"using scalar n={n_fallback}")
            return np.full(n_cells, n_fallback, dtype=np.float64)

        output_dir = getattr(cfg, 'OUTPUT_DIR', 'output/')
        cached = os.path.join(output_dir, 'lulc_mannings_lcz.tif')
        dem_path = cfg.ROUTING_DEM_PATH
        result = download_lcz_raster(
            dem_path=dem_path,
            watershed_geojson_path=getattr(
                cfg, 'OPM_WATERSHED_GEOJSON', 'output/watershed.geojson'),
            output_path=cached,
            project=getattr(cfg, 'GEE_PROJECT', None),
        )
        if result is None:
            print("  [WARN] GEE LCZ download failed; "
                  f"using scalar n={n_fallback}")
            return np.full(n_cells, n_fallback, dtype=np.float64)

        lcz_2d = align_raster_to_dem(cached, dem_path, resampling='nearest')
        csv_path = getattr(cfg, 'LCZ_LOOKUP_CSV', 'lcz_lookup.csv')
        lut = pd.read_csv(csv_path)
        code_to_n = dict(zip(lut['class_code'].astype(int),
                             lut['mannings_n'].astype(float)))
        lulc_1d = lcz_2d[s_rows, s_cols]
        n_1d = np.full(n_cells, n_fallback, dtype=np.float64)
        for code, nval in code_to_n.items():
            n_1d[lulc_1d == code] = nval

    else:
        raise ValueError(f"Unknown MANNINGS_N_SOURCE: '{source}'")

    # ── Channel override (all modes) ─────────────────────────────────────
    n_channel_cfg = getattr(cfg, 'MANNINGS_N_CHANNEL', None)
    if n_channel_cfg is not None:
        faccum_1d = grid_data['faccum_1d']
        fa = faccum_1d.get() if hasattr(faccum_1d, 'get') else np.asarray(
            faccum_1d)
        threshold = getattr(cfg, 'CHANNEL_FACCUM_THRESHOLD', None)
        if threshold is None:
            threshold = max(1, n_cells // 100)
        channel_mask = fa > threshold

        if isinstance(n_channel_cfg, dict):
            ds_idx = grid_data['ds_idx']
            ds_np = ds_idx.get() if hasattr(ds_idx, 'get') else np.asarray(
                ds_idx)
            strahler = compute_strahler_order(ds_np, n_cells)
            max_order = max(n_channel_cfg.keys())
            for ci in np.where(channel_mask)[0]:
                so = min(int(strahler[ci]), max_order)
                n_1d[ci] = n_channel_cfg.get(so, n_channel_cfg[max_order])
            order_dist = {o: int((strahler[channel_mask] == o).sum())
                          for o in sorted(set(strahler[channel_mask]))}
            print(f"  Manning's n   |  channel cells: "
                  f"{int(channel_mask.sum()):,} / {n_cells:,}  "
                  f"(threshold={threshold})")
            print(f"  Manning's n   |  Strahler order distribution: "
                  f"{order_dist}")
        else:
            n_1d[channel_mask] = float(n_channel_cfg)
            print(f"  Manning's n   |  channel cells: "
                  f"{int(channel_mask.sum()):,} / {n_cells:,}  "
                  f"(threshold={threshold})")

    print(f"  Manning's n   |  source={source}"
          f"  range=[{n_1d.min():.4f}, {n_1d.max():.4f}]"
          f"  mean={n_1d.mean():.4f}")
    return n_1d


# ---------------------------------------------------------------------------
# 8b. Per-cell land-cover field lookup (impervious fraction, root depth, …)
# ---------------------------------------------------------------------------

def _lulc_class_1d(cfg, grid_data, source):
    """
    Per-cell land-cover class codes from the cached LCZ/LULC raster.

    Reuses the SAME cache file as ``resolve_mannings_n``
    (``lulc_mannings_lcz.tif`` / ``lulc_mannings.tif``) so no extra GEE download
    happens when Manning's n already pulled the layer.

    Returns
    -------
    (class_1d, lookup_csv_path) : (n_cells,) int-ish array + CSV path,
        or (None, None) when the raster is unavailable.
    """
    import os

    s_rows = grid_data['s_rows']
    s_cols = grid_data['s_cols']
    dem_path   = cfg.ROUTING_DEM_PATH
    output_dir = getattr(cfg, 'OUTPUT_DIR', 'output/')
    geojson    = getattr(cfg, 'OPM_WATERSHED_GEOJSON', 'output/watershed.geojson')
    project    = getattr(cfg, 'GEE_PROJECT', None)

    if source == 'lcz':
        try:
            from ...gee.serves_gee import download_lcz_raster
        except ImportError:
            return None, None
        cached = os.path.join(output_dir, 'lulc_mannings_lcz.tif')
        result = download_lcz_raster(dem_path=dem_path,
                                     watershed_geojson_path=geojson,
                                     output_path=cached, project=project)
        csv_path = getattr(cfg, 'LCZ_LOOKUP_CSV', 'lcz_lookup.csv')
    else:
        try:
            from ...gee.serves_gee import download_lulc_raster
        except ImportError:
            return None, None
        cached = os.path.join(output_dir, 'lulc_mannings.tif')
        result = download_lulc_raster(dem_path=dem_path,
                                      watershed_geojson_path=geojson,
                                      output_path=cached, project=project)
        csv_path = getattr(cfg, 'LULC_LOOKUP_CSV', 'lulc_lookup.csv')

    if result is None:
        return None, None

    arr2d = align_raster_to_dem(cached, dem_path, resampling='nearest')
    _to_np = lambda a: a.get() if hasattr(a, 'get') else np.asarray(a)
    return arr2d[_to_np(s_rows), _to_np(s_cols)], csv_path


def resolve_lulc_field(cfg, grid_data, column, default, source):
    """
    Build a per-cell field by remapping land-cover class codes through *column*
    of the LCZ/LULC lookup CSV.  Used for impervious fraction and root-zone
    depth.  Cells with an unmapped class (or when the raster/column is missing)
    get *default*.  Returns (n_cells,) float64.
    """
    import pandas as pd

    n_cells = len(grid_data['s_rows'])
    class_1d, csv_path = _lulc_class_1d(cfg, grid_data, source)
    if class_1d is None:
        print(f"  [WARN] {source} raster unavailable; "
              f"'{column}' → {default} everywhere")
        return np.full(n_cells, float(default), dtype=np.float64)

    lut = pd.read_csv(csv_path)
    if column not in lut.columns:
        print(f"  [WARN] column '{column}' missing from {csv_path}; "
              f"using {default} everywhere")
        return np.full(n_cells, float(default), dtype=np.float64)

    code_to_v = dict(zip(lut['class_code'].astype(int),
                         lut[column].astype(float)))
    out = np.full(n_cells, float(default), dtype=np.float64)
    for code, v in code_to_v.items():
        out[class_1d == code] = v
    return out


def resolve_impervious_fraction(cfg, grid_data):
    """
    Per-cell impervious fraction Imp ∈ [0,1] from ``IMPERVIOUS_SOURCE``.

      'lcz' / 'lulc' → impervious_fraction column of the matching lookup CSV
      'raster'       → continuous GeoTIFF at IMPERVIOUS_RASTER_PATH (bilinear)
      'none'         → zeros

    Returns (n_cells,) float64, clipped to [0,1].
    """
    import os

    source  = getattr(cfg, 'IMPERVIOUS_SOURCE', 'none').lower()
    n_cells = len(grid_data['s_rows'])
    s_rows  = grid_data['s_rows']
    s_cols  = grid_data['s_cols']

    if source == 'none':
        return np.zeros(n_cells, dtype=np.float64)

    if source == 'raster':
        path = getattr(cfg, 'IMPERVIOUS_RASTER_PATH', None)
        if not path or not os.path.isfile(path):
            raise FileNotFoundError(
                f"IMPERVIOUS_SOURCE='raster' but IMPERVIOUS_RASTER_PATH "
                f"not found: {path}"
            )
        arr2d = align_raster_to_dem(path, cfg.ROUTING_DEM_PATH,
                                    resampling='bilinear')
        _to_np = lambda a: a.get() if hasattr(a, 'get') else np.asarray(a)
        imp = arr2d[_to_np(s_rows), _to_np(s_cols)].astype(np.float64)
        imp[~np.isfinite(imp)] = 0.0
    elif source in ('lcz', 'lulc'):
        imp = resolve_lulc_field(cfg, grid_data, 'impervious_fraction',
                                 0.0, source)
    else:
        raise ValueError(f"Unknown IMPERVIOUS_SOURCE: '{source}'")

    imp = np.clip(imp, 0.0, 1.0)
    print(f"  Impervious    |  source={source}"
          f"  range=[{imp.min():.2f}, {imp.max():.2f}]"
          f"  mean={imp.mean():.2f}  (>0: {int((imp > 0).sum()):,}/{n_cells:,} cells)")
    return imp


# ---------------------------------------------------------------------------
# 10.  Channel (river) cross-section geometry
# ---------------------------------------------------------------------------

def build_channel_geometry(cfg, grid_data):
    """
    Per-cell channel geometry for ``CHANNEL_ROUTING`` (Workstream 1).

    Returns three NumPy arrays of shape ``(n_cells,)`` in topological order:

      chan_mask_1d  : bool   – True on channel cells (faccum > threshold), the SAME
                               cells the Manning's-n channel override uses.
      width_1d      : float  – flow width [m]: ``cell_size`` on overland cells,
                               a Strahler-order width (``CHANNEL_WIDTH_BY_ORDER``)
                               on channel cells.
      store_area_1d : float  – depth-from-volume denominator [m²]: ``cell_area`` on
                               overland cells (depth = V/cell_area, unchanged), and
                               ``width · flow-length`` on channel cells so depth is
                               the channel-reach depth V/(B·L).

    With ``CHANNEL_ROUTING`` False (or no channel cells / empty width table) every
    value reduces to the wide-sheet defaults (width = cell_size, store_area =
    cell_area), so routing is bit-for-bit unchanged.

    Reuses the channel threshold convention from ``resolve_mannings_n`` and
    ``compute_strahler_order`` for the network order — no new network analysis.
    """
    n_cells   = int(grid_data['n_cells'])
    cell_size = float(grid_data['cell_size'])
    cell_area = float(grid_data['cell_area'])
    dist_1d   = np.asarray(grid_data['dist_1d'], dtype=np.float64)

    width_1d      = np.full(n_cells, cell_size, dtype=np.float64)
    store_area_1d = np.full(n_cells, cell_area, dtype=np.float64)
    chan_mask_1d  = np.zeros(n_cells, dtype=bool)

    if not getattr(cfg, 'CHANNEL_ROUTING', False):
        return chan_mask_1d, width_1d, store_area_1d

    # Channel mask: same faccum threshold as the Manning's-n channel override.
    fa = grid_data['faccum_1d']
    fa = fa.get() if hasattr(fa, 'get') else np.asarray(fa)
    threshold = getattr(cfg, 'CHANNEL_FACCUM_THRESHOLD', None)
    if threshold is None:
        threshold = max(1, n_cells // 100)
    chan_mask_1d = fa > threshold

    width_by_order = getattr(cfg, 'CHANNEL_WIDTH_BY_ORDER', None)
    if not width_by_order:
        print("  [WARN] CHANNEL_ROUTING on but CHANNEL_WIDTH_BY_ORDER empty; "
              "channel width defaults to cell_size (no confinement).")
    else:
        ds_idx = grid_data['ds_idx']
        ds_np  = ds_idx.get() if hasattr(ds_idx, 'get') else np.asarray(ds_idx)
        order  = compute_strahler_order(ds_np, n_cells)
        max_o  = max(int(o) for o in width_by_order.keys())
        # Strahler-order → width LUT (orders above max reuse max; unspecified
        # intermediate orders keep cell_size = no confinement).
        width_lut = np.full(max_o + 1, cell_size, dtype=np.float64)
        for o, w in width_by_order.items():
            if 0 <= int(o) <= max_o:
                width_lut[int(o)] = float(w)
        order_clip = np.minimum(order, max_o).astype(np.intp)
        width_1d   = np.where(chan_mask_1d, width_lut[order_clip], width_1d)

    # Channel storage footprint = width × channel length through the cell.
    store_area_1d = np.where(chan_mask_1d, width_1d * dist_1d, store_area_1d)

    n_chan = int(chan_mask_1d.sum())
    if n_chan:
        wch = width_1d[chan_mask_1d]
        print(f"  Channel routing|  {n_chan:,}/{n_cells:,} cells "
              f"(faccum>{threshold:g})  width=[{wch.min():.1f}, {wch.max():.1f}] m")
    else:
        print(f"  Channel routing|  no cells exceed faccum threshold "
              f"{threshold:g}; routing as wide sheet everywhere.")
    return chan_mask_1d, width_1d, store_area_1d
