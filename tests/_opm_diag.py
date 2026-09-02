"""
_opm_diag.py
============
Shared helpers for the diagnostic test scripts (05/06/07) so they honour the
ACTIVE config — the scenario output folder (config.OUTPUT_DIR) and the selected
precipitation method (including the IMERG GEE source) — instead of hardcoding
'output/' and the test-data gauges.

Importable because Python puts the running script's directory (tests/) on
sys.path[0]; the diagnostic scripts also insert REPO before importing this.
"""

from pathlib import Path

import config

REPO = Path(__file__).resolve().parent.parent


def _abs(p):
    """Return an absolute Path for *p* (resolved against REPO if relative)."""
    if p is None:
        return None
    p = Path(p)
    return p if p.is_absolute() else REPO / p


def output_dir():
    """Scenario output folder from config.OUTPUT_DIR (absolute Path)."""
    return _abs(config.OUTPUT_DIR)


def resolve_precip(cfg=config):
    """
    Resolve ``(gauge_csv, ts_csv, event_date)`` for the active PRECIP_METHOD.

      imerg_*      → the downloaded IMERG folder gauges/timeseries
                     (ensures the data exists via imerg_gee.ensure_imerg_data)
      thiessen/idw → cfg.PRECIP_GAUGE_FILE / cfg.PRECIP_TIMESERIES_FILE
      uniform      → (None, None, event_date)

    ``event_date`` prefers cfg.SERVES_TARGET_DATE; for IMERG it falls back to the
    date part of IMERG_START_LOCAL.  Paths are returned absolute (or None).

    The diagnostic scripts use this instead of hardcoding the test-data event so
    the gauge geometry they visualise matches the zones the engine actually
    builds (e.g. 25 IMERG pixel pseudo-gauges, not 7 field gauges).
    """
    method = getattr(cfg, 'PRECIP_METHOD', 'uniform').lower()
    event_date = getattr(cfg, 'SERVES_TARGET_DATE', None)

    if method.startswith('imerg'):
        from hydroflow.gee.imerg_gee import ensure_imerg_data
        gauge_csv, ts_csv = ensure_imerg_data(cfg)
        if event_date is None:
            start = getattr(cfg, 'IMERG_START_LOCAL', None)
            if start:
                event_date = str(start).split()[0]   # 'YYYY-MM-DD'
        return _abs(gauge_csv), _abs(ts_csv), event_date

    if method in ('thiessen', 'idw'):
        return (_abs(getattr(cfg, 'PRECIP_GAUGE_FILE', None)),
                _abs(getattr(cfg, 'PRECIP_TIMESERIES_FILE', None)),
                event_date)

    return None, None, event_date   # uniform


def nearest_station_raster(gauge_csv, ws_mask, transform):
    """
    Build the nearest-station partition as a 2-D int raster (zone id per watershed
    cell, -1 elsewhere), identical to the ``cell_polygon`` PrecipEngine computes
    (KDTree over station centroids).  Use this for diagnostic maps so the drawn
    zones and per-zone values match the model — including boundary strips owned by
    stations that sit outside the watershed.

    Returns (poly_raster, n_stations).
    """
    import numpy as np
    import pandas as pd
    from scipy.spatial import KDTree

    g = pd.read_csv(gauge_csv)
    gxy = g[['easting_m', 'northing_m']].values.astype(float)

    rows, cols = np.where(ws_mask)
    cx = transform.c + (cols + 0.5) * transform.a
    cy = transform.f + (rows + 0.5) * transform.e
    _, nearest = KDTree(gxy).query(np.column_stack([cx, cy]), k=1)

    poly = np.full(ws_mask.shape, -1, dtype=np.int32)
    poly[rows, cols] = nearest
    return poly, len(g)
