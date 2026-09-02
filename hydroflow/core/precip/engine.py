"""
precip_input.py
===============
Modular precipitation engine for the kinematic-wave router.

Supported PRECIP_METHOD values:  'uniform' | 'thiessen' | 'idw'

Internal unit: m/s (rate), converted from mm-depth/interval at load time.
Spatial weights are precomputed once at initialisation; per-step cost is
O(log T) searchsorted interpolation + O(N·G) matrix-vector multiply.

File formats
------------
precipitation/gauges.csv
    gauge_id,name,easting_m,northing_m
    G01,Sundarijal,335000,3090000
    ...

precipitation/timeseries.csv
    time_s,G01,G02,...
    0,0.0,0.0,...
    300,1.2,0.8,...
    ...
  - time_s   : seconds from simulation start (monotonically increasing)
  - values   : mm depth falling during the interval ending at that time row
  - Rain is zero before row 0 and after the last row (fill_value=0 in interp)
"""

import numpy as np
import pandas as pd


# ── Spatial weight builders ──────────────────────────────────────────────────

def _build_thiessen_weights(cell_coords, gauge_coords):
    """
    Nearest-neighbour (Voronoi / Thiessen) weights.

    Returns
    -------
    W       : (n_cells, n_gauges) 0/1 matrix — each row has exactly one 1.
    nearest : (n_cells,) int array — index of nearest gauge per cell.
    """
    from scipy.spatial import KDTree
    _, nearest = KDTree(gauge_coords).query(cell_coords, k=1)
    W = np.zeros((len(cell_coords), len(gauge_coords)), dtype=np.float64)
    W[np.arange(len(cell_coords)), nearest] = 1.0
    return W, nearest


def _build_idw_weights(cell_coords, gauge_coords, power=2.0):
    """
    Inverse-distance weighting.

    Returns (n_cells, n_gauges) matrix; rows sum to 1.
    Coincident points (dist == 0) are given full weight.
    """
    diff = cell_coords[:, np.newaxis, :] - gauge_coords[np.newaxis, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=2))          # (n_cells, n_gauges)

    # Coincident-point guard: if a cell sits exactly on a gauge, assign all
    # weight to that gauge.
    coincident = dist == 0.0                          # (n_cells, n_gauges)
    any_coincident = coincident.any(axis=1)           # (n_cells,)

    dist[dist == 0.0] = 1e-10                         # avoid divide-by-zero
    inv_d = 1.0 / (dist ** power)
    W = inv_d / inv_d.sum(axis=1, keepdims=True)

    # Override rows where a cell coincides with a gauge
    W[any_coincident] = coincident[any_coincident].astype(np.float64)

    return W


# ── Main engine ──────────────────────────────────────────────────────────────

class PrecipEngine:
    """
    Unified precipitation engine used by both kinematic_wave_router.py and
    the animation test.

    Parameters
    ----------
    cfg       : config module (or any object with the required attributes)
    grid_data : dict returned by initialise_grid(); needs:
                  ws_mask, s_rows, s_cols, nrows, ncols, n_cells, transform
    """

    def __init__(self, cfg, grid_data):
        self._method  = getattr(cfg, 'PRECIP_METHOD', 'uniform').lower()
        self._s_rows  = grid_data['s_rows']
        self._s_cols  = grid_data['s_cols']
        self._nrows   = grid_data['nrows']
        self._ncols   = grid_data['ncols']
        self._n_cells = grid_data['n_cells']

        if self._method == 'uniform':
            self._weight_method = 'uniform'
            self._init_uniform(cfg)
        elif self._method in ('thiessen', 'idw'):
            # Source and weighting are the same for plain gauge methods.
            self._weight_method = self._method
            self._init_gauge(cfg, grid_data,
                             cfg.PRECIP_GAUGE_FILE, cfg.PRECIP_TIMESERIES_FILE)
        elif self._method in ('imerg_thiessen', 'imerg_idw'):
            # IMERG source: download (if needed) gridded pseudo-gauges, then apply
            # the weighting named in the suffix.  Reuses the gauge machinery.
            self._weight_method = self._method.split('_', 1)[1]   # 'thiessen'|'idw'
            # Derive IMERG window from EVENT_START_UTC + TOTAL_SIMULATION_TIME_HOURS.
            _evt   = getattr(cfg, 'EVENT_START_UTC', None)
            _sim_h = float(getattr(cfg, 'TOTAL_SIMULATION_TIME_HOURS', 0) or 0)
            _off   = float(getattr(cfg, 'IMERG_UTC_OFFSET_HOURS', 0.0))
            if _evt:
                from datetime import datetime, timedelta as _td
                _utc_start = datetime.fromisoformat(str(_evt).strip())
                _utc_end   = _utc_start + _td(hours=_sim_h)
                if not getattr(cfg, 'IMERG_START_LOCAL', None):
                    cfg.IMERG_START_LOCAL = (_utc_start + _td(hours=_off)).strftime("%Y-%m-%d %H:%M")
                    print(f"  IMERG_START_LOCAL  → {cfg.IMERG_START_LOCAL}  (EVENT_START_UTC + offset)")
                if not getattr(cfg, 'IMERG_END_LOCAL', None):
                    cfg.IMERG_END_LOCAL = (_utc_end + _td(hours=_off)).strftime("%Y-%m-%d %H:%M")
                    print(f"  IMERG_END_LOCAL    → {cfg.IMERG_END_LOCAL}  (start + {_sim_h:.0f}h)")
            from ...gee.imerg_gee import ensure_imerg_data
            gauge_file, ts_file = ensure_imerg_data(cfg)
            self._init_gauge(cfg, grid_data, gauge_file, ts_file)
        else:
            raise ValueError(
                f"PRECIP_METHOD='{self._method}' is not recognised. "
                "Choose 'uniform', 'thiessen', 'idw', 'imerg_thiessen', "
                "or 'imerg_idw'."
            )

        print(f"  PrecipEngine    |  method={self._method}"
              f"  |  gauges={self._n_gauges}")

    # ── Initialisation paths ─────────────────────────────────────────────────

    def _init_uniform(self, cfg):
        """
        Encode the uniform event as a 1-gauge, all-ones-weight system.
        This reuses the same data model as the gauge path with zero extra
        branching in the hot time loop.
        """
        rate_ms    = cfg.RAIN_INTENSITY_MM_HR / (1000.0 * 3600.0)
        duration_s = cfg.RAIN_DURATION_HOURS * 3600.0
        eps        = 1e-6  # tiny gap so rain stops sharply at duration_s

        # Three-point step function: on → on → off
        self._time_s   = np.array([0.0, duration_s - eps, duration_s])
        self._rates_ms = np.array([[rate_ms],
                                   [rate_ms],
                                   [0.0]])        # shape (3, 1)
        self._weights  = np.ones((self._n_cells, 1), dtype=np.float64)
        self._n_gauges = 1
        self._cell_polygon = None   # no zones for uniform rainfall

    def _drop_outside_stations(self, gauges_df, gauge_ids, grid_data):
        """
        Return the subset of *gauge_ids* whose centroid lands on an active
        watershed cell.  Inside-ness is tested against the watershed cell set
        (s_rows/s_cols) rather than a polygon, so it works for any caller.
        Falls back to the full list (with a warning) if none are inside.
        """
        t = grid_data['transform']
        mask = np.zeros((self._nrows, self._ncols), dtype=bool)
        mask[self._s_rows, self._s_cols] = True

        keep = []
        for gid in gauge_ids:
            e = float(gauges_df.loc[gid, 'easting_m'])
            n = float(gauges_df.loc[gid, 'northing_m'])
            col = int((e - t.c) / t.a)
            row = int((n - t.f) / t.e)
            if 0 <= row < self._nrows and 0 <= col < self._ncols \
                    and mask[row, col]:
                keep.append(gid)

        n_drop = len(gauge_ids) - len(keep)
        if not keep:
            print(f"  [WARN] PRECIP_EXCLUDE_OUTSIDE_STATIONS: all "
                  f"{len(gauge_ids)} stations are outside the watershed; "
                  f"keeping all.")
            return gauge_ids
        if n_drop:
            print(f"  Stations        |  excluded {n_drop} outside watershed "
                  f"({len(keep)}/{len(gauge_ids)} kept)")
        return keep

    def _init_gauge(self, cfg, grid_data, gauge_file, timeseries_file):
        """
        Load gauges.csv + timeseries.csv, convert mm depth → m/s rates,
        and build the spatial weight matrix (Thiessen or IDW).

        The weighting scheme is taken from ``self._weight_method`` ('thiessen' or
        'idw'), which is independent of ``self._method`` (so an 'imerg_*' source can
        reuse this path).  ``gauge_file``/``timeseries_file`` are passed explicitly
        because IMERG reads from a downloaded folder rather than PRECIP_GAUGE_FILE.
        """
        gauges_df = pd.read_csv(gauge_file).set_index('gauge_id')
        ts_df     = pd.read_csv(timeseries_file)

        gauge_ids = gauges_df.index.tolist()

        # Validate: every gauge in metadata must have a column in timeseries
        missing = set(gauge_ids) - set(ts_df.columns)
        if missing:
            raise ValueError(
                f"Gauges {sorted(missing)} are listed in "
                f"{gauge_file} but have no matching column in "
                f"{timeseries_file}."
            )

        # Optionally drop stations whose centroid falls outside the watershed.
        # Done here (once) so the same station set drives both the rainfall
        # weights and the per-zone SD partition (cell_polygon).
        if getattr(cfg, 'PRECIP_EXCLUDE_OUTSIDE_STATIONS', False):
            gauge_ids = self._drop_outside_stations(gauges_df, gauge_ids, grid_data)
            gauges_df = gauges_df.loc[gauge_ids]

        time_s_raw = ts_df['time_s'].values.astype(np.float64)

        # Warn if timeseries is shorter than the simulation — the router already
        # returns zero rainfall after the last gauge record, but the user should
        # know so they can check whether truncation is intentional.
        _sim_h = getattr(cfg, 'TOTAL_SIMULATION_TIME_HOURS', None)
        if _sim_h is not None:
            _sim_end_s = float(_sim_h) * 3600.0
            if time_s_raw[-1] < _sim_end_s - 1.0:
                _gap_h = (_sim_end_s - time_s_raw[-1]) / 3600.0
                print(f"  [WARN] Gauge timeseries ends at "
                      f"t={time_s_raw[-1]/3600:.2f}h but simulation runs "
                      f"{_sim_h:.1f}h — zero rainfall assumed for the "
                      f"remaining {_gap_h:.2f}h.")
            if time_s_raw[0] > 1.0:
                _pre_h = time_s_raw[0] / 3600.0
                print(f"  [WARN] Gauge timeseries starts at t={_pre_h:.2f}h "
                      f"(not t=0) — zero rainfall assumed before first record.")

        # mm depth per interval → m/s rate
        # interval_s[i] = duration of the interval ending at row i
        # For the last row, repeat the second-to-last interval length.
        intervals = np.diff(time_s_raw)
        if len(intervals) == 0:
            intervals = np.array([1.0])           # single-row edge case
        intervals = np.append(intervals, intervals[-1])   # (T,)

        depth_mm  = ts_df[gauge_ids].values.astype(np.float64)   # (T, G)
        rates_ms  = np.maximum(
            depth_mm / 1000.0 / intervals[:, np.newaxis], 0.0
        )                                                          # (T, G)

        self._time_s    = time_s_raw   # (T,)
        self._rates_ms  = rates_ms     # (T, G)
        self._n_gauges  = len(gauge_ids)
        self._gauge_ids = gauge_ids

        print(f"  Gauge timeseries|  T={len(time_s_raw)} rows"
              f"  |  t=[{time_s_raw[0]:.0f}s … {time_s_raw[-1]:.0f}s]")

        # ── Projected cell coordinates ─────────────────────────────────────
        t       = grid_data['transform']
        cell_x  = t.c + (self._s_cols + 0.5) * t.a
        cell_y  = t.f + (self._s_rows + 0.5) * t.e
        cell_xy = np.column_stack([cell_x, cell_y])         # (n_cells, 2)

        gauge_xy = gauges_df[['easting_m', 'northing_m']].values.astype(np.float64)

        power = getattr(cfg, 'PRECIP_IDW_POWER', 2.0)
        if self._weight_method == 'thiessen':
            self._weights, self._cell_polygon = _build_thiessen_weights(cell_xy, gauge_xy)
        else:
            self._weights = _build_idw_weights(cell_xy, gauge_xy, power)
            # For IDW: zone cells by nearest gauge for per-polygon VSA sandbox
            from scipy.spatial import KDTree
            _, self._cell_polygon = KDTree(gauge_xy).query(cell_xy, k=1)

        # Sanity check: rows should sum to ~1
        row_sums = self._weights.sum(axis=1)
        if not np.allclose(row_sums, 1.0, atol=1e-9):
            raise RuntimeError(
                "Weight matrix rows do not sum to 1.  "
                f"min={row_sums.min():.6f}  max={row_sums.max():.6f}"
            )

    # ── Runtime accessors ────────────────────────────────────────────────────

    def _interp_gauges(self, t_seconds):
        """
        Vectorised linear interpolation of all gauges to *t_seconds*.

        Returns (n_gauges,) array of rainfall rates in m/s.
        Outside the time range → 0 (fill_value=0, no extrapolation).
        """
        # Outside range → zero
        if t_seconds < self._time_s[0] or t_seconds > self._time_s[-1]:
            return np.zeros(self._n_gauges, dtype=np.float64)

        idx   = np.searchsorted(self._time_s, t_seconds, side='right') - 1
        idx   = np.clip(idx, 0, len(self._time_s) - 2)

        t0    = self._time_s[idx]
        t1    = self._time_s[idx + 1]
        alpha = 0.0 if t1 == t0 else (t_seconds - t0) / (t1 - t0)
        alpha = float(np.clip(alpha, 0.0, 1.0))

        return (1.0 - alpha) * self._rates_ms[idx] + alpha * self._rates_ms[idx + 1]

    def get_field_1d(self, t_seconds):
        """
        Rainfall rate at each active cell in topological order.

        Returns
        -------
        rain_1d : 1-D float64 array, shape (n_cells,), units m/s
        """
        gauge_rates = self._interp_gauges(t_seconds)   # (n_gauges,)
        return self._weights @ gauge_rates              # (n_cells,)

    def get_field_2d(self, t_seconds):
        """
        Rainfall rate as a full 2-D grid (nrows × ncols), units m/s.
        Cells outside the watershed are NaN.
        Used by the animation script for spatial display.
        """
        rain_1d = self.get_field_1d(t_seconds)
        rain_2d = np.full((self._nrows, self._ncols), np.nan, dtype=np.float64)
        rain_2d[self._s_rows, self._s_cols] = rain_1d
        return rain_2d

    def is_raining(self, t_seconds):
        """True if any active cell receives non-zero rain at *t_seconds*."""
        return bool(self.get_field_1d(t_seconds).max() > 0.0)

    @property
    def cell_polygon(self):
        """Per-cell nearest-gauge zone index (n_cells,) int, or None for uniform."""
        return self._cell_polygon

    @property
    def rain_end_seconds(self):
        """Last timestamp with non-zero rain (approximate)."""
        return float(self._time_s[-1])
