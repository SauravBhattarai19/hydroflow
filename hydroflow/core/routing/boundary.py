# -*- coding: utf-8 -*-
"""
boundary.py
===========
Time-varying upstream inflow boundary condition(s) for the routing time loop.

Injects an externally-specified discharge hydrograph Q(t) [m³/s] at one or more
cells, so the router can (a) account for upstream flow entering the domain and
(b) run pure routing with no rainfall (set RAIN_INTENSITY_MM_HR=0).

Each boundary point is optionally snapped to the highest-flow-accumulation
(channel) cell within a small radius of the requested location, then mapped to
its position in the topological cell array.  At runtime ``rate_1d(t)`` returns a
per-cell inflow-rate array [m³/s] (zero everywhere except the boundary cells);
duplicate cells (two BCs snapping to the same cell) are summed.

Config (cfg.ROUTING_INFLOW_BC): None (disabled) or a list of dicts, e.g.
    {"name": "us1",              # optional label
     "lat": 27.7, "lon": 85.3,   # location: lat/lon OR row/col OR easting/northing
     "csv": "inflow_us1.csv",    # time series: columns (time_hr | time_s) + Q_m3s
     "snap_to_channel": True,     # snap to max-faccum cell within radius
     "snap_radius_cells": 3}
"""

import os

import numpy as np
import pandas as pd

from ...utils import gpu_utils


def _resolve_rowcol(spec, transform, target_crs, nrows, ncols):
    """Return (row, col) for a BC spec from row/col, easting/northing or lat/lon."""
    if "row" in spec and "col" in spec:
        row, col = int(spec["row"]), int(spec["col"])
    elif "easting" in spec and "northing" in spec:
        e, n = float(spec["easting"]), float(spec["northing"])
        col = int((e - transform.c) / transform.a)
        row = int((n - transform.f) / transform.e)
    elif "lat" in spec and "lon" in spec:
        from pyproj import Transformer
        tr = Transformer.from_crs("EPSG:4326", target_crs, always_xy=True)
        e, n = tr.transform(float(spec["lon"]), float(spec["lat"]))
        col = int((e - transform.c) / transform.a)
        row = int((n - transform.f) / transform.e)
    else:
        raise ValueError(
            "Inflow BC needs a location: lat/lon, row/col, or easting/northing."
        )
    if not (0 <= row < nrows and 0 <= col < ncols):
        raise ValueError(
            f"Inflow BC location (row={row}, col={col}) is outside the grid "
            f"({nrows}×{ncols}).  Check the coordinates / CRS."
        )
    return row, col


def _load_hydrograph(csv_path):
    """Load a BC hydrograph CSV → (times_s, Q_m3s) float arrays.

    Accepts a time column named 'time_s' or 'time_hr' and a discharge column
    named 'Q_m3s' or 'Q' (case-insensitive).
    """
    df = pd.read_csv(csv_path)
    cols = {c.lower(): c for c in df.columns}
    if "time_s" in cols:
        times_s = df[cols["time_s"]].to_numpy(dtype=np.float64)
    elif "time_hr" in cols:
        times_s = df[cols["time_hr"]].to_numpy(dtype=np.float64) * 3600.0
    else:
        raise ValueError(
            f"{csv_path}: need a 'time_s' or 'time_hr' column (found {list(df.columns)})."
        )
    if "q_m3s" in cols:
        q = df[cols["q_m3s"]].to_numpy(dtype=np.float64)
    elif "q" in cols:
        q = df[cols["q"]].to_numpy(dtype=np.float64)
    else:
        raise ValueError(
            f"{csv_path}: need a 'Q_m3s' (or 'Q') discharge column (found {list(df.columns)})."
        )
    order = np.argsort(times_s)
    return times_s[order], np.maximum(q[order], 0.0)


class InflowBoundary:
    """Upstream inflow boundary condition engine (see module docstring)."""

    def __init__(self, cfg, grid_data):
        specs = getattr(cfg, "ROUTING_INFLOW_BC", None)
        if isinstance(specs, dict):
            specs = [specs]
        specs = list(specs) if specs else []

        self._xp = grid_data.get("xp", np)
        self._dtype = gpu_utils.get_dtype(cfg)
        self._n_cells = grid_data["n_cells"]
        self._points = []   # list of dicts: name, pos, times_s, q

        if not specs:
            return

        nrows, ncols = grid_data["nrows"], grid_data["ncols"]
        transform = grid_data["transform"]
        target_crs = getattr(cfg, "TARGET_CRS_EPSG", None)
        s_rows = gpu_utils.to_cpu(grid_data["s_rows"]).astype(np.int64)
        s_cols = gpu_utils.to_cpu(grid_data["s_cols"]).astype(np.int64)
        faccum_1d = gpu_utils.to_cpu(grid_data["faccum_1d"]).astype(np.float64)

        # position-of-(row,col) lookup (−1 off-mask) and faccum grid for snapping
        pos_2d = np.full((nrows, ncols), -1, dtype=np.int64)
        pos_2d[s_rows, s_cols] = np.arange(self._n_cells, dtype=np.int64)
        faccum_2d = np.full((nrows, ncols), -np.inf, dtype=np.float64)
        faccum_2d[s_rows, s_cols] = faccum_1d

        T = float(getattr(cfg, "TOTAL_SIMULATION_TIME_HOURS", 0.0) or 0.0) * 3600.0

        for i, spec in enumerate(specs):
            name = spec.get("name") or f"bc{i + 1}"
            row, col = _resolve_rowcol(spec, transform, target_crs, nrows, ncols)

            snap = spec.get("snap_to_channel", True)
            radius = int(spec.get("snap_radius_cells", 3))
            if snap and radius > 0:
                r0, r1 = max(0, row - radius), min(nrows, row + radius + 1)
                c0, c1 = max(0, col - radius), min(ncols, col + radius + 1)
                window = faccum_2d[r0:r1, c0:c1]
                if not np.isfinite(window).any():
                    raise ValueError(
                        f"Inflow BC '{name}': no active watershed cell within "
                        f"{radius} cells of (row={row}, col={col})."
                    )
                loc = np.unravel_index(np.argmax(window), window.shape)
                row, col = r0 + int(loc[0]), c0 + int(loc[1])

            pos = int(pos_2d[row, col])
            if pos < 0:
                raise ValueError(
                    f"Inflow BC '{name}' maps to (row={row}, col={col}) which is "
                    "outside the watershed.  Enable snap_to_channel or pick an "
                    "on-mask cell."
                )

            times_s, q = _load_hydrograph(spec["csv"])
            if T > 0 and times_s[-1] < T - 1.0:
                print(f"  [WARN] Inflow BC '{name}' series ends at "
                      f"t={times_s[-1] / 3600:.2f}h but simulation runs "
                      f"{T / 3600:.1f}h — last value held for the remainder.")
            self._points.append(dict(name=name, pos=pos, row=row, col=col,
                                     times_s=times_s, q=q, q_peak=float(q.max())))

        # Device-side index buffer + a reusable scatter target
        self._pos_dev = self._xp.asarray(
            np.array([p["pos"] for p in self._points], dtype=np.int64))
        self._buf = self._xp.zeros(self._n_cells, dtype=self._dtype)

        pts = ", ".join(f"{p['name']}@(r{p['row']},c{p['col']}) "
                        f"peak={p['q_peak']:.3g}" for p in self._points)
        print(f"  InflowBoundary  |  {len(self._points)} point(s)  |  {pts}")

    @property
    def active(self):
        """True when at least one boundary point is configured."""
        return bool(self._points)

    @property
    def positions(self):
        """Topological cell positions of the boundary points (list of int)."""
        return [p["pos"] for p in self._points]

    def rate_1d(self, t_seconds):
        """Per-cell inflow rate [m³/s], shape (n_cells,), on the active device.

        Zero everywhere except the boundary cells.  Each point's Q(t) is a
        linear interpolation of its hydrograph that HOLDS the end values outside
        the series range (np.interp default).  The returned array is a reused
        buffer — valid until the next call; callers consume it within one step.
        """
        q_vals = np.array(
            [float(np.interp(t_seconds, p["times_s"], p["q"])) for p in self._points],
            dtype=np.float64)
        self._buf.fill(0)
        gpu_utils.scatter_add(self._buf, self._pos_dev,
                              self._xp.asarray(q_vals, dtype=self._dtype))
        return self._buf
