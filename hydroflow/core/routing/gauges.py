# -*- coding: utf-8 -*-
"""
gauges.py
=========
Virtual gauges: record depth / discharge / velocity time series at a handful of
points (e.g. hydrological stations) sampled every OUTPUT_INTERVAL and written to
``{OUTPUT_DIR}/gauges.csv``.

This is the memory-cheap way to get fine-cadence hydrographs at specific
locations for arrival-time / travel-time validation, without saving the full
per-cell field stack.  Each gauge is snapped to the nearest channel (max
flow-accumulation) cell, exactly like ``ROUTING_INFLOW_BC``.

Config (cfg.ROUTING_GAUGES): None (disabled) or a list of dicts:
    {"name": "Betrawati", "lat": 27.974, "lon": 85.185,
     "snap_to_channel": True, "snap_radius_cells": 5}
"""

import csv
import os

import numpy as np

from ...utils import gpu_utils
from .boundary import _resolve_rowcol

_EPS = 1e-12


class GaugeRecorder:
    """Record depth/discharge/velocity at fixed points over time (see module docstring)."""

    def __init__(self, cfg, grid_data):
        specs = getattr(cfg, "ROUTING_GAUGES", None)
        if isinstance(specs, dict):
            specs = [specs]
        specs = list(specs) if specs else []

        self._xp = grid_data.get("xp", np)
        self._out_dir = getattr(cfg, "OUTPUT_DIR", "output/")
        self._gauges = []      # (name, pos, row, col, faccum)
        self._rows = []        # accumulated [t_s, d0,q0,v0, d1,q1,v1, ...]

        if not specs:
            return

        nrows, ncols = grid_data["nrows"], grid_data["ncols"]
        transform = grid_data["transform"]
        target_crs = getattr(cfg, "TARGET_CRS_EPSG", None)
        s_rows = gpu_utils.to_cpu(grid_data["s_rows"]).astype(np.int64)
        s_cols = gpu_utils.to_cpu(grid_data["s_cols"]).astype(np.int64)
        faccum = gpu_utils.to_cpu(grid_data["faccum_1d"]).astype(np.float64)

        pos_2d = np.full((nrows, ncols), -1, dtype=np.int64)
        pos_2d[s_rows, s_cols] = np.arange(len(s_rows), dtype=np.int64)
        fac_2d = np.full((nrows, ncols), -np.inf, dtype=np.float64)
        fac_2d[s_rows, s_cols] = faccum

        for i, spec in enumerate(specs):
            name = spec.get("name") or f"gauge{i + 1}"
            row, col = _resolve_rowcol(spec, transform, target_crs, nrows, ncols)
            if spec.get("snap_to_channel", True):
                rad = int(spec.get("snap_radius_cells", 5))
                r0, r1 = max(0, row - rad), min(nrows, row + rad + 1)
                c0, c1 = max(0, col - rad), min(ncols, col + rad + 1)
                win = fac_2d[r0:r1, c0:c1]
                if np.isfinite(win).any():
                    loc = np.unravel_index(np.argmax(win), win.shape)
                    row, col = r0 + int(loc[0]), c0 + int(loc[1])
            pos = int(pos_2d[row, col])
            if pos < 0:
                print(f"  [WARN] gauge '{name}' maps off-watershed; skipped.")
                continue
            self._gauges.append((name, pos, row, col, float(faccum[pos])))

        if self._gauges:
            self._pos_dev = self._xp.asarray(
                np.array([g[1] for g in self._gauges], dtype=np.int64))
            info = ", ".join(f"{n}@(r{r},c{c},fac{f:.0f})"
                             for n, _, r, c, f in self._gauges)
            print(f"  GaugeRecorder   |  {len(self._gauges)} gauge(s)  |  {info}")

    @property
    def active(self):
        return bool(self._gauges)

    def record(self, t_seconds, depth_1d, Q_out_1d, A_xs_1d, xp):
        """Sample the gauge cells this output step (device gather → few scalars)."""
        pos = self._pos_dev
        dep = gpu_utils.to_cpu(depth_1d[pos])
        q = gpu_utils.to_cpu(Q_out_1d[pos])
        a = gpu_utils.to_cpu(A_xs_1d[pos])
        vel = q / np.maximum(a, _EPS)
        row = [float(t_seconds)]
        for k in range(len(self._gauges)):
            row += [float(dep[k]), float(q[k]), float(vel[k])]
        self._rows.append(row)

    def save(self):
        """Write {OUTPUT_DIR}/gauges.csv.  Returns the path (or None)."""
        if not self._gauges or not self._rows:
            return None
        os.makedirs(self._out_dir, exist_ok=True)
        path = os.path.join(self._out_dir, "gauges.csv")
        header = ["time_s", "time_hr"]
        for n, _, _, _, _ in self._gauges:
            header += [f"{n}_depth_m", f"{n}_Q_m3s", f"{n}_vel_ms"]
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            for r in self._rows:
                t = r[0]
                w.writerow([f"{t:.1f}", f"{t / 3600.0:.5f}"]
                           + [f"{v:.4f}" for v in r[1:]])
        print(f"  Gauges saved    → {path}  ({len(self._rows)} rows, "
              f"{len(self._gauges)} gauges)")
        return path
