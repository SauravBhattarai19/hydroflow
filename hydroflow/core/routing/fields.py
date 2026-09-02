# -*- coding: utf-8 -*-
"""
fields.py
=========
Spatiotemporal field recorder for the routing time loop, plus loaders.

When cfg.SAVE_FIELDS is on, the router records per-cell depth / velocity /
discharge (and optionally volume) at each OUTPUT_INTERVAL (every FIELD_STRIDE-th
record) and writes a compact archive so plots, animations, and at-any-cell
hydrographs can be built afterwards:

    {OUTPUT_DIR}/fields/
        fields.npz        times_s[T], s_rows[N], s_cols[N], <var>[T, N]  (float32)
        fields_meta.json  nrows, ncols, transform[6], crs, cell_size, units, …

Cells are stored in the router's 1-D topological order; ``field_to_2d`` (below)
scatters a (N,) or (T, N) array back onto the (nrows, ncols) grid for plotting.
"""

import json
import os

import numpy as np

from ...utils import gpu_utils

_EPS = 1e-12

_UNITS = {
    "depth": "m",
    "velocity": "m/s",
    "discharge": "m3/s",
    "volume": "m3",
}


class FieldRecorder:
    """Accumulate per-cell fields over time and write them to a compact archive."""

    def __init__(self, cfg, grid_data):
        allowed = ("depth", "velocity", "discharge", "volume")
        self._vars = [v for v in getattr(cfg, "FIELD_VARS", ["depth"]) if v in allowed]
        self._stride = max(1, int(getattr(cfg, "FIELD_STRIDE", 1)))
        self._out_dir = (getattr(cfg, "FIELD_OUTPUT_DIR", None)
                         or os.path.join(getattr(cfg, "OUTPUT_DIR", "output/"), "fields"))

        self._s_rows = gpu_utils.to_cpu(grid_data["s_rows"]).astype(np.int32)
        self._s_cols = gpu_utils.to_cpu(grid_data["s_cols"]).astype(np.int32)
        self._nrows = int(grid_data["nrows"])
        self._ncols = int(grid_data["ncols"])
        self._cell_size = float(grid_data["cell_size"])
        t = grid_data["transform"]
        self._transform = [t.a, t.b, t.c, t.d, t.e, t.f]
        self._crs = str(getattr(cfg, "TARGET_CRS_EPSG", "") or "")

        self._times = []
        self._store = {v: [] for v in self._vars}
        self._counter = 0

        print(f"  FieldRecorder   |  vars={self._vars}  stride={self._stride}  "
              f"→ {self._out_dir}")

    @property
    def active(self):
        return bool(self._vars)

    def record(self, t_seconds, depth_1d, Q_out_1d, A_xs_1d, volume_1d, xp):
        """Record the current step's fields (called from the router's output block).

        Only every FIELD_STRIDE-th call is retained.  Arrays are transferred to
        host and stored as float32 to halve the archive size.
        """
        self._counter += 1
        if (self._counter - 1) % self._stride != 0:
            return

        self._times.append(float(t_seconds))
        for v in self._vars:
            if v == "depth":
                arr = depth_1d
            elif v == "discharge":
                arr = Q_out_1d
            elif v == "velocity":
                arr = Q_out_1d / xp.maximum(A_xs_1d, _EPS)
            else:  # volume
                arr = volume_1d
            self._store[v].append(gpu_utils.to_cpu(arr).astype(np.float32))

    def save(self):
        """Write fields.npz + fields_meta.json.  No-op if nothing was recorded."""
        if not self._times:
            print("  FieldRecorder   |  nothing recorded — no archive written.")
            return None

        os.makedirs(self._out_dir, exist_ok=True)
        npz_path = os.path.join(self._out_dir, "fields.npz")
        meta_path = os.path.join(self._out_dir, "fields_meta.json")

        arrays = {
            "times_s": np.asarray(self._times, dtype=np.float64),
            "s_rows": self._s_rows,
            "s_cols": self._s_cols,
        }
        for v in self._vars:
            arrays[v] = np.stack(self._store[v], axis=0)   # (T, N) float32
        np.savez_compressed(npz_path, **arrays)

        meta = {
            "nrows": self._nrows,
            "ncols": self._ncols,
            "cell_size": self._cell_size,
            "transform": self._transform,   # affine [a, b, c, d, e, f]
            "crs": self._crs,
            "n_times": len(self._times),
            "n_cells": int(self._s_rows.size),
            "vars": list(self._vars),
            "units": {v: _UNITS.get(v, "") for v in self._vars},
            "dtype": "float32",
            "stride": self._stride,
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        size_mb = os.path.getsize(npz_path) / 1e6
        print(f"  Field archive   → {npz_path}  "
              f"({len(self._times)} times × {self._s_rows.size} cells, {size_mb:.1f} MB)")
        return npz_path


# ── Loaders / plotting helpers ────────────────────────────────────────────────

def load_field_archive(path):
    """Load a field archive.

    *path* may be the fields.npz file or the directory containing it.  Returns a
    dict with keys: ``times_s`` (T,), ``s_rows``/``s_cols`` (N,), each recorded
    variable (T, N), and ``meta`` (the JSON sidecar, if present).
    """
    if os.path.isdir(path):
        npz_path = os.path.join(path, "fields.npz")
    else:
        npz_path = path
    meta_path = os.path.join(os.path.dirname(npz_path), "fields_meta.json")

    with np.load(npz_path) as data:
        out = {k: data[k] for k in data.files}
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            out["meta"] = json.load(f)
    return out


def field_to_2d(values, meta_or_archive, fill=np.nan):
    """Scatter a (N,) or (T, N) 1-D-topological array onto the 2-D grid.

    *meta_or_archive* is either the meta dict / archive dict from
    ``load_field_archive`` or any object exposing ``nrows``/``ncols`` plus
    ``s_rows``/``s_cols``.  Returns (nrows, ncols) for (N,), or
    (T, nrows, ncols) for (T, N), with ``fill`` outside the watershed.
    """
    d = meta_or_archive
    meta = d.get("meta", d) if isinstance(d, dict) else d
    nrows = int(meta["nrows"]); ncols = int(meta["ncols"])
    s_rows = d["s_rows"] if isinstance(d, dict) else getattr(d, "s_rows")
    s_cols = d["s_cols"] if isinstance(d, dict) else getattr(d, "s_cols")

    values = np.asarray(values)
    if values.ndim == 1:
        out = np.full((nrows, ncols), fill, dtype=np.float64)
        out[s_rows, s_cols] = values
        return out
    T = values.shape[0]
    out = np.full((T, nrows, ncols), fill, dtype=np.float64)
    out[:, s_rows, s_cols] = values
    return out
