"""
precip_input_gpu.py
===================
GPU-accelerated PrecipEngine subclass.

Overrides:
  - Weight matrix storage (Thiessen → nearest-gauge index; IDW → GPU matrix)
  - get_field_1d  → returns a CuPy array on the GPU
  - get_field_2d  → transfers GPU result back to CPU for 2-D raster output

_interp_gauges stays on CPU: cp.searchsorted does not accept Python float
scalars, and the gauge count is tiny (n_gauges << n_cells) so the overhead
is negligible.
"""

import numpy as np
import cupy as cp

from ...utils import gpu_utils
from .engine import PrecipEngine


class PrecipEngineGPU(PrecipEngine):
    """
    Drop-in GPU replacement for PrecipEngine.

    Constructed the same way as the CPU version; the parent __init__ builds
    the CPU weight matrix first, then _gpu_init() converts it to GPU format.
    """

    def __init__(self, cfg, grid_data):
        # Build everything on CPU first (loads gauge files, builds weights)
        super().__init__(cfg, grid_data)
        self._gpu_init()

    # ── GPU weight conversion ─────────────────────────────────────────────────

    def _gpu_init(self):
        """
        Convert CPU weight representations to GPU-optimised equivalents.

        Thiessen → nearest-gauge index array (fancy index; 157× faster than
                   a full n_cells × n_gauges matrix-vector multiply).
        IDW      → full weight matrix transferred to GPU.
        Uniform  → scalar broadcast; no matrix needed.
        """
        if self._weight_method == 'uniform':
            # Scalar broadcast; _weights is a (n_cells, 1) ones matrix from
            # the parent — not used in get_field_1d; nothing to transfer.
            pass

        elif self._weight_method == 'thiessen':
            # Each row has exactly one 1 — replace with its column index.
            W_cpu = self._weights                          # (n_cells, n_gauges)
            nearest = np.argmax(W_cpu, axis=1).astype(np.int32)
            self._nearest_idx_gpu = cp.asarray(nearest)
            self._weights = None   # free CPU memory

        else:  # 'idw'
            self._weights_gpu = cp.asarray(self._weights)
            self._weights = None

    # ── Runtime accessor (returns CuPy array) ────────────────────────────────

    def get_field_1d(self, t_seconds):
        """
        Rainfall rate at each active cell [m/s], returned as a CuPy array.

        _interp_gauges runs on CPU (tiny; n_gauges rows of linear interp).
        """
        gauge_rates_np = self._interp_gauges(t_seconds)   # (n_gauges,) numpy

        if self._weight_method == 'uniform':
            # Scalar broadcast — fastest possible path.
            rate = float(gauge_rates_np[0])
            return cp.full(self._n_cells, rate, dtype=np.float64)

        elif self._weight_method == 'thiessen':
            # Fancy index: gauge_rates_gpu[nearest_idx_gpu]
            # H2D transfer of gauge_rates is ~13 µs; negligible vs step time.
            gr = cp.asarray(gauge_rates_np)
            return gr[self._nearest_idx_gpu]              # (n_cells,) CuPy

        else:  # 'idw'
            gr = cp.asarray(gauge_rates_np)
            return self._weights_gpu @ gr                 # (n_cells,) CuPy

    # ── 2-D raster output (CPU) ───────────────────────────────────────────────

    def get_field_2d(self, t_seconds):
        """
        Rainfall rate as a full 2-D grid (nrows × ncols), units m/s.
        Cells outside the watershed are NaN.
        Returns a NumPy array (for plotting / animation scripts).
        """
        rain_1d_cpu = gpu_utils.to_cpu(self.get_field_1d(t_seconds))
        rain_2d = np.full((self._nrows, self._ncols), np.nan, dtype=np.float64)
        rain_2d[self._s_rows, self._s_cols] = rain_1d_cpu
        return rain_2d
