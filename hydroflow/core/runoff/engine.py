# -*- coding: utf-8 -*-
"""
runoff_input.py
===============
Modular runoff generation engine for the kinematic-wave router.

Sits between PrecipEngine (rainfall in [m/s]) and the time loop (volume update),
transforming raw rainfall into effective surface runoff based on the selected mode.

Modes (set via config.RUNOFF_SOURCE):
  'none'        – all rainfall is direct runoff (default, backward compatible)
  'coefficient' – multiply rainfall by static spatial Cf raster [0–1]
  'raster'      – read pre-computed runoff raster time series [m/s]
  'scs_cn'      – SCS Curve Number method with per-cell CN raster
  'vsa_opm'     – Variable Source Area: Pradhan & Ogden (2010) OPM

Usage in the time loop (forward Euler):
    source_1d = runoff_engine.get_effective_1d(t_s, rain_1d)   # current state
    runoff_engine.update_state(rain_1d, dt)                     # advance state
    rain_vol = source_1d * cell_area * dt

Reference:
    Pradhan, N.R. and Ogden, F.L. (2010). Development of a one-parameter variable
    source area runoff model for ungauged basins. Advances in Water Resources,
    33(5), pp.572–584.
"""

import os

import numpy as np
import pandas as pd
import rasterio

from .vsa import VsaOpmMixin




# ─────────────────────────────────────────────────────────────────────────────
class RunoffEngine(VsaOpmMixin):
    """
    Unified runoff generation engine.

    Parameters
    ----------
    cfg       : config module
    grid_data : dict returned by kinematic_wave_router.initialise_grid()
                Must contain 'faccum_1d' for 'vsa_opm' mode.
    """

    def __init__(self, cfg, grid_data):
        mode = getattr(cfg, 'RUNOFF_SOURCE', 'none').lower()
        self._mode    = mode
        self._n_cells = grid_data['n_cells']
        self._s_rows  = grid_data['s_rows']
        self._s_cols  = grid_data['s_cols']
        self._nrows   = grid_data['nrows']
        self._ncols   = grid_data['ncols']

        if mode == 'none':
            pass
        elif mode == 'coefficient':
            self._init_coefficient(cfg, grid_data)
        elif mode == 'raster':
            self._init_raster(cfg, grid_data)
        elif mode == 'scs_cn':
            self._init_scs_cn(cfg, grid_data)
        elif mode == 'vsa_opm':
            self._init_vsa_opm(cfg, grid_data)
        else:
            raise ValueError(
                f"RUNOFF_SOURCE='{mode}' is not recognised. "
                "Valid options: 'none', 'coefficient', 'raster', 'scs_cn', 'vsa_opm'."
            )

        print(f"  RunoffEngine    |  mode='{mode}'")

    # ── Public interface ──────────────────────────────────────────────────────

    def update_state(self, rain_1d, dt):
        """
        Advance internal state by one timestep.

        Called AFTER get_effective_1d (forward Euler): the current state
        determines this step's runoff, then state advances for the next step.

        Parameters
        ----------
        rain_1d : (n_cells,) float64 [m/s]  raw rainfall from PrecipEngine
        dt      : float  timestep [s]
        """
        if self._mode == 'vsa_opm':
            self._update_opm_sandbox(rain_1d, dt)
        elif self._mode == 'scs_cn':
            self._update_scs_cn(rain_1d, dt)
        # 'none', 'coefficient', 'raster': stateless → no-op

    def get_effective_1d(self, t_seconds, rain_1d):
        """
        Effective runoff rate [m/s] per active cell, shape (n_cells,).

        Uses the state set by the PREVIOUS update_state call (forward Euler).
        At t=0 uses the initial state set during __init__.

        Parameters
        ----------
        t_seconds : float  current simulation time [s]
        rain_1d   : (n_cells,) float64 [m/s]  raw rainfall from PrecipEngine
        """
        if self._mode == 'none':
            return rain_1d
        elif self._mode == 'coefficient':
            return rain_1d * self._Cf_1d
        elif self._mode == 'raster':
            return self._interp_raster(t_seconds)
        elif self._mode == 'scs_cn':
            return self._get_scs_rate()
        elif self._mode == 'vsa_opm':
            return self._opm_effective_runoff(rain_1d)

    def get_effective_2d(self, t_seconds, rain_1d):
        """
        2-D runoff map (nrows × ncols), NaN outside watershed.
        Used by the animation script for overlay rendering.
        """
        eff_1d = self.get_effective_1d(t_seconds, rain_1d)
        out = np.full((self._nrows, self._ncols), np.nan, dtype=np.float64)
        out[self._s_rows, self._s_cols] = eff_1d
        return out

    def is_active(self, t_seconds):
        """True if any cell is generating non-zero runoff this step."""
        if self._mode == 'none':
            return False
        if self._mode == 'vsa_opm':
            # Runoff can come from the VSA, impervious cells, or Horton's own
            # infiltration-excess — any of which makes the engine "active".
            # Note: self._horton_on (not the sandbox's recharge cap) is the
            # right test here, since the sandbox cap alone (self._sandbox_
            # infilt_on) doesn't add any separate output runoff by itself.
            return (bool(self._vsa_mask.any())
                    or bool((self._imperv_1d > 0).any())
                    or self._horton_on)
        if self._mode == 'coefficient':
            return bool((self._Cf_1d > 0).any())
        if self._mode == 'raster':
            return bool((self._interp_raster(t_seconds) > 0).any())
        if self._mode == 'scs_cn':
            return bool((self._delta_Pe_m > 0).any())
        return False

    # ── Mode: 'coefficient' ───────────────────────────────────────────────────

    def _init_coefficient(self, cfg, grid_data):
        path = cfg.RUNOFF_COEFFICIENT_PATH
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"RUNOFF_COEFFICIENT_PATH '{path}' not found. "
                "Generate it with tools/generate_runoff_examples.py."
            )
        with rasterio.open(path) as src:
            arr = src.read(1).astype(np.float64)
            nd  = src.nodata
        if nd is not None:
            arr[arr == nd] = 0.0
        arr = np.clip(arr, 0.0, 1.0)
        s_rows, s_cols = grid_data['s_rows'], grid_data['s_cols']
        self._Cf_1d = arr[s_rows, s_cols]

    # ── Mode: 'raster' ────────────────────────────────────────────────────────

    def _init_raster(self, cfg, grid_data):
        manifest_path = cfg.RUNOFF_RASTER_MANIFEST
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(
                f"RUNOFF_RASTER_MANIFEST '{manifest_path}' not found."
            )
        mf = pd.read_csv(manifest_path)
        s_rows, s_cols = grid_data['s_rows'], grid_data['s_cols']

        self._raster_times = mf['time_s'].values.astype(np.float64)
        self._raster_cache = {}
        for _, row in mf.iterrows():
            t_s  = float(row['time_s'])
            fpath = row['filepath']
            with rasterio.open(fpath) as src:
                arr = src.read(1).astype(np.float64)
                nd  = src.nodata
            if nd is not None:
                arr[arr == nd] = 0.0
            arr = np.maximum(arr, 0.0)
            self._raster_cache[t_s] = arr[s_rows, s_cols]

    def _interp_raster(self, t_seconds):
        times = self._raster_times
        idx   = np.searchsorted(times, t_seconds, side='right') - 1
        idx   = int(np.clip(idx, 0, len(times) - 2))
        t0, t1 = times[idx], times[idx + 1]
        r0 = self._raster_cache[t0]
        r1 = self._raster_cache[t1]
        w  = (t_seconds - t0) / (t1 - t0) if t1 > t0 else 0.0
        return r0 + w * (r1 - r0)

    # ── Mode: 'scs_cn' ────────────────────────────────────────────────────────

    def _init_scs_cn(self, cfg, grid_data):
        path = cfg.RUNOFF_CN_PATH
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"RUNOFF_CN_PATH '{path}' not found. "
                "Generate it with tools/generate_runoff_examples.py."
            )
        with rasterio.open(path) as src:
            arr = src.read(1).astype(np.float64)
            nd  = src.nodata
        if nd is not None:
            arr[arr == nd] = 75.0
        arr = np.clip(arr, 1.0, 100.0)
        s_rows, s_cols = grid_data['s_rows'], grid_data['s_cols']
        CN_1d = arr[s_rows, s_cols]

        Ia_factor     = getattr(cfg, 'RUNOFF_SCS_Ia_FACTOR', 0.2)
        S_1d          = 25400.0 / CN_1d - 254.0          # [mm] max retention
        self._Ia_1d   = Ia_factor * S_1d                  # [mm] initial abstraction
        self._S_1d    = S_1d

        self._cumrain_mm  = np.zeros(self._n_cells, dtype=np.float64)
        self._Pe_mm_old   = np.zeros(self._n_cells, dtype=np.float64)
        self._scs_rate_ms = np.zeros(self._n_cells, dtype=np.float64)  # [m/s]

    def _scs_formula(self, P_mm):
        """SCS-CN accumulated effective rainfall [mm] from cumulative P [mm]."""
        excess = P_mm - self._Ia_1d
        return np.where(
            excess > 0,
            (excess ** 2) / (excess + self._S_1d),
            0.0,
        )

    def _update_scs_cn(self, rain_1d, dt):
        """Advance SCS-CN state and store instantaneous runoff rate [m/s]."""
        self._cumrain_mm += rain_1d * dt * 1000.0      # m/s → mm
        Pe_new = self._scs_formula(self._cumrain_mm)
        delta  = np.maximum(Pe_new - self._Pe_mm_old, 0.0)   # [mm] this step
        self._scs_rate_ms = (delta / 1000.0) / dt if dt > 0 \
            else np.zeros(self._n_cells)                       # [m/s]
        self._Pe_mm_old = Pe_new

    def _get_scs_rate(self):
        """Return SCS effective runoff rate [m/s] computed by last update_state."""
        return self._scs_rate_ms
