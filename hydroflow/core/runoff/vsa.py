# -*- coding: utf-8 -*-
"""
vsa.py — the VSA-OPM runoff mechanics (Pradhan & Ogden 2010).

VsaOpmMixin holds every method of the runoff engine that implements the
variable-source-area sandbox (single and per-polygon), Green-Ampt
infiltration-excess and the impervious/urban shedding composition.
It is mixed into RunoffEngine (runoff.engine); all state lives on the
engine instance.
"""

import os

import numpy as np
import rasterio

from ...utils import gpu_utils
from ..io_utils import raster_band_1d
from .soil import (
    OPM_Q_MIN,
    resolve_sd_params,
    resolve_zone_divides,
    per_zone_sd_from_raster,
    usda_psi_m,
)


class VsaOpmMixin:
    """VSA / Green-Ampt / impervious mechanics of the runoff engine."""


    def _opm_effective_runoff(self, rain_1d):
        """
        Unified per-cell effective runoff [m/s] for vsa_opm mode:

            runoff = rain · [ Imp + (1 − Imp) · max(in_VSA, infil_excess_frac) ]

        - Impervious fraction Imp always sheds rain (urban contributes regardless
          of A_t).
        - Pervious fraction sheds rain if the cell is in the VSA (saturation
          excess) OR rainfall beats the Green-Ampt infiltration capacity
          (Hortonian / infiltration excess).  `max` caps shedding at 100 % of rain.
        With OPM_INFILTRATION='none' and no impervious layer this reduces exactly
        to the original `rain · in_VSA`.
        """
        xp = self._xp
        # excess_frac is computed whenever the shared Green-Ampt machinery is
        # active (self._ga_active = horton_on OR sandbox_infilt_on), since the
        # sandbox's own recharge cap (_divide_infiltration) needs the same f_p
        # physics.  But it may only be ADDED to the output / reported as Horton
        # runoff when the user actually selected 'horton' in RUNOFF_MECHANISMS
        # (self._horton_on) — otherwise a user who only wants the sandbox
        # capped (without reporting Horton) would silently get Horton runoff
        # leaking into the output.
        if self._ga_active:
            f_p = self._ga_ksat * (1.0 + self._ga_psi * self._ga_dtheta0
                                   / xp.maximum(self._ga_F, self._GA_F_FLOOR))
            excess      = xp.maximum(rain_1d - f_p, 0.0)
            excess_frac = xp.where(rain_1d > 0.0,
                                   excess / xp.maximum(rain_1d, 1e-30), 0.0)
        else:
            excess_frac = 0.0
        effective_excess_frac = excess_frac if self._horton_on else 0.0
        pervious_frac = xp.where(self._vsa_mask, 1.0, effective_excess_frac)

        # ── Mechanism decomposition (per-cell rates [m/s]) ───────────────────
        # Stash the three runoff components so the router can accumulate the
        # Dunne / Horton / impervious split.  By construction
        #     imperv + dunne + horton == rain·(Imp + (1−Imp)·pervious_frac)
        # i.e. they sum EXACTLY to the effective runoff returned below.
        #   • impervious : urban shedding,            rain·Imp
        #   • Dunne      : saturation-excess (in VSA), rain·(1−Imp)         on VSA cells
        #   • Horton     : infiltration-excess,        rain·(1−Imp)·effective_excess_frac off VSA
        imp  = self._imperv_1d
        perv = 1.0 - imp
        self._last_imperv_rate = rain_1d * imp
        self._last_dunne_rate  = rain_1d * perv * xp.where(self._vsa_mask, 1.0, 0.0)
        self._last_horton_rate = rain_1d * perv * xp.where(self._vsa_mask, 0.0, effective_excess_frac)

        return rain_1d * (self._imperv_1d
                          + (1.0 - self._imperv_1d) * pervious_frac)

    def get_opm_diagnostics(self):
        """
        Return current OPM state as a dict (vsa_opm mode only).

        Per-polygon mode: z_m, SD_max_t, A_t_m2 are (n_polygons,) arrays.
        Single mode:      z_m, SD_max_t, A_t_m2 are scalars.
        VSA_m2 and VSA_fraction are always scalar (catchment-wide totals).
        """
        if self._mode != 'vsa_opm':
            return {}
        d = {
            "z_m":          self._opm_z.copy() if self._per_polygon else self._opm_z,
            "SD_max_t":     self._opm_SD_max.copy() if self._per_polygon else self._opm_SD_max,
            "A_t_m2":       self._opm_A_t.copy() if self._per_polygon else self._opm_A_t,
            "VSA_m2":       float(self._vsa_mask.sum()) * self._cell_area,
            "VSA_fraction": float(self._vsa_mask.sum()) / self._n_cells,
            "per_polygon":  self._per_polygon,
        }
        if self._per_polygon:
            d["n_polygons"] = self._n_polygons
        return d

    # ── Mode: 'vsa_opm' ──────────────────────────────────────────────────────

    def _init_vsa_opm(self, cfg, grid_data):
        """
        Initialise the OPM Variable Source Area engine.

        Implements Pradhan & Ogden (2010) Equations 4, 5, 10, 12.
        H_a is computed from Eq 4 using the initial A_t from Eq 10.

        Per-polygon mode (Thiessen / IDW):
            When cell_polygon is available in grid_data, each precipitation zone
            gets its own sandbox (z, SD_max, A_t).  The divide cell per zone is
            the cell with minimum flow accumulation within that zone.
            Catchment-wide constants (H_a, A_1, A_outlet) remain shared.

        Runoff mechanisms:
            ``RUNOFF_MECHANISMS`` (config) lists which of {'vsa','horton',
            'impervious'} are active.  They are orthogonal: VSA (Dunne) builds the
            saturated-area mask, Horton (Green-Ampt) the infiltration-excess
            fraction, impervious the urban shed fraction.  When 'vsa' is absent the
            whole OPM sandbox is skipped and the VSA mask is all-False, so runoff is
            rain·[Imp + (1−Imp)·infil_excess].  Absent key → legacy behaviour
            (VSA on; Horton from OPM_INFILTRATION; impervious from IMPERVIOUS_SOURCE).
        """
        # ── Mechanism selection (orthogonal toggles) ─────────────────────────
        mechs = getattr(cfg, 'RUNOFF_MECHANISMS', None)
        if mechs is None:
            mechs = ['vsa']
            if getattr(cfg, 'OPM_INFILTRATION', 'none').lower() == 'green_ampt':
                mechs.append('horton')
            if getattr(cfg, 'IMPERVIOUS_SOURCE', 'none').lower() != 'none':
                mechs.append('impervious')
        mechs = [str(m).lower().strip() for m in mechs]
        self._vsa_on        = 'vsa' in mechs
        self._horton_on     = 'horton' in mechs
        self._impervious_on = 'impervious' in mechs
        print(f"  RunoffEngine    |  mechanisms: "
              f"{[m for m in ('vsa', 'horton', 'impervious') if m in mechs] or 'none'}")

        Q_max = float(cfg.OPM_Q_MAX)
        if self._vsa_on and Q_max <= OPM_Q_MIN:
            raise ValueError(
                f"OPM_Q_MAX={Q_max} m³/s must be > {OPM_Q_MIN} m³/s (Q_min)."
            )

        cell_area = float(grid_data['cell_area'])
        cell_size = float(grid_data['cell_size'])
        slope_1d  = grid_data['slope_1d']
        faccum_1d = grid_data['faccum_1d']   # (n_cells,) cell counts

        self._cell_area    = cell_area
        self._cell_size    = cell_size

        # Upslope contributing area per cell [m²] — computed once
        self._upslope_area = faccum_1d * cell_area   # (n_cells,) [m²]

        # ── OPM scalars ───────────────────────────────────────────────────────
        params = resolve_sd_params(cfg, cell_size)
        SD_max_initial  = params['sd_max']
        sd_min          = params['sd_min']
        phi             = params['phi']
        ksat_ms         = params['ksat_ms']

        self._sd_min  = sd_min
        self._phi     = phi
        self._ksat_ms = ksat_ms

        # A_1: upslope area of a single divide cell [m²]
        A_1 = cell_area

        # A_outlet: total catchment area [m²]
        A_outlet = float(faccum_1d[-1]) * cell_area

        # Eq 10 — initial threshold contributing area from single discharge measurement.
        # Only meaningful when VSA is active (needs a valid Q_max); placeholder otherwise.
        if self._vsa_on:
            A_t_init = A_outlet / (1.0 - np.log(OPM_Q_MIN / Q_max))
            ratio    = A_t_init / (A_t_init - A_1)   # Eq 4 ratio (shared)
        else:
            A_t_init = A_outlet
            ratio    = 1.0

        # Store catchment-wide constants
        self._opm_A_1      = A_1
        self._opm_A_outlet = A_outlet
        self._opm_A_t_init = A_t_init

        # Extensibility hook: records which A_t initialisation method was used.
        self._at_method = 'single_measurement'

        # ── Backend (numpy / cupy) for the vectorised hot path ────────────────
        # _upslope_area is already CuPy in GPU mode (built from CuPy faccum_1d),
        # so get_xp picks the right module for every per-cell/per-zone op below.
        self._xp = gpu_utils.get_xp(self._upslope_area)
        xp = self._xp

        # ── Impervious fraction (urban areas shed rain regardless of A_t) ─────
        # Resolved only when the 'impervious' mechanism is active (skipping it also
        # avoids the LCZ/LULC download); otherwise zero everywhere.
        from ..routing import surface as _ru
        if self._impervious_on:
            imperv_np = _ru.resolve_impervious_fraction(cfg, grid_data)   # (n_cells,)
        else:
            imperv_np = np.zeros(self._n_cells, dtype=np.float64)
        self._imperv_1d = xp.asarray(imperv_np)

        # ── Green-Ampt per-cell infiltration setup ────────────────────────────
        # Two independent consumers of the same Green-Ampt physics:
        #   • self._horton_on         (RUNOFF_MECHANISMS) — is Horton's own
        #     infiltration-excess ADDED to the output runoff / partition?
        #   • self._sandbox_infilt_on (cfg.OPM_INFILTRATION, authoritative,
        #     independent of RUNOFF_MECHANISMS) — is the VSA sandbox's own
        #     recharge (see _divide_infiltration) CAPPED by infiltration
        #     capacity, or given rainfall uncapped?  This used to be silently
        #     tied to self._horton_on, which made 'vsa'-alone runs recharge the
        #     sandbox with 100% of rainfall (unphysical — see
        #     outputs collection/combinations_100m/figures/mechanism_spatial/).
        # self._ga_active gates whether the shared GA parameter machinery
        # (psi/ksat/dtheta0/F) is resolved and advanced at all, since both
        # consumers above read the same underlying per-cell state.
        self._sandbox_infilt_on = str(getattr(cfg, 'OPM_INFILTRATION', 'none')).lower() == 'green_ampt'
        self._ga_active         = self._horton_on or self._sandbox_infilt_on
        self._GA_F_FLOOR        = 1e-9   # m — floor on F so f_p is finite at F=0
        if self._ga_active:
            # Wetting-front suction ψ [m] per cell (scalar or SoilGrids texture).
            psi_m = self._resolve_ga_psi_m(
                cfg, grid_data, float(getattr(cfg, 'OPM_GA_SUCTION_M', 0.15)))
            # Vertical surface infiltration capacity (NOT the lateral OPM_K_SAT).
            kv_scalar = float(getattr(cfg, 'OPM_GA_KSAT_MMHR', 12.0))   # mm/hr

            s_rows = grid_data['s_rows']
            s_cols = grid_data['s_cols']

            # Per-cell vertical Ksat [mm/hr].  Uniform scalar, or gridded
            # HiHydroSoil v2.0 Ksat (source='gee'/'raster'); nodata→scalar.
            kv_mmhr = self._resolve_ga_ksat_mmhr(cfg, grid_data, kv_scalar)
            kv_ms_1d = kv_mmhr / 1000.0 / 3600.0                       # → m/s

            # Root-zone depth Z_r per cell from the SAME land-cover source that
            # built the SERVES deficit raster, so Δθ₀ = deficit / Z_r is consistent.
            _n_source = getattr(cfg, 'MANNINGS_N_SOURCE', 'lulc').lower()
            _lc_src   = 'lcz' if _n_source == 'lcz' else 'lulc'
            zr_default = float(getattr(cfg, 'OPM_SD_MAX_INITIAL', 0.5)) or 0.5
            zr_np = np.maximum(
                _ru.resolve_lulc_field(cfg, grid_data, 'root_zone_depth_m',
                                       zr_default, _lc_src), 1e-3)

            # Δθ₀ = initial moisture deficit (porosity − θ) per cell.
            #   From the SERVES deficit raster: Δθ₀ = deficit / Z_r.
            #   Fallback (no raster / nodata): watershed-scalar SD_max ÷ mean Z_r.
            _fallback = float(SD_max_initial) / float(zr_np.mean())
            dtheta0_np = None
            deficit_raster = params.get('deficit_raster')
            if deficit_raster:
                dfc = raster_band_1d(deficit_raster, s_rows, s_cols)
                if dfc is not None:
                    with np.errstate(invalid='ignore', divide='ignore'):
                        dtheta0_np = dfc / zr_np
            if dtheta0_np is None:
                dtheta0_np = np.full(self._n_cells, _fallback, dtype=np.float64)
            _bad = ~np.isfinite(dtheta0_np)
            if _bad.any():
                dtheta0_np[_bad] = _fallback
            dtheta0_np = np.clip(dtheta0_np, 0.0, 1.0)

            self._ga_psi     = xp.asarray(psi_m)
            self._ga_ksat    = xp.asarray(kv_ms_1d)
            self._ga_dtheta0 = xp.asarray(dtheta0_np)
            self._ga_F       = xp.zeros(self._n_cells, dtype=np.float64)
            print(f"  Green-Ampt    |  K_v(mm/hr)=[{kv_mmhr.min():.2f}, "
                  f"{kv_mmhr.max():.2f}] mean={kv_mmhr.mean():.2f}"
                  f"  psi(m)=[{psi_m.min():.3f}, {psi_m.max():.3f}]"
                  f"  dtheta0=[{dtheta0_np.min():.3f}, {dtheta0_np.max():.3f}]")
        else:
            self._ga_psi = self._ga_ksat = self._ga_dtheta0 = self._ga_F = None

        # ── VSA (Dunne) sandbox ───────────────────────────────────────────────
        # When the 'vsa' mechanism is off there is no saturation-excess: the VSA
        # mask is permanently empty and the OPM sandbox is never built/updated, so
        # runoff = rain·[Imp + (1−Imp)·infil_excess].  The per-cell Green-Ampt F
        # still advances (Horton needs it); only the divide sandbox is skipped.
        if not self._vsa_on:
            self._per_polygon = False
            self._vsa_mask    = xp.zeros(self._n_cells, dtype=bool)
            self._opm_z = self._opm_SD_max = self._opm_A_t = 0.0
            print("  OPM           |  VSA disabled — runoff from impervious + "
                  "infiltration-excess only (no saturation-excess sandbox)")
            return

        # ── Per-polygon vs single-sandbox branching ───────────────────────────
        cell_polygon = grid_data.get('cell_polygon')
        use_per_polygon = getattr(cfg, 'OPM_PER_POLYGON', True)

        if cell_polygon is not None and use_per_polygon:
            # ── Per-polygon mode ──────────────────────────────────────────────
            # Each precipitation zone (nearest-gauge region) gets its own
            # sandbox state so that spatially variable rainfall drives local
            # VSA expansion independently.
            cell_polygon = np.asarray(cell_polygon).ravel()
            # One zone per precipitation gauge — not cell_polygon.max()+1, which
            # undercounts when trailing gauges have no nearest cell (e.g. IMERG
            # edge pixels).  Aligning n_polygons with the gauge count keeps the
            # per-polygon state arrays in step with sd_max_per_polygon (one value
            # per gauge from SERVES) and with downstream consumers that index by
            # gauge id.  Empty zones are handled by the fallback divide below.
            precip_engine = grid_data.get('precip_engine')
            n_gauges      = getattr(precip_engine, '_n_gauges', None)
            n_polygons    = int(n_gauges) if n_gauges \
                else int(cell_polygon.max()) + 1

            # Ensure NumPy for the init loop (faccum_1d / slope_1d may be CuPy)
            _to_np = lambda a: a.get() if hasattr(a, 'get') else np.asarray(a)
            faccum_np = _to_np(faccum_1d)
            slope_np  = _to_np(slope_1d)
            dem = grid_data['dem']
            s_rows = grid_data['s_rows']
            s_cols = grid_data['s_cols']
            transform = grid_data['transform']

            divide_idx, divide_candidates, divide_xy = resolve_zone_divides(
                cell_polygon, n_polygons, faccum_np, dem, s_rows, s_cols, transform)
            slope_divide = slope_np[divide_idx]

            self._per_polygon          = True
            self._n_polygons           = n_polygons
            self._cell_polygon         = cell_polygon
            self._polygon_divide_idx   = divide_idx
            self._polygon_slope_divide = slope_divide

            # Per-zone SD_max from the deficit raster, reduced over each zone's
            # OWN watershed cells (same partition as rainfall).  Zones with no
            # deficit data fall back to the watershed SD_max.
            deficit_raster = params.get('deficit_raster')
            reducer = getattr(cfg, 'OPM_SD_REDUCER', 'mean').lower()
            if deficit_raster:
                sd_init_arr = per_zone_sd_from_raster(
                    deficit_raster, cell_polygon, n_polygons,
                    s_rows, s_cols, reducer, sd_min, SD_max_initial,
                    divide_candidates=divide_candidates, divide_xy=divide_xy)
                n_real = int((np.abs(sd_init_arr - SD_max_initial) > 1e-9).sum())
                print(f"                |  Per-zone SD ({reducer}) over watershed "
                      f"cells: {n_real}/{n_polygons} zones populated, "
                      f"range=[{sd_init_arr.min():.3f}, {sd_init_arr.max():.3f}] m")
            else:
                sd_init_arr = np.full(n_polygons, SD_max_initial,
                                      dtype=np.float64)
            self._SD_max_initial = sd_init_arr

            self._ksat_ms = np.full(n_polygons, ksat_ms, dtype=np.float64)

            Rf_init_arr = sd_min / sd_init_arr
            H_a_arr = ratio * np.log(Rf_init_arr)
            self._opm_H_a = H_a_arr

            # Per-polygon state arrays
            self._opm_z      = np.zeros(n_polygons, dtype=np.float64)
            self._opm_SD_max = sd_init_arr.copy()
            self._opm_A_t    = np.full(n_polygons, A_t_init, dtype=np.float64)

            # Initial VSA mask
            A_t_per_cell = self._opm_A_t[cell_polygon]
            if hasattr(self._upslope_area, '__cuda_array_interface__'):
                import cupy as _cp
                A_t_per_cell = _cp.asarray(A_t_per_cell)
            self._vsa_mask = self._upslope_area > A_t_per_cell

            print(f"  OPM           |  A_outlet={A_outlet:.3e} m²"
                  f"  A_t_init={A_t_init:.3e} m²")
            print(f"                |  H_a={H_a_arr}  phi={phi}  Q_max={Q_max} m³/s")
            print(f"                |  SD_max_init={sd_init_arr}")
            print(f"                |  Per-polygon mode: {n_polygons} zones")
            print(f"                |  Initial VSA={self._vsa_mask.sum():,} cells"
                  f" ({100*self._vsa_mask.mean():.1f}% of watershed)")

            # Move per-zone state + indices onto the active backend so the
            # vectorised sandbox update runs entirely in numpy OR cupy.
            self._cell_polygon         = xp.asarray(cell_polygon)
            self._polygon_divide_idx   = xp.asarray(divide_idx)
            self._polygon_slope_divide = xp.asarray(slope_divide)
            self._SD_max_initial       = xp.asarray(sd_init_arr)
            self._ksat_ms              = xp.asarray(self._ksat_ms)
            self._opm_H_a              = xp.asarray(H_a_arr)
            self._opm_z                = xp.asarray(self._opm_z)
            self._opm_SD_max           = xp.asarray(self._opm_SD_max)
            self._opm_A_t              = xp.asarray(self._opm_A_t)
        else:
            # ── Single-sandbox mode (uniform rainfall) ────────────────────────
            self._per_polygon  = False
            _to_np = lambda a: a.get() if hasattr(a, 'get') else np.asarray(a)
            faccum_np = _to_np(faccum_1d)
            slope_np  = _to_np(slope_1d)
            dem = grid_data['dem']
            s_rows = grid_data['s_rows']
            s_cols = grid_data['s_cols']
            min_fa = faccum_np.min()
            candidates = np.where(faccum_np == min_fa)[0]
            elev = dem[s_rows[candidates], s_cols[candidates]]
            divide_cell = candidates[elev.argmax()]
            self._slope_divide = float(slope_np[divide_cell])
            self._divide_cell  = int(divide_cell)

            self._SD_max_initial = SD_max_initial  # scalar
            Rf_init = sd_min / SD_max_initial
            H_a = ratio * np.log(Rf_init)
            self._opm_H_a = H_a  # scalar

            # Scalar state variables
            self._opm_z      = 0.0
            self._opm_SD_max = SD_max_initial
            self._opm_A_t    = A_t_init

            self._vsa_mask = self._upslope_area > A_t_init

            print(f"  OPM           |  A_outlet={A_outlet:.3e} m²"
                  f"  A_t_init={A_t_init:.3e} m²")
            print(f"                |  H_a={H_a:.4f}  SD_max_init={SD_max_initial} m"
                  f"  phi={phi}  Q_max={Q_max} m³/s")
            print(f"                |  Initial VSA={self._vsa_mask.sum():,} cells"
                  f" ({100*self._vsa_mask.mean():.1f}% of watershed)")

    # ── OPM sandbox update ────────────────────────────────────────────────────

    def _resolve_ga_ksat_mmhr(self, cfg, grid_data, kv_scalar):
        """
        Per-cell Green-Ampt vertical Ksat [mm/hr].

          'scalar' → uniform kv_scalar.
          'gee'    → HiHydroSoil v2.0 Ksat downloaded aligned to the routing DEM
                     (cached); nodata/≤0 cells fall back to kv_scalar.
          'raster' → pre-computed mm/hr GeoTIFF at OPM_GA_KSAT_RASTER.
        OPM_GA_KSAT_SCALE multiplies the gridded values (calibration knob).
        """
        n      = self._n_cells
        source = getattr(cfg, 'OPM_GA_KSAT_SOURCE', 'scalar').lower()
        scale  = float(getattr(cfg, 'OPM_GA_KSAT_SCALE', 1.0))
        if source == 'scalar':
            return np.full(n, kv_scalar, dtype=np.float64)

        path = getattr(cfg, 'OPM_GA_KSAT_RASTER', None) \
            or os.path.join(getattr(cfg, 'OUTPUT_DIR', 'output/'), 'ksat_hihydro.tif')

        if source == 'gee':
            try:
                from ...gee.serves_gee import download_ksat_raster
                got = download_ksat_raster(
                    dem_path=cfg.ROUTING_DEM_PATH,
                    watershed_geojson_path=getattr(
                        cfg, 'OPM_WATERSHED_GEOJSON', 'output/watershed.geojson'),
                    output_path=path,
                    project=getattr(cfg, 'GEE_PROJECT', None))
            except Exception as exc:
                print(f"  [WARN] Ksat download failed ({exc}); "
                      f"scalar K_v={kv_scalar} mm/hr")
                got = None
            if not got:
                return np.full(n, kv_scalar, dtype=np.float64)
        elif source == 'raster':
            if not (path and os.path.isfile(path)):
                print(f"  [WARN] OPM_GA_KSAT_RASTER not found: {path}; "
                      f"scalar K_v={kv_scalar} mm/hr")
                return np.full(n, kv_scalar, dtype=np.float64)
        else:
            raise ValueError(f"Unknown OPM_GA_KSAT_SOURCE: '{source}'")

        kv = raster_band_1d(path, grid_data['s_rows'], grid_data['s_cols'])
        if kv is None:
            print("  [WARN] Ksat raster grid ≠ routing grid; "
                  f"scalar K_v={kv_scalar} mm/hr")
            return np.full(n, kv_scalar, dtype=np.float64)
        kv = kv * scale
        bad = ~np.isfinite(kv) | (kv <= 0.0)
        if bad.any():
            # Fill HiHydroSoil gaps with the valid-cell median (consistent with
            # the surrounding soil) rather than the fixed scalar, which can be
            # an order of magnitude off and create a spurious discontinuity.
            good = ~bad
            fill = float(np.median(kv[good])) if good.any() else kv_scalar
            kv[bad] = fill
            print(f"  Ksat gaps     |  {int(bad.sum()):,}/{n:,} nodata cells "
                  f"filled with median {fill:.2f} mm/hr")
        return kv

    def _resolve_ga_psi_m(self, cfg, grid_data, psi_scalar):
        """
        Per-cell Green-Ampt suction ψ [m].

          'scalar'  → uniform psi_scalar.
          'texture' → SoilGrids sand/clay (2-band raster, DEM-aligned) → USDA
                      class → Rawls (1983) ψ.  nodata cells fall back to psi_scalar.
        """
        n      = self._n_cells
        source = getattr(cfg, 'OPM_GA_SUCTION_SOURCE', 'scalar').lower()
        if source != 'texture':
            return np.full(n, psi_scalar, dtype=np.float64)

        path = os.path.join(getattr(cfg, 'OUTPUT_DIR', 'output/'), 'texture_sandclay.tif')
        try:
            from ...gee.serves_gee import download_texture_raster
            got = download_texture_raster(
                dem_path=cfg.ROUTING_DEM_PATH,
                watershed_geojson_path=getattr(
                    cfg, 'OPM_WATERSHED_GEOJSON', 'output/watershed.geojson'),
                output_path=path,
                soil_depth_band=getattr(cfg, 'OPM_SOILGRIDS_DEPTH', 'b30'),
                project=getattr(cfg, 'GEE_PROJECT', None))
        except Exception as exc:
            print(f"  [WARN] texture download failed ({exc}); scalar psi={psi_scalar} m")
            got = None
        if not got:
            return np.full(n, psi_scalar, dtype=np.float64)

        with rasterio.open(got) as src:
            sand2d = src.read(1).astype(np.float64)
            clay2d = src.read(2).astype(np.float64)
            nodata = src.nodata
        _to_np = lambda a: a.get() if hasattr(a, 'get') else np.asarray(a)
        sr = _to_np(grid_data['s_rows']); sc = _to_np(grid_data['s_cols'])
        if sr.max() >= sand2d.shape[0] or sc.max() >= sand2d.shape[1]:
            print("  [WARN] texture raster grid ≠ routing grid; "
                  f"scalar psi={psi_scalar} m")
            return np.full(n, psi_scalar, dtype=np.float64)

        sand = sand2d[sr, sc]; clay = clay2d[sr, sc]
        bad = ~np.isfinite(sand) | ~np.isfinite(clay) | (sand + clay <= 0)
        if nodata is not None:
            bad |= (sand == nodata) | (clay == nodata)
        psi = usda_psi_m(np.clip(sand, 0, 100), np.clip(clay, 0, 100))
        if bad.any():
            psi[bad] = psi_scalar
        print(f"  GA suction    |  psi(m)=[{psi.min():.3f}, {psi.max():.3f}] "
              f"mean={psi.mean():.3f}  (texture/Rawls)")
        return psi

    def _update_opm_sandbox(self, rain_1d, dt):
        """Advance OPM sandbox state by one timestep (dispatches to mode)."""
        if self._vsa_on:
            if self._per_polygon:
                self._update_opm_sandbox_per_polygon(rain_1d, dt)
            else:
                self._update_opm_sandbox_single(rain_1d, dt)
        # Advance per-cell Green-Ampt cumulative infiltration AFTER the sandbox
        # has used the current-step F (forward Euler — matches get_effective_1d).
        # Runs even with VSA off, since the Horton mechanism depends on F.
        self._update_ga_F(rain_1d, dt)

    def _update_ga_F(self, rain_1d, dt):
        """Advance per-cell Green-Ampt cumulative infiltration F [m]."""
        if not self._ga_active:
            return
        xp  = self._xp
        f_p = self._ga_ksat * (1.0 + self._ga_psi * self._ga_dtheta0
                               / xp.maximum(self._ga_F, self._GA_F_FLOOR))
        f = xp.minimum(rain_1d, f_p)          # infiltration rate on pervious soil
        self._ga_F = self._ga_F + f * dt

    def _divide_infiltration(self, rain_1d):
        """
        Effective recharge rate [m/s] feeding each zone's sandbox at its divide
        cell.  Gated by self._sandbox_infilt_on (cfg.OPM_INFILTRATION),
        independent of whether Horton's own runoff is being reported
        (self._horton_on) — the sandbox's water balance should reflect a real
        infiltration limit whenever the user asks for one, regardless of which
        mechanisms are selected in RUNOFF_MECHANISMS.  Impervious cells never
        recharge the water table, capped or not.
        Returns a (n_polygons,) array indexed by polygon_divide_idx.
        """
        xp    = self._xp
        idx   = self._polygon_divide_idx
        P_div = rain_1d[idx]
        if not self._sandbox_infilt_on:
            return (1.0 - self._imperv_1d[idx]) * P_div
        f_p = self._ga_ksat[idx] * (1.0 + self._ga_psi[idx] * self._ga_dtheta0[idx]
                                    / xp.maximum(self._ga_F[idx], self._GA_F_FLOOR))
        f_div = xp.minimum(P_div, f_p)
        return (1.0 - self._imperv_1d[idx]) * f_div

    def _update_opm_sandbox_per_polygon(self, rain_1d, dt):
        """
        Per-polygon sandbox update — fully vectorised over zones (numpy / cupy).

        Each precipitation zone has its own divide cell, saturated-zone thickness
        z[p], SD_max[p] and threshold area A_t[p].  Only the *infiltrated* depth
        at the divide raises z (Green-Ampt); the VSA mask is then rebuilt from
        per-cell polygon assignments.
        """
        xp    = self._xp
        f_div = self._divide_infiltration(rain_1d)             # (n_polygons,)

        q_b = (self._ksat_ms * self._polygon_slope_divide
               * self._opm_z * self._cell_size)
        dV  = (f_div * self._cell_area - q_b) * dt
        dz  = dV / (self._cell_area * self._phi)
        self._opm_z = xp.maximum(0.0, self._opm_z + dz)

        self._opm_SD_max = xp.maximum(self._sd_min,
                                      self._SD_max_initial - self._opm_z)

        Rf_t  = self._sd_min / self._opm_SD_max
        denom = self._opm_H_a - xp.log(Rf_t)
        # Guard the near-zero denominator before dividing (xp.where evaluates
        # both branches, so the divisor must be finite even where unused).
        denom_safe = xp.where(xp.abs(denom) < 1e-12, 1.0, denom)
        new_A_t    = xp.where(xp.abs(denom) < 1e-12, self._opm_A_t_init,
                              self._opm_H_a * self._opm_A_1 / denom_safe)
        self._opm_A_t = xp.clip(new_A_t, self._opm_A_1, self._opm_A_outlet)

        # Vectorised VSA mask rebuild: each cell uses its polygon's A_t
        A_t_per_cell   = self._opm_A_t[self._cell_polygon]
        self._vsa_mask = self._upslope_area > A_t_per_cell

    def _update_opm_sandbox_single(self, rain_1d, dt):
        """
        Single-sandbox update (original Pradhan & Ogden 2010, Eq 12), with
        optional Green-Ampt limiting of the divide-cell infiltration.
        """
        di    = self._divide_cell
        P_div = float(rain_1d[di])
        if self._sandbox_infilt_on:
            F_d  = float(self._ga_F[di])
            kv   = float(self._ga_ksat[di])
            dth0 = float(self._ga_dtheta0[di])
            f_p  = kv * (1.0 + float(self._ga_psi[di]) * dth0 / max(F_d, self._GA_F_FLOOR))
            f_div = (1.0 - float(self._imperv_1d[di])) * min(P_div, f_p)
        else:
            f_div = (1.0 - float(self._imperv_1d[di])) * P_div

        q_b = self._ksat_ms * self._slope_divide * self._opm_z * self._cell_size
        dV  = (f_div * self._cell_area - q_b) * dt
        dz  = dV / (self._cell_area * self._phi)
        self._opm_z = max(0.0, self._opm_z + dz)

        self._opm_SD_max = max(self._sd_min, self._SD_max_initial - self._opm_z)

        Rf_t  = self._sd_min / self._opm_SD_max
        denom = self._opm_H_a - np.log(Rf_t)
        if abs(denom) < 1e-12:
            new_A_t = self._opm_A_t_init
        else:
            new_A_t = self._opm_H_a * self._opm_A_1 / denom

        self._opm_A_t = float(np.clip(new_A_t, self._opm_A_1, self._opm_A_outlet))
        self._vsa_mask = self._upslope_area > self._opm_A_t
