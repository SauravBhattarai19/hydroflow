"""
vsa_opm.py
==========
Standalone One-Parameter Model (OPM) for Variable Source Area (VSA) runoff.

Implements Pradhan & Ogden (2010):
  - Eq 10: initial threshold area A_t from single discharge measurement
  - Eq  4: constant scaling factor H_a
  - Eq 12: Sandbox water balance at the catchment divide (Darcy drainage)
  - Eq  5: dynamic threshold area A_t(t)
  - Eq  9: VSA = cells with upslope_area > A_t

Usage
-----
    python vsa_opm.py

Reads:  output/clipped_dem.tif, output/clipped_flow_accumulation.tif,
        output/flow_direction.tif, output/watershed.tif,
        precipitation/gauges.csv, precipitation/timeseries.csv  (via config.py)
Writes: output/vsa_opm_results.csv

All parameters are read from config.py:
    OPM_SD_MAX_INITIAL  [m]      initial max soil moisture deficit
    OPM_Q_MAX           [m³/s]   initial observed outlet discharge
    OPM_PHI             [-]      drainable porosity (default 0.35)
"""

import os
import time
import math

import numpy as np
import pandas as pd

from .routing import terrain as ru
from .precip import PrecipEngine

# ── OPM physical constants ────────────────────────────────────────────────────
SD_MIN = 0.001          # m   minimum saturation deficit (default floor)
Q_MIN  = 0.001          # m³/s  minimum discharge (Eq 10)


def _resolve_sd_params(cfg, cell_size):
    """Resolve OPM soil parameters — delegates to runoff.soil.resolve_sd_params."""
    from .runoff.soil import resolve_sd_params as _resolve
    return _resolve(cfg, cell_size)


# ─────────────────────────────────────────────────────────────────────────────
def run_opm(cfg):
    """
    Run the OPM/VSA loop and return a DataFrame of results.

    Parameters
    ----------
    cfg : module  (imported config)

    Returns
    -------
    df : pd.DataFrame  with columns: time_s, SD_max_t, A_t_m2, VSA_m2
    """
    print("=" * 60)
    print("OPM / VSA  —  Variable Source Area Runoff Generation")
    print("    Pradhan & Ogden (2010)")
    print("=" * 60)

    # ── 1. Load rasters (reuses routing_utils) ───────────────────────────────
    dem, fdir, faccum, ws_mask, transform, nodata_dem, cell_size = \
        ru.load_rasters(cfg)
    nrows, ncols = dem.shape
    cell_area    = cell_size ** 2

    # ── 2. Slope grid ────────────────────────────────────────────────────────
    print("  Computing slope grid...")
    slope_2d = ru.compute_slope_grid(
        dem, fdir, ws_mask, cell_size, cfg.MIN_SLOPE, nodata_dem
    )

    # ── 3. Topological order ─────────────────────────────────────────────────
    print("  Building topological order...")
    s_rows, s_cols, outlet_rc = ru.topological_order(faccum, fdir, ws_mask)
    n_cells = len(s_rows)

    # ── 4. Extract 1-D arrays ────────────────────────────────────────────────
    slope_1d  = slope_2d[s_rows, s_cols]
    faccum_1d = faccum[s_rows, s_cols].astype(np.float64)  # cell counts

    # Divide cell: highest elevation among cells with minimum faccum
    min_fa = faccum_1d.min()
    candidates = np.where(faccum_1d == min_fa)[0]
    elev_cand = dem[s_rows[candidates], s_cols[candidates]]
    divide_cell = candidates[elev_cand.argmax()]
    slope_divide = float(slope_1d[divide_cell])

    # Upslope contributing area per cell [m²]
    upslope_area_1d = faccum_1d * cell_area

    # ── 5. Build minimal grid_data for PrecipEngine ──────────────────────────
    grid_data = {
        "s_rows"   : s_rows,
        "s_cols"   : s_cols,
        "nrows"    : nrows,
        "ncols"    : ncols,
        "n_cells"  : n_cells,
        "ws_mask"  : ws_mask,
        "transform": transform,
    }
    print("  Building precipitation engine...")
    precip_engine = PrecipEngine(cfg, grid_data)

    # ── 6. OPM Initialisation ─────────────────────────────────────────────────

    params = _resolve_sd_params(cfg, cell_size)
    SD_max_initial  = params['sd_max']
    sd_min          = params['sd_min']
    phi             = params['phi']
    deficit_raster  = params['deficit_raster']
    K_MS            = params['ksat_ms']
    Q_max = float(cfg.OPM_Q_MAX)

    if Q_max <= Q_MIN:
        raise ValueError(
            f"OPM_Q_MAX={Q_max} m³/s must be > {Q_MIN} m³/s (Q_min constant)."
        )

    # A_1: upslope area of single divide cell [m²]
    A_1 = cell_area

    # A_outlet: total catchment area [m²] (outlet cell has highest faccum)
    A_outlet = float(faccum[outlet_rc]) * cell_area

    # Eq 10 — initial threshold contributing area (single discharge measurement)
    A_t_init = A_outlet / (1.0 - math.log(Q_MIN / Q_max))

    ratio = A_t_init / (A_t_init - A_1)

    print()
    print(f"  OPM parameters:")
    print(f"    SD_max_initial = {SD_max_initial}")
    print(f"    Q_max          = {Q_max:.4f} m³/s")
    print(f"    phi (porosity) = {phi:.3f}")
    print(f"    K_sat          = {K_MS*86400:.1f} m/day = {K_MS:.2e} m/s")
    print(f"    A_1            = {A_1:.1f} m²  (single cell)")
    print(f"    A_outlet       = {A_outlet:.3e} m²  (total catchment)")
    print(f"    A_t_init [Eq10]= {A_t_init:.3e} m²")

    # ── Per-polygon vs single-sandbox ─────────────────────────────────────
    cell_polygon = precip_engine.cell_polygon
    use_per_polygon = getattr(cfg, 'OPM_PER_POLYGON', True)

    if cell_polygon is not None and use_per_polygon:
        # Per-polygon mode: each precipitation zone gets its own sandbox
        cell_polygon = np.asarray(cell_polygon).ravel()
        n_polygons   = int(cell_polygon.max()) + 1

        from .runoff.soil import resolve_zone_divides, per_zone_sd_from_raster
        poly_divide_idx, poly_divide_candidates, poly_divide_xy = resolve_zone_divides(
            cell_polygon, n_polygons, faccum_1d, dem, s_rows, s_cols, transform)
        poly_slope_divide = slope_1d[poly_divide_idx]

        # Per-zone SD_max from the deficit raster over each zone's own watershed
        # cells — the SAME path as the production engine
        # (runoff_input._init_vsa_opm), so this diagnostic matches the real run
        # instead of falling back to a uniform watershed SD_max.
        reducer = getattr(cfg, 'OPM_SD_REDUCER', 'mean').lower()
        if deficit_raster:
            sd_init_arr = per_zone_sd_from_raster(
                deficit_raster, cell_polygon, n_polygons,
                s_rows, s_cols, reducer, sd_min, SD_max_initial,
                divide_candidates=poly_divide_candidates, divide_xy=poly_divide_xy)
        else:
            sd_init_arr = np.full(n_polygons, SD_max_initial, dtype=np.float64)

        ksat_arr = np.full(n_polygons, K_MS, dtype=np.float64)

        Rf_init_arr = sd_min / sd_init_arr
        H_a_arr = ratio * np.log(Rf_init_arr)

        z_arr      = np.zeros(n_polygons)
        SD_max_arr = sd_init_arr.copy()
        A_t_arr    = np.full(n_polygons, A_t_init)

        per_polygon = True
        print(f"    Per-polygon mode: {n_polygons} zones")
        print(f"    SD_max_init = {sd_init_arr}")
        print(f"    K_sat       = {[f'{v*86400:.2f}' for v in ksat_arr]} m/day")
        print(f"    H_a         = {H_a_arr}")
    else:
        per_polygon = False

        Rf_init = sd_min / SD_max_initial
        H_a = ratio * math.log(Rf_init)
        sd_init_arr = None
        H_a_arr = None

        print(f"    slope_divide   = {slope_divide:.6f} m/m")
        print(f"    Rf_init        = {Rf_init:.6f}")
        print(f"    H_a    [Eq 4]  = {H_a:.6f}  (negative)")

        z        = 0.0
        A_t      = A_t_init
        SD_max_t = SD_max_initial

    print()

    # ── 7. Time loop ──────────────────────────────────────────────────────────
    dt      = float(cfg.TIME_STEP_SECONDS)
    n_steps = int(cfg.TOTAL_SIMULATION_TIME_HOURS * 3600.0 / dt)

    _out_interval = getattr(cfg, 'OUTPUT_INTERVAL_SECONDS', None)
    write_every   = max(1, round((_out_interval or dt) / dt))

    print(f"  Time loop: {n_steps:,} steps × {dt} s  "
          f"(recording every {write_every} steps)\n")

    results     = []
    t_wall_start = time.time()

    for step in range(n_steps):
        t_s = step * dt

        # ── Get rainfall for this step ─────────────────────────────────────
        rain_1d = precip_engine.get_field_1d(t_s)    # [m/s], (n_cells,)

        if per_polygon:
            # ── Per-polygon VSA ────────────────────────────────────────────
            A_t_per_cell = A_t_arr[cell_polygon]
            vsa_mask     = upslope_area_1d > A_t_per_cell
            VSA_m2       = float(vsa_mask.sum()) * cell_area

            # Record (use mean SD_max and A_t for backward-compatible CSV)
            if (step + 1) % write_every == 0 or step == n_steps - 1:
                results.append({
                    "time_s"  : t_s + dt,
                    "SD_max_t": float(SD_max_arr.mean()),
                    "A_t_m2"  : float(A_t_arr.mean()),
                    "VSA_m2"  : VSA_m2,
                })

            # Update each polygon's sandbox independently
            for p in range(n_polygons):
                P_div = float(rain_1d[poly_divide_idx[p]])
                z_p   = z_arr[p]

                q_b = ksat_arr[p] * poly_slope_divide[p] * z_p * cell_size
                dV  = (P_div * cell_area - q_b) * dt
                dz  = dV / (cell_area * phi)
                z_p = max(0.0, z_p + dz)
                z_arr[p] = z_p

                sd_p = max(sd_min, sd_init_arr[p] - z_p)
                SD_max_arr[p] = sd_p

                Rf_t  = sd_min / sd_p
                ha_p  = H_a_arr[p]
                denom = ha_p - math.log(Rf_t)
                if abs(denom) < 1e-12:
                    pass   # keep current A_t_arr[p]
                else:
                    new_A_t = ha_p * A_1 / denom
                    A_t_arr[p] = float(np.clip(new_A_t, A_1, A_outlet))

            # Progress (use mean values for display)
            SD_max_t = float(SD_max_arr.mean())
            A_t      = float(A_t_arr.mean())

        else:
            # ── Single-sandbox VSA (original) ──────────────────────────────
            P_divide = float(rain_1d[0])

            vsa_mask = upslope_area_1d > A_t
            VSA_m2   = float(vsa_mask.sum()) * cell_area

            if (step + 1) % write_every == 0 or step == n_steps - 1:
                results.append({
                    "time_s"  : t_s + dt,
                    "SD_max_t": SD_max_t,
                    "A_t_m2"  : A_t,
                    "VSA_m2"  : VSA_m2,
                })

            q_b = K_MS * slope_divide * z * cell_size
            dV  = (P_divide * cell_area - q_b) * dt
            dz  = dV / (cell_area * phi)
            z   = max(0.0, z + dz)

            SD_max_t = max(sd_min, SD_max_initial - z)

            Rf_t  = sd_min / SD_max_t
            denom = H_a - math.log(Rf_t)
            if abs(denom) < 1e-12:
                pass
            else:
                new_A_t = H_a * A_1 / denom
                A_t     = float(np.clip(new_A_t, A_1, A_outlet))

        # Progress reporting
        if (step + 1) % max(1, n_steps // 10) == 0:
            elapsed = time.time() - t_wall_start
            pct     = 100.0 * (step + 1) / n_steps
            print(f"  {pct:5.1f}%  t={t_s/3600:.3f}h  "
                  f"SD_max={SD_max_t:.5f} m  "
                  f"A_t={A_t:.3e} m²  "
                  f"VSA={VSA_m2/1e6:.2f} km²  "
                  f"wall={elapsed:.1f}s")

    elapsed_total = time.time() - t_wall_start
    print(f"\n  OPM loop finished in {elapsed_total:.1f}s  "
          f"({len(results)} records)\n")

    # ── 8. Write output CSV ───────────────────────────────────────────────────
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(cfg.OUTPUT_DIR, "vsa_opm_results.csv")
    df = pd.DataFrame(results, columns=["time_s", "SD_max_t", "A_t_m2", "VSA_m2"])
    df.to_csv(out_path, index=False, float_format="%.6g")

    # Summary
    if per_polygon:
        z_final = float(z_arr.mean())
    else:
        z_final = z
    print(f"  Results written → {out_path}")
    print(f"  Final state  :  z={z_final:.4f} m  "
          f"SD_max={SD_max_t:.5f} m  "
          f"A_t={A_t:.3e} m²  "
          f"VSA={VSA_m2/1e6:.3f} km²")
    print(f"  Max VSA: {df['VSA_m2'].max()/1e6:.3f} km²  "
          f"at t={df.loc[df['VSA_m2'].idxmax(), 'time_s']/3600:.2f} h")
    print(f"  Min A_t: {df['A_t_m2'].min():.3e} m²  "
          f"Min SD_max: {df['SD_max_t'].min():.5f} m")
    if per_polygon:
        print(f"  Per-polygon z final: {z_arr}")
        print(f"  Per-polygon SD_max:  {SD_max_arr}")
        print(f"  Per-polygon A_t:     {A_t_arr}")
    print()

    return df


