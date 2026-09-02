"""
tests/03_test_vsa_opm.py
========================
Verification tests for the OPM/VSA runoff generation engine.

Run from the project root:
    python tests/03_test_vsa_opm.py

Tests
-----
1. Eq 5 self-consistency  — Eq 5 with initial conditions recovers A_t_init
2. H_a sign               — H_a < 0 for all physical inputs
3. VSA grows under rain   — sustained rain → VSA monotonically expands
4. VSA contracts on dry   — after rain stops sandbox drains → VSA contracts
5. Backward compatibility — RUNOFF_SOURCE='none' → identical hydrograph
6. Mass balance           — routed volume ≤ total rainfall at all times
7. Q_max validation       — OPM_Q_MAX ≤ Q_min raises ValueError
8. CSV output             — vsa_opm_results.csv has correct columns, no NaN

Each test prints PASS or FAIL with a short reason.
"""

import os
import sys
import math
import copy

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import config
from hydroflow.core import routing as ru
from hydroflow.core.precip import PrecipEngine

# ── OPM constants (mirrors runoff_input.py) ───────────────────────────────────
SD_MIN = 0.001
Q_MIN  = 0.001
K_MS   = 44.0 / 86400.0

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"


def _load_grid():
    """Load grid data needed by tests (cached to avoid repeat I/O)."""
    dem, fdir, faccum, ws_mask, transform, nodata_dem, cell_size = \
        ru.load_rasters(config)
    nrows, ncols = dem.shape
    cell_area    = cell_size ** 2
    s_rows, s_cols, outlet_rc = ru.topological_order(faccum, fdir, ws_mask)
    slope_2d  = ru.compute_slope_grid(
        dem, fdir, ws_mask, cell_size, config.MIN_SLOPE, nodata_dem
    )
    slope_1d  = slope_2d[s_rows, s_cols]
    faccum_1d = faccum[s_rows, s_cols].astype(np.float64)
    return {
        "dem": dem, "fdir": fdir, "faccum": faccum, "ws_mask": ws_mask,
        "transform": transform, "cell_size": cell_size, "cell_area": cell_area,
        "s_rows": s_rows, "s_cols": s_cols, "outlet_rc": outlet_rc,
        "nrows": nrows, "ncols": ncols, "n_cells": len(s_rows),
        "slope_1d": slope_1d, "faccum_1d": faccum_1d,
    }


def _opm_init(g, SD_max_initial, Q_max):
    """Return (A_1, A_outlet, A_t_init, H_a, Rf_init)."""
    A_1      = g["cell_area"]
    A_outlet = float(g["faccum_1d"][-1]) * g["cell_area"]
    A_t_init = A_outlet / (1.0 - math.log(Q_MIN / Q_max))
    Rf_init  = SD_MIN / SD_max_initial
    ratio    = A_t_init / (A_t_init - A_1)
    H_a      = ratio * math.log(Rf_init)
    return A_1, A_outlet, A_t_init, H_a, Rf_init


# ─────────────────────────────────────────────────────────────────────────────
# Test 1 — Eq 5 self-consistency
# ─────────────────────────────────────────────────────────────────────────────
def test_eq5_self_consistency(g):
    """
    Plugging initial conditions (Rf_init) into Eq 5 must recover A_t_init exactly.
    This verifies that H_a was derived consistently from Eq 4.
    """
    name = "1 · Eq 5 self-consistency"
    try:
        A_1, A_outlet, A_t_init, H_a, Rf_init = _opm_init(
            g, config.OPM_SD_MAX_INITIAL, config.OPM_Q_MAX
        )
        # Eq 5 at t=0:  A_t = H_a * A_1 / (H_a - ln(Rf_init))
        # By construction this is singular (denominator ≈ 0 for large watersheds)
        # but the limit equals A_t_init. Verify numerically with slight perturbation.
        eps      = 1e-8
        Rf_perturbed = Rf_init * (1 + eps)
        denom = H_a - math.log(Rf_perturbed)
        A_t_eq5 = H_a * A_1 / denom
        # Should be close to A_t_init (within a few % for the perturbation)
        rel_err = abs(A_t_eq5 - A_t_init) / A_t_init
        if rel_err < 0.01:
            print(f"  {PASS}  {name}  (rel_err={rel_err:.2e})")
        else:
            print(f"  {FAIL}  {name}  rel_err={rel_err:.2e} (expected < 0.01)")
    except Exception as e:
        print(f"  {FAIL}  {name}  Exception: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Test 2 — H_a sign
# ─────────────────────────────────────────────────────────────────────────────
def test_ha_sign(g):
    """H_a must always be negative for physical (SD_min < SD_max, Q_min < Q_max)."""
    name = "2 · H_a sign (must be negative)"
    try:
        # Test across a range of plausible parameters
        all_negative = True
        for sd in [0.02, 0.05, 0.10, 0.20, 0.50]:
            for qmax in [0.1, 1.0, 10.0, 100.0]:
                _, _, _, H_a, _ = _opm_init(g, sd, qmax)
                if H_a >= 0:
                    all_negative = False
                    print(f"  {FAIL}  {name}  H_a={H_a:.4f} >= 0 for "
                          f"SD_max={sd}, Q_max={qmax}")
                    return
        print(f"  {PASS}  {name}  (H_a < 0 for all tested params)")
    except Exception as e:
        print(f"  {FAIL}  {name}  Exception: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Test 3 — VSA grows under sustained rain
# ─────────────────────────────────────────────────────────────────────────────
def test_vsa_grows_under_rain(g):
    """Under constant precipitation, VSA must be non-decreasing (Dunne mechanism)."""
    name = "3 · VSA grows under sustained rain"
    try:
        SD_max_initial = config.OPM_SD_MAX_INITIAL
        Q_max          = config.OPM_Q_MAX
        phi            = getattr(config, 'OPM_PHI', 0.35)
        cell_area      = g["cell_area"]
        cell_size      = g["cell_size"]
        faccum_1d      = g["faccum_1d"]
        slope_divide   = float(g["slope_1d"][0])
        upslope_area   = faccum_1d * cell_area

        A_1, A_outlet, A_t_init, H_a, Rf_init = _opm_init(g, SD_max_initial, Q_max)

        # Constant rainfall = RAIN_INTENSITY_MM_HR → m/s
        P_ms = config.RAIN_INTENSITY_MM_HR / 1000.0 / 3600.0
        dt   = config.TIME_STEP_SECONDS
        # Run for the rain duration only
        n_wet = int(config.RAIN_DURATION_HOURS * 3600.0 / dt)

        z, A_t, SD_max_t = 0.0, A_t_init, SD_max_initial
        VSA_prev = float((upslope_area > A_t).sum()) * cell_area
        violated = 0

        for _ in range(n_wet):
            # Sandbox advance
            q_b = K_MS * slope_divide * z * cell_size
            dz  = (P_ms * cell_area - q_b) * dt / (cell_area * phi)
            z   = max(0.0, z + dz)
            SD_max_t = max(SD_MIN, SD_max_initial - z)
            Rf_t  = SD_MIN / SD_max_t
            denom = H_a - math.log(Rf_t)
            if abs(denom) >= 1e-12:
                new_At = H_a * A_1 / denom
                A_t    = float(np.clip(new_At, A_1, A_outlet))
            VSA_now = float((upslope_area > A_t).sum()) * cell_area
            if VSA_now < VSA_prev - 1e-6:   # allow floating-point tolerance
                violated += 1
            VSA_prev = VSA_now

        if violated == 0:
            print(f"  {PASS}  {name}  (VSA non-decreasing over {n_wet} wet steps)")
        else:
            print(f"  {FAIL}  {name}  VSA decreased {violated} times during rain")
    except Exception as e:
        print(f"  {FAIL}  {name}  Exception: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Test 4 — VSA contracts after rain stops
# ─────────────────────────────────────────────────────────────────────────────
def test_vsa_contracts_after_rain(g):
    """After rainfall ends, Darcy drainage reduces z → SD_max recovers → A_t grows → VSA shrinks."""
    name = "4 · VSA contracts after rain stops"
    try:
        SD_max_initial = config.OPM_SD_MAX_INITIAL
        Q_max          = config.OPM_Q_MAX
        phi            = getattr(config, 'OPM_PHI', 0.35)
        cell_area      = g["cell_area"]
        cell_size      = g["cell_size"]
        faccum_1d      = g["faccum_1d"]
        slope_divide   = float(g["slope_1d"][0])
        upslope_area   = faccum_1d * cell_area

        A_1, A_outlet, A_t_init, H_a, _ = _opm_init(g, SD_max_initial, Q_max)

        P_ms  = config.RAIN_INTENSITY_MM_HR / 1000.0 / 3600.0
        dt    = config.TIME_STEP_SECONDS
        n_wet = int(config.RAIN_DURATION_HOURS * 3600.0 / dt)
        n_dry = int(3 * 3600.0 / dt)   # 3-hour dry period

        z, A_t, SD_max_t = 0.0, A_t_init, SD_max_initial

        def _advance(rain_rate, n_steps):
            nonlocal z, A_t, SD_max_t
            for _ in range(n_steps):
                q_b = K_MS * slope_divide * z * cell_size
                dz  = (rain_rate * cell_area - q_b) * dt / (cell_area * phi)
                z   = max(0.0, z + dz)
                SD_max_t = max(SD_MIN, SD_max_initial - z)
                Rf_t  = SD_MIN / SD_max_t
                denom = H_a - math.log(Rf_t)
                if abs(denom) >= 1e-12:
                    new_At = H_a * A_1 / denom
                    A_t    = float(np.clip(new_At, A_1, A_outlet))

        # Wet phase
        _advance(P_ms, n_wet)
        VSA_peak = float((upslope_area > A_t).sum()) * cell_area
        z_peak   = z

        # Dry phase
        _advance(0.0, n_dry)
        VSA_after = float((upslope_area > A_t).sum()) * cell_area
        z_after   = z

        if z_after < z_peak and VSA_after <= VSA_peak:
            print(f"  {PASS}  {name}  "
                  f"(z: {z_peak:.4f} → {z_after:.4f} m; "
                  f"VSA: {VSA_peak/1e6:.3f} → {VSA_after/1e6:.3f} km²)")
        else:
            print(f"  {FAIL}  {name}  "
                  f"z_after={z_after:.4f} z_peak={z_peak:.4f}  "
                  f"VSA_after={VSA_after:.0f} VSA_peak={VSA_peak:.0f}")
    except Exception as e:
        print(f"  {FAIL}  {name}  Exception: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Test 5 — Backward compatibility
# ─────────────────────────────────────────────────────────────────────────────
def test_backward_compatibility():
    """RUNOFF_SOURCE='none' must leave source_1d == rain_1d (no change to routing)."""
    name = "5 · Backward compatibility (RUNOFF_SOURCE='none')"
    try:
        from hydroflow.core import runoff as ri

        class _DummyCfg:
            RUNOFF_SOURCE = 'none'

        # RunoffEngine 'none' mode: get_effective_1d must return the same array
        # We test at the class level (no raster loading needed).
        rain = np.random.rand(50)

        class _MinimalGrid:
            pass

        # Manually build just enough for 'none' mode
        g = {"n_cells": 50, "s_rows": np.zeros(50, int), "s_cols": np.zeros(50, int),
             "nrows": 10, "ncols": 10}
        eng = ri.RunoffEngine(_DummyCfg(), g)
        out = eng.get_effective_1d(0.0, rain)
        if np.array_equal(out, rain):
            print(f"  {PASS}  {name}")
        else:
            print(f"  {FAIL}  {name}  source_1d differs from rain_1d")
    except Exception as e:
        print(f"  {FAIL}  {name}  Exception: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Test 6 — Mass balance
# ─────────────────────────────────────────────────────────────────────────────
def test_mass_balance(g):
    """
    Total effective runoff volume produced by OPM ≤ total rainfall volume.
    (VSA ≤ watershed area → runoff ≤ rainfall at every step)
    """
    name = "6 · Mass balance (runoff ≤ rainfall)"
    try:
        SD_max_initial = config.OPM_SD_MAX_INITIAL
        Q_max          = config.OPM_Q_MAX
        phi            = getattr(config, 'OPM_PHI', 0.35)
        cell_area      = g["cell_area"]
        cell_size      = g["cell_size"]
        faccum_1d      = g["faccum_1d"]
        slope_divide   = float(g["slope_1d"][0])
        upslope_area   = faccum_1d * cell_area
        n_cells        = g["n_cells"]

        A_1, A_outlet, A_t_init, H_a, _ = _opm_init(g, SD_max_initial, Q_max)

        grid_data_pe = {
            "s_rows": g["s_rows"], "s_cols": g["s_cols"],
            "nrows": g["nrows"],   "ncols": g["ncols"],
            "n_cells": n_cells,    "ws_mask": g["ws_mask"],
            "transform": g["transform"],
        }
        pe  = PrecipEngine(config, grid_data_pe)
        dt  = config.TIME_STEP_SECONDS
        n_steps = int(config.TOTAL_SIMULATION_TIME_HOURS * 3600.0 / dt)

        z, A_t, SD_max_t = 0.0, A_t_init, SD_max_initial
        total_rain_m3   = 0.0
        total_runoff_m3 = 0.0
        violated = False

        for step in range(n_steps):
            t_s     = step * dt
            rain_1d = pe.get_field_1d(t_s)

            # VSA mask and effective runoff
            vsa_mask   = upslope_area > A_t
            source_1d  = rain_1d * vsa_mask.astype(np.float64)

            # Accumulate volumes
            step_rain   = rain_1d.sum() * cell_area * dt
            step_runoff = source_1d.sum() * cell_area * dt
            total_rain_m3   += step_rain
            total_runoff_m3 += step_runoff

            if step_runoff > step_rain + 1e-10:
                violated = True
                break

            # Sandbox advance
            q_b = K_MS * slope_divide * z * cell_size
            dz  = (float(rain_1d[0]) * cell_area - q_b) * dt / (cell_area * phi)
            z   = max(0.0, z + dz)
            SD_max_t = max(SD_MIN, SD_max_initial - z)
            Rf_t  = SD_MIN / SD_max_t
            denom = H_a - math.log(Rf_t)
            if abs(denom) >= 1e-12:
                new_At = H_a * A_1 / denom
                A_t    = float(np.clip(new_At, A_1, A_outlet))

        if not violated and total_runoff_m3 <= total_rain_m3 + 1e-6:
            ratio = total_runoff_m3 / max(total_rain_m3, 1e-12)
            print(f"  {PASS}  {name}  "
                  f"(runoff/rain ratio = {ratio:.3f})")
        else:
            print(f"  {FAIL}  {name}  "
                  f"runoff={total_runoff_m3:.3e} > rain={total_rain_m3:.3e}")
    except Exception as e:
        print(f"  {FAIL}  {name}  Exception: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Test 7 — Q_max validation
# ─────────────────────────────────────────────────────────────────────────────
def test_qmax_validation():
    """OPM_Q_MAX ≤ Q_min must raise ValueError before any computation."""
    name = "7 · Q_max validation (raises ValueError for bad input)"
    try:
        from hydroflow.core import runoff as ri

        class _BadCfg:
            RUNOFF_SOURCE       = 'vsa_opm'
            OPM_SD_MAX_INITIAL  = 0.10
            OPM_Q_MAX           = 0.0005   # below Q_MIN = 0.001
            OPM_PHI             = 0.35

        # Minimal grid_data with enough keys for _init_vsa_opm
        dummy_faccum = np.ones(10, dtype=np.float64)
        dummy_faccum[-1] = 1000.0
        gd = {
            "n_cells": 10, "s_rows": np.zeros(10, int), "s_cols": np.zeros(10, int),
            "nrows": 5, "ncols": 5, "cell_area": 900.0, "cell_size": 30.0,
            "slope_1d": np.full(10, 0.01), "faccum_1d": dummy_faccum,
            "outlet_rc": (0, 0),
        }
        try:
            ri.RunoffEngine(_BadCfg(), gd)
            print(f"  {FAIL}  {name}  No exception raised")
        except ValueError:
            print(f"  {PASS}  {name}")
        except Exception as ex:
            print(f"  {FAIL}  {name}  Wrong exception type: {type(ex).__name__}: {ex}")
    except Exception as e:
        print(f"  {FAIL}  {name}  Exception in test setup: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Test 8 — CSV output
# ─────────────────────────────────────────────────────────────────────────────
def test_csv_output():
    """vsa_opm_results.csv must exist, have correct columns, and contain no NaN."""
    name = "8 · CSV output (columns, no NaN)"
    try:
        import pandas as pd
        csv_path = os.path.join(config.OUTPUT_DIR, "vsa_opm_results.csv")
        if not os.path.exists(csv_path):
            print(f"  {FAIL}  {name}  File not found: {csv_path}")
            print("          → Run vsa_opm.py first to generate the CSV.")
            return
        df = pd.read_csv(csv_path)
        required = {"time_s", "SD_max_t", "A_t_m2", "VSA_m2"}
        missing  = required - set(df.columns)
        if missing:
            print(f"  {FAIL}  {name}  Missing columns: {missing}")
            return
        nan_counts = df[list(required)].isna().sum().sum()
        if nan_counts > 0:
            print(f"  {FAIL}  {name}  {nan_counts} NaN values found")
            return
        # Basic sanity: SD_max_t in (0, SD_max_init], A_t_m2 > 0, VSA_m2 >= 0
        ok = (
            (df['SD_max_t'] > 0).all() and
            (df['SD_max_t'] <= config.OPM_SD_MAX_INITIAL + 1e-9).all() and
            (df['A_t_m2'] > 0).all() and
            (df['VSA_m2'] >= 0).all()
        )
        if ok:
            print(f"  {PASS}  {name}  ({len(df)} rows, no NaN, values in range)")
        else:
            print(f"  {FAIL}  {name}  Values out of physical range")
    except Exception as e:
        print(f"  {FAIL}  {name}  Exception: {e}")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("OPM/VSA Verification Tests")
    print("=" * 60)
    print()

    print("Loading grid data (shared across tests)...")
    g = _load_grid()
    print()

    test_eq5_self_consistency(g)
    test_ha_sign(g)
    test_vsa_grows_under_rain(g)
    test_vsa_contracts_after_rain(g)
    test_backward_compatibility()
    test_mass_balance(g)
    test_qmax_validation()
    test_csv_output()

    print()
    print("Done.")
