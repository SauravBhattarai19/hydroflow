"""
06_test_at_sensitivity.py
=========================
Sensitivity analysis of A_t to Q_max, and per-polygon A_t dynamics.

Shows Dr. Nawa:
  1. How ln() compresses Q_max → A_t_init  (Eq 10)
  2. How little initial VSA changes with Q_max
  3. How A_t(t) converges regardless of Q_max within a few hours
  4. Per-polygon A_t evolution under actual rainfall
  5. How A_t_init is defined: shared watershed-level, not per-polygon

Usage
-----
  cd /path/to/OPM
  python tests/06_test_at_sensitivity.py
"""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import re
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
from _opm_diag import output_dir, resolve_precip

# Outputs + gauges/event follow the active config (scenario folder, precip method)
OUTPUT = output_dir()
GAUGE_CSV, TS_CSV, EVENT_DATE = resolve_precip(config)

Q_MIN = 0.001  # model constant [m³/s]


def main():
    from hydroflow.core.routing import router as kwr
    import rasterio

    print("=" * 65)
    print("  A_t Sensitivity to Q_max  |  Per-Polygon Analysis")
    print("=" * 65)

    # ── Load grid to get real faccum and cell geometry ────────────────────
    orig_backend = config.BACKEND
    orig_qmax    = config.OPM_Q_MAX
    orig_date    = getattr(config, 'SERVES_TARGET_DATE', None)
    config.BACKEND = 'cpu'
    if EVENT_DATE:
        config.SERVES_TARGET_DATE = EVENT_DATE
    # No precip override — respect config.PRECIP_METHOD.

    grid_data = kwr.initialise_grid(config)

    faccum_1d  = grid_data['faccum_1d']
    s_rows     = grid_data['s_rows']
    s_cols     = grid_data['s_cols']
    cell_area  = grid_data['cell_area']
    cell_size  = grid_data['cell_size']
    n_cells    = grid_data['n_cells']

    upslope_area = faccum_1d * cell_area
    A_outlet = float(faccum_1d[-1]) * cell_area

    precip_engine = grid_data['precip_engine']
    cell_polygon  = grid_data.get('cell_polygon')
    if cell_polygon is not None:
        cell_polygon = np.asarray(cell_polygon).ravel()
        n_poly = int(cell_polygon.max()) + 1
    else:
        n_poly = 1

    # Get per-polygon areas
    poly_areas_km2 = []
    for p in range(n_poly):
        poly_areas_km2.append(float((cell_polygon == p).sum()) * cell_area / 1e6)

    # ══════════════════════════════════════════════════════════════════════
    # Panel 1 & 2: Analytical A_t_init and VSA % vs Q_max
    # ══════════════════════════════════════════════════════════════════════
    q_max_range = np.logspace(-2, 3, 200)  # 0.01 to 1000 m³/s

    at_init_vals = A_outlet / (1.0 - np.log(Q_MIN / q_max_range))
    vsa_frac_vals = np.array([
        float((upslope_area > at).sum()) / n_cells * 100
        for at in at_init_vals
    ])

    # ══════════════════════════════════════════════════════════════════════
    # Panel 3: Run OPM sandbox at different Q_max → A_t(t) convergence
    # ══════════════════════════════════════════════════════════════════════
    test_qmax = [0.1, 1.0, 10.0, 100.0, 500.0]
    dt = float(config.TIME_STEP_SECONDS)
    total_s = config.TOTAL_SIMULATION_TIME_HOURS * 3600.0
    n_steps = int(total_s / dt)

    # We'll track mean A_t across polygons for each Q_max
    at_histories = {}
    vsa_histories = {}

    # Get SD params once (they don't change with Q_max)
    from hydroflow.core.runoff import resolve_sd_params, OPM_SD_MIN
    sd_params = resolve_sd_params(config, cell_size)
    SD_max_initial = sd_params['sd_max']
    sd_min = sd_params['sd_min']
    phi = sd_params['phi']
    sd_max_per_poly = sd_params['sd_max_per_polygon']
    ksat_ms = sd_params['ksat_ms']

    if sd_max_per_poly is not None and len(sd_max_per_poly) == n_poly:
        sd_init_arr = np.array(sd_max_per_poly)
    else:
        sd_init_arr = np.full(n_poly, SD_max_initial)

    # Per-polygon divide cell info
    faccum_np = np.asarray(faccum_1d)
    slope_np  = np.asarray(grid_data['slope_1d'])
    divide_idx   = np.empty(n_poly, dtype=np.intp)
    slope_divide = np.empty(n_poly)
    # Catchment-wide fallback divide for zones with no watershed cells (possible
    # with gridded IMERG stations whose buffered footprint includes empty pixels).
    _global_divide = int(faccum_np.argmin())
    for p in range(n_poly):
        local_idx = np.where(cell_polygon == p)[0]
        if local_idx.size == 0:
            divide_idx[p] = _global_divide
            slope_divide[p] = float(slope_np[_global_divide])
            continue
        best = local_idx[faccum_np[local_idx].argmin()]
        divide_idx[p] = best
        slope_divide[p] = float(slope_np[best])

    ksat_arr = np.full(n_poly, ksat_ms)

    A_1 = cell_area

    print(f"\n  Running OPM sandbox at {len(test_qmax)} Q_max values ...")
    print(f"  A_outlet = {A_outlet:.3e} m² = {A_outlet/1e6:.1f} km²")
    print(f"  n_polygons = {n_poly}")
    print(f"  Polygon areas: {[f'{a:.1f}' for a in poly_areas_km2]} km²")
    print(f"  SD_max_init per polygon: {[f'{v:.3f}' for v in sd_init_arr]}")
    print()

    # Record every 10 minutes for the time series
    record_interval = max(1, int(600 / dt))

    for q_max in test_qmax:
        A_t_init = A_outlet / (1.0 - math.log(Q_MIN / q_max))
        ratio = A_t_init / (A_t_init - A_1)
        Rf_init = sd_min / sd_init_arr
        H_a = ratio * np.log(Rf_init)

        z = np.zeros(n_poly)
        SD_max = sd_init_arr.copy()
        A_t = np.full(n_poly, A_t_init)

        init_vsa = float((upslope_area > A_t_init).sum()) / n_cells * 100

        t_hrs = [0.0]
        at_per_poly = [A_t.copy()]
        vsa_pcts = [init_vsa]

        for step in range(1, n_steps + 1):
            t = step * dt
            rain_1d = precip_engine.get_field_1d(t)

            for p in range(n_poly):
                P_div = float(rain_1d[divide_idx[p]])
                q_b = ksat_arr[p] * slope_divide[p] * z[p] * cell_size
                dV = (P_div * cell_area - q_b) * dt
                dz = dV / (cell_area * phi)
                z[p] = max(0.0, z[p] + dz)

                SD_max_p = max(sd_min, sd_init_arr[p] - z[p])
                SD_max[p] = SD_max_p

                Rf_t = sd_min / SD_max_p
                denom = H_a[p] - math.log(Rf_t)
                if abs(denom) < 1e-12:
                    A_t[p] = A_t_init
                else:
                    new_At = H_a[p] * A_1 / denom
                    A_t[p] = float(np.clip(new_At, A_1, A_outlet))

            if step % record_interval == 0:
                t_hrs.append(t / 3600.0)
                at_per_poly.append(A_t.copy())
                A_t_per_cell = A_t[cell_polygon]
                vsa = float((upslope_area > A_t_per_cell).sum()) / n_cells * 100
                vsa_pcts.append(vsa)

        at_histories[q_max] = (np.array(t_hrs), np.array(at_per_poly))
        vsa_histories[q_max] = (np.array(t_hrs), np.array(vsa_pcts))
        print(f"    Q_max={q_max:>7.1f}  A_t_init={A_t_init:.3e} m²"
              f"  init_VSA={init_vsa:.2f}%  final_VSA={vsa_pcts[-1]:.1f}%")

    # ══════════════════════════════════════════════════════════════════════
    # Figure
    # ══════════════════════════════════════════════════════════════════════
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)

    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(test_qmax)))
    poly_colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3',
                   '#ff7f00', '#a65628', '#f781bf']

    # ── Panel 1: A_t_init vs Q_max ───────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.semilogx(q_max_range, at_init_vals / 1e6, 'k-', lw=2)
    for i, q in enumerate(test_qmax):
        at_val = A_outlet / (1.0 - math.log(Q_MIN / q))
        ax1.plot(q, at_val / 1e6, 'o', color=colors[i], markersize=10,
                 markeredgecolor='black', zorder=5)
        ax1.annotate(f"{at_val/1e6:.1f} km²", (q, at_val/1e6),
                     textcoords='offset points', xytext=(8, 5), fontsize=8)

    ax1.axhline(A_outlet / 1e6, color='red', ls='--', lw=1, alpha=0.5,
                label=f'A_outlet = {A_outlet/1e6:.1f} km²')
    ax1.set_xlabel(r"$Q_{max}$ [m³/s]", fontsize=11)
    ax1.set_ylabel(r"$A_t^{init}$ [km²]", fontsize=11)
    ax1.set_title(r"Eq 10:  $A_t^{init} = \frac{A_{outlet}}{1 - \ln(Q_{min}/Q_{max})}$",
                  fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, ls='--', alpha=0.3)

    # Annotate the log compression
    ax1.text(0.5, 0.05,
             r"$Q_{max}$ changes 10000$\times$,"
             r" but $A_t^{init}$ changes only ~2.5$\times$"
             "\n(logarithmic compression)",
             transform=ax1.transAxes, fontsize=9, ha='center',
             bbox=dict(facecolor='lightyellow', alpha=0.9, pad=4))

    # ── Panel 2: Initial VSA % vs Q_max ──────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.semilogx(q_max_range, vsa_frac_vals, 'k-', lw=2)
    for i, q in enumerate(test_qmax):
        at_val = A_outlet / (1.0 - math.log(Q_MIN / q))
        vsa_val = float((upslope_area > at_val).sum()) / n_cells * 100
        ax2.plot(q, vsa_val, 'o', color=colors[i], markersize=10,
                 markeredgecolor='black', zorder=5,
                 label=f'Q={q:.1f} → VSA={vsa_val:.2f}%')

    ax2.set_xlabel(r"$Q_{max}$ [m³/s]", fontsize=11)
    ax2.set_ylabel("Initial VSA [% of watershed]", fontsize=11)
    ax2.set_title("Initial Saturated Area vs Q_max", fontsize=12,
                  fontweight='bold')
    ax2.legend(fontsize=8, loc='upper left')
    ax2.grid(True, ls='--', alpha=0.3)

    # ── Panel 3: VSA(t) convergence for different Q_max ──────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    for i, q in enumerate(test_qmax):
        t_hrs, vsa_pcts = vsa_histories[q]
        ax3.plot(t_hrs, vsa_pcts, color=colors[i], lw=2,
                 label=f'Q_max={q:.1f}')

    ax3.set_xlabel("Time [hr]", fontsize=11)
    ax3.set_ylabel("VSA [% of watershed]", fontsize=11)
    ax3.set_title("VSA Evolution — Convergence Regardless of Q_max",
                  fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, ls='--', alpha=0.3)

    # ── Panel 4: Mean A_t(t) convergence ─────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    for i, q in enumerate(test_qmax):
        t_hrs, at_arr = at_histories[q]
        mean_at = at_arr.mean(axis=1) / 1e6  # mean across polygons
        ax4.plot(t_hrs, mean_at, color=colors[i], lw=2,
                 label=f'Q_max={q:.1f}')

    ax4.set_xlabel("Time [hr]", fontsize=11)
    ax4.set_ylabel(r"Mean $A_t$ across polygons [km²]", fontsize=11)
    ax4.set_title(r"$A_t(t)$ Convergence — Different Start, Same Destination",
                  fontsize=12, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(True, ls='--', alpha=0.3)

    # ── Panel 5: Per-polygon A_t(t) at Q_max = current config ───────────
    ax5 = fig.add_subplot(gs[2, 0])
    q_ref = orig_qmax
    if q_ref not in at_histories:
        q_ref = test_qmax[len(test_qmax) // 2]
    t_hrs, at_arr = at_histories[q_ref]
    for p in range(n_poly):
        ax5.plot(t_hrs, at_arr[:, p] / 1e6, color=poly_colors[p % 7],
                 lw=1.8, label=f'P{p} (SD={sd_init_arr[p]:.3f}m,'
                 f' {poly_areas_km2[p]:.0f}km²)')

    ax5.set_xlabel("Time [hr]", fontsize=11)
    ax5.set_ylabel(r"$A_t$ [km²]", fontsize=11)
    ax5.set_title(f"Per-Polygon A_t(t)  |  Q_max={q_ref:.1f} m³/s",
                  fontsize=12, fontweight='bold')
    ax5.legend(fontsize=7, loc='upper right')
    ax5.grid(True, ls='--', alpha=0.3)

    # ── Panel 6: Explanation text ────────────────────────────────────────
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.axis('off')

    text = (
        r"$\bf{How\ A_t\ is\ initialized\ per\ polygon}$" + "\n\n"
        r"$\bullet$ $A_t^{init}$ is computed from Eq 10 using:" + "\n"
        f"     A_outlet = {A_outlet/1e6:.1f} km² (whole watershed)\n"
        f"     Q_max = config value (single outlet discharge)\n\n"
        r"$\bullet$ ALL polygons start with the SAME $A_t^{init}$" + "\n"
        r"     (it is a topographic threshold, not a polygon area)" + "\n\n"
        r"$\bullet$ The VSA mask compares each cell's GLOBAL" + "\n"
        r"     upslope area against its polygon's $A_t$" + "\n"
        r"     → cells near channels (high faccum) saturate first" + "\n\n"
        r"$\bullet$ $H_a$ differs by polygon (different $SD_{max}$)" + "\n"
        r"     → $A_t$ dynamics diverge once rain starts" + "\n\n"
        r"$\bullet$ $\bf{Key\ finding:}$ Q$_{max}$ barely matters" + "\n"
        r"     because ln() compresses 10000× Q range into" + "\n"
        r"     ~2.5× $A_t$ range, and rainfall dynamics dominate" + "\n"
        r"     within the first few hours"
    )
    ax6.text(0.05, 0.95, text, transform=ax6.transAxes,
             fontsize=10.5, va='top', family='monospace',
             bbox=dict(facecolor='lightyellow', alpha=0.9, pad=12,
                       edgecolor='gray'))

    fig.suptitle(r"OPM Sensitivity: $A_t$ vs $Q_{max}$  |  Per-Polygon Dynamics",
                 fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    out = OUTPUT / "at_sensitivity_qmax.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"\n  Saved: {out}")

    # ── Print summary table ──────────────────────────────────────────────
    print(f"\n  {'Q_max':>10}  {'A_t_init (km²)':>14}  {'Init VSA %':>10}"
          f"  {'Final VSA %':>11}  {'A_t ratio':>10}")
    print("  " + "─" * 60)
    at_ref = A_outlet / (1.0 - math.log(Q_MIN / test_qmax[0]))
    for q in test_qmax:
        at_val = A_outlet / (1.0 - math.log(Q_MIN / q))
        init_vsa = float((upslope_area > at_val).sum()) / n_cells * 100
        _, vsa_pcts = vsa_histories[q]
        print(f"  {q:>10.1f}  {at_val/1e6:>14.2f}  {init_vsa:>10.2f}"
              f"  {vsa_pcts[-1]:>11.1f}  {at_val/at_ref:>10.2f}x")

    config.BACKEND = orig_backend
    config.OPM_Q_MAX = orig_qmax
    config.SERVES_TARGET_DATE = orig_date
    print("\nDone.")


if __name__ == "__main__":
    main()
