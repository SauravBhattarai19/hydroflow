"""
07_test_at_faccum_analysis.py
=============================
Why VSA never exceeds ~27% even when A_t drops to "just 5 cells."

Shows:
  1. Flow accumulation distribution — 72% of cells are headwaters (faccum ≤ 5)
  2. A_t(t) per polygon on LOG scale — reveals the actual dynamics
  3. VSA ceiling curve: max possible VSA as a function of A_t
  4. Spatial map: which cells CAN saturate vs which NEVER will

Usage
-----
  cd /path/to/OPM
  python tests/07_test_at_faccum_analysis.py
"""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import re
import math
import numpy as np
import pandas as pd
import rasterio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

import config
from _opm_diag import output_dir, resolve_precip

# Outputs + gauges/event follow the active config (scenario folder, precip method)
OUTPUT = output_dir()
GAUGE_CSV, TS_CSV, EVENT_DATE = resolve_precip(config)

Q_MIN = 0.001
POLY_COLORS = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3',
               '#ff7f00', '#a65628', '#f781bf']


def main():
    from hydroflow.core.routing import router as kwr

    print("=" * 65)
    print("  A_t + Flow Accumulation Analysis")
    print("=" * 65)

    # ── Initialize grid ──────────────────────────────────────────────────
    orig_backend = config.BACKEND
    orig_date    = getattr(config, 'SERVES_TARGET_DATE', None)
    orig_qmax    = config.OPM_Q_MAX
    config.BACKEND = 'cpu'
    if EVENT_DATE:
        config.SERVES_TARGET_DATE = EVENT_DATE
    # No precip override — respect config.PRECIP_METHOD.

    grid_data = kwr.initialise_grid(config)

    re_engine = grid_data['runoff_engine']
    pe        = grid_data['precip_engine']
    faccum_1d = np.asarray(grid_data['faccum_1d'])
    s_rows    = grid_data['s_rows']
    s_cols    = grid_data['s_cols']
    nrows     = grid_data['nrows']
    ncols     = grid_data['ncols']
    cell_area = grid_data['cell_area']
    cell_size = grid_data['cell_size']
    n_cells   = grid_data['n_cells']
    A_1       = cell_area
    A_outlet  = float(faccum_1d[-1]) * cell_area
    upslope_area = faccum_1d * cell_area

    n_poly = re_engine._n_polygons
    cell_polygon = re_engine._cell_polygon

    # Load DEM and watershed for spatial maps
    with rasterio.open(config.ROUTING_DEM_PATH) as src:
        dem = src.read(1).astype(np.float64)
        nodata = src.nodata
    with rasterio.open(config.ROUTING_WATERSHED_MASK_PATH) as src:
        ws = src.read(1) > 0
    if nodata is not None:
        ws &= (dem != nodata)
    dem_masked = np.where(ws, dem, np.nan)

    # Crop to watershed bounds
    rows_ws, cols_ws = np.where(ws)
    pad = 5
    r0 = max(0, rows_ws.min() - pad)
    r1 = min(ws.shape[0], rows_ws.max() + pad + 1)
    c0 = max(0, cols_ws.min() - pad)
    c1 = min(ws.shape[1], cols_ws.max() + pad + 1)
    bb = (r0, r1, c0, c1)

    def crop(arr):
        return arr[r0:r1, c0:c1]

    dt = float(config.TIME_STEP_SECONDS)
    total_s = config.TOTAL_SIMULATION_TIME_HOURS * 3600.0
    n_steps = int(total_s / dt)

    # ══════════════════════════════════════════════════════════════════════
    # 1. Flow accumulation distribution
    # ══════════════════════════════════════════════════════════════════════
    print("\n  Flow accumulation distribution:")
    print(f"    faccum=1 (ridge tops): {(faccum_1d==1).sum():,} = "
          f"{(faccum_1d==1).sum()/n_cells*100:.1f}%")
    print(f"    faccum 1-5 (headwaters): {(faccum_1d<=5).sum():,} = "
          f"{(faccum_1d<=5).sum()/n_cells*100:.1f}%")
    print(f"    faccum > 5 (can saturate): {(faccum_1d>5).sum():,} = "
          f"{(faccum_1d>5).sum()/n_cells*100:.1f}%")

    # ══════════════════════════════════════════════════════════════════════
    # 2. Run simulation, record per-polygon A_t and VSA
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n  Running {n_steps} time steps ...")

    record_interval = max(1, int(600 / dt))  # every 10 minutes
    t_hrs = [0.0]
    at_history = [re_engine._opm_A_t.copy()]
    z_history = [re_engine._opm_z.copy()]
    vsa_history = [float(re_engine._vsa_mask.mean()) * 100]
    rain_history = [np.zeros(n_poly)]

    for step in range(1, n_steps + 1):
        t = step * dt
        rain_1d = pe.get_field_1d(t)
        re_engine.update_state(rain_1d, dt)

        if step % record_interval == 0:
            t_hrs.append(t / 3600.0)
            at_history.append(re_engine._opm_A_t.copy())
            z_history.append(re_engine._opm_z.copy())
            mask = re_engine._vsa_mask
            if hasattr(mask, 'get'):
                mask = mask.get()
            vsa_history.append(float(mask.mean()) * 100)

            rain_pp = np.zeros(n_poly)
            for p in range(n_poly):
                m = cell_polygon == p
                if m.any():                       # empty zones (IMERG edge pixels)
                    rain_pp[p] = float(rain_1d[m].mean()) * 3.6e6
            rain_history.append(rain_pp)

    t_hrs = np.array(t_hrs)
    at_arr = np.array(at_history)      # (T, n_poly)
    z_arr = np.array(z_history)        # (T, n_poly)
    vsa_arr = np.array(vsa_history)    # (T,)
    rain_arr = np.array(rain_history)  # (T, n_poly)

    print(f"    Peak VSA: {vsa_arr.max():.1f}%  at t={t_hrs[vsa_arr.argmax()]:.1f}h")
    print(f"    Min A_t:  {at_arr.min()/1e6:.4f} km² = "
          f"{at_arr.min()/cell_area:.1f} cells")

    # ══════════════════════════════════════════════════════════════════════
    # 3. VSA ceiling curve: max possible VSA for any A_t value
    # ══════════════════════════════════════════════════════════════════════
    at_sweep = np.logspace(np.log10(cell_area), np.log10(A_outlet), 500)
    vsa_ceiling = np.array([
        (upslope_area > at_val).sum() / n_cells * 100
        for at_val in at_sweep
    ])

    # ══════════════════════════════════════════════════════════════════════
    # 4. Spatial map: max-possible saturated cells
    # ══════════════════════════════════════════════════════════════════════
    # Cells that CAN saturate when A_t = 5 cells (typical minimum)
    at_min_typical = 5 * cell_area
    can_saturate = upslope_area > at_min_typical

    can_sat_2d = np.full((nrows, ncols), np.nan)
    can_sat_2d[s_rows, s_cols] = can_saturate.astype(float)

    # Peak VSA mask
    peak_idx = vsa_arr.argmax()

    # ══════════════════════════════════════════════════════════════════════
    # FIGURE: 6 panels
    # ══════════════════════════════════════════════════════════════════════
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)

    # ── Panel 1: Flow accumulation histogram ─────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])

    bins = [1, 2, 3, 4, 5, 6, 10, 20, 50, 100, 500, 1000, 5000, 70000]
    counts, _ = np.histogram(faccum_1d, bins=bins)
    pcts = counts / n_cells * 100

    bar_labels = []
    for i in range(len(bins) - 1):
        if bins[i + 1] - bins[i] == 1:
            bar_labels.append(f"{bins[i]}")
        else:
            bar_labels.append(f"{bins[i]}-{bins[i+1]-1}")

    bar_colors = ['#d73027' if bins[i] <= 5 else '#4575b4'
                  for i in range(len(bins) - 1)]
    bars = ax1.bar(range(len(counts)), pcts, color=bar_colors, edgecolor='black',
                   linewidth=0.5)

    ax1.set_xticks(range(len(bar_labels)))
    ax1.set_xticklabels(bar_labels, rotation=45, ha='right', fontsize=8)
    ax1.set_ylabel("% of cells", fontsize=11)
    ax1.set_xlabel("Flow accumulation (# contributing cells)", fontsize=10)
    ax1.set_title("Flow Accumulation Distribution", fontsize=12,
                  fontweight='bold')

    headwater_pct = (faccum_1d <= 5).sum() / n_cells * 100
    ax1.axvline(4.5, color='black', ls='--', lw=1.5, alpha=0.7)
    ax1.text(4.7, max(pcts) * 0.9,
             f"← {headwater_pct:.0f}% headwaters\n   (faccum ≤ 5)\n"
             f"   NEVER saturate",
             fontsize=9, fontweight='bold', color='#d73027')
    ax1.text(5.5, max(pcts) * 0.7,
             f"{100-headwater_pct:.0f}% can saturate →",
             fontsize=9, fontweight='bold', color='#4575b4')

    legend_patches = [
        mpatches.Patch(color='#d73027', label='Cannot saturate (faccum ≤ 5)'),
        mpatches.Patch(color='#4575b4', label='Can saturate (faccum > 5)'),
    ]
    ax1.legend(handles=legend_patches, fontsize=8, loc='upper right')
    ax1.grid(True, ls='--', alpha=0.3, axis='y')

    # ── Panel 2: VSA ceiling curve ───────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.semilogx(at_sweep / 1e6, vsa_ceiling, 'k-', lw=2,
                 label='Max possible VSA')

    # Mark actual A_t values at key times
    markers = [(0, 'o', 't=0'), (peak_idx, '*', f't={t_hrs[peak_idx]:.0f}h (peak)'),
               (-1, 's', f't={t_hrs[-1]:.0f}h')]
    for idx, mkr, lbl in markers:
        at_mean = at_arr[idx].mean() / 1e6
        vsa_val = vsa_arr[idx]
        ax2.plot(at_mean, vsa_val, mkr, markersize=12, color='red',
                 markeredgecolor='black', zorder=10, label=f'{lbl}: VSA={vsa_val:.1f}%')

    ax2.axhline(100 - headwater_pct, color='gray', ls=':', lw=1,
                label=f'Ceiling ≈ {100-headwater_pct:.0f}% (all faccum>5 saturated)')
    ax2.set_xlabel(r"$A_t$ [km²]  (log scale)", fontsize=11)
    ax2.set_ylabel("VSA [% of watershed]", fontsize=11)
    ax2.set_title(r"VSA Ceiling: Maximum Possible VSA at Given $A_t$",
                  fontsize=12, fontweight='bold')
    ax2.legend(fontsize=8, loc='center left')
    ax2.grid(True, ls='--', alpha=0.3)
    ax2.set_ylim(-1, 35)

    # ── Panel 3: A_t(t) per polygon — LOG SCALE ─────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    for p in range(n_poly):
        ax3.semilogy(t_hrs, at_arr[:, p] / 1e6,
                     color=POLY_COLORS[p % 7], lw=1.8,
                     label=f'P{p} (SD={re_engine._SD_max_initial[p]:.3f}m)')

    ax3.axhline(A_1 / 1e6, color='gray', ls=':', lw=1,
                label=f'A_1 = {A_1/1e6:.6f} km² (1 cell)')
    ax3.axhline(5 * A_1 / 1e6, color='orange', ls='--', lw=1,
                label=f'5 cells = {5*A_1/1e6:.4f} km²')
    ax3.set_xlabel("Time [hr]", fontsize=11)
    ax3.set_ylabel(r"$A_t$ [km²]  (log scale)", fontsize=11)
    ax3.set_title(r"$A_t(t)$ per Polygon — Log Scale Reveals Dynamics",
                  fontsize=12, fontweight='bold')
    ax3.legend(fontsize=7, loc='upper right', ncol=2)
    ax3.grid(True, ls='--', alpha=0.3, which='both')

    # ── Panel 4: z(t) per polygon ────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    for p in range(n_poly):
        ax4.plot(t_hrs, z_arr[:, p], color=POLY_COLORS[p % 7], lw=1.8,
                 label=f'P{p}')
        ax4.axhline(re_engine._SD_max_initial[p], color=POLY_COLORS[p % 7],
                     ls=':', lw=0.8, alpha=0.5)

    ax4.set_xlabel("Time [hr]", fontsize=11)
    ax4.set_ylabel("z [m]  (saturated zone thickness)", fontsize=11)
    ax4.set_title("Saturated Zone z(t)  |  dotted = SD_max (full saturation)",
                  fontsize=12, fontweight='bold')
    ax4.legend(fontsize=7, loc='upper left', ncol=2)
    ax4.grid(True, ls='--', alpha=0.3)

    # ── Panel 5: Spatial map — can saturate vs never saturates ───────────
    ax5 = fig.add_subplot(gs[2, 0])
    dem_c = crop(dem_masked)
    ax5.imshow(dem_c, cmap='terrain', alpha=0.3, interpolation='nearest')

    can_c = crop(can_sat_2d)
    never = np.where(can_c == 0, 1.0, np.nan)
    can   = np.where(can_c == 1, 1.0, np.nan)

    ax5.imshow(never, cmap=ListedColormap(['#fee0d2']),
               alpha=0.6, interpolation='nearest')
    ax5.imshow(can, cmap=ListedColormap(['#3182bd']),
               alpha=0.5, interpolation='nearest')

    can_count = int(np.nansum(can_c == 1))
    never_count = int(np.nansum(can_c == 0))
    legend_patches = [
        mpatches.Patch(color='#3182bd', alpha=0.6,
                       label=f'Can saturate ({can_count:,} cells, '
                       f'{can_count/n_cells*100:.1f}%)'),
        mpatches.Patch(color='#fee0d2', alpha=0.6,
                       label=f'Never saturates ({never_count:,} cells, '
                       f'{never_count/n_cells*100:.1f}%)'),
    ]
    ax5.legend(handles=legend_patches, fontsize=8, loc='lower left',
               framealpha=0.9)
    ax5.set_title("Topographic VSA Potential  |  faccum > 5 threshold",
                  fontsize=12, fontweight='bold')
    ax5.set_xticks([]); ax5.set_yticks([])

    # ── Panel 6: VSA(t) + rainfall ───────────────────────────────────────
    ax6 = fig.add_subplot(gs[2, 1])
    ax6_rain = ax6.twinx()

    ax6.plot(t_hrs, vsa_arr, 'k-', lw=2.5, label='VSA %', zorder=5)
    ax6.axhline(100 - headwater_pct, color='gray', ls=':', lw=1, alpha=0.7)
    ax6.fill_between(t_hrs, vsa_arr, alpha=0.15, color='red')

    mean_rain = rain_arr.mean(axis=1)
    ax6_rain.bar(t_hrs, mean_rain, width=t_hrs[1]-t_hrs[0] if len(t_hrs) > 1 else 0.5,
                 color='#6baed6', alpha=0.4, label='Mean rainfall')
    ax6_rain.invert_yaxis()
    ax6_rain.set_ylabel("Rain [mm/hr]", fontsize=9, color='#6baed6')

    ax6.set_xlabel("Time [hr]", fontsize=11)
    ax6.set_ylabel("VSA [% of watershed]", fontsize=11)
    ax6.set_title("VSA Evolution with Rainfall", fontsize=12,
                  fontweight='bold')
    ax6.set_ylim(0, 35)
    ax6.grid(True, ls='--', alpha=0.3)

    ax6.text(0.98, 0.95,
             f"Peak VSA = {vsa_arr.max():.1f}%\n"
             f"Ceiling ≈ {100-headwater_pct:.0f}%\n"
             f"(72% cells are\n headwaters with\n faccum ≤ 5)",
             transform=ax6.transAxes, fontsize=8, va='top', ha='right',
             bbox=dict(facecolor='white', alpha=0.85, pad=3))

    fig.suptitle(
        "Why VSA Peaks at ~27%: Topographic Control on Saturation\n"
        "72% of cells are ridge-top headwaters (faccum ≤ 5) that can never "
        "saturate from below",
        fontsize=13, fontweight='bold', y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    out = OUTPUT / "at_faccum_analysis.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"\n  Saved: {out}")

    config.BACKEND = orig_backend
    config.OPM_Q_MAX = orig_qmax
    config.SERVES_TARGET_DATE = orig_date
    print("Done.")


if __name__ == "__main__":
    main()
