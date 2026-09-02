"""
tests/04_test_vsa_animation.py
==============================
Animated visualisation of the VSA-OPM runoff generation + kinematic-wave routing.

Layout (4-panel)
----------------
  LEFT (full height) – Watershed map
      • Hillshade background
      • Water depth overlay    (YlGnBu, semi-transparent)
      • VSA saturated cells    (YlOrRd, semi-transparent)  ← new
      • Outlet star + gauge markers
      • Time / rain / VSA% annotation

  TOP-RIGHT    – Outlet hydrograph (Q vs time)

  MID-RIGHT    – VSA evolution: saturated area [km²] vs time
                 + A_t threshold line on secondary y-axis

  BOT-RIGHT    – Sandbox state: SD_max [m] and z (saturated thickness) [m]

Output: output/vsa_routing_animation.gif

Requires: config.RUNOFF_SOURCE = 'vsa_opm'
"""

import os
import sys
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LightSource
import rasterio

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import config
from hydroflow.core import routing as ru
from hydroflow.core import runoff as ri

# ── Constants ────────────────────────────────────────────────────────────────
K_MS   = 44.0 / 86400.0
SD_MIN = 0.001
Q_MIN  = 0.001

_out_interval       = getattr(config, 'OUTPUT_INTERVAL_SECONDS', None)
FRAME_EVERY_N_STEPS = max(1, round((_out_interval or config.TIME_STEP_SECONDS * 10)
                                   / config.TIME_STEP_SECONDS))
GIF_FPS    = 6
GIF_DPI    = 110
OUTPUT_GIF = os.path.join(config.OUTPUT_DIR, "vsa_routing_animation.gif")
BG         = "#0f1117"

# ─────────────────────────────────────────────────────────────────────────────
# Guard: script is designed for vsa_opm mode
# ─────────────────────────────────────────────────────────────────────────────
_rsrc = getattr(config, 'RUNOFF_SOURCE', 'none').lower()
if _rsrc != 'vsa_opm':
    print(f"[WARNING] RUNOFF_SOURCE='{_rsrc}'.  This animation is designed for "
          "'vsa_opm'. VSA overlay will be empty — set RUNOFF_SOURCE='vsa_opm' "
          "in config.py for full visualisation.")

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Load rasters + build grid
# ─────────────────────────────────────────────────────────────────────────────
print("Loading rasters and building grid …")
dem, fdir, faccum, ws_mask, transform, nodata_dem, cell_size = ru.load_rasters(config)
nrows, ncols = dem.shape
cell_area    = cell_size ** 2

slope_2d  = ru.compute_slope_grid(dem, fdir, ws_mask, cell_size, config.MIN_SLOPE, nodata_dem)
s_rows, s_cols, outlet_rc = ru.topological_order(faccum, fdir, ws_mask)
ds_idx    = ru.build_downstream_map(s_rows, s_cols, fdir, ws_mask, nrows, ncols)
n_cells   = len(s_rows)
outlet_pos = n_cells - 1

slope_1d  = slope_2d[s_rows, s_cols]
faccum_1d = faccum[s_rows, s_cols].astype(np.float64)

# ─────────────────────────────────────────────────────────────────────────────
# 2.  Build precipitation engine
# ─────────────────────────────────────────────────────────────────────────────
from hydroflow.core.precip import PrecipEngine
grid_data_pe = {
    "s_rows": s_rows, "s_cols": s_cols,
    "nrows": nrows,   "ncols": ncols,
    "n_cells": n_cells, "ws_mask": ws_mask, "transform": transform,
}
precip_engine = PrecipEngine(config, grid_data_pe)

# ─────────────────────────────────────────────────────────────────────────────
# 3.  Build RunoffEngine (VSA-OPM)
# ─────────────────────────────────────────────────────────────────────────────
grid_data_full = {
    **grid_data_pe,
    "cell_area": cell_area, "cell_size": cell_size,
    "slope_1d": slope_1d,   "faccum_1d": faccum_1d,
    "outlet_rc": outlet_rc,
}
runoff_engine = ri.RunoffEngine(config, grid_data_full)
_is_vsa = (_rsrc == 'vsa_opm')

# ─────────────────────────────────────────────────────────────────────────────
# 4.  DEM hillshade + map geometry
# ─────────────────────────────────────────────────────────────────────────────
with rasterio.open(config.ROUTING_DEM_PATH) as src:
    dem_bg = src.read(1).astype(float)
if nodata_dem is not None:
    dem_bg[dem_bg == nodata_dem] = np.nan
dem_bg[~np.isfinite(dem_bg)] = np.nan

ls        = LightSource(azdeg=315, altdeg=45)
dem_valid = np.where(np.isnan(dem_bg), 0, dem_bg)
hillshade = ls.hillshade(dem_valid, vert_exag=2, dx=cell_size, dy=cell_size)

left   = transform.c
right  = transform.c + ncols * transform.a
top    = transform.f
bottom = transform.f + nrows * transform.e
map_extent = [left, right, bottom, top]

ws_ri, ws_ci = np.where(ws_mask)
pad  = max(5, int(max(ws_ri.max()-ws_ri.min(), ws_ci.max()-ws_ci.min()) * 0.02))
ws_x_min = left + (ws_ci.min() - pad) * transform.a
ws_x_max = left + (ws_ci.max() + 1 + pad) * transform.a
ws_y_min = top  + (ws_ri.max() + 1 + pad) * transform.e
ws_y_max = top  + (ws_ri.min() - pad)     * transform.e

outlet_x = left + (outlet_rc[1] + 0.5) * transform.a
outlet_y = top  + (outlet_rc[0] + 0.5) * transform.e

# ─────────────────────────────────────────────────────────────────────────────
# 5.  Gauge metadata (if thiessen / idw)
# ─────────────────────────────────────────────────────────────────────────────
_precip_method = getattr(config, 'PRECIP_METHOD', 'uniform').lower()
_gauge_mode    = (_precip_method in ('thiessen', 'idw'))

if _gauge_mode:
    gauges_df   = pd.read_csv(config.PRECIP_GAUGE_FILE).set_index('gauge_id')
    gauge_ids   = gauges_df.index.tolist()
    gauge_names = gauges_df['name'].tolist()
    gauge_xs    = gauges_df['easting_m'].values.astype(float)
    gauge_ys    = gauges_df['northing_m'].values.astype(float)
    n_gauges    = len(gauge_ids)
    GAUGE_COLORS = [plt.get_cmap('tab10')(i) for i in range(n_gauges)]

# ─────────────────────────────────────────────────────────────────────────────
# 6.  Simulation loop — capture frames
# ─────────────────────────────────────────────────────────────────────────────
dt      = float(config.TIME_STEP_SECONDS)
n_mann  = config.MANNINGS_N
n_steps = int(config.TOTAL_SIMULATION_TIME_HOURS * 3600.0 / dt)

print(f"\nRunning simulation + capturing frames …")
print(f"  Steps: {n_steps:,}  |  Frame every {FRAME_EVERY_N_STEPS} steps  "
      f"|  VSA mode: {_is_vsa}")

volume_1d = np.zeros(n_cells, dtype=np.float64)
Q_out_1d  = np.zeros(n_cells, dtype=np.float64)

# Frame storage
frame_times_hr = []
frame_depths   = []     # (nrows, ncols) water depth [m]
frame_vsa      = []     # (nrows, ncols) VSA mask (float, NaN outside ws)
frame_vsa_km2  = []     # scalar VSA area [km²]
frame_At_km2   = []     # scalar A_t [km²]
frame_SD       = []     # scalar SD_max [m]
frame_z        = []     # scalar z [m]

# Hydrograph
hydro_t_hr = []
hydro_Q    = []

t_wall = time.time()

for step in range(n_steps):
    t_s = step * dt

    rain_1d = precip_engine.get_field_1d(t_s)

    # ── Forward Euler: query current state → advance ──────────────────────
    if _is_vsa:
        source_1d = runoff_engine.get_effective_1d(t_s, rain_1d)
        diag      = runoff_engine.get_opm_diagnostics()
        runoff_engine.update_state(rain_1d, dt)
    else:
        source_1d = rain_1d
        diag      = {"VSA_m2": 0.0, "A_t_m2": 0.0, "SD_max_t": config.OPM_SD_MAX_INITIAL, "z_m": 0.0}

    # ── Routing ───────────────────────────────────────────────────────────
    inflow_1d = np.zeros(n_cells, dtype=np.float64)
    valid_ds  = ds_idx >= 0
    np.add.at(inflow_1d, ds_idx[valid_ds], Q_out_1d[valid_ds])

    depth_1d    = np.maximum(volume_1d / cell_area, config.MIN_DEPTH_M)
    velocity_1d = ru.mannings_velocity(depth_1d, slope_1d, n_mann)
    Q_out_1d    = ru.cell_discharge(depth_1d, velocity_1d, cell_size)
    Q_out_1d    = ru.flux_limiter(Q_out_1d, volume_1d, dt)

    volume_1d = np.maximum(
        volume_1d + source_1d * cell_area * dt + inflow_1d * dt - Q_out_1d * dt,
        0.0,
    )

    t_end_hr = (t_s + dt) / 3600.0
    hydro_t_hr.append(t_end_hr)
    hydro_Q.append(float(Q_out_1d[outlet_pos]))

    # ── Capture frame ─────────────────────────────────────────────────────
    if step % FRAME_EVERY_N_STEPS == 0 or step == n_steps - 1:
        # Depth map
        d2d = np.full((nrows, ncols), np.nan)
        d2d[s_rows, s_cols] = volume_1d / cell_area
        d2d[~ws_mask] = np.nan
        frame_depths.append(d2d.copy())

        # VSA map — 1.0 where saturated, NaN elsewhere
        if _is_vsa:
            vsa_mask_1d = runoff_engine._vsa_mask   # current mask
            v2d = np.full((nrows, ncols), np.nan)
            v2d[s_rows[vsa_mask_1d], s_cols[vsa_mask_1d]] = 1.0
        else:
            v2d = np.full((nrows, ncols), np.nan)
        frame_vsa.append(v2d.copy())

        frame_times_hr.append(t_end_hr)
        frame_vsa_km2.append(diag.get("VSA_m2", 0.0) / 1e6)

        # Per-polygon mode returns arrays; use mean for scalar time-series
        _at = diag.get("A_t_m2", 0.0)
        _sd = diag.get("SD_max_t", config.OPM_SD_MAX_INITIAL)
        _zv = diag.get("z_m", 0.0)
        frame_At_km2.append(float(np.mean(_at)) / 1e6 if hasattr(_at, '__len__') else _at / 1e6)
        frame_SD.append(float(np.mean(_sd)) if hasattr(_sd, '__len__') else _sd)
        frame_z.append(float(np.mean(_zv)) if hasattr(_zv, '__len__') else _zv)

        if (step // FRAME_EVERY_N_STEPS) % 10 == 0:
            vsa_frac = diag.get("VSA_m2", 0.0) / (cell_area * n_cells) * 100
            _sd_disp = float(np.mean(_sd)) if hasattr(_sd, '__len__') else _sd
            print(f"  Frame {len(frame_depths):3d}  t={t_end_hr:.2f}h"
                  f"  Q={hydro_Q[-1]:.2f} m³/s"
                  f"  VSA={vsa_frac:.1f}%"
                  f"  SD={_sd_disp:.4f}m"
                  f"  wall={time.time()-t_wall:.1f}s")

n_frames = len(frame_depths)
print(f"\nCaptured {n_frames} frames in {time.time()-t_wall:.1f}s")

# ── Convert to arrays ─────────────────────────────────────────────────────
hydro_t_hr    = np.array(hydro_t_hr)
hydro_Q       = np.array(hydro_Q)
frame_times_hr = np.array(frame_times_hr)
frame_vsa_km2  = np.array(frame_vsa_km2)
frame_At_km2   = np.array(frame_At_km2)
frame_SD       = np.array(frame_SD)
frame_z        = np.array(frame_z)

# ── Total watershed area ──────────────────────────────────────────────────
A_ws_km2 = n_cells * cell_area / 1e6

# ─────────────────────────────────────────────────────────────────────────────
# 7.  Build figure
# ─────────────────────────────────────────────────────────────────────────────
print("Rendering animation …")

max_depth_vis = max(
    np.nanpercentile(d, 99) for d in frame_depths if np.any(np.isfinite(d))
)
max_depth_vis = max(max_depth_vis, 1e-4)
peak_Q        = max(float(hydro_Q.max()), 1e-3)

fig = plt.figure(figsize=(22, 10), facecolor=BG)
gs  = GridSpec(3, 2, figure=fig,
               width_ratios=[1.4, 1],
               height_ratios=[1, 1, 1],
               hspace=0.45, wspace=0.32)

ax_map = fig.add_subplot(gs[:, 0])    # full-height left
ax_hyd = fig.add_subplot(gs[0, 1])    # top-right: hydrograph
ax_vsa = fig.add_subplot(gs[1, 1])    # mid-right: VSA area
ax_sd  = fig.add_subplot(gs[2, 1])    # bot-right: SD_max / z

all_ax = [ax_map, ax_hyd, ax_vsa, ax_sd]
fig.patch.set_facecolor(BG)
for ax in all_ax:
    ax.set_facecolor(BG)
    ax.tick_params(colors="white", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")

# ─────────────────────────────────────────────────────────────────────────────
# 8.  Map panel
# ─────────────────────────────────────────────────────────────────────────────
ax_map.imshow(hillshade, cmap="gray", vmin=0, vmax=1,
              aspect="equal", alpha=0.55, extent=map_extent, origin="upper")
ax_map.imshow(np.where(ws_mask, 0.5, np.nan), cmap="Greys", vmin=0, vmax=1,
              aspect="equal", alpha=0.15, extent=map_extent, origin="upper")

# Water depth layer
depth_im = ax_map.imshow(
    frame_depths[0], cmap="YlGnBu",
    vmin=0, vmax=max_depth_vis,
    aspect="equal", alpha=0.80,
    extent=map_extent, origin="upper",
)

# VSA overlay layer (YlOrRd: yellow → orange → red for saturated cells)
vsa_im = ax_map.imshow(
    frame_vsa[0], cmap="YlOrRd",
    vmin=0, vmax=1,
    aspect="equal", alpha=0.45,
    extent=map_extent, origin="upper",
)

ax_map.set_xlim(ws_x_min, ws_x_max)
ax_map.set_ylim(ws_y_min, ws_y_max)

# Outlet
ax_map.plot(outlet_x, outlet_y, marker="*", color="#FF4444",
            markersize=12, zorder=10, linestyle="None", label="Outlet")

# Gauge markers
gauge_scat_dyn = None
if _gauge_mode:
    for gi in range(n_gauges):
        ax_map.plot(gauge_xs[gi], gauge_ys[gi],
                    marker="^", color=GAUGE_COLORS[gi],
                    markersize=9, zorder=12, linestyle="None")
        ax_map.annotate(gauge_names[gi],
                        xy=(gauge_xs[gi], gauge_ys[gi]),
                        xytext=(6, 5), textcoords="offset points",
                        color=GAUGE_COLORS[gi], fontsize=7, fontweight="bold")
    init_rain = precip_engine._interp_gauges(0.0) * 3600.0 * 1000.0
    max_rain_mmhr = max(float(init_rain.max()), 0.1)
    # dynamically-sized bubbles  (no init data needed now, updated per frame)
    gauge_scat_dyn = ax_map.scatter(
        gauge_xs, gauge_ys,
        s=40, c=[list(c) for c in GAUGE_COLORS],
        alpha=0.55, zorder=11, linewidths=1.5, edgecolors="white",
    )

ax_map.legend(loc="lower right", facecolor="#1a1a2e", labelcolor="white",
              fontsize=8, framealpha=0.75)

# Depth colorbar
cbar_d = fig.colorbar(depth_im, ax=ax_map, fraction=0.025, pad=0.01)
cbar_d.set_label("Depth [m]", color="white", fontsize=8)
cbar_d.ax.yaxis.set_tick_params(color="white")
plt.setp(cbar_d.ax.yaxis.get_ticklabels(), color="white", fontsize=7)

ax_map.set_title("Water Depth  +  VSA (orange = saturated)",
                 color="white", fontsize=11, pad=6)
ax_map.set_xlabel("Easting  [m]",  color="#aaa", fontsize=8)
ax_map.set_ylabel("Northing  [m]", color="#aaa", fontsize=8)
ax_map.ticklabel_format(style="sci", axis="both", scilimits=(3, 3))

# Annotations
time_text = ax_map.text(
    0.03, 0.97, "", transform=ax_map.transAxes,
    color="white", fontsize=10, va="top",
    bbox=dict(boxstyle="round,pad=0.3", fc="#1a1a2e", alpha=0.85),
)
vsa_text = ax_map.text(
    0.03, 0.88, "", transform=ax_map.transAxes,
    color="#FF9900", fontsize=9, va="top",
    bbox=dict(boxstyle="round,pad=0.3", fc="#1a1a2e", alpha=0.85),
)
rain_text = ax_map.text(
    0.03, 0.80, "", transform=ax_map.transAxes,
    color="#FFD700", fontsize=9, va="top",
    bbox=dict(boxstyle="round,pad=0.3", fc="#1a1a2e", alpha=0.85),
)

# ─────────────────────────────────────────────────────────────────────────────
# 9.  Hydrograph panel
# ─────────────────────────────────────────────────────────────────────────────
rain_end_hr = (precip_engine.rain_end_seconds / 3600.0
               if _precip_method != 'uniform'
               else config.RAIN_DURATION_HOURS)

ax_hyd.set_facecolor(BG)
ax_hyd.set_xlim(0, config.TOTAL_SIMULATION_TIME_HOURS)
ax_hyd.set_ylim(0, peak_Q * 1.15)
ax_hyd.set_xlabel("Time  [h]", color="#aaa", fontsize=8)
ax_hyd.set_ylabel("Q  [m³/s]", color="#aaa", fontsize=8)
ax_hyd.set_title("Outlet Hydrograph", color="white", fontsize=10, pad=4)
ax_hyd.grid(True, color="#333", linewidth=0.5, linestyle="--")
ax_hyd.axvspan(0, rain_end_hr, alpha=0.12, color="#4fc3f7", label="Rain period")
ax_hyd.legend(loc="upper left", facecolor="#1a1a2e", labelcolor="white",
              fontsize=7, framealpha=0.7)

hyd_line,   = ax_hyd.plot([], [], color="#00e5ff", linewidth=1.8)
vline_hyd    = ax_hyd.axvline(0, color="#FF4444", linewidth=1.2, linestyle="--")
peak_annot   = ax_hyd.annotate("", xy=(0, 0), xytext=(0, 0),
                                color="#FF4444", fontsize=7,
                                arrowprops=dict(arrowstyle="->", color="#FF4444"))

# ─────────────────────────────────────────────────────────────────────────────
# 10. VSA evolution panel
# ─────────────────────────────────────────────────────────────────────────────
ax_vsa.set_facecolor(BG)
ax_vsa.set_xlim(0, config.TOTAL_SIMULATION_TIME_HOURS)
_vsa_max = max(float(frame_vsa_km2.max()), 0.1)
ax_vsa.set_ylim(0, _vsa_max * 1.15)
ax_vsa.set_xlabel("Time  [h]", color="#aaa", fontsize=8)
ax_vsa.set_ylabel("VSA  [km²]", color="#FF9900", fontsize=8)
ax_vsa.set_title("Variable Source Area (VSA) Evolution", color="white", fontsize=10, pad=4)
ax_vsa.tick_params(axis='y', colors="#FF9900")
ax_vsa.grid(True, color="#333", linewidth=0.5, linestyle="--")
ax_vsa.axhline(A_ws_km2, color="#aaaaaa", linewidth=0.8, linestyle=":",
               label=f"Watershed ({A_ws_km2:.0f} km²)")
ax_vsa.axvspan(0, rain_end_hr, alpha=0.10, color="#4fc3f7")

# Secondary y-axis: A_t [km²]
ax_at = ax_vsa.twinx()
ax_at.set_facecolor(BG)
_At_max = max(float(frame_At_km2.max()), 0.1)
ax_at.set_ylim(0, _At_max * 1.15)
ax_at.set_ylabel("A_t  [km²]", color="#88CCFF", fontsize=8)
ax_at.tick_params(axis='y', colors="#88CCFF", labelsize=7)
for spine in ax_at.spines.values():
    spine.set_edgecolor("#444")

vsa_line,  = ax_vsa.plot([], [], color="#FF9900", linewidth=2.0, label="VSA area")
At_line,   = ax_at.plot([], [], color="#88CCFF", linewidth=1.4,
                        linestyle="--", label="A_t threshold")
vline_vsa  = ax_vsa.axvline(0, color="#FF4444", linewidth=1.2, linestyle="--")

# Combined legend
lines1, labs1 = ax_vsa.get_legend_handles_labels()
lines2, labs2 = ax_at.get_legend_handles_labels()
ax_vsa.legend(lines1 + lines2, labs1 + labs2,
              loc="upper left", facecolor="#1a1a2e", labelcolor="white",
              fontsize=7, framealpha=0.7)

# ─────────────────────────────────────────────────────────────────────────────
# 11. Sandbox state panel (SD_max and z)
# ─────────────────────────────────────────────────────────────────────────────
ax_sd.set_facecolor(BG)
ax_sd.set_xlim(0, config.TOTAL_SIMULATION_TIME_HOURS)
_sd_max_val = float(config.OPM_SD_MAX_INITIAL)
ax_sd.set_ylim(0, _sd_max_val * 1.15)
ax_sd.set_xlabel("Time  [h]", color="#aaa", fontsize=8)
ax_sd.set_ylabel("SD_max  [m]", color="#77DD77", fontsize=8)
ax_sd.set_title("Sandbox Water Balance  (Divide Cell)", color="white", fontsize=10, pad=4)
ax_sd.tick_params(axis='y', colors="#77DD77")
ax_sd.grid(True, color="#333", linewidth=0.5, linestyle="--")
ax_sd.axhline(SD_MIN, color="#888888", linewidth=0.8, linestyle=":",
              label=f"SD_min = {SD_MIN} m")
ax_sd.axvspan(0, rain_end_hr, alpha=0.10, color="#4fc3f7")

# Secondary y-axis: z [m]
ax_z = ax_sd.twinx()
ax_z.set_facecolor(BG)
_z_max = max(float(frame_z.max()), 1e-4) * 1.2
ax_z.set_ylim(0, _z_max)
ax_z.set_ylabel("z (sat. thickness)  [m]", color="#FF88AA", fontsize=8)
ax_z.tick_params(axis='y', colors="#FF88AA", labelsize=7)
for spine in ax_z.spines.values():
    spine.set_edgecolor("#444")

sd_line,  = ax_sd.plot([], [], color="#77DD77", linewidth=2.0, label="SD_max(t)")
z_line,   = ax_z.plot([], [], color="#FF88AA",  linewidth=1.4,
                      linestyle="--", label="z(t)")
vline_sd  = ax_sd.axvline(0, color="#FF4444", linewidth=1.2, linestyle="--")

lines3, labs3 = ax_sd.get_legend_handles_labels()
lines4, labs4 = ax_z.get_legend_handles_labels()
ax_sd.legend(lines3 + lines4, labs3 + labs4,
             loc="upper right", facecolor="#1a1a2e", labelcolor="white",
             fontsize=7, framealpha=0.7)

# ─────────────────────────────────────────────────────────────────────────────
# 12. Pre-compute gauge rain at frame times (gauge mode)
# ─────────────────────────────────────────────────────────────────────────────
if _gauge_mode:
    _max_rain_mmhr_dyn = 0.1
    gauge_mmhr_frames  = np.zeros((n_frames, n_gauges))
    for fi, t_hr in enumerate(frame_times_hr):
        r = precip_engine._interp_gauges(t_hr * 3600.0) * 3600.0 * 1000.0
        gauge_mmhr_frames[fi] = r
    _max_rain_mmhr_dyn = max(float(gauge_mmhr_frames.max()), 0.1)

# ─────────────────────────────────────────────────────────────────────────────
# 13. Animation update function
# ─────────────────────────────────────────────────────────────────────────────
def update(fi):
    t_hr = float(frame_times_hr[fi])

    # ── Map: depth ────────────────────────────────────────────────────────────
    depth_im.set_data(np.clip(frame_depths[fi], 0, max_depth_vis))

    # ── Map: VSA overlay ──────────────────────────────────────────────────────
    vsa_im.set_data(frame_vsa[fi])

    # ── Annotations ──────────────────────────────────────────────────────────
    vsa_km2   = frame_vsa_km2[fi]
    vsa_frac  = vsa_km2 / A_ws_km2 * 100.0
    sd_now    = frame_SD[fi]
    At_now    = frame_At_km2[fi]

    time_text.set_text(f"t = {t_hr:.2f} h")
    vsa_text.set_text(f"VSA = {vsa_km2:.1f} km²  ({vsa_frac:.1f}%)\n"
                      f"A_t = {At_now:.2f} km²")
    is_raining = precip_engine.is_raining(t_hr * 3600.0)
    rain_text.set_text("☂  Raining" if is_raining else "☀  Dry period")

    # ── Hydrograph ────────────────────────────────────────────────────────────
    mask = hydro_t_hr <= t_hr
    hyd_line.set_data(hydro_t_hr[mask], hydro_Q[mask])
    vline_hyd.set_xdata([t_hr, t_hr])
    if mask.any():
        lpi = int(np.argmax(hydro_Q[mask]))
        lpQ = hydro_Q[mask][lpi]
        lpt = hydro_t_hr[mask][lpi]
        if lpQ > 0.01:
            peak_annot.xy     = (lpt, lpQ)
            peak_annot.xytext = (min(lpt + 0.3, config.TOTAL_SIMULATION_TIME_HOURS * 0.9),
                                 lpQ * 0.80)
            peak_annot.set_text(f"Peak\n{lpQ:.1f} m³/s")
        else:
            peak_annot.set_text("")

    # ── VSA evolution ─────────────────────────────────────────────────────────
    fmask = frame_times_hr <= t_hr
    vsa_line.set_data(frame_times_hr[fmask], frame_vsa_km2[fmask])
    At_line.set_data( frame_times_hr[fmask], frame_At_km2[fmask])
    vline_vsa.set_xdata([t_hr, t_hr])

    # ── Sandbox state ─────────────────────────────────────────────────────────
    sd_line.set_data(frame_times_hr[fmask], frame_SD[fmask])
    z_line.set_data( frame_times_hr[fmask], frame_z[fmask])
    vline_sd.set_xdata([t_hr, t_hr])

    artists = [depth_im, vsa_im, hyd_line, vline_hyd, peak_annot,
               vsa_line, At_line, vline_vsa,
               sd_line, z_line, vline_sd,
               time_text, vsa_text, rain_text]

    # ── Gauge bubbles ─────────────────────────────────────────────────────────
    if _gauge_mode and gauge_scat_dyn is not None:
        cur_mmhr = gauge_mmhr_frames[fi]
        sizes    = (cur_mmhr / _max_rain_mmhr_dyn) * 600 + 40
        gauge_scat_dyn.set_sizes(sizes)
        artists.append(gauge_scat_dyn)

    return artists


# ─────────────────────────────────────────────────────────────────────────────
# 14. Render and save GIF
# ─────────────────────────────────────────────────────────────────────────────
anim = animation.FuncAnimation(
    fig, update,
    frames=n_frames,
    interval=1000 // GIF_FPS,
    blit=True,
)

os.makedirs(config.OUTPUT_DIR, exist_ok=True)
writer = animation.PillowWriter(fps=GIF_FPS)
anim.save(OUTPUT_GIF, writer=writer, dpi=GIF_DPI,
          savefig_kwargs={"facecolor": fig.get_facecolor()})
plt.close(fig)

print(f"\n  Animation saved → {OUTPUT_GIF}")
print(f"  Frames: {n_frames}   FPS: {GIF_FPS}   DPI: {GIF_DPI}")
print(f"\n  VSA summary:")
print(f"    Initial VSA : {frame_vsa_km2[0]:.2f} km² ({frame_vsa_km2[0]/A_ws_km2*100:.1f}%)")
print(f"    Peak VSA    : {frame_vsa_km2.max():.2f} km² ({frame_vsa_km2.max()/A_ws_km2*100:.1f}%)"
      f"  at t={frame_times_hr[int(np.argmax(frame_vsa_km2))]:.2f} h")
print(f"    Final VSA   : {frame_vsa_km2[-1]:.2f} km² ({frame_vsa_km2[-1]/A_ws_km2*100:.1f}%)")
print(f"    Min SD_max  : {frame_SD.min():.5f} m  (SD_min = {SD_MIN} m)")
print(f"    Max z       : {frame_z.max():.4f} m")
