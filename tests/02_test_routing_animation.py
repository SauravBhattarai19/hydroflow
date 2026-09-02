"""
tests/02_test_routing_animation.py
===================================
Visualises the kinematic-wave routing model spatially AND temporally.

Layout (uniform mode)
---------------------
  LEFT  – 2-D water depth map over the watershed
  RIGHT – Outlet hydrograph building up in real time

Layout (thiessen / idw mode)  ← extra panel unlocked
-----------------------------------------------------
  LEFT        – 2-D water depth map + gauge location markers (▲)
                 Marker SIZE scales with current rain intensity at that gauge.
  TOP-RIGHT   – Outlet hydrograph
  BOTTOM-RIGHT– Gauge precipitation timeseries (one line / gauge)
                 Vertical cursor stays in sync with the hydrograph cursor.
                 Current mm/hr value per gauge annotated next to the cursor.

Output:  output/routing_animation.gif
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

# ── Make parent directory importable ──────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import config
from hydroflow.core import routing as ru
from hydroflow.core import precip as pi
from hydroflow.core.routing.router import initialise_grid

# ─────────────────────────────────────────────────────────────────────────────
# SETTINGS
# ─────────────────────────────────────────────────────────────────────────────
_out_interval       = getattr(config, 'OUTPUT_INTERVAL_SECONDS', None)
FRAME_EVERY_N_STEPS = max(1, round((_out_interval or config.TIME_STEP_SECONDS * 10)
                                   / config.TIME_STEP_SECONDS))
GIF_FPS             = 8
GIF_DPI             = 110
OUTPUT_GIF          = os.path.join(config.OUTPUT_DIR, "routing_animation.gif")
BG                  = "#0f1117"   # dark background colour

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Initialise grid (loads rasters, builds topo order + PrecipEngine)
# ─────────────────────────────────────────────────────────────────────────────
print("Initialising routing grid …")
grid_data = initialise_grid(config)

s_rows     = grid_data["s_rows"]
s_cols     = grid_data["s_cols"]
slope_1d   = grid_data["slope_1d"]
ds_idx     = grid_data["ds_idx"]
n_cells    = grid_data["n_cells"]
outlet_pos = grid_data["outlet_pos"]
cell_area  = grid_data["cell_area"]
nrows      = grid_data["nrows"]
ncols      = grid_data["ncols"]
ws_mask    = grid_data["ws_mask"]

dx             = grid_data["cell_size"]
dt             = config.TIME_STEP_SECONDS
n_mann         = config.MANNINGS_N
precip_engine  = grid_data["precip_engine"]
n_steps        = int(config.TOTAL_SIMULATION_TIME_HOURS * 3600.0 / dt)

_precip_method = getattr(config, 'PRECIP_METHOD', 'uniform').lower()
_gauge_mode    = (_precip_method in ('thiessen', 'idw'))

print(f"Total steps : {n_steps:,}  |  frames : {n_steps // FRAME_EVERY_N_STEPS}"
      f"  |  precip method : {_precip_method}")

# ─────────────────────────────────────────────────────────────────────────────
# 2.  Load gauge metadata (gauge mode only)
# ─────────────────────────────────────────────────────────────────────────────
if _gauge_mode:
    gauges_df   = pd.read_csv(config.PRECIP_GAUGE_FILE).set_index('gauge_id')
    gauge_ids   = gauges_df.index.tolist()
    gauge_names = gauges_df['name'].tolist()
    gauge_xs    = gauges_df['easting_m'].values.astype(float)
    gauge_ys    = gauges_df['northing_m'].values.astype(float)
    n_gauges    = len(gauge_ids)
    # One distinct colour per gauge
    GAUGE_COLORS = [plt.get_cmap('tab10')(i) for i in range(n_gauges)]
    print(f"  Gauges loaded   |  {n_gauges} gauges: {gauge_names}")

# ─────────────────────────────────────────────────────────────────────────────
# 3.  Load DEM → hillshade + geographic extent
# ─────────────────────────────────────────────────────────────────────────────
with rasterio.open(config.ROUTING_DEM_PATH) as src:
    dem_bg     = src.read(1).astype(float)
    transform  = src.transform
    nodata_dem = src.nodata

if nodata_dem is not None:
    dem_bg[dem_bg == nodata_dem] = np.nan
dem_bg[~np.isfinite(dem_bg)] = np.nan

ls        = LightSource(azdeg=315, altdeg=45)
dem_valid = np.where(np.isnan(dem_bg), 0, dem_bg)
hillshade = ls.hillshade(dem_valid, vert_exag=2, dx=dx, dy=dx)

# imshow extent = [left, right, bottom, top] in projected metres
left   = transform.c
right  = transform.c + ncols * transform.a
top    = transform.f
bottom = transform.f + nrows * transform.e   # transform.e < 0
map_extent = [left, right, bottom, top]

# Watershed bounding box for axis zoom
ws_rows_idx, ws_cols_idx = np.where(ws_mask)
pad_cells = max(5, int(max(ws_rows_idx.max() - ws_rows_idx.min(),
                           ws_cols_idx.max() - ws_cols_idx.min()) * 0.02))
ws_x_min = left + (ws_cols_idx.min() - pad_cells) * transform.a
ws_x_max = left + (ws_cols_idx.max() + 1 + pad_cells) * transform.a
ws_y_min = top  + (ws_rows_idx.max() + 1 + pad_cells) * transform.e
ws_y_max = top  + (ws_rows_idx.min() - pad_cells)     * transform.e

outlet_x = left + (grid_data["outlet_rc"][1] + 0.5) * transform.a
outlet_y = top  + (grid_data["outlet_rc"][0] + 0.5) * transform.e

# ─────────────────────────────────────────────────────────────────────────────
# 4.  Run the time loop, capturing frames
# ─────────────────────────────────────────────────────────────────────────────
print("Running simulation and capturing frames …")

volume_1d  = np.zeros(n_cells, dtype=np.float64)
Q_out_1d   = np.zeros(n_cells, dtype=np.float64)

frame_times_hr = []
frame_depths   = []
hydro_times_hr = []
hydro_Q        = []

t_wall = time.time()

for step in range(n_steps):
    t_seconds = step * dt

    rain_1d   = precip_engine.get_field_1d(t_seconds)

    inflow_1d = np.zeros(n_cells, dtype=np.float64)
    valid_ds  = ds_idx >= 0
    np.add.at(inflow_1d, ds_idx[valid_ds], Q_out_1d[valid_ds])

    depth_1d    = np.maximum(volume_1d / cell_area, config.MIN_DEPTH_M)
    velocity_1d = ru.mannings_velocity(depth_1d, slope_1d, n_mann)
    Q_out_1d    = ru.cell_discharge(depth_1d, velocity_1d, dx)
    Q_out_1d    = ru.flux_limiter(Q_out_1d, volume_1d, dt)

    volume_1d = np.maximum(
        volume_1d + rain_1d * cell_area * dt + inflow_1d * dt - Q_out_1d * dt,
        0.0
    )

    t_end_hr = (t_seconds + dt) / 3600.0
    hydro_times_hr.append(t_end_hr)
    hydro_Q.append(Q_out_1d[outlet_pos])

    if step % FRAME_EVERY_N_STEPS == 0 or step == n_steps - 1:
        depth_2d               = np.full((nrows, ncols), np.nan)
        depth_2d[s_rows, s_cols] = volume_1d / cell_area
        depth_2d[~ws_mask]     = np.nan
        frame_depths.append(depth_2d.copy())
        frame_times_hr.append(t_end_hr)

        if (step // FRAME_EVERY_N_STEPS) % 10 == 0:
            print(f"  Frame {len(frame_depths):3d}  |  t={t_end_hr:.2f}h"
                  f"  |  Q={hydro_Q[-1]:.3f} m³/s"
                  f"  |  wall={time.time()-t_wall:.1f}s")

print(f"Captured {len(frame_depths)} frames in {time.time()-t_wall:.1f}s")

# ─────────────────────────────────────────────────────────────────────────────
# 5.  Pre-compute gauge precipitation at every frame time (gauge mode)
# ─────────────────────────────────────────────────────────────────────────────
# gauge_mmhr_at_frames  shape (n_frames, n_gauges)  units mm/hr
# dense_gauge_mmhr      shape (T_knots, n_gauges)   for static line drawing
if _gauge_mode:
    n_fr = len(frame_times_hr)
    gauge_mmhr_at_frames = np.zeros((n_fr, n_gauges), dtype=np.float64)
    for fi, t_hr in enumerate(frame_times_hr):
        r = precip_engine._interp_gauges(t_hr * 3600.0)   # m/s per gauge
        gauge_mmhr_at_frames[fi] = r * 1000.0 * 3600.0    # → mm/hr

    # Static lines use the engine's internal time knots (linear interpolation
    # between CSV rows, so knots are the exact breakpoints — no denser sampling needed)
    dense_t_s  = precip_engine._time_s                     # (T,)
    dense_t_hr = dense_t_s / 3600.0
    dense_gauge_mmhr = np.zeros((len(dense_t_s), n_gauges), dtype=np.float64)
    for ti, ts in enumerate(dense_t_s):
        r = precip_engine._interp_gauges(ts)
        dense_gauge_mmhr[ti] = r * 1000.0 * 3600.0

    max_rain_mmhr = max(float(dense_gauge_mmhr.max()), 0.1)

# ─────────────────────────────────────────────────────────────────────────────
# 6.  Build figure
# ─────────────────────────────────────────────────────────────────────────────
print("Rendering animation …")

hydro_times_hr = np.array(hydro_times_hr)
hydro_Q        = np.array(hydro_Q)
n_frames       = len(frame_depths)

max_depth_vis = max(
    np.nanpercentile(d, 99) for d in frame_depths if np.any(~np.isnan(d))
)
max_depth_vis = max(max_depth_vis, 1e-4)
peak_Q        = max(hydro_Q.max(), 1e-3)

# ── Figure layout ─────────────────────────────────────────────────────────
if _gauge_mode:
    # 2 columns: map spans full height | stacked hydro + precip panel
    fig = plt.figure(figsize=(19, 9), facecolor=BG)
    gs  = GridSpec(2, 2, figure=fig,
                   width_ratios=[1.3, 1], height_ratios=[1, 1],
                   hspace=0.40, wspace=0.32)
    ax_map  = fig.add_subplot(gs[:, 0])
    ax_hyd  = fig.add_subplot(gs[0, 1])
    ax_prec = fig.add_subplot(gs[1, 1])
    all_axes = [ax_map, ax_hyd, ax_prec]
else:
    fig, (ax_map, ax_hyd) = plt.subplots(
        1, 2, figsize=(14, 6), facecolor=BG,
        gridspec_kw={"width_ratios": [1.2, 1]}
    )
    ax_prec  = None
    all_axes = [ax_map, ax_hyd]

fig.patch.set_facecolor(BG)
for ax in all_axes:
    ax.set_facecolor(BG)
    ax.tick_params(colors="white", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")

# ─────────────────────────────────────────────────────────────────────────────
# 7.  Spatial map panel
# ─────────────────────────────────────────────────────────────────────────────
ax_map.imshow(
    hillshade, cmap="gray", vmin=0, vmax=1,
    aspect="equal", alpha=0.55,
    extent=map_extent, origin="upper",
)
ax_map.imshow(
    np.where(ws_mask, 0.5, np.nan), cmap="Greys", vmin=0, vmax=1,
    aspect="equal", alpha=0.15,
    extent=map_extent, origin="upper",
)
depth_im = ax_map.imshow(
    frame_depths[0], cmap="YlGnBu",
    vmin=0, vmax=max_depth_vis,
    aspect="equal", alpha=0.85,
    extent=map_extent, origin="upper",
)
ax_map.set_xlim(ws_x_min, ws_x_max)
ax_map.set_ylim(ws_y_min, ws_y_max)

# Outlet star
ax_map.plot(outlet_x, outlet_y, marker="*", color="#FF4444",
            markersize=12, zorder=10, label="Outlet", linestyle="None")

# ── Gauge markers (gauge mode) ─────────────────────────────────────────────
# Default stubs (overwritten below if gauge mode)
gauge_scat_dyn = None

if _gauge_mode:
    # Static triangle pin + name label for each gauge
    for gi in range(n_gauges):
        ax_map.plot(gauge_xs[gi], gauge_ys[gi],
                    marker="^", color=GAUGE_COLORS[gi],
                    markersize=9, zorder=12, linestyle="None",
                    label=gauge_names[gi])
        ax_map.annotate(
            gauge_names[gi],
            xy=(gauge_xs[gi], gauge_ys[gi]),
            xytext=(6, 5), textcoords="offset points",
            color=GAUGE_COLORS[gi], fontsize=7, fontweight="bold",
            zorder=13,
        )
    # Dynamic scatter: circle size ∝ current rain intensity
    # (grows during rain, shrinks to minimum during dry period)
    init_sizes = (gauge_mmhr_at_frames[0] / max_rain_mmhr) * 600 + 40
    gauge_scat_dyn = ax_map.scatter(
        gauge_xs, gauge_ys,
        s=init_sizes,
        c=[list(c) for c in GAUGE_COLORS],
        alpha=0.55, zorder=11,
        linewidths=1.5, edgecolors="white",
    )

ax_map.legend(loc="lower right", facecolor="#1a1a2e", labelcolor="white",
              fontsize=7.5, framealpha=0.75)

cbar = fig.colorbar(depth_im, ax=ax_map, fraction=0.035, pad=0.02)
cbar.set_label("Water Depth  [m]", color="white", fontsize=9)
cbar.ax.yaxis.set_tick_params(color="white")
plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

ax_map.set_title("Watershed Water Depth", color="white", fontsize=11, pad=6)
ax_map.set_xlabel("Easting  [m]",  color="#aaa", fontsize=8)
ax_map.set_ylabel("Northing  [m]", color="#aaa", fontsize=8)
ax_map.ticklabel_format(style="sci", axis="both", scilimits=(3, 3))

time_text = ax_map.text(
    0.03, 0.97, "", transform=ax_map.transAxes,
    color="white", fontsize=10, va="top",
    bbox=dict(boxstyle="round,pad=0.3", fc="#1a1a2e", alpha=0.8),
)
rain_text = ax_map.text(
    0.03, 0.88, "", transform=ax_map.transAxes,
    color="#FFD700", fontsize=9, va="top",
    bbox=dict(boxstyle="round,pad=0.3", fc="#1a1a2e", alpha=0.8),
)

# ─────────────────────────────────────────────────────────────────────────────
# 8.  Hydrograph panel
# ─────────────────────────────────────────────────────────────────────────────
ax_hyd.set_xlim(0, config.TOTAL_SIMULATION_TIME_HOURS)
ax_hyd.set_ylim(0, peak_Q * 1.15)
ax_hyd.set_xlabel("Time  [hours]", color="#aaa", fontsize=9)
ax_hyd.set_ylabel("Q at Outlet  [m³/s]", color="#aaa", fontsize=9)
ax_hyd.set_title("Outlet Hydrograph", color="white", fontsize=11, pad=6)
ax_hyd.grid(True, color="#333", linewidth=0.5, linestyle="--")

if _precip_method == 'uniform':
    rain_end_hr = config.RAIN_DURATION_HOURS
    rain_label  = f"Rainfall ({config.RAIN_INTENSITY_MM_HR} mm/hr, uniform)"
else:
    rain_end_hr = precip_engine.rain_end_seconds / 3600.0
    rain_label  = f"Rainfall period ({_precip_method})"
ax_hyd.axvspan(0, rain_end_hr, alpha=0.12, color="#4fc3f7", label=rain_label)
ax_hyd.legend(loc="upper right", facecolor="#1a1a2e", labelcolor="white",
              fontsize=8, framealpha=0.7)

hyd_line,  = ax_hyd.plot([], [], color="#00e5ff", linewidth=1.8, zorder=4)
peak_annot  = ax_hyd.annotate(
    "", xy=(0, 0), xytext=(0, 0),
    color="#FF4444", fontsize=8,
    arrowprops=dict(arrowstyle="->", color="#FF4444"),
)
vline = ax_hyd.axvline(0, color="#FF4444", linewidth=1.2, linestyle="--", zorder=5)

# ─────────────────────────────────────────────────────────────────────────────
# 9.  Gauge precipitation panel (gauge mode only)
# ─────────────────────────────────────────────────────────────────────────────
vline_prec     = None
gauge_val_texts = []

if _gauge_mode:
    ax_prec.set_facecolor(BG)
    ax_prec.set_xlim(0, config.TOTAL_SIMULATION_TIME_HOURS)
    ax_prec.set_ylim(0, max_rain_mmhr * 1.25)
    ax_prec.set_xlabel("Time  [hours]", color="#aaa", fontsize=9)
    ax_prec.set_ylabel("Intensity  [mm/hr]", color="#aaa", fontsize=9)
    ax_prec.set_title(
        f"Gauge Precipitation  ({_precip_method.title()} weighting)",
        color="white", fontsize=11, pad=6,
    )
    ax_prec.grid(True, color="#333", linewidth=0.5, linestyle="--")

    # Static full timeseries line per gauge
    for gi in range(n_gauges):
        ax_prec.plot(
            dense_t_hr, dense_gauge_mmhr[:, gi],
            color=GAUGE_COLORS[gi], linewidth=1.8,
            label=gauge_names[gi], zorder=4,
        )
    ax_prec.legend(loc="upper right", facecolor="#1a1a2e", labelcolor="white",
                   fontsize=8, framealpha=0.7)

    # Vertical time cursor (shared with hydrograph)
    vline_prec = ax_prec.axvline(
        0, color="#FF4444", linewidth=1.2, linestyle="--", zorder=6,
    )

    # Current-value labels: one per gauge, placed just right of the cursor.
    # Initialised off-screen; update() positions them each frame.
    for gi in range(n_gauges):
        txt = ax_prec.text(
            0, 0, "",
            color=GAUGE_COLORS[gi], fontsize=8,
            va="center", ha="left", zorder=7,
            fontweight="bold",
        )
        gauge_val_texts.append(txt)

if not _gauge_mode:
    plt.tight_layout(pad=1.5)

# ─────────────────────────────────────────────────────────────────────────────
# 10. Animation update function
# ─────────────────────────────────────────────────────────────────────────────
def update(frame_idx):
    t_hr     = frame_times_hr[frame_idx]
    depth_2d = frame_depths[frame_idx]

    # ── Spatial map ───────────────────────────────────────────────────────────
    depth_im.set_data(np.where(np.isnan(depth_2d), np.nan,
                               np.clip(depth_2d, 0, max_depth_vis)))
    time_text.set_text(f"t = {t_hr:.2f} h")
    is_raining = precip_engine.is_raining(t_hr * 3600.0)
    rain_text.set_text("☂  Raining" if is_raining else "☀  Dry period")

    # ── Hydrograph ────────────────────────────────────────────────────────────
    mask = hydro_times_hr <= t_hr
    hyd_line.set_data(hydro_times_hr[mask], hydro_Q[mask])
    vline.set_xdata([t_hr, t_hr])
    if mask.any():
        lpi = np.argmax(hydro_Q[mask])
        lpQ = hydro_Q[mask][lpi]
        lpt = hydro_times_hr[mask][lpi]
        if lpQ > 0.01:
            peak_annot.xy     = (lpt, lpQ)
            peak_annot.xytext = (lpt + 0.1, lpQ * 0.85)
            peak_annot.set_text(f"Peak\n{lpQ:.2f} m³/s")
        else:
            peak_annot.set_text("")

    artists = [depth_im, hyd_line, vline, time_text, rain_text, peak_annot]

    # ── Gauge extras (gauge mode only) ────────────────────────────────────────
    if _gauge_mode:
        cur_mmhr = gauge_mmhr_at_frames[frame_idx]   # (n_gauges,)

        # 1. Move precip cursor
        vline_prec.set_xdata([t_hr, t_hr])
        artists.append(vline_prec)

        # 2. Resize gauge bubble on map: size ∝ current intensity
        sizes = (cur_mmhr / max_rain_mmhr) * 600 + 40
        gauge_scat_dyn.set_sizes(sizes)
        artists.append(gauge_scat_dyn)

        # 3. Floating value labels on precip panel (right of cursor)
        label_x = t_hr + config.TOTAL_SIMULATION_TIME_HOURS * 0.015
        for gi, (txt, val) in enumerate(zip(gauge_val_texts, cur_mmhr)):
            txt.set_position((label_x, val))
            txt.set_text(f"{val:.1f}")
            artists.append(txt)

    return artists


# ─────────────────────────────────────────────────────────────────────────────
# 11. Render and save GIF
# ─────────────────────────────────────────────────────────────────────────────
anim = animation.FuncAnimation(
    fig,
    update,
    frames=n_frames,
    interval=1000 // GIF_FPS,
    blit=True,
)

writer = animation.PillowWriter(fps=GIF_FPS)
anim.save(OUTPUT_GIF, writer=writer, dpi=GIF_DPI,
          savefig_kwargs={"facecolor": fig.get_facecolor()})

plt.close(fig)
print(f"\n  Animation saved → {OUTPUT_GIF}")
print(f"   Frames : {n_frames}   FPS : {GIF_FPS}   DPI : {GIF_DPI}")
