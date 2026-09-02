"""
05_test_spatial_maps.py
=======================
Diagnostic visualizations:
  1. lulc_mannings_map.png   — LULC classes + Manning's n (overland/channel)
  2. sdmax_polygons_map.png  — Voronoi polygons with max-deficit cell markers
  3. vsa_propagation.gif     — Animated GIF: rain per polygon, A_t, VSA expansion

All figures are cropped to the watershed bounding box.

Usage
-----
  cd /path/to/OPM
  python tests/05_test_spatial_maps.py
"""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import re
import numpy as np
import pandas as pd
import rasterio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm
from shapely.geometry import Point, MultiPoint
from shapely.ops import voronoi_diagram
import geopandas as gpd

import config
from _opm_diag import output_dir, resolve_precip

# All inputs/outputs follow the ACTIVE config (scenario folder + precip method).
OUTPUT   = output_dir()
DEM_PATH = REPO / config.ROUTING_DEM_PATH
WS_PATH  = REPO / config.ROUTING_WATERSHED_MASK_PATH
FA_PATH  = REPO / config.ROUTING_FLOW_ACCUM_PATH
LULC_PATH = OUTPUT / "lulc_mannings.tif"
DEFICIT_PATH = OUTPUT / "deficit_serves.tif"
LUT_CSV  = REPO / config.LULC_LOOKUP_CSV
WS_JSON  = REPO / getattr(config, 'OPM_WATERSHED_GEOJSON',
                           'output/watershed.geojson')

# Gauges/timeseries/event come from the active method (IMERG pixel pseudo-gauges
# when PRECIP_METHOD='imerg_*', the configured CSV gauges otherwise).
GAUGE_CSV, TS_CSV, EVENT_DATE = resolve_precip(config)


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def load_dem_and_mask():
    with rasterio.open(DEM_PATH) as src:
        dem = src.read(1).astype(np.float64)
        transform = src.transform
        nodata = src.nodata
    with rasterio.open(WS_PATH) as src:
        ws = src.read(1) > 0
    if nodata is not None:
        ws &= (dem != nodata)
    dem_masked = np.where(ws, dem, np.nan)
    return dem_masked, ws, transform


def ws_bounds(ws):
    """Row/col bounding box of the watershed with a small pad."""
    rows, cols = np.where(ws)
    pad = 5
    r0 = max(0, rows.min() - pad)
    r1 = min(ws.shape[0], rows.max() + pad + 1)
    c0 = max(0, cols.min() - pad)
    c1 = min(ws.shape[1], cols.max() + pad + 1)
    return r0, r1, c0, c1


def crop(arr, bounds):
    r0, r1, c0, c1 = bounds
    return arr[r0:r1, c0:c1]


def load_faccum(ws):
    with rasterio.open(FA_PATH) as src:
        fa = src.read(1).astype(np.float64)
    fa[~ws] = np.nan
    return fa


def build_voronoi_gdf(gauge_csv, ws_geojson, target_crs):
    gauges = pd.read_csv(gauge_csv)
    points = [Point(x, y) for x, y
              in zip(gauges['easting_m'], gauges['northing_m'])]
    ws = gpd.read_file(ws_geojson)
    if ws.crs and str(ws.crs).upper() != target_crs.upper():
        ws = ws.to_crs(target_crs)
    ws_geom = ws.dissolve().geometry.iloc[0]
    regions = voronoi_diagram(MultiPoint(points), envelope=ws_geom.envelope)
    polygons = [None] * len(points)
    for region in regions.geoms:
        clipped = region.intersection(ws_geom)
        if clipped.is_empty:
            continue
        for i, pt in enumerate(points):
            if clipped.contains(pt):
                polygons[i] = clipped
                break
    for i in range(len(polygons)):
        if polygons[i] is None:
            polygons[i] = points[i].buffer(1.0).intersection(ws_geom)
    return gpd.GeoDataFrame(
        {'gauge_id': gauges.iloc[:, 0].values,
         'geometry': polygons}, crs=target_crs)


def rasterize_polygons(gdf, transform, shape):
    from rasterio.features import rasterize
    return rasterize(
        [(geom, i) for i, geom in enumerate(gdf.geometry)],
        out_shape=shape, transform=transform, fill=-1, dtype=np.int32)


def polygon_boundaries(poly_raster, ws):
    from scipy.ndimage import binary_erosion
    mask = (poly_raster >= 0).astype(np.uint8)
    interior = binary_erosion(mask, iterations=1)
    return mask.astype(bool) & ~interior & ws


# ══════════════════════════════════════════════════════════════════════════════
# Figure 1: LULC + Manning's n
# ══════════════════════════════════════════════════════════════════════════════

def plot_lulc_and_mannings(dem_masked, ws, transform):
    print("  Plotting LULC + Manning's n ...")
    lut = pd.read_csv(LUT_CSV)
    bb = ws_bounds(ws)

    with rasterio.open(LULC_PATH) as src:
        lulc = src.read(1)
    lulc_masked = np.where(ws, lulc, 0).astype(float)
    lulc_masked[~ws] = np.nan

    class_colors = {
        10: ('#006400', 'Tree cover'),
        20: ('#ffbb22', 'Shrubland'),
        30: ('#ffff4c', 'Grassland'),
        40: ('#f096ff', 'Cropland'),
        50: ('#fa0000', 'Built-up'),
        60: ('#b4b4b4', 'Bare/sparse'),
        70: ('#f0f0f0', 'Snow/ice'),
        80: ('#0064c8', 'Water'),
        90: ('#0096a0', 'Wetland'),
        95: ('#00cf75', 'Mangroves'),
        100: ('#fae6a0', 'Moss/lichen'),
    }
    codes = sorted(class_colors.keys())
    cmap = ListedColormap([class_colors[c][0] for c in codes])
    bounds_c = codes + [codes[-1] + 10]
    norm = BoundaryNorm(bounds_c, cmap.N)

    code_to_n = dict(zip(lut['class_code'].astype(int),
                         lut['mannings_n'].astype(float)))
    n_arr = np.full_like(lulc, np.nan, dtype=np.float64)
    for code, nval in code_to_n.items():
        n_arr[lulc == code] = nval
    n_arr[~ws] = np.nan

    fa = load_faccum(ws)
    n_cells_ws = int(ws.sum())
    threshold = getattr(config, 'CHANNEL_FACCUM_THRESHOLD', None)
    if threshold is None:
        threshold = max(1, n_cells_ws // 100)
    channels = fa > threshold
    n_channel = getattr(config, 'MANNINGS_N_CHANNEL', 0.035)
    if n_channel is not None:
        n_arr[channels] = n_channel

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    lc = crop(lulc_masked, bb)
    ax1.imshow(lc, cmap=cmap, norm=norm, interpolation='nearest')
    ax1.set_title("ESA WorldCover LULC", fontsize=13, fontweight='bold')
    patches = [mpatches.Patch(color=class_colors[c][0], label=class_colors[c][1])
               for c in codes if np.nanmax(lc == c) > 0]
    ax1.legend(handles=patches, loc='lower left', fontsize=7,
               framealpha=0.9, ncol=2)
    ax1.set_xticks([]); ax1.set_yticks([])

    nc = crop(n_arr, bb)
    im2 = ax2.imshow(nc, cmap='YlOrRd_r', interpolation='nearest',
                     vmin=0.01, vmax=0.16)
    ax2.set_title("Manning's n  (overland + channel)", fontsize=13,
                  fontweight='bold')
    fig.colorbar(im2, ax=ax2, shrink=0.75, label="Manning's n")
    ch_overlay = np.where(crop(channels, bb), 1.0, np.nan)
    ax2.imshow(ch_overlay, cmap=ListedColormap(['#1f78b4']),
               alpha=0.5, interpolation='nearest')
    ax2.set_xticks([]); ax2.set_yticks([])

    n_valid = n_arr[ws & np.isfinite(n_arr)]
    n_chan_count = int(channels[ws].sum())
    stats = (f"mean={n_valid.mean():.4f}  "
             f"range=[{n_valid.min():.3f}, {n_valid.max():.3f}]\n"
             f"channel cells: {n_chan_count:,}/{n_cells_ws:,}  "
             f"(n_ch={n_channel}  thr={threshold})")
    ax2.text(0.02, 0.02, stats, transform=ax2.transAxes, fontsize=8,
             va='bottom', bbox=dict(facecolor='white', alpha=0.8))

    plt.tight_layout()
    out = OUTPUT / "lulc_mannings_map.png"
    plt.savefig(out, dpi=180)
    plt.close()
    print(f"    Saved: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 2: Per-polygon SD_max + max-deficit locations
# ══════════════════════════════════════════════════════════════════════════════

def _geo_to_pixel(easting, northing, transform, bb):
    """Convert projected coordinates to pixel coords cropped to bounding box."""
    r0, r1, c0, c1 = bb
    col = (easting  - transform.c) / transform.a
    row = (northing - transform.f) / transform.e
    return col - c0, row - r0


def plot_sdmax_polygons(dem_masked, ws, transform):
    print("  Plotting per-polygon SD_max ...")
    bb = ws_bounds(ws)
    r0, r1, c0, c1 = bb

    if not DEFICIT_PATH.exists():
        print("    Downloading deficit raster from GEE ...")
        from hydroflow.gee.serves_gee import download_deficit_raster
        result = download_deficit_raster(
            dem_path=str(DEM_PATH),
            watershed_geojson_path=str(WS_JSON),
            output_path=str(DEFICIT_PATH),
            lookup_csv_path=str(LUT_CSV),
            target_date=EVENT_DATE or getattr(config, 'SERVES_TARGET_DATE', None),
            satellite=getattr(config, 'SERVES_SATELLITE', 'landsat'),
            search_window=getattr(config, 'SERVES_SEARCH_WINDOW', 16),
            soil_depth_band=getattr(config, 'OPM_SOILGRIDS_DEPTH', 'b30'),
            project=getattr(config, 'GEE_PROJECT', None),
        )
        if result is None:
            print("    [WARN] Could not download deficit raster. Skipping.")
            return

    with rasterio.open(DEFICIT_PATH) as src:
        deficit = src.read(1).astype(np.float64)
    deficit[~ws] = np.nan

    # Nearest-station partition — identical to the model's cell_polygon, so the
    # zones drawn here (incl. boundary strips owned by outside stations) and the
    # per-zone SD labels match what the engine uses.
    from _opm_diag import nearest_station_raster
    poly_raster, n_poly = nearest_station_raster(GAUGE_CSV, ws, transform)
    bdry = polygon_boundaries(poly_raster, ws)
    bdry_c = crop(bdry, bb)

    gauges = pd.read_csv(GAUGE_CSV)

    # The ★ marks the max-deficit cell (where the driest soil is); the label
    # shows the value the ENGINE actually uses per OPM_SD_REDUCER (max or mean
    # over the polygon's watershed cells) so the figure matches the model.
    reducer = getattr(config, 'OPM_SD_REDUCER', 'mean').lower()
    max_rows, max_cols, sd_maxes, poly_valid = [], [], [], []
    for p in range(n_poly):
        mask_p = (poly_raster == p) & ws & np.isfinite(deficit)
        if not mask_p.any():
            # Empty zone (e.g. IMERG pixel outside the basin) — no marker drawn.
            max_rows.append(0); max_cols.append(0); sd_maxes.append(0)
            poly_valid.append(False)
            continue
        poly_valid.append(True)
        deficit_p = np.where(mask_p, deficit, -np.inf)
        idx = np.unravel_index(deficit_p.argmax(), deficit_p.shape)
        max_rows.append(idx[0])
        max_cols.append(idx[1])
        vals = deficit[mask_p]
        sd_maxes.append(float(vals.max()) if reducer == 'max'
                        else float(vals.mean()))

    # ── Figure: 2 panels ─────────────────────────────────────────────────
    fig, (ax_elev, ax_sd) = plt.subplots(1, 2, figsize=(18, 9))

    # ── Left panel: Elevation + polygon boundaries + gauge stations ──────
    dem_c = crop(dem_masked, bb)
    im_elev = ax_elev.imshow(dem_c, cmap='terrain', interpolation='nearest')
    fig.colorbar(im_elev, ax=ax_elev, shrink=0.65, label="Elevation [m]")

    # Thiessen polygon boundary lines using contour (crisp vector lines)
    poly_c_img = crop(poly_raster, bb).astype(float)
    poly_c_img[~crop(ws, bb)] = np.nan
    ax_elev.contour(poly_c_img, levels=np.arange(-0.5, n_poly, 1),
                    colors='black', linewidths=1.8, zorder=5)

    for i, row in gauges.iterrows():
        gx, gy = _geo_to_pixel(row['easting_m'], row['northing_m'],
                               transform, bb)
        ax_elev.plot(gx, gy, marker='^', markersize=10, color='white',
                     markeredgecolor='black', markeredgewidth=1.2, zorder=12)
        gid = str(row['gauge_id'])
        ax_elev.annotate(gid, (gx, gy),
                         textcoords='offset points', xytext=(6, -12),
                         fontsize=7, fontweight='bold', color='black',
                         bbox=dict(facecolor='white', alpha=0.8, pad=1,
                                   edgecolor='none'), zorder=13)

    # SD_max locations on elevation map with their elevation values
    for p in range(n_poly):
        if not poly_valid[p]:
            continue
        r_local = max_rows[p] - r0
        c_local = max_cols[p] - c0
        elev_at_sdmax = float(dem_masked[max_rows[p], max_cols[p]])
        ax_elev.plot(c_local, r_local, marker='*', markersize=14,
                     color='red', markeredgecolor='black',
                     markeredgewidth=0.8, zorder=10)
        ax_elev.annotate(
            f"P{p}: {elev_at_sdmax:.0f}m\nSD={sd_maxes[p]:.3f}m ({reducer})",
            (c_local, r_local),
            textcoords='offset points', xytext=(8, 6),
            fontsize=6.5, fontweight='bold',
            bbox=dict(facecolor='white', alpha=0.85, pad=1.5,
                      edgecolor='gray'), zorder=11)

    # Per-polygon max elevation label at polygon centroid
    poly_c_full = crop(poly_raster, bb)
    for p in range(n_poly):
        mask_p = (poly_c_full == p)
        if not mask_p.any():
            continue
        elev_p = np.where(mask_p & np.isfinite(dem_c), dem_c, np.nan)
        max_elev_p = float(np.nanmax(elev_p))
        rows_p, cols_p = np.where(mask_p)
        cy, cx = rows_p.mean(), cols_p.mean()
        ax_elev.text(cx, cy, f"{max_elev_p:.0f}m",
                     fontsize=8, fontweight='bold', color='navy',
                     ha='center', va='center',
                     bbox=dict(facecolor='white', alpha=0.7, pad=1,
                               edgecolor='none'), zorder=8)

    elev_valid = dem_c[np.isfinite(dem_c)]
    ax_elev.set_title(
        r"Elevation  |  $\bigstar$ = max deficit cell  |  blue = polygon max elev",
        fontsize=11, fontweight='bold')
    ax_elev.text(0.02, 0.02,
                 f"Elev range: {elev_valid.min():.0f} – {elev_valid.max():.0f} m",
                 transform=ax_elev.transAxes, fontsize=8, va='bottom',
                 bbox=dict(facecolor='white', alpha=0.8))
    ax_elev.set_xticks([]); ax_elev.set_yticks([])

    # ── Right panel: spatial deficit + max-deficit stars + stations ───────
    deficit_c = crop(deficit, bb)
    ax_sd.imshow(dem_c, cmap='terrain', alpha=0.3, interpolation='nearest')
    im_sd = ax_sd.imshow(deficit_c, cmap='RdYlBu_r', alpha=0.75,
                         interpolation='nearest')
    fig.colorbar(im_sd, ax=ax_sd, shrink=0.65,
                 label="SM Deficit  (porosity - " + r"$\theta$" + r") $\times$ Z$_r$  [m]")

    by, bx = np.where(bdry_c)
    ax_sd.scatter(bx, by, c='black', s=0.15, alpha=0.5, zorder=5)

    for p in range(n_poly):
        if not poly_valid[p]:
            continue
        r_local = max_rows[p] - r0
        c_local = max_cols[p] - c0
        ax_sd.plot(c_local, r_local, marker='*', markersize=16,
                   color='red', markeredgecolor='black', markeredgewidth=0.8,
                   zorder=10)
        ax_sd.annotate(f"P{p}: {sd_maxes[p]:.3f}m ({reducer})",
                       (c_local, r_local),
                       textcoords='offset points', xytext=(8, 8),
                       fontsize=7.5, fontweight='bold',
                       bbox=dict(facecolor='white', alpha=0.85, pad=1.5,
                                 edgecolor='gray'), zorder=11)

    for i, row in gauges.iterrows():
        gx, gy = _geo_to_pixel(row['easting_m'], row['northing_m'],
                               transform, bb)
        ax_sd.plot(gx, gy, marker='^', markersize=10, color='white',
                   markeredgecolor='black', markeredgewidth=1.2, zorder=12)

    ax_sd.set_title(r"SM Deficit  |  $\bigstar$ = max deficit cell  |  "
                    f"label = SD_max ({reducer})  |  " + r"$\triangle$ = gauge",
                    fontsize=12, fontweight='bold')
    ax_sd.set_xticks([]); ax_sd.set_yticks([])

    plt.tight_layout()
    out = OUTPUT / "sdmax_polygons_map.png"
    plt.savefig(out, dpi=180)
    plt.close()
    print(f"    Saved: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 3: VSA propagation GIF
# ══════════════════════════════════════════════════════════════════════════════

POLY_COLORS = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3',
               '#ff7f00', '#a65628', '#f781bf']


def _make_frame(ax_map, ax_rain, ax_at,
                dem_c, vsa_2d_c, ws_c, poly_c, bdry_c, rain_per_poly,
                at_arr, sd_init_arr, t_hr, n_vsa, pct, n_poly,
                rain_hist_t, rain_hist_vals, at_hist_t, at_hist_vals,
                total_hrs):
    """Draw one frame of the GIF into the provided axes."""

    # ── Left panel: VSA map with polygon coloring ────────────────────────
    ax_map.clear()
    ax_map.imshow(dem_c, cmap='terrain', alpha=0.35, interpolation='nearest')

    # Color polygons by current rain intensity
    rain_max = max(np.nanmax(rain_hist_vals) if len(rain_hist_vals) else 1, 0.1)
    poly_rain_2d = np.full_like(dem_c, np.nan)
    for p in range(n_poly):
        poly_rain_2d[poly_c == p] = rain_per_poly[p]
    poly_rain_2d[~ws_c] = np.nan
    ax_map.imshow(poly_rain_2d, cmap='Blues', alpha=0.4,
                  interpolation='nearest', vmin=0, vmax=rain_max)

    # VSA overlay
    vsa_display = np.where(vsa_2d_c == 1, 1.0, np.nan)
    ax_map.imshow(vsa_display, cmap=ListedColormap(['#d62728']),
                  alpha=0.7, interpolation='nearest')

    # Polygon boundaries
    by, bx = np.where(bdry_c)
    ax_map.scatter(bx, by, c='black', s=0.08, alpha=0.4, zorder=5)

    ax_map.set_title(f"t = {t_hr:.1f} h", fontsize=13, fontweight='bold')
    ax_map.text(0.02, 0.98,
                f"VSA = {n_vsa:,} cells ({pct:.1f}%)",
                transform=ax_map.transAxes, fontsize=9, va='top',
                bbox=dict(facecolor='white', alpha=0.85, pad=2))

    legend_patches = [
        mpatches.Patch(color='#d62728', alpha=0.7, label='VSA (saturated)'),
        mpatches.Patch(color='#6baed6', alpha=0.5, label='Rainfall'),
    ]
    ax_map.legend(handles=legend_patches, loc='lower left', fontsize=7,
                  framealpha=0.9)
    ax_map.set_xticks([]); ax_map.set_yticks([])

    # ── Top-right: precipitation per polygon ─────────────────────────────
    ax_rain.clear()
    for p in range(n_poly):
        vals = [v[p] for v in rain_hist_vals]
        ax_rain.plot(rain_hist_t, vals, color=POLY_COLORS[p % len(POLY_COLORS)],
                     lw=1.2, label=f"P{p}")
    ax_rain.axvline(t_hr, color='red', lw=1, ls='--', alpha=0.7)
    ax_rain.set_ylabel("Rain [mm/hr]", fontsize=9)
    ax_rain.set_title("Precipitation per Polygon", fontsize=10, fontweight='bold')
    ax_rain.set_xlim(0, total_hrs)
    ax_rain.legend(fontsize=6, ncol=min(n_poly, 6), loc='upper right')
    ax_rain.grid(True, ls='--', alpha=0.3)
    ax_rain.set_xlabel("Time [hr]", fontsize=8)

    # ── Bottom-right: A_t per polygon (LOG scale) ─────────────────────────
    ax_at.clear()
    for p in range(n_poly):
        vals = [v[p] for v in at_hist_vals]
        ax_at.semilogy(at_hist_t, vals, color=POLY_COLORS[p % len(POLY_COLORS)],
                       lw=1.2, label=f"P{p}")
    ax_at.axvline(t_hr, color='red', lw=1, ls='--', alpha=0.7)
    ax_at.set_ylabel(r"$A_t$ [km$^2$]  (log)", fontsize=9)
    ax_at.set_title(r"Threshold Area $A_t$ per Polygon  (log scale)", fontsize=10,
                    fontweight='bold')
    ax_at.set_xlim(0, total_hrs)
    ax_at.legend(fontsize=6, ncol=min(n_poly, 6), loc='upper right')
    ax_at.grid(True, ls='--', alpha=0.3, which='both')
    ax_at.set_xlabel("Time [hr]", fontsize=8)


def plot_vsa_gif(dem_masked, ws, transform):
    print("  Generating VSA propagation GIF ...")
    from hydroflow.core.routing import router as kwr
    from PIL import Image
    import io

    bb = ws_bounds(ws)

    orig_backend = config.BACKEND
    orig_date    = getattr(config, 'SERVES_TARGET_DATE', None)
    # The GIF runs the full simulation; the per-step VSA update dominates, so use
    # the GPU when available (≈30× faster for a 96 h run).  Falls back to CPU.
    from hydroflow.utils import gpu_utils
    config.BACKEND = 'gpu' if gpu_utils.cupy_available() else 'cpu'
    print(f"    backend: {config.BACKEND}")
    if EVENT_DATE:
        config.SERVES_TARGET_DATE = EVENT_DATE

    # No precip override — respect config.PRECIP_METHOD so the engine's zones
    # match the gauges/Voronoi this script visualises.
    grid_data = kwr.initialise_grid(config)

    runoff_engine  = grid_data['runoff_engine']
    precip_engine  = grid_data['precip_engine']
    s_rows, s_cols = grid_data['s_rows'], grid_data['s_cols']
    nrows, ncols   = grid_data['nrows'], grid_data['ncols']
    dt = config.TIME_STEP_SECONDS
    total_s = config.TOTAL_SIMULATION_TIME_HOURS * 3600.0
    total_hrs = config.TOTAL_SIMULATION_TIME_HOURS
    n_steps = int(total_s / dt)

    # Use the engine's EXACT partition (cell_polygon) so the GIF zones match the
    # model one-to-one — including strips owned by outside stations.
    n_poly = runoff_engine._n_polygons
    _cp = runoff_engine._cell_polygon
    if hasattr(_cp, 'get'):
        _cp = _cp.get()
    poly_raster = np.full((nrows, ncols), -1, dtype=np.int32)
    poly_raster[s_rows, s_cols] = np.asarray(_cp)
    bdry = polygon_boundaries(poly_raster, ws)

    dem_c  = crop(dem_masked, bb)
    ws_c   = crop(ws, bb)
    poly_c = crop(poly_raster, bb)
    bdry_c = crop(bdry, bb)
    r0, r1, c0, c1 = bb

    # Snapshot every ~30 minutes of simulation time, capped at 200 frames
    snap_interval = max(1, int(1800 / dt))
    snap_steps = set(range(0, n_steps + 1, snap_interval))
    snap_steps.add(n_steps)

    # History buffers
    rain_hist_t    = []
    rain_hist_vals = []
    at_hist_t      = []
    at_hist_vals   = []
    frames = []

    def capture_frame(t_hr, rain_per_poly, at_arr):
        mask_np = runoff_engine._vsa_mask
        if hasattr(mask_np, 'get'):
            mask_np = mask_np.get()
        vsa_2d = np.full((nrows, ncols), np.nan)
        vsa_2d[s_rows, s_cols] = mask_np.astype(float)
        n_vsa = int(mask_np.sum())
        pct   = float(mask_np.mean() * 100)

        fig = plt.figure(figsize=(16, 8))
        gs = fig.add_gridspec(2, 2, width_ratios=[1.3, 1], hspace=0.35,
                              wspace=0.25)
        ax_map  = fig.add_subplot(gs[:, 0])
        ax_rain = fig.add_subplot(gs[0, 1])
        ax_at   = fig.add_subplot(gs[1, 1])

        sd_init = runoff_engine._SD_max_initial
        _make_frame(ax_map, ax_rain, ax_at,
                    dem_c, crop(vsa_2d, bb), ws_c, poly_c, bdry_c,
                    rain_per_poly, at_arr / 1e6, sd_init,
                    t_hr, n_vsa, pct, n_poly,
                    rain_hist_t, rain_hist_vals, at_hist_t, at_hist_vals,
                    total_hrs)

        fig.suptitle("VSA Propagation  |  Bagmati Basin",
                     fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100)
        plt.close(fig)
        buf.seek(0)
        frames.append(Image.open(buf).copy())
        buf.close()

    # Capture t=0
    rain_1d_0 = precip_engine.get_field_1d(0)
    rain_pp_0 = np.zeros(n_poly)
    for p in range(n_poly):
        mask_p = runoff_engine._cell_polygon == p
        rain_pp_0[p] = float(rain_1d_0[mask_p].mean()) * 3.6e6  # m/s → mm/hr

    at_0 = runoff_engine._opm_A_t.copy()
    rain_hist_t.append(0.0)
    rain_hist_vals.append(rain_pp_0.copy())
    at_hist_t.append(0.0)
    at_hist_vals.append(at_0.copy() / 1e6)

    capture_frame(0.0, rain_pp_0, at_0)
    print(f"    t=0.0h  frame 1")

    frame_count = 1
    for step in range(1, n_steps + 1):
        t = step * dt
        rain_1d = precip_engine.get_field_1d(t)
        runoff_engine.update_state(rain_1d, dt)

        if step in snap_steps:
            t_hr = t / 3600.0

            rain_pp = np.zeros(n_poly)
            for p in range(n_poly):
                mask_p = runoff_engine._cell_polygon == p
                rain_pp[p] = float(rain_1d[mask_p].mean()) * 3.6e6

            at_now = runoff_engine._opm_A_t.copy()

            rain_hist_t.append(t_hr)
            rain_hist_vals.append(rain_pp.copy())
            at_hist_t.append(t_hr)
            at_hist_vals.append(at_now.copy() / 1e6)

            capture_frame(t_hr, rain_pp, at_now)
            frame_count += 1
            if frame_count % 20 == 0:
                mask_np = runoff_engine._vsa_mask
                if hasattr(mask_np, 'get'):
                    mask_np = mask_np.get()
                print(f"      t={t_hr:.1f}h  frame {frame_count}  "
                      f"VSA={int(mask_np.sum()):,} cells")

    config.BACKEND = orig_backend
    config.SERVES_TARGET_DATE = orig_date

    # Save GIF
    out = OUTPUT / "vsa_propagation.gif"
    frames[0].save(
        out, save_all=True, append_images=frames[1:],
        duration=150, loop=0, optimize=True,
    )
    print(f"    Saved: {out}  ({len(frames)} frames)")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  Spatial Diagnostic Maps")
    print("=" * 60)

    if not LULC_PATH.exists():
        print(f"[ERROR] LULC raster not found: {LULC_PATH}")
        print("        Run the model first (python tools/run_all_floods.py)")
        sys.exit(1)

    dem_masked, ws, transform = load_dem_and_mask()

    plot_lulc_and_mannings(dem_masked, ws, transform)

    if GAUGE_CSV is not None:
        plot_sdmax_polygons(dem_masked, ws, transform)
    else:
        print("  [SKIP] No gauge CSV — skipping SD_max polygon map")

    if GAUGE_CSV is not None:
        plot_vsa_gif(dem_masked, ws, transform)
    else:
        print("  [SKIP] No gauge CSV — skipping VSA GIF")

    print(f"\nAll maps saved to {OUTPUT}/")


if __name__ == "__main__":
    main()
