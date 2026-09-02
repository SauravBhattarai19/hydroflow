"""
imerg_gee.py
============
NASA GPM IMERG V07 precipitation downloader for the kinematic-wave router.

Pulls half-hourly precipitation (``NASA/GPM_L3/IMERG_V07``, band ``precipitation``
in mm/hr) from Google Earth Engine over the watershed, turns each native 0.1°
(~11 km) IMERG pixel into a *pseudo-gauge* at its centroid, and writes the standard
OPM gauge format so the existing Thiessen/IDW + per-polygon VSA machinery runs
unchanged:

    <PRECIP_IMERG_DIR>/gauges.csv       gauge_id,name,easting_m,northing_m
    <PRECIP_IMERG_DIR>/timeseries.csv   time_s, <gauge cols...>   (mm depth / interval)
    <PRECIP_IMERG_DIR>/imerg_meta.json  provenance for the cache/"already detected" check

Download strategy
-----------------
A single ``ImageCollection.getRegion(region, scale)`` call returns the whole
space-time cube — each row is ``[id, longitude, latitude, time_ms, precipitation]``
where longitude/latitude are the *native pixel centroids*.  So one call yields both
the stations and their timeseries.  For very long windows / large basins the call is
split into date chunks to stay under EE's ~1.05M-value getRegion limit.

Authentication and watershed loading reuse the helpers in ``serves_gee.py``.
"""

import os
import json
import math
import logging
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import ee
    GEE_AVAILABLE = True
except ImportError:
    GEE_AVAILABLE = False

# IMERG native grid: 0.1° ≈ 11.132 km.  Each image spans 30 min.
_IMERG_PIXEL_DEG = 0.1
_INTERVAL_S      = 1800              # 30-minute IMERG cadence
_INTERVAL_H      = 0.5              # hours per interval (rate mm/hr → depth mm)
_DEG_PER_M       = 1.0 / 111320.0  # rough degrees-per-metre at the equator
# getRegion aborts past ~1,048,576 returned values (rows × cols).  Stay well under.
_SAFE_GETREGION_VALUES = 800_000
_GETREGION_COLS        = 5          # id, lon, lat, time, precipitation


# ── Config helpers ────────────────────────────────────────────────────────────

def _resolve_paths(cfg):
    """Return (imerg_dir, gauges_path, timeseries_path, meta_path)."""
    imerg_dir = getattr(cfg, 'PRECIP_IMERG_DIR', None) \
        or os.path.join(getattr(cfg, 'OUTPUT_DIR', 'output/'), 'imerg/')
    return (
        imerg_dir,
        os.path.join(imerg_dir, 'gauges.csv'),
        os.path.join(imerg_dir, 'timeseries.csv'),
        os.path.join(imerg_dir, 'imerg_meta.json'),
    )


def _local_to_utc(local_str, offset_hours):
    """Parse a local civil time string and convert to a UTC datetime (tz-aware)."""
    for fmt in ('%Y-%m-%d %H:%M', '%Y-%m-%d %H:%M:%S', '%Y-%m-%d'):
        try:
            naive = datetime.strptime(local_str.strip(), fmt)
            break
        except ValueError:
            continue
    else:
        raise ValueError(
            f"Could not parse IMERG date '{local_str}'. "
            "Use 'YYYY-MM-DD HH:MM' (local civil time)."
        )
    return (naive - timedelta(hours=float(offset_hours))).replace(tzinfo=timezone.utc)


def _watershed_bounds_4326(geojson_path):
    """Return (minx, miny, maxx, maxy) of the watershed in EPSG:4326."""
    import geopandas as gpd
    gdf = gpd.read_file(geojson_path).to_crs("EPSG:4326")
    minx, miny, maxx, maxy = gdf.total_bounds
    return float(minx), float(miny), float(maxx), float(maxy)


# ── Core download ─────────────────────────────────────────────────────────────

def download_imerg(cfg):
    """
    Download IMERG over the watershed and write gauges.csv / timeseries.csv /
    imerg_meta.json into PRECIP_IMERG_DIR.

    Returns (gauges_path, timeseries_path) on success; raises on failure.
    """
    if not GEE_AVAILABLE:
        raise RuntimeError(
            "earthengine-api is not installed but PRECIP_METHOD='imerg_*' was "
            "requested. Install it (pip install earthengine-api) or pre-populate "
            "the IMERG folder with gauges.csv/timeseries.csv."
        )

    from .auth import authenticate as _authenticate

    imerg_dir, gauges_path, ts_path, meta_path = _resolve_paths(cfg)
    project   = getattr(cfg, 'GEE_PROJECT', None)
    if project and not os.environ.get('GEE_PROJECT'):
        os.environ['GEE_PROJECT'] = project
    if not _authenticate(project):
        raise RuntimeError("Google Earth Engine authentication failed.")

    dataset    = getattr(cfg, 'IMERG_DATASET', 'NASA/GPM_L3/IMERG_V07')
    band       = getattr(cfg, 'IMERG_BAND', 'precipitation')
    offset_h    = float(getattr(cfg, 'IMERG_UTC_OFFSET_HOURS', 0.0))
    start_local = getattr(cfg, 'IMERG_START_LOCAL', None)
    end_local   = getattr(cfg, 'IMERG_END_LOCAL',   None)
    buffer_m    = float(getattr(cfg, 'IMERG_BBOX_BUFFER_M', 11132.0))
    target_crs  = getattr(cfg, 'TARGET_CRS_EPSG', 'EPSG:32645')
    geojson     = getattr(cfg, 'OPM_WATERSHED_GEOJSON', 'output/watershed.geojson')

    # Auto-derive window from EVENT_START_UTC + TOTAL_SIMULATION_TIME_HOURS.
    _evt   = getattr(cfg, 'EVENT_START_UTC', None)
    _sim_h = float(getattr(cfg, 'TOTAL_SIMULATION_TIME_HOURS', 0) or 0)
    if _evt:
        _utc_s = datetime.fromisoformat(str(_evt).strip())
        if not start_local:
            start_local = (_utc_s + timedelta(hours=offset_h)).strftime("%Y-%m-%d %H:%M")
            logger.info("IMERG_START_LOCAL auto-derived: %s", start_local)
        if not end_local and _sim_h:
            end_local = (_utc_s + timedelta(hours=_sim_h + offset_h)).strftime("%Y-%m-%d %H:%M")
            logger.info("IMERG_END_LOCAL auto-derived: %s", end_local)

    if not start_local or not end_local:
        raise ValueError(
            "IMERG download requires EVENT_START_UTC + TOTAL_SIMULATION_TIME_HOURS "
            "in §1 of config.py, or explicit IMERG_START_LOCAL / IMERG_END_LOCAL."
        )

    start_utc = _local_to_utc(start_local, offset_h)
    end_utc   = _local_to_utc(end_local,   offset_h)
    if end_utc <= start_utc:
        raise ValueError("IMERG end must be after start (check EVENT_START_UTC / IMERG_END_LOCAL).")

    # ── Buffered watershed bbox → ee.Geometry.Rectangle ──────────────────────
    minx, miny, maxx, maxy = _watershed_bounds_4326(geojson)
    buf_deg = buffer_m * _DEG_PER_M
    minx -= buf_deg; maxx += buf_deg
    miny -= buf_deg; maxy += buf_deg
    region = ee.Geometry.Rectangle([minx, miny, maxx, maxy], proj='EPSG:4326',
                                   geodesic=False)

    print("=" * 60)
    print("IMERG DOWNLOAD  (Google Earth Engine)")
    print("=" * 60)
    print(f"  Dataset      : {dataset}  band='{band}'")
    print(f"  Window local : {start_local} → {end_local}  (UTC{offset_h:+g}h)")
    print(f"  Window UTC   : {start_utc:%Y-%m-%d %H:%M} → {end_utc:%Y-%m-%d %H:%M}")
    print(f"  BBox 4326    : [{minx:.3f}, {miny:.3f}, {maxx:.3f}, {maxy:.3f}]")

    # ── Build the collection; probe scale + image count ──────────────────────
    ic_full = (ee.ImageCollection(dataset).select(band)
               .filterDate(start_utc.isoformat(), end_utc.isoformat())
               .filterBounds(region))

    n_images = int(ic_full.size().getInfo())
    if n_images == 0:
        raise RuntimeError(
            f"IMERG returned 0 images for {start_utc} → {end_utc}. "
            "Check the date window and watershed coverage."
        )
    scale = float(ee.Image(ic_full.first()).projection().nominalScale().getInfo())

    # ── Estimate pixel count to decide on date-chunking ──────────────────────
    nx = math.ceil((maxx - minx) / _IMERG_PIXEL_DEG) + 2
    ny = math.ceil((maxy - miny) / _IMERG_PIXEL_DEG) + 2
    n_pixels_est = max(1, nx * ny)
    imgs_per_chunk = max(1, _SAFE_GETREGION_VALUES //
                         (n_pixels_est * _GETREGION_COLS))
    print(f"  Scale        : {scale:.1f} m   images={n_images}   "
          f"~pixels={n_pixels_est}")

    # ── Fetch the space-time cube (single call or date-chunked) ──────────────
    frames = []
    if n_images <= imgs_per_chunk:
        frames.append(_get_region_df(ic_full, region, scale, band))
    else:
        chunk_s = imgs_per_chunk * _INTERVAL_S
        a = start_utc
        n_chunks = math.ceil((end_utc - start_utc).total_seconds() / chunk_s)
        print(f"  Chunking     : {n_chunks} date windows "
              f"(~{imgs_per_chunk} images each)")
        while a < end_utc:
            b = min(a + timedelta(seconds=chunk_s), end_utc)
            ic_c = (ee.ImageCollection(dataset).select(band)
                    .filterDate(a.isoformat(), b.isoformat())
                    .filterBounds(region))
            frames.append(_get_region_df(ic_c, region, scale, band))
            a = b
    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=['longitude', 'latitude', 'time'])
    if df.empty:
        raise RuntimeError("IMERG getRegion returned no usable pixels.")

    # ── Stations = unique pixel centroids ────────────────────────────────────
    df['lon_r'] = df['longitude'].round(3)
    df['lat_r'] = df['latitude'].round(3)

    stations = (df[['lon_r', 'lat_r']].drop_duplicates()
                .sort_values(['lat_r', 'lon_r'], ascending=[False, True])
                .reset_index(drop=True))
    stations['gauge_id'] = [f"IMG{i + 1:03d}" for i in range(len(stations))]
    stations['name'] = [f"IMERG {lat:.3f},{lon:.3f}"
                        for lat, lon in zip(stations['lat_r'], stations['lon_r'])]

    # Reproject centroids (lon/lat) → easting/northing in the target CRS
    from pyproj import Transformer
    tf = Transformer.from_crs("EPSG:4326", target_crs, always_xy=True)
    east, north = tf.transform(stations['lon_r'].values, stations['lat_r'].values)
    stations['easting_m'] = east
    stations['northing_m'] = north

    id_lookup = stations.set_index(['lon_r', 'lat_r'])['gauge_id']
    df = df.join(id_lookup, on=['lon_r', 'lat_r'])

    # ── Regular 30-min time axis (gap-free) → depth per interval ──────────────
    start_ms = start_utc.timestamp() * 1000.0
    df['time_s'] = ((df['time'].astype(float) - start_ms) / 1000.0).round().astype(int)
    df[band] = pd.to_numeric(df[band], errors='coerce')

    # Build the regular 30-min axis from the OBSERVED phase, not from multiples of
    # 1800 s off start_utc.  IMERG images sit on the UTC :00/:30 grid, so a fractional
    # UTC offset (e.g. Nepal +5:45) puts every image at start+900, start+2700, …  —
    # inheriting the data's own phase keeps real values on the grid and fills only
    # genuine internal gaps with 0.
    ts_min = int(df['time_s'].min())
    ts_max = int(df['time_s'].max())
    full_index = pd.Index(np.arange(ts_min, ts_max + _INTERVAL_S, _INTERVAL_S),
                          name='time_s')
    n_steps = len(full_index)

    rate = (df.pivot_table(index='time_s', columns='gauge_id', values=band,
                           aggfunc='mean')
              .reindex(index=full_index, columns=stations['gauge_id'])
              .fillna(0.0))                       # mm/hr, gap-free
    depth_mm = (rate * _INTERVAL_H).clip(lower=0.0)   # mm per 30-min interval

    # ── Write outputs ────────────────────────────────────────────────────────
    os.makedirs(imerg_dir, exist_ok=True)
    stations[['gauge_id', 'name', 'easting_m', 'northing_m']].to_csv(
        gauges_path, index=False)

    ts_out = depth_mm.reset_index()
    ts_out.to_csv(ts_path, index=False)

    meta = {
        'dataset':            dataset,
        'band':               band,
        'start_local':        start_local,
        'end_local':          end_local,
        'utc_offset_hours':   offset_h,
        'start_utc':          start_utc.isoformat(),
        'end_utc':            end_utc.isoformat(),
        'scale_m':            scale,
        'n_stations':         int(len(stations)),
        'n_timesteps':        int(n_steps),
        'target_crs':         target_crs,
        'downloaded_at':      datetime.now(timezone.utc).isoformat(),
    }
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    mean_total = float(depth_mm.sum(axis=0).mean())
    print(f"  Stations     : {len(stations)} pseudo-gauges")
    print(f"  Timesteps    : {n_steps} × {_INTERVAL_S}s")
    print(f"  Mean total   : {mean_total:.1f} mm per station over the window")
    print(f"  Written      : {gauges_path}")
    print(f"               : {ts_path}")
    print("=" * 60)

    return gauges_path, ts_path


def _get_region_df(ic, region, scale, band):
    """Run getRegion and return a DataFrame with native column names."""
    arr = ic.getRegion(region, scale).getInfo()
    header, rows = arr[0], arr[1:]
    return pd.DataFrame(rows, columns=header)[['longitude', 'latitude', 'time', band]]


# ── Cache-aware entry point used by PrecipEngine ──────────────────────────────

def ensure_imerg_data(cfg):
    """
    Return (gauges_path, timeseries_path) for the IMERG folder, downloading from
    GEE only when the folder is empty (or PRECIP_IMERG_FORCE_DOWNLOAD is True).

    If the CSVs already exist, a warning is printed and they are reused as-is
    (a stale window vs. config triggers an extra warning but never an auto-refetch).
    """
    imerg_dir, gauges_path, ts_path, meta_path = _resolve_paths(cfg)
    force = bool(getattr(cfg, 'PRECIP_IMERG_FORCE_DOWNLOAD', False))

    have_data = (os.path.isfile(gauges_path) and os.path.getsize(gauges_path) > 0
                 and os.path.isfile(ts_path) and os.path.getsize(ts_path) > 0)

    if have_data and not force:
        print(f"  [IMERG] Precip data already detected in '{imerg_dir}' — "
              f"skipping GEE download.")
        print(f"          Set PRECIP_IMERG_FORCE_DOWNLOAD=True (or delete the "
              f"folder) to refetch.")
        _warn_on_window_mismatch(cfg, meta_path)
        return gauges_path, ts_path

    if force and have_data:
        print(f"  [IMERG] PRECIP_IMERG_FORCE_DOWNLOAD=True — refetching into "
              f"'{imerg_dir}'.")

    return download_imerg(cfg)


def _warn_on_window_mismatch(cfg, meta_path):
    """Warn (do not refetch) if cached IMERG meta disagrees with current config."""
    if not os.path.isfile(meta_path):
        return
    try:
        with open(meta_path) as f:
            meta = json.load(f)
    except (OSError, ValueError):
        return

    mismatches = []
    for key, attr in (('start_local', 'IMERG_START_LOCAL'),
                      ('end_local', 'IMERG_END_LOCAL'),
                      ('utc_offset_hours', 'IMERG_UTC_OFFSET_HOURS')):
        cfg_val = getattr(cfg, attr, None)
        if cfg_val is not None and str(meta.get(key)) != str(cfg_val):
            mismatches.append(f"{attr}: cached={meta.get(key)} config={cfg_val}")
    if mismatches:
        print("  [IMERG][WARNING] Cached data was fetched for a DIFFERENT window:")
        for m in mismatches:
            print(f"                   {m}")
        print("                   Delete the folder or set "
              "PRECIP_IMERG_FORCE_DOWNLOAD=True to refetch.")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    # python -m vsa_opm.gee.imerg_gee <config.yaml|json> [--force]
    import sys
    from ..config import OpmConfig

    logging.basicConfig(level=logging.INFO, format='%(message)s')
    args = [a for a in sys.argv[1:] if a != '--force']
    if not args:
        sys.exit("usage: python -m vsa_opm.gee.imerg_gee <config.yaml|json> [--force]")
    cfg = OpmConfig.from_file(args[0])
    if '--force' in sys.argv:
        cfg.PRECIP_IMERG_FORCE_DOWNLOAD = True
    ensure_imerg_data(cfg)
