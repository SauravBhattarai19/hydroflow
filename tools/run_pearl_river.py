#!/usr/bin/env python
"""
run_pearl_river.py
===================
One-off, single-config run of the VSA-OPM model on the Pearl River basin
(Mississippi-Louisiana), Feb 2020 flood — a brand-new, never-run-before basin.

Applies the best config found by tools/run_vsa_param_sweep.py on the Nepal
test basin (diffusive routing, vsa+impervious mechanisms, OPM_SD_MIN=0.1,
OPM_K_SAT=0.044 m/day, channel routing on), but with SD_max/phi sourced live
from satellite data (OPM_SD_SOURCE='gee', SERVES NDVI+SoilGrids) instead of a
manual sweep value.

Unlike tools/run_combinations.py / run_vsa_param_sweep.py, this basin has no
pre-existing DEM, watershed, gauge network or observed discharge — everything
is built from scratch:
  1. Download a raw DEM from GEE (NASADEM, area-averaged to 100m) covering a
     generous bounding box around the whole basin.
  2. Build an OpmConfig pointing at that DEM + the chosen outlet.
  3. run_pipeline(cfg, stages=("process_dem", "routing")) does the rest:
     terrain analysis, outlet snap + delineation, IMERG download, and all the
     lazy GEE soil/LULC downloads (SERVES deficit, HiHydroSoil Ksat, SoilGrids
     texture, LCZ Manning's-n/impervious), then the diffusive-wave route.

No PRECIP_GAUGE_FILE / observed discharge exists for this basin, so NSE/pbias
are not computed — the deliverable is the hydrograph + mass balance +
Dunne/Horton/impervious partition.

Output: outputs collection/pearl_river_2020_flood/

Usage
-----
    python tools/run_pearl_river.py
"""

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))

from hydroflow import OpmConfig, run_pipeline
from hydroflow.gee.dem_gee import download_dem

# ── Basin-specific constants (see docs/plan for sourcing) ────────────────────
GEE_PROJECT = 'ee-sauravbhattarai1999'

# Generous bounding box around the whole Pearl River basin (headwaters near
# 32.9N/-89.0W down to the Gulf coast) — padding costs terrain-analysis
# compute, but a too-tight box would truncate the delineated watershed.
# (Verified via a diagnostic delineation-only run: the outlet consistently
# resolves to an ~8,800 km^2 sub-basin regardless of box size — the shortfall
# vs. the official ~17,024 km^2 gauge drainage area is flat-terrain D8
# tributary mis-routing at 100m, not a bounding-box truncation artifact.
# Accepted as-is for this exploratory run; see plan/chat for the discussion.)
DEM_BBOX = (-91.2, 30.2, -88.3, 33.3)   # (min_lon, min_lat, max_lon, max_lat)
DEM_SCALE_M = 100.0

TARGET_CRS_EPSG = "EPSG:32616"          # WGS84 UTM 16N
OUTLET_LATLON = (30.79324, -89.82091)   # USGS 02489500, Pearl River near Bogalusa, LA
                                        # (upstream of the East/West Pearl delta
                                        #  split, where D8 delineation breaks down)

OUTPUT_DIR = str(REPO_ROOT / "outputs collection" / "pearl_river_2020_flood")

EVENT_START_UTC = "2020-02-08 00:00"    # 2-day lead-in before the Feb 10 rain onset
TOTAL_SIMULATION_TIME_HOURS = 288.0     # 12 days -> Feb 20 (rise + Feb 17 crest + early recession)
IMERG_UTC_OFFSET_HOURS = -6.0           # US Central Standard Time (no DST in Feb)

OPM_Q_MAX = 573.0                       # ~20,217 cfs, Feb median historical baseflow at Bogalusa


def main():
    dem_path = os.path.join(OUTPUT_DIR, "raw_dem_pearl.tif")
    print(f"[1/2] Downloading DEM ({DEM_BBOX}, {DEM_SCALE_M}m, {TARGET_CRS_EPSG}) ...")
    result = download_dem(DEM_BBOX, TARGET_CRS_EPSG, DEM_SCALE_M, dem_path, project=GEE_PROJECT)
    if result is None:
        sys.exit("[ERROR] DEM download failed — see log above.")
    print(f"      -> {dem_path}")

    cfg = OpmConfig(
        DEM_PATH=dem_path,
        TARGET_CRS_EPSG=TARGET_CRS_EPSG,
        OUTPUT_POINT=OUTLET_LATLON,
        OUTPUT_DIR=OUTPUT_DIR,

        EVENT_START_UTC=EVENT_START_UTC,
        TOTAL_SIMULATION_TIME_HOURS=TOTAL_SIMULATION_TIME_HOURS,
        IMERG_UTC_OFFSET_HOURS=IMERG_UTC_OFFSET_HOURS,
        GEE_PROJECT=GEE_PROJECT,

        PRECIP_METHOD="imerg_thiessen",

        RUNOFF_SOURCE="vsa_opm",
        RUNOFF_MECHANISMS=["vsa", "impervious"],
        ROUTING_SCHEME="diffusive",
        DIFFUSION_THETA=1.0,
        CHANNEL_ROUTING=True,

        OPM_SD_MIN=0.1,
        OPM_K_SAT=0.044,
        OPM_SD_SOURCE="gee",
        OPM_SD_REDUCER="max",
        OPM_GA_KSAT_SOURCE="gee",
        OPM_GA_SUCTION_SOURCE="texture",
        MANNINGS_N_SOURCE="lcz",
        IMPERVIOUS_SOURCE="lcz",
        OPM_INFILTRATION="green_ampt",
        OPM_Q_MAX=OPM_Q_MAX,

        BACKEND="gpu",
    )
    cfg.update_output_paths()
    cfg.validate()

    print(f"\n[2/2] Running pipeline -> {OUTPUT_DIR}")
    results = run_pipeline(cfg, stages=("process_dem", "routing"))

    df = results.get("hydrograph_df")
    if df is not None:
        peak_q = df["Q_m3s"].max()
        peak_t_hr = df.loc[df["Q_m3s"].idxmax(), "time_hr"]
        print(f"\nPeak modelled discharge: {peak_q:.1f} m^3/s at t={peak_t_hr:.1f} h")
    print(f"Hydrograph: {results.get('hydrograph_csv')}")
    print(f"Done. Outputs in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
