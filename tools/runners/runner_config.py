"""
runner_config.py
================
ONE place for every knob the flood runner uses.  Edit HERE — the single entry
point (tools/runner.py) reads this file, looks at PRECIP_METHOD, and runs the
right pipeline.  You never have to hunt through the scripts again.

This is separate from the model's own config.py:
  • config.py        → the hydrologic model defaults (DEM, OUTPUT_DIR, IMERG
                        window, …) used when you run the model standalone.
  • runner_config.py → only the batch runner: which rainfall source, where the
                        .gag files live, and where ALL results land.
"""

from pathlib import Path

# Repo root = two levels above this file (tools/runners/ → tools/ → repo).
REPO_ROOT = Path(__file__).resolve().parent.parent.parent


# ══════════════════════════════════════════════════════════════════════════════
# 0.  THE MASTER SWITCH  —  which rainfall source to run
# ══════════════════════════════════════════════════════════════════════════════
#
#   Gauge (.gag) sources  → handled by the GAUGE pipeline (gauge.py)
#       'thiessen'   Thiessen polygons over the .gag gauges
#       'idw'        inverse-distance weighting over the .gag gauges
#       'uniform'    single basin-average rainfall
#
#   IMERG (GEE) sources   → handled by the IMERG pipeline (imerg.py)
#       'imerg_thiessen'  Thiessen over IMERG pixel pseudo-gauges
#       'imerg_idw'       IDW over IMERG pixel pseudo-gauges
#
# Set this one value; runner.py dispatches to the correct pipeline automatically.
PRECIP_METHOD = "thiessen"


# ══════════════════════════════════════════════════════════════════════════════
# 1.  OUTPUT LOCATION  —  ONE folder for ALL methods AND ALL events
# ══════════════════════════════════════════════════════════════════════════════
#
# Every flood event (every .gag) writes its results into THIS one folder, tagged
# by event date — exactly like the station runner:
#     <OUTPUT_DIR>/hydrograph_<tag>.csv
#     <OUTPUT_DIR>/comparison_<tag>.png
#     <OUTPUT_DIR>/summary_all_floods.csv
#     <OUTPUT_DIR>/timing.csv
# The shared watershed rasters (process_dem) and the per-event intermediates
# (downloaded IMERG, date-specific deficit raster) also live here; the
# intermediates are overwritten event-to-event, the tagged results are kept.
#
# This OVERRIDES config.OUTPUT_DIR for the batch run, so change the scenario
# folder in this one place.  (Relative paths are resolved against the repo root.)
OUTPUT_DIR = "outputs collection/test_muskinghum"


# ══════════════════════════════════════════════════════════════════════════════
# 2.  INPUT DATA  —  where the runner READS from
# ══════════════════════════════════════════════════════════════════════════════

# Folder holding the GSSHA rainfall files (*.gag) AND the observed flows
# (discharge_<tag>.csv).  Both pipelines scan this for events.
GSSHA_DIR = REPO_ROOT / "test_data" / "gssha_format"

# Where the GAUGE pipeline writes the converted OPM CSVs (gauges/timeseries per
# event).  IMERG ignores this — it downloads its own pseudo-gauges.
OPM_DIR = REPO_ROOT / "test_data" / "opm_format"


# ══════════════════════════════════════════════════════════════════════════════
# 3.  RUNTIME OPTIONS
# ══════════════════════════════════════════════════════════════════════════════

# Compute backend for the router.  Both pipelines currently require a GPU.
BACKEND = "gpu"
