# Quickstart

Go from a DEM to a routed hydrograph in a few lines. You need one input: a
digital elevation model (GeoTIFF) and the latitude/longitude of your outlet.

## Python API

```python
from hydroflow import Config, run_pipeline

cfg = Config(
    DEM_PATH="dem.tif",
    OUTPUT_DIR="results/",
    OUTPUT_POINT=(27.632, 85.293),   # (lat, lon) of the basin outlet
    TARGET_CRS_EPSG="EPSG:32645",    # a metric CRS for your area

    # a simple design storm, routed as runoff
    PRECIP_METHOD="uniform",         # or the code: 0
    RAIN_INTENSITY_MM_HR=20.0,
    RAIN_DURATION_HOURS=3.0,
    TOTAL_SIMULATION_TIME_HOURS=12.0,
    RUNOFF_SOURCE="none",            # route rainfall directly (0)
    ROUTING_SCHEME="kinematic",      # kinematic | diffusive | muskingum
)
cfg.update_output_paths()

results = run_pipeline(cfg, stages=("process_dem", "routing"))
print(results["hydrograph_csv"])     # results/hydrograph.csv
```

`run_pipeline` runs the requested **stages** in order and returns a dict of
output paths and in-memory DataFrames. `process_dem` writes the watershed and
flow grids; `routing` writes `hydrograph.csv` plus an always-on mass-balance
report.

!!! tip "Strings or integer codes"
    Every fixed-choice option accepts its string **or** a short integer code —
    `PRECIP_METHOD="uniform"` and `PRECIP_METHOD=0` are identical. See
    [Configuration](configuration.md) or run `hydroflow list-options`.

## CLI

The same run, driven by a config file:

```bash
hydroflow init-config -o my_run.yaml     # write a template with every parameter
# edit DEM_PATH, OUTPUT_POINT, TARGET_CRS_EPSG, OUTPUT_DIR …
hydroflow validate -c my_run.yaml        # pre-flight checks
hydroflow run -c my_run.yaml             # process_dem + routing
```

Config files may be `.yaml`, `.json`, or a legacy flat `.py` settings module.

## What you get

| Output | File |
|---|---|
| Watershed mask + boundary | `watershed.tif`, `watershed.geojson` |
| Flow grids | `flow_direction.tif`, `clipped_flow_accumulation.tif` |
| Clipped DEM | `clipped_dem.tif` |
| Outlet hydrograph | `hydrograph.csv` |
| Mass-balance check | printed each run (optionally `mass_balance.csv`) |

Next: work through full, runnable [examples](examples.md) with plots.
