# Examples

Every example below is runnable — code and output come from real runs against
live Google Earth Engine data. They build on each other: local DEM → DEM
straight from Earth Engine → adding an upstream boundary condition →
spatially-varying Manning's n → a capstone combining everything.

## 1. Delineate the watershed from a DEM

Run just the `process_dem` stage to reproject, pit-fill, compute D8 flow
direction/accumulation, and delineate the catchment draining to your outlet.

```python
from hydroflow import Config, run_pipeline, plot_watershed

cfg = Config(
    DEM_PATH="dem_250.tif",
    OUTPUT_DIR="results/",
    OUTPUT_POINT=(27.632222, 85.293333),   # (lat, lon)
    TARGET_CRS_EPSG="EPSG:32645",
)
cfg.update_output_paths()

out = run_pipeline(cfg, stages=("process_dem",))
print(out["watershed_geojson"])   # results/watershed.geojson

fig, ax = plot_watershed(out)
fig.savefig("watershed.png")
```

![Delineated watershed](assets/img/watershed.png)

`plot_watershed` accepts the `run_pipeline()` result dict directly, a
`Config`, or an `OUTPUT_DIR` path — no need to hand-write the
rasterio/geopandas overlay yourself.

## 2. Get a DEM from Earth Engine and delineate

No local DEM yet? Skip `DEM_PATH` and give a bounding box instead — hydroflow
downloads, area-averages and reprojects a DEM from Google Earth Engine before
delineating. Requires `pip install hydroflow[gee]` and authentication (see
[Configuration](configuration.md)).

```python
import hydroflow
from hydroflow import Config, run_pipeline, plot_watershed

print(hydroflow.describe_available_dems())   # browse options — no [gee] needed just to look

cfg = Config(
    DEM_BOUNDS_WGS84=(85.05, 27.55, 85.55, 27.90),  # (min_lon, min_lat, max_lon, max_lat)
    DEM_SOURCE="nasadem",                            # see hydroflow list-dems
    OUTPUT_DIR="results/",
    OUTPUT_POINT=(27.632222, 85.293333),             # (lat, lon), inside the box
    TARGET_CRS_EPSG="EPSG:32645",
    GEE_PROJECT="your-gee-project",
)
cfg.update_output_paths()

out = run_pipeline(cfg, stages=("process_dem",))
fig, ax = plot_watershed(out)
fig.savefig("watershed.png")
```

![Watershed from a GEE-downloaded DEM](assets/img/dem_gee_watershed.png)

This delineates the real ~584 km² Bagmati watershed above Kathmandu — the
download is cached to `results/raw_dem_gee.tif`, so re-running the same
config skips straight to delineation.

## 3. Add an upstream boundary-condition hydrograph

[`ROUTING_INFLOW_BC`](configuration.md#boundary-conditions) injects an
external discharge hydrograph Q(t) at one or more points, so you can route
water entering from upstream of your domain — set `RAIN_INTENSITY_MM_HR=0`
for pure routing driven only by the boundary condition. The location can be
lat/lon; `snap_to_channel` (on by default) automatically finds the nearest
high-flow-accumulation cell within a few cells, so it doesn't need to be
pixel-perfect.

```python
import numpy as np, pandas as pd
from hydroflow import Config, run_pipeline, plot_hydrograph

# Synthetic upstream inflow: rises to a 12 m3/s peak at t=1h, recedes to a
# 2 m3/s baseflow by t=3h.
t_hr = np.linspace(0, 6, 73)
Q = np.where(t_hr <= 1, 2 + 10 * t_hr,
     np.where(t_hr <= 3, 12 - 5 * (t_hr - 1), 2.0))
pd.DataFrame({"time_hr": t_hr, "Q_m3s": Q}).to_csv("results/inflow_upstream.csv", index=False)

cfg = Config(
    DEM_BOUNDS_WGS84=(85.3809, 27.7625, 85.4773, 27.8223),
    DEM_SOURCE="nasadem",
    OUTPUT_DIR="results/",
    OUTPUT_POINT=(27.77772, 85.42399),
    TARGET_CRS_EPSG="EPSG:32645",
    GEE_PROJECT="your-gee-project",
    RAIN_INTENSITY_MM_HR=0,
    PRECIP_METHOD="uniform",
    RUNOFF_SOURCE="none",              # pure routing — no rainfall-generated runoff
    ROUTING_INFLOW_BC=[{
        "name": "upstream",
        "lat": 27.80667, "lon": 85.39830,   # anywhere near the upstream channel
        "csv": "results/inflow_upstream.csv",
    }],
    TOTAL_SIMULATION_TIME_HOURS=6.0,
    ADAPTIVE_TIMESTEP=True,            # recommended for point-source BCs on steep terrain
)
cfg.update_output_paths()
out = run_pipeline(cfg, stages=("process_dem", "routing"))

fig, ax = plot_hydrograph(out, label="outlet Q(t)")
fig.savefig("bc_hydrograph.png")
```

![Outlet hydrograph from an upstream boundary condition](assets/img/bc_routing_hydrograph.png)

The injected pulse peaks at 12 m³/s; by the outlet it arrives attenuated and
delayed (peak ≈ 11.4 m³/s around t≈1.7h) — exactly what kinematic-wave
translation/attenuation should do to a wave routed through ~5 km of channel.

!!! tip "Use ADAPTIVE_TIMESTEP for boundary-condition runs"
    A small discharge routed through a fixed, coarse timestep on steep
    terrain can trigger heavy CFL flux-limiting and stall numerically.
    `ADAPTIVE_TIMESTEP=True` (with the default `CFL_TARGET=0.85`) lets the
    router shrink the timestep automatically and is the more robust choice
    whenever you're not routing a basin-wide rainfall event.

## 4. Manning's n by elevation

There's no dedicated `MANNINGS_N_SOURCE="elevation"` — instead,
`mannings_n_from_dem` generates a Manning's-n raster from your DEM using an
elevation rule, and you point the existing `MANNINGS_N_SOURCE="raster"` path
at it.

```python
from hydroflow import Config, run_pipeline, mannings_n_from_dem, plot_raster, plot_hydrograph

cfg = Config(
    DEM_BOUNDS_WGS84=(85.3809, 27.7625, 85.4773, 27.8223),
    DEM_SOURCE="nasadem",
    OUTPUT_DIR="results/",
    OUTPUT_POINT=(27.77772, 85.42399),
    TARGET_CRS_EPSG="EPSG:32645",
    GEE_PROJECT="your-gee-project",
)
cfg.update_output_paths()
out = run_pipeline(cfg, stages=("process_dem",))

n_path = mannings_n_from_dem(
    dem_path=out["clipped_dem"],
    rule=[(1700, 0.035), (2000, 0.06), (2300, 0.10), (float("inf"), 0.14)],
    output_path="results/mannings_n_elev.tif",
)
fig, ax = plot_raster(n_path, cmap="YlOrBr", label="Manning's n")
fig.savefig("mannings_n_elevation.png")

cfg.MANNINGS_N_SOURCE = "raster"
cfg.MANNINGS_N_RASTER_PATH = n_path
cfg.PRECIP_METHOD = "uniform"
cfg.RAIN_INTENSITY_MM_HR = 20.0
cfg.RAIN_DURATION_HOURS = 1.0
cfg.RUNOFF_SOURCE = "none"
cfg.TOTAL_SIMULATION_TIME_HOURS = 5.0
cfg.ADAPTIVE_TIMESTEP = True
out.update(run_pipeline(cfg, stages=("routing",)))

fig2, ax2 = plot_hydrograph(out, label="outlet Q(t)")
fig2.savefig("mannings_n_hydrograph.png")
```

![Elevation-based Manning's n raster](assets/img/mannings_n_elevation.png)
![Hydrograph routed with elevation-based roughness](assets/img/mannings_n_hydrograph.png)

`rule` accepts a list of ascending `(upper_bound_elev, n)` breakpoints (as
above), a `{(min, max): n}` dict of bins, or any callable
`f(elevation_array) -> n_array` for a continuous relationship. The router's
own log confirms the raster reached it: `Manning's n | source=raster
range=[0.035, 0.140]`.

## 5. Capstone: everything together

GEE DEM download, elevation-based Manning's n, an upstream boundary
condition, and full VSA-OPM physics (saturation-excess + Green-Ampt +
impervious shedding) — all in one run.

```python
from hydroflow import (Config, run_pipeline, mannings_n_from_dem,
                        plot_watershed, plot_hydrograph, plot_mass_balance)

cfg = Config(
    DEM_BOUNDS_WGS84=(85.3809, 27.7625, 85.4773, 27.8223),
    DEM_SOURCE="nasadem",
    OUTPUT_DIR="results/",
    OUTPUT_POINT=(27.77772, 85.42399),
    TARGET_CRS_EPSG="EPSG:32645",
    GEE_PROJECT="your-gee-project",
)
cfg.update_output_paths()
out = run_pipeline(cfg, stages=("process_dem",))

n_path = mannings_n_from_dem(
    dem_path=out["clipped_dem"],
    rule=[(1700, 0.035), (2000, 0.06), (2300, 0.10), (float("inf"), 0.14)],
    output_path="results/mannings_n_elev.tif",
)

cfg.MANNINGS_N_SOURCE = "raster"
cfg.MANNINGS_N_RASTER_PATH = n_path
cfg.PRECIP_METHOD = "uniform"
cfg.RAIN_INTENSITY_MM_HR = 15.0
cfg.RAIN_DURATION_HOURS = 1.0
cfg.RUNOFF_SOURCE = "vsa_opm"                    # or 4
cfg.RUNOFF_MECHANISMS = ["vsa", "horton", "impervious"]
cfg.OPM_INFILTRATION = "green_ampt"
cfg.ROUTING_INFLOW_BC = [{
    "name": "upstream", "lat": 27.80667, "lon": 85.39830,
    "csv": "results/inflow_upstream.csv",        # from example 3
}]
cfg.TOTAL_SIMULATION_TIME_HOURS = 8.0
cfg.ADAPTIVE_TIMESTEP = True
out.update(run_pipeline(cfg, stages=("routing",)))

plot_watershed(out)[0].savefig("capstone_watershed.png")
plot_hydrograph(out, label="outlet Q(t)")[0].savefig("capstone_hydrograph.png")
plot_mass_balance(out)[0].savefig("capstone_massbalance.png")
```

![Capstone watershed](assets/img/capstone_watershed.png)
![Capstone hydrograph](assets/img/capstone_hydrograph.png)
![Capstone mass balance](assets/img/capstone_massbalance.png)

The hydrograph shows both sources' fingerprints — the rainfall-driven VSA-OPM
runoff peaks quickly, riding on top of the slower upstream BC pulse — and the
mass balance still closes exactly, tracking rainfall-runoff and
boundary-condition inflow as two independent, separately-accounted-for
sources of water into the domain.
