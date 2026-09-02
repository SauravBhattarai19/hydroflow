# Configuration

Every run is parameterised by a single [`Config`][hydroflow.Config] object. It
is a plain mutable object with sensible, offline-friendly defaults — override
only what you need, by keyword, attribute, or a config file.

```python
from hydroflow import Config

cfg = Config(DEM_PATH="dem.tif", BACKEND="gpu")   # keyword
cfg.OUTPUT_DIR = "results/"                        # attribute
cfg.update_output_paths()                          # re-sync derived paths
```

`Config` is the single source of truth for the whole model; `OpmConfig` is an
alias of it. Load from / save to files:

```python
cfg = Config.from_file("run.yaml")   # .yaml | .json | legacy flat .py
cfg.save("run.json")
```

!!! note "After changing `OUTPUT_DIR`"
    Call `cfg.update_output_paths()` to re-base all derived output paths
    (the CLI and `run_pipeline` do this for you).

## Options: string or integer code

Fixed-choice options accept their canonical **string** or a 0-based **integer
code** — both are normalised to the string and validated on assignment:

```python
Config(PRECIP_METHOD="thiessen", ROUTING_SCHEME="muskingum")   # strings
Config(PRECIP_METHOD=1,          ROUTING_SCHEME=2)             # codes — identical
```

Discover them anytime with `hydroflow list-options` or
`Config.describe_options()`:

| Option | Codes |
|---|---|
| `PRECIP_METHOD` | `0` uniform · `1` thiessen · `2` idw · `3` imerg_thiessen · `4` imerg_idw |
| `RUNOFF_SOURCE` | `0` none · `1` coefficient · `2` raster · `3` scs_cn · `4` vsa_opm |
| `ROUTING_SCHEME` | `0` kinematic · `1` diffusive · `2` muskingum |
| `DELINEATION_ENGINE` | `0` pysheds · `1` pyflwdir |
| `BACKEND` | `0` cpu · `1` gpu |
| `GPU_PRECISION` | `0` float64 · `1` float32 |
| `OPM_INFILTRATION` | `0` none · `1` green_ampt |
| `OPM_GA_SUCTION_SOURCE` | `0` scalar · `1` texture |
| `OPM_GA_KSAT_SOURCE` | `0` scalar · `1` gee · `2` raster |
| `IMPERVIOUS_SOURCE` | `0` none · `1` lcz · `2` lulc · `3` raster |
| `OPM_SD_SOURCE` | `0` manual · `1` gee |
| `OPM_SD_REDUCER` | `0` mean · `1` max · `2` divide |
| `SERVES_SATELLITE` | `0` landsat · `1` sentinel2 · `2` modis |
| `OPM_SOILGRIDS_DEPTH` | `0` b0 · `1` b10 · `2` b30 · `3` b60 · `4` b100 · `5` b200 |
| `MANNINGS_N_SOURCE` | `0` scalar · `1` lulc · `2` lcz · `3` raster |
| `RUNOFF_MECHANISMS` (list) | `0` vsa · `1` horton · `2` impervious |

## Key parameter groups

=== "Event & grid"

    | Parameter | Meaning |
    |---|---|
    | `DEM_PATH` | input DEM (GeoTIFF) |
    | `OUTPUT_POINT` | `(lat, lon)` of the basin outlet |
    | `TARGET_CRS_EPSG` | metric CRS for the run, e.g. `"EPSG:32645"` |
    | `OUTPUT_DIR` | where results are written |
    | `CELL_SIZE` | `None` → auto-detect from DEM |

=== "Precipitation"

    | Parameter | Meaning |
    |---|---|
    | `PRECIP_METHOD` | forcing method (see table) |
    | `RAIN_INTENSITY_MM_HR`, `RAIN_DURATION_HOURS` | uniform design storm |
    | `PRECIP_GAUGE_FILE`, `PRECIP_TIMESERIES_FILE` | gauge data for thiessen/idw |
    | `EVENT_START_UTC`, `GEE_PROJECT` | needed for IMERG forcing |

=== "Runoff"

    | Parameter | Meaning |
    |---|---|
    | `RUNOFF_SOURCE` | generation method (see table) |
    | `RUNOFF_MECHANISMS` | subset of `vsa`/`horton`/`impervious` |
    | `OPM_SD_MAX_INITIAL`, `OPM_PHI`, `OPM_K_SAT` | VSA-OPM sandbox parameters |
    | `OPM_INFILTRATION` | Green-Ampt on/off |

=== "Routing"

    | Parameter | Meaning |
    |---|---|
    | `ROUTING_SCHEME` | `kinematic`/`diffusive`/`muskingum` |
    | `MANNINGS_N`, `MANNINGS_N_SOURCE` | roughness |
    | `CHANNEL_ROUTING` | confined rectangular channel cells |
    | `TIME_STEP_SECONDS`, `ADAPTIVE_TIMESTEP` | time stepping |

The full, commented list of every parameter and its default lives in the
[`Config` API reference](api.md).

## Earth Engine

GEE-backed inputs (IMERG rainfall, SERVES deficit, SoilGrids, LULC/LCZ) are
optional and lazily imported. Set a project and authenticate one of three ways,
tried in order:

1. `GOOGLE_APPLICATION_CREDENTIALS` (service-account JSON)
2. a `key.json` next to the package / repo root / cwd
3. `GEE_PROJECT` (config attribute or environment variable)

`Config.validate()` errors early if a GEE-backed option is selected without a
project, so offline runs never surprise you.
