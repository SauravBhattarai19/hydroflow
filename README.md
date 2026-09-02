# hydroflow — distributed hydrological + hydrodynamic model

A distributed, physics-based rainfall–runoff and flood-routing model built around
a **Variable Source Area (VSA)** runoff scheme, Green-Ampt infiltration, and
explicit kinematic / diffusive-wave / Muskingum–Cunge channel routing — driven by open satellite
data (SERVES soil moisture, IMERG precipitation, SoilGrids, LULC/LCZ) via
Google Earth Engine.

## 📖 Learn the model — interactive course

**[sauravbhattarai19.github.io/One-Parameter-Hydrologic-Model](https://sauravbhattarai19.github.io/One-Parameter-Hydrologic-Model/)**

This is the place to actually understand how the model works. It's a free,
open-source interactive textbook (built from the [`study/`](study) folder) that
walks through the physics from the ground up, with in-browser simulations —
no installation required:

| Chapter | Topic |
|---|---|
| 1 | Digital Elevation Models & Flow Direction (D8, flow accumulation, pit filling) |
| 2 | Watershed Delineation (catchment boundaries, stream networks, pour points) |
| 3 | Rainfall–Runoff Generation (VSA, Green-Ampt, impervious shedding, satellite pipeline) |
| 4 | Kinematic-Wave Routing (Saint-Venant simplification, Manning's equation) |
| 5 | Diffusive-Wave Routing (backwater effects, GSSHA-style conveyance, numerical diffusion) |
| 6 | Muskingum–Cunge Routing (variable-parameter; numerical diffusion tuned to physical diffusivity D=Q/(2·B·S₀), grid-independent) |

If you're new to this repo, start there before digging into the code below.

## Installation

The model is a pip-installable package (`hydroflow`):

```bash
pip install hydroflow          # core (CPU), from PyPI
pip install hydroflow[gpu]     # + CuPy/CUDA acceleration
pip install hydroflow[gee]     # + Google Earth Engine (IMERG, SERVES, SoilGrids, LULC/LCZ)

# or from a checkout of this repo:
pip install .
pip install .[gpu]
pip install .[gee]
```

## Running the model

One core drives three interfaces:

**1. Python API**

```python
from hydroflow import Config, run_pipeline

cfg = Config(DEM_PATH="dem.tif", OUTPUT_DIR="results/")
cfg.update_output_paths()
results = run_pipeline(cfg, stages=("process_dem", "routing"))
```

**2. CLI** — config-file driven (YAML, JSON, or a legacy flat `.py` module):

```bash
hydroflow init-config -o my_run.yaml     # template with every parameter
hydroflow validate -c my_run.yaml        # pre-flight checks
hydroflow run -c my_run.yaml             # process_dem + routing
hydroflow list-options                   # show fixed-choice options + integer codes
```

See [`configs/example_config.yaml`](configs/example_config.yaml) for the
repository's research scenario in CLI form.

### Choosing options — string *or* integer code

Every fixed-choice option accepts either its string value **or** a short
integer code, so these are equivalent:

```python
Config(PRECIP_METHOD="thiessen", ROUTING_SCHEME="muskingum", BACKEND="gpu")
Config(PRECIP_METHOD=1,          ROUTING_SCHEME=2,           BACKEND=1)
```

The value is always normalised to (and saved as) the canonical string. Run
`hydroflow list-options` for the full table; the most common ones:

| Option | Codes |
|---|---|
| `PRECIP_METHOD` | `0` uniform · `1` thiessen · `2` idw · `3` imerg_thiessen · `4` imerg_idw |
| `RUNOFF_SOURCE` | `0` none · `1` coefficient · `2` raster · `3` scs_cn · `4` vsa_opm |
| `ROUTING_SCHEME` | `0` kinematic · `1` diffusive · `2` muskingum |
| `DELINEATION_ENGINE` | `0` pysheds · `1` pyflwdir |
| `BACKEND` | `0` cpu · `1` gpu |
| `RUNOFF_MECHANISMS` (list) | `0` vsa · `1` horton · `2` impervious |

**3. QGIS plugin** — see [`qgis_plugin/README.md`](qgis_plugin/README.md); the
plugin imports the same `hydroflow` package (pip-installed or vendored in the
plugin zip).

For the repository's batch research workflows, configure
[`config.py`](config.py) (legacy scenario module) and run
[`tools/runner.py`](tools/runner.py), which dispatches by `PRECIP_METHOD`
(Thiessen, IDW, or IMERG-driven variants).

Outputs (hydrograph, mass-balance diagnostics, rasters) are written under the
scenario's `OUTPUT_DIR`.  Optional Earth Engine integration
(`OPM_SD_SOURCE='gee'`, IMERG precipitation) needs a GEE service account — see
[`hydroflow/gee/serves_gee.py`](hydroflow/gee/serves_gee.py) and `test_ee_auth.sh`
for setup/verification.

> Note: the runoff-method value `RUNOFF_SOURCE="vsa_opm"` and the `vsa_opm`
> pipeline stage name the Pradhan & Ogden (2010) VSA-OPM scheme (the science),
> not the package — they are unrelated to the package name.

## Repository layout

- `hydroflow/` — the pip-installable model package
  - `core/` — the science (QGIS-free)
    - `routing/` — `terrain.py` (D8/slopes/topological order), `hydraulics.py`
      (Manning + diffusive-wave + Muskingum–Cunge kernels), `surface.py`
      (Manning's n, channel geometry, impervious), `router.py` (the time loop),
      `reporting.py` (hydrograph/mass balance), `boundary.py` (inflow BC),
      `gauges.py` (virtual gauges), `fields.py` (per-cell field archive),
      `gpu.py` (CuPy variants)
    - `runoff/` — `engine.py` (RunoffEngine dispatcher), `vsa.py` (VSA-OPM /
      Green-Ampt / impervious mechanics), `soil.py` (SD_max/phi/suction
      resolution), `gpu.py`
    - `precip/` — `engine.py` (uniform/Thiessen/IDW/IMERG), `gpu.py`
    - `dem_processing.py` — watershed preprocessing (pysheds)
    - `opm.py` — standalone OPM runner; `io_utils.py` — shared raster helpers
  - `gee/` — Earth Engine integrations (`auth.py`, `serves_gee.py`,
    `imerg_gee.py`, `dem_gee.py`)
  - `cli/` — the `hydroflow` command-line interface
  - `config.py` — `Config` (aliased `OpmConfig`), the single configuration object
  - `pipeline.py` — stage orchestration shared by API, CLI and plugin
- `config.py` — legacy research-scenario settings used by `tools/` and `tests/`
- `configs/` — example CLI config files
- `tools/` — batch runners, sensitivity/OFAT sweeps, experiment combinations
- `study/` — source for the interactive course site above
- `docs/` — LaTeX lecture notes companion to the course
- `qgis_plugin/` — QGIS front-end (UI + QThread worker importing `hydroflow`)
