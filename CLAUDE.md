# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A distributed, physics-based rainfall–runoff and flood-routing model (VSA-OPM,
Pradhan & Ogden 2010): Variable Source Area saturation-excess runoff +
Green-Ampt infiltration-excess (Horton) + impervious urban shedding, feeding
explicit grid-based kinematic / diffusive-wave / Muskingum–Cunge channel
routing. Optional Google Earth Engine forcing (IMERG rainfall, SERVES soil
deficit, SoilGrids, LULC/LCZ). The science is pure NumPy/SciPy/rasterio — no
QGIS, no Qt in `hydroflow/core/`.

The pip package is **`hydroflow`** (`import hydroflow`). It was renamed from
`vsa_opm` — see "Naming & back-compat" below; the old names still work.

## Current focus

Scope is the **`hydroflow` package only** — the hydrologic model and its Python
package (core science, CLI, QGIS plugin). The active goal is making the
package better and more user-friendly: cleaner packaging/distribution,
clearer docs and examples, a smoother install/config/run experience, and
generally lowering the barrier for a new user to go from "pip install" to a
working run. Treat this as the standing task for this repo going forward.

Out of scope: `runs/trishuli_*`, `runs/trishuli_avaflow*`, and any
`r.avaflow`/GLOF/debris-flow material (e.g. `plan.md` at repo root) — that is
an unrelated study that happens to share this workspace, not part of the
`hydroflow` package.

## Naming & back-compat (the vsa_opm → hydroflow rename)

The package was renamed `vsa_opm` → **`hydroflow`** (v0.1.0). Nothing in the
science changed; only names moved:

- Import package is now `hydroflow/` (was `vsa_opm/`). Internal imports are all
  relative, so the rename was a directory move.
- The config class is now **`Config`** (was `OpmConfig`); `OpmConfig` is kept as
  an alias.
- A shim keeps old imports alive: `import vsa_opm`, `from vsa_opm.core… import …`
  and `OpmConfig` still work (with a `DeprecationWarning`), via
  `hydroflow/_compat.py` (a `sys.meta_path` finder) + a thin `vsa_opm/` shim
  package. Prefer `hydroflow` in new code.
- **Unchanged on purpose** (these name the science, not the package): the
  `RUNOFF_SOURCE="vsa_opm"` value, the `vsa_opm` pipeline stage, and
  `runoff_engine._mode == 'vsa_opm'`.
- The QGIS plugin still imports `vsa_opm` and vendors `_vendor/vsa_opm`; it keeps
  working through the shim. Re-vendoring the plugin under the new name is a
  pending follow-up.

## Install & environment

```bash
pip install hydroflow          # core (CPU), from PyPI (currently on TestPyPI)
pip install hydroflow[gpu]     # + CuPy/CUDA 12.x
pip install hydroflow[gee]     # + earthengine-api
pip install .                  # or from a checkout of this repo
```

Batch research workflows expect a conda env named `opm`
(`conda run -n opm python ...`).

## Running the model

Three interfaces, one core, all driven by a `Config` (or any object with the
same attributes) through `hydroflow.pipeline.run_pipeline`:

- **Python API**: `from hydroflow import Config, run_pipeline`
- **CLI**: `hydroflow init-config -o run.yaml` → `hydroflow validate -c run.yaml` →
  `hydroflow run -c run.yaml [--stages process_dem routing] [--backend cpu|gpu]`.
  `hydroflow list-options` prints every fixed-choice option and its integer code.
  Config files may be `.yaml`, `.json`, or a legacy flat `.py` settings module.
  (`vsa-opm` is kept as an alias of the `hydroflow` command.)
- **QGIS plugin** (`qgis_plugin/`): a 5-tab dialog + Processing algorithms that
  build a `Config` and call the same pipeline in a `QThread`.

Batch runner: edit `config.py` (legacy scenario module) and/or
`tools/runners/runner_config.py`, then `python tools/runner.py` — it dispatches
by `PRECIP_METHOD` to the gauge or IMERG pipeline over every flood event.

## Tests

There is no single test runner. Two disjoint suites:

- `tests/NN_*.py` — **standalone scripts**, not pytest. Run individually from
  the repo root, e.g. `python tests/03_test_vsa_opm.py`. They print PASS/FAIL
  per check and double as demos (several emit GIFs/PNGs into `tests/_demo_out/`).
- `qgis_plugin/tests/` — **pytest**, no QGIS needed for `test_config_bridge.py`
  (`pytest qgis_plugin/tests/test_config_bridge.py -v`); `test_runner.py` needs
  rasters under `output/`.

## Architecture

### The config object is the contract

`hydroflow/config.py::Config` (aliased `OpmConfig`) is the single source of truth
for every knob. Every core function (`dem_processing.main`, `initialise_grid`,
`run_time_loop`, `run_opm`, …) accepts any object exposing these attributes —
that duck-typing is why the legacy flat `config.py` module can be passed directly
as `cfg`. When adding a parameter, add it here first; `__init__` rejects unknown
kwargs, and mutable defaults (`RUNOFF_MECHANISMS`, `CHANNEL_WIDTH_BY_ORDER`,
`FIELD_VARS`) are deep-copied per instance. After changing `OUTPUT_DIR`, call
`update_output_paths()` to re-sync all derived paths (the pipeline does this for
you via `prepare_output_dir`). The QGIS plugin re-exports this same class through
`qgis_plugin/bridge/config_bridge.py` for backward compatibility — do not
redefine it there.

**Fixed-choice options take a string or an integer code.** Every enum-like knob
(`PRECIP_METHOD`, `RUNOFF_SOURCE`, `ROUTING_SCHEME`, `BACKEND`,
`DELINEATION_ENGINE`, the Manning/infiltration/impervious/SD sources, …) accepts
its canonical string *or* a 0-based integer code, normalised to the string on
assignment via `Config.__setattr__`. The registries are `_ENUM_CHOICES` and
`_ENUM_LIST` at the top of `config.py`; when adding a new enum option, add its
ordered choices there so codes and validation come for free (order is API —
appending is safe, reordering renumbers the codes). `Config.describe_options()`
renders the table (also the `hydroflow list-options` CLI command). Because the
value is normalised at assignment, `validate()` no longer re-checks enum
membership — only cross-field rules.

### Pipeline stages

`run_pipeline(cfg, stages=...)` orchestrates ordered stages, forwarding
`on_log` / `on_progress` / `is_cancelled` callbacks so each interface handles its
own threading/stdout. Stages:
- `process_dem` — `core/dem_processing.py`: reproject → fill → D8 flow
  dir/accum → delineate watershed. Writes `clipped_dem.tif`,
  `flow_direction.tif`, `clipped_flow_accumulation.tif`, `watershed.tif/.geojson`
  to `OUTPUT_DIR`. Engine selectable via `DELINEATION_ENGINE` (`pysheds` default
  — must stay byte-identical for existing callers — or opt-in `pyflwdir`, which
  fixes flow collapse across large flat reservoirs).
- `routing` — `core/routing/router.py`: `initialise_grid` → `run_time_loop` →
  `save_hydrograph`.
- `vsa_opm` — `core/opm.py::run_opm`, the standalone OPM runner.

### Runoff generation (`core/runoff/`)

`RunoffEngine` (`engine.py`) sits between precipitation [m/s] and the routing
time loop, dispatching on `RUNOFF_SOURCE`:
`none | coefficient | raster | scs_cn | vsa_opm`. It follows a **forward-Euler
contract**: call `get_effective_1d(t, rain)` (uses previous state) *then*
`update_state(rain, dt)`. The `vsa_opm` mode lives in `vsa.py` (`VsaOpmMixin`,
the sandbox water-balance / VSA / Green-Ampt / impervious mechanics), with
`soil.py` resolving SD_max / phi / suction (scalar, GEE/SERVES, or raster).
`RUNOFF_MECHANISMS` (`vsa`/`horton`/`impervious`) is an orthogonal, composable
subset.

### Routing (`core/routing/`)

Explicit grid solver in `router.py` (the time loop). Supporting modules:
`terrain.py` (D8 slopes, topological upstream-first order, downstream map),
`hydraulics.py` (Manning + diffusive-wave + Muskingum–Cunge kernels),
`surface.py` (Manning's n, channel geometry, impervious), `reporting.py`
(hydrograph + always-on mass balance). `ROUTING_SCHEME` selects
`kinematic | diffusive | muskingum` (all three wired in the time loop). Cells
are flattened to 1-D arrays in topological order for fast indexing; the outlet
is the last (highest-accumulation) cell.

Optional routing add-ons (all off by default, config-gated):
- `boundary.py` (`ROUTING_INFLOW_BC`) — inject an external Q(t) hydrograph at
  one or more cells (lat/lon, row/col, or easting/northing; snapped to channel),
  so the router can run pure downstream routing with no rainfall.
- `gauges.py` (`ROUTING_GAUGES`) — virtual gauges written to `gauges.csv`
  (depth/Q/velocity at named points every `OUTPUT_INTERVAL`).
- `fields.py` (`SAVE_FIELDS`) — compact per-cell depth/velocity/discharge archive
  (`fields.npz` + `fields_meta.json`) for post-hoc plots/animations.

### CPU/GPU backend selection

`BACKEND='gpu'` + `gpu_utils.cupy_available()` swaps in the `gpu.py` variants
(each of `routing/`, `runoff/`, `precip/` has one) at `initialise_grid` time,
setting `xp = cupy`; otherwise NumPy. GPU is requested-but-unavailable →
automatic CPU fallback with a warning. Never assume a GPU is present.

### Google Earth Engine

Optional and lazily imported. `gee/auth.py` initializes EE from
`GOOGLE_APPLICATION_CREDENTIALS`, then a `key.json` next to the gee module / repo
root / cwd, then `GEE_PROJECT` (config attr or env var). `key.json` is **never
committed**. All GEE-backed options degrade gracefully — scalar/manual/gauge
settings run fully offline, and `Config.validate()` errors early if a
GEE-backed source is selected without a project. `gee/dem_gee.py` downloads a
NASADEM (area-averaged, auto-tiled/mosaicked) to bootstrap a basin that has no
local DEM yet.

## Repo-specific notes

- `config.py` at the repo root is the **legacy research scenario module** (values
  only), used by `tools/` and `tests/`. It is distinct from
  `hydroflow/config.py::Config`. Both are loadable by the CLI.
- `study/` is a separate Next.js MDX interactive textbook (the public course
  site); `docs/` is the LaTeX companion. They are documentation, not the model.
- The plugin still ships a vendored copy of the core package under
  `_vendor/vsa_opm/` (pre-rename name) and imports `vsa_opm`; it works via the
  shim. `bridge.ensure_core()` puts the repo-root package on `sys.path` in
  dev/symlink mode. Rebuild with `./build_windows_plugin.sh`. Re-vendoring the
  plugin as `hydroflow` is a pending follow-up.
- Land-cover lookups (`lulc_lookup.csv`, `lcz_lookup.csv`) ship inside the
  package at `hydroflow/data/` and are the config defaults.
- Packaging lives in `pyproject.toml` (`hydroflow` dist, `hydroflow`/`vsa-opm`
  console scripts, `[gpu]`/`[gee]` extras), plus `LICENSE` (MIT) and
  `MANIFEST.in`. Build with `python -m build`; `v0.1.0` is tagged and on
  TestPyPI (real PyPI upload deferred).
