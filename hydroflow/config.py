# -*- coding: utf-8 -*-
"""
config.py
=========
Config — the single configuration object for the hydroflow model.

Every simulation function in the package (process_dem.main, initialise_grid,
run_time_loop, run_opm, …) accepts any object exposing these attributes, so
Config is the canonical way to parameterise a run from the Python API, the
CLI (via a YAML/JSON file) or the QGIS plugin UI.

Fixed-choice options accept either their string value or a short integer code
(e.g. ``PRECIP_METHOD="uniform"`` and ``PRECIP_METHOD=0`` are equivalent);
the value is always normalised and stored as the canonical string.  See
``_ENUM_CHOICES`` below or run ``Config.describe_options()``.

Usage
-----
    from hydroflow import Config

    cfg = Config(DEM_PATH="/path/to/dem.tif", BACKEND="gpu")   # or BACKEND=1
    cfg.OUTPUT_DIR = "/path/to/results"
    cfg.update_output_paths()

    # or from a config file (YAML / JSON / legacy python module):
    cfg = Config.from_file("config.yaml")

``OpmConfig`` is available as an alias of ``Config``.
"""

import json
import os


def _data_path(filename):
    """Absolute path of a lookup file shipped inside the package (hydroflow/data)."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", filename)


# ═════════════════════════════════════════════════════════════════════════════
# Fixed-choice option registry
# ═════════════════════════════════════════════════════════════════════════════
# Each entry maps a config attribute to its ordered list of valid string values.
# The list index is the option's integer code, so users may pass either the
# string or the integer (both are accepted anywhere a config value is set):
#     PRECIP_METHOD="thiessen"  ≡  PRECIP_METHOD=1
# Values are always normalised to (and stored as) the canonical string, so
# saved YAML/JSON and logging stay human-readable.  Order is API — appending is
# safe, reordering renumbers the codes.
_ENUM_CHOICES = {
    "PRECIP_METHOD":         ["uniform", "thiessen", "idw", "imerg_thiessen", "imerg_idw"],
    "RUNOFF_SOURCE":         ["none", "coefficient", "raster", "scs_cn", "vsa_opm"],
    "ROUTING_SCHEME":        ["kinematic", "diffusive", "muskingum"],
    "DELINEATION_ENGINE":    ["pysheds", "pyflwdir"],
    "BACKEND":               ["cpu", "gpu"],
    "GPU_PRECISION":         ["float64", "float32"],
    "OPM_INFILTRATION":      ["none", "green_ampt"],
    "OPM_GA_SUCTION_SOURCE": ["scalar", "texture"],
    "OPM_GA_KSAT_SOURCE":    ["scalar", "gee", "raster"],
    "IMPERVIOUS_SOURCE":     ["none", "lcz", "lulc", "raster"],
    "OPM_SD_SOURCE":         ["manual", "gee"],
    "OPM_SD_REDUCER":        ["mean", "max", "divide"],
    "SERVES_SATELLITE":      ["landsat", "sentinel2", "modis"],
    "OPM_SOILGRIDS_DEPTH":   ["b0", "b10", "b30", "b60", "b100", "b200"],
    "MANNINGS_N_SOURCE":     ["scalar", "lulc", "lcz", "raster"],
}

# Options whose value is a *list* of choices (each element normalised the same
# way as a scalar enum), rather than a single choice.
_ENUM_LIST = {
    "RUNOFF_MECHANISMS": ["vsa", "horton", "impervious"],
}


def _normalize_enum(key, value, choices):
    """Normalise a single fixed-choice value to its canonical string.

    Accepts the canonical string (case-insensitive) or an integer code (int or
    a digit-string like "1").  Raises ValueError listing the valid choices and
    their codes on anything else.
    """
    # Integer code (real int, or a bare digit string) → index into choices.
    if isinstance(value, bool):
        pass  # bools are ints in Python; treat as invalid below
    elif isinstance(value, int):
        if 0 <= value < len(choices):
            return choices[value]
        raise ValueError(
            f"{key}: integer code {value} out of range; valid codes are "
            f"{_choices_help(choices)}."
        )
    elif isinstance(value, str) and value.strip().isdigit():
        idx = int(value.strip())
        if 0 <= idx < len(choices):
            return choices[idx]
        raise ValueError(
            f"{key}: integer code {idx} out of range; valid codes are "
            f"{_choices_help(choices)}."
        )

    if isinstance(value, str):
        v = value.strip().lower()
        if v in choices:
            return v
    raise ValueError(
        f"{key}: {value!r} is not a valid choice; use one of "
        f"{_choices_help(choices)}."
    )


def _choices_help(choices):
    """'0=uniform, 1=thiessen, …' helper string for error/help messages."""
    return ", ".join(f"{i}={c}" for i, c in enumerate(choices))


class Config:
    """
    All model knobs as a mutable plain object.

    Defaults are conservative and machine-independent so the model runs
    "out of the box"; every value can be overridden individually (keyword
    arguments, attribute assignment, or a YAML/JSON config file).

    Fixed-choice options (see ``_ENUM_CHOICES``) accept either their string
    value or an integer code and are normalised to the canonical string on
    assignment, so ``Config(BACKEND=1)`` and ``Config(BACKEND="gpu")`` are
    identical.  ``OpmConfig`` is an alias of this class.
    """

    # ═════════════════════════════════════════════════════════════════════════
    # 1.  EVENT & SCENARIO
    # ═════════════════════════════════════════════════════════════════════════
    DEM_PATH: str = ""
    TARGET_CRS_EPSG: str = "EPSG:32645"
    OUTPUT_POINT: tuple = (27.632222, 85.293333)   # (lat, lon)
    OUTPUT_DIR: str = "output/"

    # DEM delineation engine.  'pysheds' (default) is the original engine and
    # must stay byte-identical for all existing callers.  'pyflwdir' is an
    # opt-in alternative (priority-flood fill) that correctly resolves flow
    # across large flat reservoirs/lakes where pysheds' fill+resolve_flats can
    # collapse flow accumulation (see vsa_opm/core/dem_processing.py).
    DELINEATION_ENGINE: str = "pysheds"   # 'pysheds' | 'pyflwdir'

    # EVENT_START_UTC: "YYYY-MM-DD HH:MM" UTC — single source of truth for the
    # event date (SERVES antecedent query + IMERG download window).
    EVENT_START_UTC = None
    TOTAL_SIMULATION_TIME_HOURS: float = 96.0
    IMERG_UTC_OFFSET_HOURS: float = 5.75           # NPT = UTC+5:45

    # Land-cover lookup CSVs (root-zone depth, Manning's n, impervious fraction).
    # Defaults resolve to the copies shipped inside the package (vsa_opm/data/).
    LULC_LOOKUP_CSV: str = _data_path("lulc_lookup.csv")   # ESA WorldCover
    LCZ_LOOKUP_CSV: str = _data_path("lcz_lookup.csv")     # WUDAPT LCZ

    # Google Earth Engine cloud project (IMERG / SERVES / LULC / LCZ download).
    GEE_PROJECT = None

    # ═════════════════════════════════════════════════════════════════════════
    # 2.  WATERSHED PRE-PROCESSING OUTPUTS  (written by core.process_dem)
    # ═════════════════════════════════════════════════════════════════════════
    ROUTING_DEM_PATH: str = "output/clipped_dem.tif"
    ROUTING_FLOW_DIR_PATH: str = "output/flow_direction.tif"
    ROUTING_FLOW_ACCUM_PATH: str = "output/clipped_flow_accumulation.tif"
    ROUTING_WATERSHED_MASK_PATH: str = "output/watershed.tif"
    OPM_WATERSHED_GEOJSON: str = "output/watershed.geojson"

    # ═════════════════════════════════════════════════════════════════════════
    # 3.  PRECIPITATION
    # ═════════════════════════════════════════════════════════════════════════
    RAIN_INTENSITY_MM_HR: float = 20.0
    RAIN_DURATION_HOURS: float = 3.0

    # 'uniform' | 'thiessen' | 'idw' | 'imerg_thiessen' | 'imerg_idw'
    PRECIP_METHOD: str = "uniform"
    PRECIP_GAUGE_FILE: str = ""
    PRECIP_TIMESERIES_FILE: str = ""
    PRECIP_IDW_POWER: float = 2.0
    PRECIP_EXCLUDE_OUTSIDE_STATIONS: bool = False

    # IMERG source (used when PRECIP_METHOD='imerg_thiessen'/'imerg_idw').
    IMERG_START_LOCAL = None       # None → auto from EVENT_START_UTC
    IMERG_END_LOCAL = None
    PRECIP_IMERG_DIR: str = "output/imerg/"
    IMERG_DATASET: str = "NASA/GPM_L3/IMERG_V07"
    IMERG_BAND: str = "precipitation"
    PRECIP_IMERG_FORCE_DOWNLOAD: bool = False
    IMERG_BBOX_BUFFER_M: float = 11132.0

    # ═════════════════════════════════════════════════════════════════════════
    # 4.  RUNOFF GENERATION
    # ═════════════════════════════════════════════════════════════════════════
    # 'none'|'coefficient'|'raster'|'scs_cn'|'vsa_opm'
    RUNOFF_SOURCE: str = "none"
    RUNOFF_COEFFICIENT_PATH: str = ""
    RUNOFF_RASTER_MANIFEST: str = ""
    RUNOFF_CN_PATH: str = ""
    RUNOFF_SCS_Ia_FACTOR: float = 0.2

    # ═════════════════════════════════════════════════════════════════════════
    # 5.  OPM / VSA PARAMETERS  (used when RUNOFF_SOURCE='vsa_opm')
    # ═════════════════════════════════════════════════════════════════════════
    # Which runoff-generation mechanisms are active (orthogonal subset).
    #   'vsa' | 'horton' | 'impervious'
    RUNOFF_MECHANISMS = ["vsa", "horton", "impervious"]

    OPM_SD_MAX_INITIAL: float = 0.10   # root zone depth D [m] (physical height)
    OPM_Q_MAX: float = 100.0           # observed baseflow / initial discharge [m³/s]
    OPM_PHI: float = 0.35              # drainable porosity [-]
    OPM_K_SAT: float = 44.0            # lateral saturated conductivity [m/day]
    OPM_PER_POLYGON: bool = True

    # ── Infiltration (Green-Ampt) ────────────────────────────────────────────
    OPM_INFILTRATION: str = "none"          # 'none' | 'green_ampt'
    OPM_GA_SUCTION_SOURCE: str = "scalar"   # 'scalar' | 'texture'
    OPM_GA_SUCTION_M: float = 0.15          # wetting-front suction head ψ [m]
    OPM_GA_KSAT_SOURCE: str = "scalar"      # 'scalar' | 'gee' | 'raster'
    OPM_GA_KSAT_MMHR: float = 12.0          # vertical surface Ksat [mm/hr]
    OPM_GA_KSAT_RASTER = None               # None → auto {OUTPUT_DIR}/ksat_hihydro.tif
    OPM_GA_KSAT_SCALE: float = 1.0

    # ── Impervious fraction (urban shedding) ─────────────────────────────────
    IMPERVIOUS_SOURCE: str = "none"         # 'none'|'lcz'|'lulc'|'raster'
    IMPERVIOUS_RASTER_PATH = None

    # ── Baseflow ─────────────────────────────────────────────────────────────
    OPM_BASEFLOW: bool = False

    # ═════════════════════════════════════════════════════════════════════════
    # 6.  SERVES / GEE SOIL-MOISTURE DEFICIT  (SD_max & phi from satellite)
    # ═════════════════════════════════════════════════════════════════════════
    OPM_SD_SOURCE: str = "manual"           # 'manual' | 'gee'
    OPM_SD_REDUCER: str = "mean"            # 'mean' | 'max' | 'divide'
    OPM_DEFICIT_RASTER = None               # None → auto {OUTPUT_DIR}/deficit_serves_{date}.tif
    SERVES_SATELLITE: str = "landsat"       # 'landsat' | 'sentinel2' | 'modis'
    SERVES_SEARCH_WINDOW: int = 30          # days backward from EVENT_START_UTC
    OPM_SOILGRIDS_DEPTH: str = "b30"        # 'b0' 'b10' 'b30' 'b60' 'b100' 'b200'
    # Legacy / backward-compat (older model builds read this if present).
    SERVES_TARGET_DATE = None

    # ═════════════════════════════════════════════════════════════════════════
    # 7.  MANNING'S ROUGHNESS  (kinematic-wave routing)
    # ═════════════════════════════════════════════════════════════════════════
    MANNINGS_N_SOURCE: str = "scalar"       # 'scalar'|'lulc'|'lcz'|'raster'
    MANNINGS_N: float = 0.09                # uniform fallback / nodata default
    MANNINGS_N_LULC_PATH: str = "gee"       # 'gee' → download ESA WorldCover
    MANNINGS_N_RASTER_PATH = None
    # Channel roughness override for cells above CHANNEL_FACCUM_THRESHOLD.
    # float → uniform channel n | dict{order:n} → per Strahler order | None → off.
    MANNINGS_N_CHANNEL = 0.035
    CHANNEL_FACCUM_THRESHOLD = None         # None → auto (top 1% of cells)

    # ═════════════════════════════════════════════════════════════════════════
    # 8.  GRID & NUMERICAL LIMITS
    # ═════════════════════════════════════════════════════════════════════════
    CELL_SIZE = None                        # None → auto-detect from DEM

    # ── Routing scheme ───────────────────────────────────────────────────────
    ROUTING_SCHEME: str = "kinematic"       # 'kinematic' | 'diffusive'
    DIFFUSION_THETA: float = 1.0            # diffusion weight θ∈[0,1]

    # ── Channel (river) cross-section routing ────────────────────────────────
    CHANNEL_ROUTING: bool = False
    CHANNEL_WIDTH_BY_ORDER = {1: 3.0, 2: 5.0, 3: 8.0, 4: 12.0,
                              5: 18.0, 6: 28.0, 7: 45.0, 8: 70.0}   # m

    # ── Upstream inflow boundary condition(s) ────────────────────────────────
    # Inject an external discharge hydrograph Q(t) [m³/s] at one or more cells so
    # upstream flow entering the domain is routed.  Set RAIN_INTENSITY_MM_HR=0
    # (PRECIP_METHOD='uniform', RUNOFF_SOURCE='none') for a pure routing run driven
    # only by these boundary conditions.
    # None → disabled.  Otherwise a list of dicts, one per inflow point:
    #   {"name": "us1",              # optional label (auto-named if absent)
    #    "lat": 27.7, "lon": 85.3,   # location: lat/lon  OR  "row"/"col"  OR "easting"/"northing"
    #    "csv": "inflow_us1.csv",    # time series: columns (time_hr | time_s) + Q_m3s
    #    "snap_to_channel": True,     # snap to highest-flow-accumulation cell within radius
    #    "snap_radius_cells": 3}
    ROUTING_INFLOW_BC = None

    # ── Virtual gauges (record depth/discharge time series at points) ─────────
    # None → disabled.  Otherwise a list of dicts, one per gauge, sampled every
    # OUTPUT_INTERVAL and written to {OUTPUT_DIR}/gauges.csv.  Same location keys
    # and channel-snapping as ROUTING_INFLOW_BC:
    #   {"name": "Betrawati", "lat": 27.974, "lon": 85.185,
    #    "snap_to_channel": True, "snap_radius_cells": 5}
    ROUTING_GAUGES = None

    # ── Time stepping ────────────────────────────────────────────────────────
    TIME_STEP_SECONDS: float = 2.0
    OUTPUT_INTERVAL_SECONDS: int = 600

    # ── Adaptive CFL timestep ────────────────────────────────────────────────
    ADAPTIVE_TIMESTEP: bool = False
    CFL_TARGET: float = 0.85
    CFL_DT_MAX = 5.0                        # None → OUTPUT_INTERVAL_SECONDS
    CFL_DT_MIN: float = 0.01
    CFL_DT_GROW: float = 1.5

    # ── Numerical floors ─────────────────────────────────────────────────────
    MIN_SLOPE: float = 1e-4
    MIN_DEPTH_M: float = 1e-6
    MAX_DEPTH_M: float = 10.0               # display use only

    # ═════════════════════════════════════════════════════════════════════════
    # 9.  OUTPUTS
    # ═════════════════════════════════════════════════════════════════════════
    HYDROGRAPH_CSV: str = "output/hydrograph.csv"
    MASS_BALANCE_REPORT: bool = True
    MASS_BALANCE_CSV: str = "output/mass_balance.csv"

    # ── Spatial field output (per-cell depth / velocity / discharge over time) ─
    # When enabled, the router records these fields at each OUTPUT_INTERVAL and
    # writes a compact {OUTPUT_DIR}/fields/fields.npz (float32) + fields_meta.json
    # so plots / animations / at-any-cell hydrographs can be made afterwards.
    SAVE_FIELDS: bool = False
    FIELD_VARS = ["depth", "velocity", "discharge"]  # subset of depth|velocity|discharge|volume
    FIELD_STRIDE: int = 1            # record every Nth OUTPUT_INTERVAL (memory control)
    FIELD_OUTPUT_DIR = None          # None → {OUTPUT_DIR}/fields/  (resolved at runtime)

    # ═════════════════════════════════════════════════════════════════════════
    # 10. COMPUTE BACKEND
    # ═════════════════════════════════════════════════════════════════════════
    BACKEND: str = "cpu"                    # 'cpu' | 'gpu'
    GPU_PRECISION: str = "float64"          # 'float32' | 'float64'

    # ─────────────────────────────────────────────────────────────────────────

    def __setattr__(self, key, value):
        """Normalise fixed-choice options (string or integer code) on assign.

        Any attribute listed in ``_ENUM_CHOICES`` / ``_ENUM_LIST`` is validated
        and stored as its canonical string here, so integer codes work through
        every entry path (keyword args, attribute assignment, from_dict,
        from_file).  All other attributes pass through unchanged.
        """
        if key in _ENUM_CHOICES:
            value = _normalize_enum(key, value, _ENUM_CHOICES[key])
        elif key in _ENUM_LIST:
            choices = _ENUM_LIST[key]
            if isinstance(value, (str, int)):
                value = [value]           # tolerate a single element
            value = [_normalize_enum(key, v, choices) for v in value]
        object.__setattr__(self, key, value)

    def __init__(self, **kwargs):
        """
        Create a Config, optionally overriding defaults via keyword args.

        Example
        -------
            cfg = Config(DEM_PATH="/data/dem.tif", BACKEND="gpu")   # or BACKEND=1
        """
        # Give each instance its own copy of the mutable class-level defaults
        # so in-place edits (cfg.RUNOFF_MECHANISMS.append(...)) can never leak
        # into other configs.
        self.RUNOFF_MECHANISMS = list(self.RUNOFF_MECHANISMS)
        self.CHANNEL_WIDTH_BY_ORDER = dict(self.CHANNEL_WIDTH_BY_ORDER)
        self.FIELD_VARS = list(self.FIELD_VARS)
        for key, value in kwargs.items():
            if not hasattr(self, key):
                raise AttributeError(
                    f"Config has no attribute '{key}'.  "
                    f"Check hydroflow/config.py for valid parameter names."
                )
            setattr(self, key, value)

    def update_output_paths(self):
        """
        Sync every OUTPUT_DIR-derived path to the current OUTPUT_DIR.
        Call this after setting OUTPUT_DIR.
        """
        d = self.OUTPUT_DIR
        self.ROUTING_DEM_PATH = os.path.join(d, "clipped_dem.tif")
        self.ROUTING_FLOW_DIR_PATH = os.path.join(d, "flow_direction.tif")
        self.ROUTING_FLOW_ACCUM_PATH = os.path.join(d, "clipped_flow_accumulation.tif")
        self.ROUTING_WATERSHED_MASK_PATH = os.path.join(d, "watershed.tif")
        self.OPM_WATERSHED_GEOJSON = os.path.join(d, "watershed.geojson")
        self.PRECIP_IMERG_DIR = os.path.join(d, "imerg/")
        self.HYDROGRAPH_CSV = os.path.join(d, "hydrograph.csv")
        self.MASS_BALANCE_CSV = os.path.join(d, "mass_balance.csv")

    # ── Construction from dicts / files ──────────────────────────────────────

    @classmethod
    def from_dict(cls, data):
        """
        Build a Config from a plain dict of {PARAMETER: value}.

        If OUTPUT_DIR is given, all derived paths are re-based onto it first;
        explicitly provided derived paths then take precedence.
        """
        data = dict(data or {})
        unknown = [k for k in data if not hasattr(cls, k)]
        if unknown:
            raise AttributeError(
                "Unknown config parameter(s): " + ", ".join(sorted(unknown))
                + ".  Check hydroflow/config.py for valid names."
            )
        cfg = cls()
        if "OUTPUT_DIR" in data:
            cfg.OUTPUT_DIR = data.pop("OUTPUT_DIR")
            cfg.update_output_paths()
        for key, value in data.items():
            setattr(cfg, key, value)
        return cfg

    @classmethod
    def from_file(cls, path):
        """
        Load a config file.  Supported formats:

        - ``.yaml`` / ``.yml``  (requires PyYAML)
        - ``.json``
        - ``.py``   — a legacy flat settings module (UPPERCASE names only),
                      e.g. the repository's historical config.py.
        """
        path = str(path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Config file not found: {path}")
        ext = os.path.splitext(path)[1].lower()

        if ext in (".yaml", ".yml"):
            try:
                import yaml
            except ImportError as exc:
                raise ImportError(
                    "PyYAML is required for YAML config files "
                    "(pip install pyyaml), or use JSON instead."
                ) from exc
            with open(path) as f:
                data = yaml.safe_load(f) or {}
            return cls.from_dict(data)

        if ext == ".json":
            with open(path) as f:
                data = json.load(f)
            return cls.from_dict(data)

        if ext == ".py":
            import importlib.util
            spec = importlib.util.spec_from_file_location("_hydroflow_user_config", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            data = {k: v for k, v in vars(module).items()
                    if k.isupper() and hasattr(cls, k)}
            # .py configs define derived paths explicitly, so no re-basing.
            cfg = cls()
            for key, value in data.items():
                setattr(cfg, key, value)
            return cfg

        raise ValueError(f"Unsupported config format '{ext}' (use .yaml, .json or .py)")

    def save(self, path):
        """Write the current configuration to a YAML or JSON file."""
        path = str(path)
        ext = os.path.splitext(path)[1].lower()
        data = self.to_dict()
        # Tuples don't round-trip through YAML/JSON; store as lists.
        data = {k: (list(v) if isinstance(v, tuple) else v) for k, v in data.items()}
        if ext in (".yaml", ".yml"):
            import yaml
            with open(path, "w") as f:
                yaml.safe_dump(data, f, sort_keys=False, default_flow_style=False)
        elif ext == ".json":
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
        else:
            raise ValueError(f"Unsupported config format '{ext}' (use .yaml or .json)")
        return path

    def validate(self):
        """
        Basic sanity checks before starting a run.

        Raises
        ------
        ValueError  with a descriptive message listing all failed checks.
        """
        errors = []

        if not self.DEM_PATH or not os.path.exists(self.DEM_PATH):
            errors.append(f"DEM_PATH not found: '{self.DEM_PATH}'")

        if self.TIME_STEP_SECONDS <= 0:
            errors.append(f"TIME_STEP_SECONDS must be > 0 (got {self.TIME_STEP_SECONDS})")

        if self.TOTAL_SIMULATION_TIME_HOURS <= 0:
            errors.append("TOTAL_SIMULATION_TIME_HOURS must be > 0")

        if self.MANNINGS_N <= 0:
            errors.append(f"MANNINGS_N must be > 0 (got {self.MANNINGS_N})")

        # Gauge CSVs are only required for the file-based interpolation methods.
        if self.PRECIP_METHOD in ("thiessen", "idw"):
            if not self.PRECIP_GAUGE_FILE or not os.path.exists(self.PRECIP_GAUGE_FILE):
                errors.append(f"PRECIP_GAUGE_FILE not found: '{self.PRECIP_GAUGE_FILE}'")
            if not self.PRECIP_TIMESERIES_FILE or not os.path.exists(self.PRECIP_TIMESERIES_FILE):
                errors.append(f"PRECIP_TIMESERIES_FILE not found: '{self.PRECIP_TIMESERIES_FILE}'")

        # IMERG / GEE methods need a project and an event date to derive the window.
        if self.PRECIP_METHOD in ("imerg_thiessen", "imerg_idw"):
            if not (self.GEE_PROJECT or os.environ.get("GEE_PROJECT")):
                errors.append("IMERG precipitation needs GEE_PROJECT (or the GEE_PROJECT env var).")
            if not (self.EVENT_START_UTC or self.IMERG_START_LOCAL):
                errors.append("IMERG precipitation needs EVENT_START_UTC (or IMERG_START_LOCAL).")

        if self.RUNOFF_SOURCE == "vsa_opm":
            if self.OPM_Q_MAX <= 0.001:
                errors.append(f"OPM_Q_MAX must be > 0.001 m³/s (got {self.OPM_Q_MAX})")
            if not (0 < self.OPM_PHI < 1):
                errors.append(f"OPM_PHI must be in (0, 1) (got {self.OPM_PHI})")

        # GEE-backed parameter sources need a project.
        _needs_gee = (
            self.OPM_SD_SOURCE == "gee"
            or self.OPM_GA_KSAT_SOURCE == "gee"
            or self.OPM_GA_SUCTION_SOURCE == "texture"
            or self.MANNINGS_N_SOURCE in ("lulc", "lcz")
            or self.IMPERVIOUS_SOURCE in ("lulc", "lcz")
        )
        if _needs_gee and self.RUNOFF_SOURCE == "vsa_opm":
            if not (self.GEE_PROJECT or os.environ.get("GEE_PROJECT")):
                errors.append(
                    "A GEE-backed option is selected (SERVES SD, gridded Ksat, "
                    "texture suction, or LULC/LCZ Manning's/impervious) but "
                    "GEE_PROJECT is not set."
                )

        # Note: fixed-choice options (BACKEND, DELINEATION_ENGINE, PRECIP_METHOD,
        # ROUTING_SCHEME, …) are validated at assignment via _ENUM_CHOICES, so no
        # membership check is needed here.

        # Upstream inflow boundary condition(s): a list of point specs, each
        # needing an existing hydrograph CSV and a resolvable location.
        if self.ROUTING_INFLOW_BC is not None:
            specs = self.ROUTING_INFLOW_BC
            if isinstance(specs, dict):
                specs = [specs]
            if not isinstance(specs, (list, tuple)):
                errors.append("ROUTING_INFLOW_BC must be a list of dicts (or a single dict).")
            else:
                for i, spec in enumerate(specs):
                    if not isinstance(spec, dict):
                        errors.append(f"ROUTING_INFLOW_BC[{i}] must be a dict.")
                        continue
                    csv = spec.get("csv")
                    if not csv or not os.path.exists(csv):
                        errors.append(f"ROUTING_INFLOW_BC[{i}] csv not found: '{csv}'")
                    has_loc = (("lat" in spec and "lon" in spec)
                               or ("row" in spec and "col" in spec)
                               or ("easting" in spec and "northing" in spec))
                    if not has_loc:
                        errors.append(
                            f"ROUTING_INFLOW_BC[{i}] needs a location: "
                            "lat/lon, row/col, or easting/northing."
                        )

        if self.SAVE_FIELDS:
            _allowed = {"depth", "velocity", "discharge", "volume"}
            _bad = [v for v in self.FIELD_VARS if v not in _allowed]
            if _bad:
                errors.append(
                    f"FIELD_VARS has unknown entries {_bad}; "
                    f"allowed: {sorted(_allowed)}."
                )
            if not self.FIELD_VARS:
                errors.append("SAVE_FIELDS is on but FIELD_VARS is empty.")
            if int(self.FIELD_STRIDE) < 1:
                errors.append(f"FIELD_STRIDE must be >= 1 (got {self.FIELD_STRIDE}).")

        if errors:
            raise ValueError(
                "Config validation failed:\n" + "\n".join(f"  • {e}" for e in errors)
            )

    def to_dict(self):
        """Return all public attributes as a plain dict (for logging/saving)."""
        keys = [k for k in vars(Config)
                if not k.startswith("_") and not callable(getattr(Config, k))]
        return {k: getattr(self, k) for k in keys}

    @classmethod
    def describe_options(cls):
        """Return a human-readable table of every fixed-choice option's codes.

        Lists each option that accepts an integer code alongside its canonical
        strings, e.g. ``PRECIP_METHOD : 0=uniform  1=thiessen  …``.  Handy for
        the CLI (`hydroflow list-options`) and docs.
        """
        lines = ["Fixed-choice options (pass the string OR the integer code):", ""]
        width = max(len(k) for k in list(_ENUM_CHOICES) + list(_ENUM_LIST))
        for key, choices in _ENUM_CHOICES.items():
            codes = "  ".join(f"{i}={c}" for i, c in enumerate(choices))
            lines.append(f"  {key:<{width}} : {codes}")
        lines.append("")
        for key, choices in _ENUM_LIST.items():
            codes = "  ".join(f"{i}={c}" for i, c in enumerate(choices))
            lines.append(f"  {key:<{width}} : (list) {codes}")
        return "\n".join(lines)

    def __repr__(self):
        items = ", ".join(f"{k}={v!r}" for k, v in self.to_dict().items())
        return f"Config({items})"


# ── Alias ────────────────────────────────────────────────────────────────────
# `OpmConfig` is a convenience alias of `Config` (the VSA-OPM heritage name),
# used by the QGIS bridge and some tools/tests.  Prefer `Config` in new code.
OpmConfig = Config
