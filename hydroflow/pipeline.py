# -*- coding: utf-8 -*-
"""
pipeline.py
===========
Pure-Python orchestration of the VSA-OPM pipeline stages — no QGIS, no Qt.

Every interface (Python API, CLI, QGIS plugin worker) drives the model
through :func:`run_pipeline`.  Interface-specific concerns (threads, signals,
stdout capture) stay in the caller.

Stages (in order):
    'process_dem'  – DEM preprocessing (core.dem_processing.main)
    'routing'      – kinematic/diffusive-wave routing
                     (initialise_grid → run_time_loop → save_hydrograph)
    'vsa_opm'      – standalone OPM run (core.opm.run_opm)
"""

import os

DEFAULT_STAGES = ("process_dem", "routing")

#: (progress base, progress span) per stage, matching the historical plugin UI.
_STAGE_PROGRESS = {
    "process_dem": (0, 30),
    "routing": (30, 65),
    "vsa_opm": (30, 65),
}


class PipelineCancelled(Exception):
    """Raised internally when the caller's is_cancelled() returns True."""


def prepare_output_dir(cfg, log=print):
    """
    Resolve cfg.OUTPUT_DIR to an ABSOLUTE path and pre-create it, so relative
    defaults are never created under a read-only working directory (e.g. QGIS
    launched from C:\\Program Files).  Re-syncs the derived paths so
    ROUTING_*/hydrograph/etc. are absolute too.
    """
    try:
        cfg.OUTPUT_DIR = os.path.abspath(cfg.OUTPUT_DIR or "output")
        if hasattr(cfg, "update_output_paths"):
            cfg.update_output_paths()
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
        log(f"[INFO] Output directory: {cfg.OUTPUT_DIR}")
    except OSError as exc:
        raise RuntimeError(
            f"Cannot create the output directory '{cfg.OUTPUT_DIR}'. "
            "Pick a writable folder (e.g. under your user/Documents folder)."
        ) from exc


# ── Individual stages ──────────────────────────────────────────────────────────

def stage_process_dem(cfg):
    """Run DEM preprocessing.  Returns a dict of generated file paths."""
    from .core import dem_processing

    dem_processing.main(cfg)

    d = cfg.OUTPUT_DIR
    return {
        "watershed_tif": os.path.join(d, "watershed.tif"),
        "watershed_geojson": os.path.join(d, "watershed.geojson"),
        "clipped_dem": os.path.join(d, "clipped_dem.tif"),
        "flow_direction": os.path.join(d, "flow_direction.tif"),
        "flow_accumulation": os.path.join(d, "clipped_flow_accumulation.tif"),
    }


def stage_routing(cfg, log=print, progress=None, is_cancelled=None):
    """Run routing: initialise_grid → run_time_loop → save_hydrograph."""
    from .core.routing import router as kwr

    base, span = _STAGE_PROGRESS["routing"]
    _emit = progress or (lambda p: None)

    log("  Initialising grid …")
    _emit(base + span // 4)
    grid_data = kwr.initialise_grid(cfg)

    if is_cancelled and is_cancelled():
        raise PipelineCancelled()

    log("  Running time loop …")
    _emit(base + span // 2)
    hydrograph = kwr.run_time_loop(grid_data, cfg)

    log("  Saving hydrograph …")
    _emit(base + span - 5)
    df = kwr.save_hydrograph(hydrograph, cfg)

    return {
        "hydrograph_csv": cfg.HYDROGRAPH_CSV,
        "hydrograph_df": df,
    }


def stage_vsa_opm(cfg):
    """Run the standalone VSA-OPM model."""
    from .core import opm

    df = opm.run_opm(cfg)
    return {
        "vsa_opm_csv": os.path.join(cfg.OUTPUT_DIR, "vsa_opm_results.csv"),
        "vsa_opm_df": df,
    }


# ── Orchestrator ───────────────────────────────────────────────────────────────

def run_pipeline(cfg, stages=DEFAULT_STAGES, on_log=None, on_progress=None,
                 is_cancelled=None):
    """
    Run the requested pipeline *stages* with configuration *cfg*.

    Parameters
    ----------
    cfg : OpmConfig (or any object with the same attributes)
    stages : sequence of str
        Subset of ('process_dem', 'routing', 'vsa_opm'), run in the given order.
    on_log : callable(str), optional
        Receives human-readable log lines (default: print).
    on_progress : callable(int), optional
        Receives 0–100 progress percentages.
    is_cancelled : callable() -> bool, optional
        Polled between stages; returning True aborts gracefully.

    Returns
    -------
    dict with the accumulated stage results (file paths + DataFrames), or the
    partial results accumulated so far if the run was cancelled.
    """
    log = on_log or print
    emit = on_progress or (lambda p: None)
    stages = list(stages)

    unknown = [s for s in stages if s not in _STAGE_PROGRESS]
    if unknown:
        raise ValueError(f"Unknown pipeline stage(s): {unknown}")

    prepare_output_dir(cfg, log=log)

    result = {}
    n = len(stages)
    try:
        for idx, stage in enumerate(stages):
            if is_cancelled and is_cancelled():
                log("[INFO] Run cancelled by user.")
                return result

            log(f"\n{'=' * 60}")
            log(f"  STAGE {idx + 1}/{n}: {stage.upper()}")
            log(f"{'=' * 60}")

            base, span = _STAGE_PROGRESS[stage]
            if stage == "process_dem":
                emit(base + span // 3)
                result.update(stage_process_dem(cfg))
                emit(base + span)
            elif stage == "routing":
                result.update(stage_routing(cfg, log=log, progress=emit,
                                            is_cancelled=is_cancelled))
                emit(base + span)
            elif stage == "vsa_opm":
                emit(base + span // 2)
                result.update(stage_vsa_opm(cfg))
                emit(base + span)
    except PipelineCancelled:
        log("[INFO] Run cancelled by user.")
        return result

    emit(100)
    log("\n[INFO] All stages complete.")
    return result
