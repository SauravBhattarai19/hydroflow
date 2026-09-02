"""
gauge.py
========
Flood-runner pipeline for GAUGE rainfall (PRECIP_METHOD = 'thiessen' / 'idw' /
'uniform').  Rainfall comes from each .gag file.

Entry point: run().  Normally invoked by ../runner.py, not directly.

Stages
------
  [0] process_dem  – reproject / fill / flow-dir / accumulation / watershed
                     (CPU-only via pysheds; runs once into OUTPUT_DIR)
  [1] GAG → OPM    – convert .gag rainfall files to OPM CSVs   (CPU-only)
  [2] GPU routing  – initialise_grid + run_time_loop

Output  (ALL events in ONE folder — runner_config.OUTPUT_DIR — tagged by event)
  <OUTPUT_DIR>/hydrograph_<EVENT>.csv
  <OUTPUT_DIR>/comparison_<EVENT>.png
  <OUTPUT_DIR>/summary_all_floods.csv
  <OUTPUT_DIR>/timing.csv
"""

import json
import re
import sys
import time
from datetime import timedelta
from pathlib import Path

import pandas as pd

from . import runner_config as rcfg
from .runner_config import REPO_ROOT, GSSHA_DIR, OPM_DIR, BACKEND, PRECIP_METHOD
from .common import (
    hms, apply_output_dir, run_process_dem,
    parse_gag, write_opm_csvs, gag_event_start, gag_sim_hours,
    load_discharge_csv, compute_metrics, make_plot, print_summary_table,
)


# ══════════════════════════════════════════════════════════════════════════════
# Router (one event)
# ══════════════════════════════════════════════════════════════════════════════

def run_router(config, gauge_csv: Path, ts_csv: Path, sim_hours: float,
               event_start_local=None) -> tuple:
    """Run the kinematic-wave router for one event. Returns (hydrograph, t_init, t_loop)."""
    from hydroflow.core.routing import router as kwr

    config.PRECIP_METHOD               = PRECIP_METHOD
    config.PRECIP_GAUGE_FILE           = str(gauge_csv.relative_to(REPO_ROOT))
    config.PRECIP_TIMESERIES_FILE      = str(ts_csv.relative_to(REPO_ROOT))
    config.TOTAL_SIMULATION_TIME_HOURS = sim_hours
    config.BACKEND                     = BACKEND

    # Convert .gag local time (NPT = UTC+5:45) → UTC for EVENT_START_UTC.
    # SERVES antecedent date and IMERG window both derive from this.
    if event_start_local is not None:
        offset_h = float(getattr(config, 'IMERG_UTC_OFFSET_HOURS', 0.0))
        event_utc = event_start_local - timedelta(hours=offset_h)
        config.EVENT_START_UTC = event_utc.strftime('%Y-%m-%d %H:%M')

    t0 = time.perf_counter()
    grid_data = kwr.initialise_grid(config)
    t_init    = time.perf_counter() - t0

    t0 = time.perf_counter()
    hydrograph = kwr.run_time_loop(grid_data, config)
    t_loop     = time.perf_counter() - t0

    return hydrograph, t_init, t_loop


# ══════════════════════════════════════════════════════════════════════════════
# Pipeline
# ══════════════════════════════════════════════════════════════════════════════

def run(output_dir=None, overrides=None, skip_process_dem=False):
    """Run the gauge pipeline over every .gag event.

    Parameters (all optional — defaults preserve standalone runner behaviour)
    ------------------------------------------------------------------------
    output_dir       : str — results folder (defaults to runner_config.OUTPUT_DIR).
    overrides        : dict — {config_attr: value} applied AFTER the OUTPUT_DIR
                       cascade, so a sweep can inject parameter values per run.
    skip_process_dem : bool — reuse pre-seeded watershed rasters instead of
                       re-deriving them (see common.run_process_dem).
    """
    from hydroflow.utils import gpu_utils
    import config

    # ONE folder for everything (overrides config.OUTPUT_DIR, cascades all paths).
    out_dir    = apply_output_dir(config, output_dir or rcfg.OUTPUT_DIR)
    # Inject sweep parameter overrides AFTER the cascade so they take precedence.
    for k, v in (overrides or {}).items():
        setattr(config, k, v)
    output_dir = REPO_ROOT / out_dir.rstrip("/")

    gag_files = sorted(GSSHA_DIR.glob("*.gag"))
    if not gag_files:
        print(f"No .gag files found in {GSSHA_DIR}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    if not gpu_utils.cupy_available():
        print("[ERROR] CuPy not available — this pipeline requires a GPU.")
        sys.exit(1)

    import cupy as cp
    mem = cp.cuda.Device(0).mem_info
    print(f"GPU detected: {mem[1]/1e9:.1f} GB VRAM  ({mem[0]/1e9:.1f} GB free)")
    print(f"Output folder: {out_dir}  |  method={PRECIP_METHOD}")

    # ── Stage 0: process_dem (once, shared watershed) ────────────────────────
    print("\n" + "=" * 68)
    print("  STAGE 0 — process_dem  (CPU only, one-time)")
    print("=" * 68)
    t_dem = run_process_dem(config, out_dir, skip_if_exists=skip_process_dem)
    print(f"\n  process_dem finished in  {hms(t_dem)}")

    all_metrics  = []
    summary_rows = []
    catalogue    = []

    for gag_path in gag_files:
        date_tag   = re.search(r'(\d{6}_\d{6})', gag_path.stem)
        date_tag   = date_tag.group(1) if date_tag else gag_path.stem
        event_name = date_tag
        disc_path  = GSSHA_DIR / f"discharge_{date_tag}.csv"

        print("\n" + "=" * 68)
        print(f"  EVENT: {event_name}")
        print("=" * 68)

        # ── Stage 1: GAG → OPM ───────────────────────────────────────────────
        print(f"\n[1] GAG → OPM  ({gag_path.name})")
        t_gag_0     = time.perf_counter()
        gag_data    = parse_gag(gag_path)
        opm_dir     = OPM_DIR / event_name
        write_opm_csvs(gag_data, opm_dir)
        t_gag       = time.perf_counter() - t_gag_0
        event_start = gag_event_start(gag_data)   # naive datetime in local NPT
        sim_hours   = gag_sim_hours(gag_data)
        event_end   = event_start + timedelta(hours=sim_hours)
        print(f"    Start: {event_start}  |  Duration: {sim_hours:.0f} h  "
              f"|  Converted in {hms(t_gag)}")

        # Write event metadata (local + UTC) alongside the OPM CSVs.
        # .gag timestamps are in Nepal local time (NPT = UTC+5:45).
        offset_h  = float(getattr(config, 'IMERG_UTC_OFFSET_HOURS', 0.0))
        start_utc = event_start - timedelta(hours=offset_h)
        end_utc   = event_end   - timedelta(hours=offset_h)
        meta = {
            'event_tag':        event_name,
            'gag_file':         gag_path.name,
            'start_local':      event_start.strftime('%Y-%m-%d %H:%M'),
            'end_local':        event_end.strftime('%Y-%m-%d %H:%M'),
            'start_utc':        start_utc.strftime('%Y-%m-%d %H:%M'),
            'end_utc':          end_utc.strftime('%Y-%m-%d %H:%M'),
            'sim_hours':        sim_hours,
            'utc_offset_hours': offset_h,
            'precip_source':    'gauge',
        }
        with open(opm_dir / 'event_meta.json', 'w') as _f:
            json.dump(meta, _f, indent=2)
        catalogue.append(meta)
        print(f"    UTC start: {start_utc.strftime('%Y-%m-%d %H:%M')} "
              f" (local - {offset_h}h offset)")

        gauge_csv = opm_dir / "gauges.csv"
        ts_csv    = opm_dir / "timeseries.csv"

        # ── Stage 2: GPU routing ──────────────────────────────────────────────
        print(f"\n[2] GPU routing  (BACKEND='{BACKEND}')")
        config.RUN_TAG = event_name   # tags mass_balance / partition CSV rows
        hydrograph, t_init, t_loop = run_router(
            config, gauge_csv, ts_csv, sim_hours,
            event_start_local=event_start,   # full NPT datetime → converted to UTC inside
        )
        print(f"    init={hms(t_init)}  loop={hms(t_loop)}  "
              f"total={hms(t_init + t_loop)}")

        # Save hydrograph (tagged) into the shared folder
        hyd_csv = output_dir / f"hydrograph_{event_name}.csv"
        pd.DataFrame({
            "time_s":  [h[0] for h in hydrograph],
            "time_hr": [h[0] / 3600 for h in hydrograph],
            "Q_m3s":   [h[1] for h in hydrograph],
        }).to_csv(hyd_csv, index=False)
        print(f"    Hydrograph saved → {hyd_csv.relative_to(REPO_ROOT)}")

        summary_rows.append({
            "event":  event_name,
            "sim_h":  sim_hours,
            "t_init": t_init,
            "t_loop": t_loop,
        })

        # ── Stage 3: Observed data + metrics ─────────────────────────────────
        if not disc_path.exists():
            print(f"\n  [WARN] {disc_path.name} not found — skipping plot.")
            continue

        print(f"\n[3] Metrics & plot")
        otl     = load_discharge_csv(disc_path, event_start)
        metrics = compute_metrics(otl, hydrograph)
        print(f"    NSE={metrics['nse']:.3f}  PBIAS={metrics['pbias']:.1f}%  "
              f"Obs Qp={metrics['obs_peak_Q']:.1f} m³/s  "
              f"Mod Qp={metrics['mod_peak_Q']:.1f} m³/s")

        out_png = output_dir / f"comparison_{event_name}.png"
        make_plot(event_name, event_start, otl, hydrograph, ts_csv, metrics, out_png)

        all_metrics.append({
            "event":       event_name,
            "start":       event_start.strftime("%Y-%m-%d"),
            "nse":         round(metrics["nse"], 3),
            "pbias_pct":   round(metrics["pbias"], 1),
            "obs_peak_Q":  round(metrics["obs_peak_Q"], 1),
            "obs_peak_hr": round(metrics["obs_peak_s"] / 3600, 2),
            "mod_peak_Q":  round(metrics["mod_peak_Q"], 1),
            "mod_peak_hr": round(metrics["mod_peak_s"] / 3600, 2),
        })

    # ── Grand summary ─────────────────────────────────────────────────────────
    print_summary_table(t_dem, summary_rows)

    timing_csv = output_dir / "timing.csv"
    pd.DataFrame([{
        "event":       r["event"],
        "sim_hours":   r["sim_h"],
        "init_s":      round(r["t_init"], 2),
        "loop_s":      round(r["t_loop"], 2),
        "total_s":     round(r["t_init"] + r["t_loop"], 2),
    } for r in summary_rows]).to_csv(timing_csv, index=False)
    print(f"\n  Timing CSV saved → {timing_csv.relative_to(REPO_ROOT)}")

    if all_metrics:
        metrics_csv = output_dir / "summary_all_floods.csv"
        pd.DataFrame(all_metrics).to_csv(metrics_csv, index=False)
        print(f"  Metrics CSV saved → {metrics_csv.relative_to(REPO_ROOT)}")

    # Event catalogue: all events with local + UTC start times in one place.
    if catalogue:
        cat_csv = output_dir / "event_catalogue.csv"
        pd.DataFrame(catalogue)[
            ['event_tag', 'gag_file', 'start_local', 'end_local',
             'start_utc', 'end_utc', 'sim_hours', 'utc_offset_hours', 'precip_source']
        ].to_csv(cat_csv, index=False)
        print(f"  Event catalogue → {cat_csv.relative_to(REPO_ROOT)}")

    print("\nAll done.")
