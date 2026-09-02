"""
imerg.py
========
Flood-runner pipeline for IMERG rainfall (PRECIP_METHOD = 'imerg_thiessen' /
'imerg_idw').  Instead of the gauge rainfall, it downloads **NASA IMERG** from
GEE for each event's own date window.  The .gag file is read ONLY for:
  • the event date window  → IMERG_START_LOCAL / IMERG_END_LOCAL
  • the simulation length   → TOTAL_SIMULATION_TIME_HOURS
The observed discharge CSV is still used for the comparison chart.

Entry point: run().  Normally invoked by ../runner.py, not directly.

Output  (ALL events in ONE folder — runner_config.OUTPUT_DIR — tagged by event,
exactly like the gauge pipeline):
  <OUTPUT_DIR>/hydrograph_<EVENT>.csv
  <OUTPUT_DIR>/comparison_<EVENT>.png
  <OUTPUT_DIR>/summary_all_floods.csv
  <OUTPUT_DIR>/timing.csv

The shared watershed rasters (process_dem, once) and per-event intermediates
(downloaded IMERG in imerg/, date-specific deficit_serves.tif) also live in the
folder; the intermediates are re-downloaded/overwritten per event (each event
has its own date window), the tagged results are kept.
"""

import json
import re
import sys
import time
from datetime import datetime, timedelta

import pandas as pd

from . import runner_config as rcfg
from .runner_config import REPO_ROOT, GSSHA_DIR, BACKEND, PRECIP_METHOD
from .common import (
    hms, apply_output_dir, run_process_dem,
    parse_gag, gag_event_start, gag_sim_hours,
    load_discharge_csv, compute_metrics, make_plot, print_summary_table,
)


# ══════════════════════════════════════════════════════════════════════════════
# Per-event window + router
# ══════════════════════════════════════════════════════════════════════════════

def apply_event(config, start_local, end_local, sim_hours, tag):
    """
    Set ONLY the per-event knobs (IMERG window, sim length, SERVES date).
    Each event gets its own imerg_{tag}/ sub-folder so downloads are preserved
    and the natural cache check works without FORCE_DOWNLOAD.
    """
    config.PRECIP_METHOD               = PRECIP_METHOD
    config.IMERG_START_LOCAL           = start_local
    config.IMERG_END_LOCAL             = end_local
    config.PRECIP_IMERG_FORCE_DOWNLOAD = False          # cache per event folder
    config.PRECIP_IMERG_DIR            = config.OUTPUT_DIR + f"imerg_{tag}/"
    config.TOTAL_SIMULATION_TIME_HOURS = sim_hours
    config.BACKEND                     = BACKEND

    # Convert .gag local time (NPT = UTC+5:45) → UTC for EVENT_START_UTC.
    # start_local is "YYYY-MM-DD HH:MM" in Nepal local time from the .gag file.
    try:
        offset_h = float(getattr(config, 'IMERG_UTC_OFFSET_HOURS', 0.0))
        _local_dt = datetime.strptime(start_local.strip(), "%Y-%m-%d %H:%M")
        config.EVENT_START_UTC = (_local_dt - timedelta(hours=offset_h)).strftime("%Y-%m-%d %H:%M")
    except (ValueError, AttributeError):
        config.EVENT_START_UTC = start_local   # fallback if format unexpected


def run_router_imerg(config):
    from hydroflow.core.routing import router as kwr
    t0 = time.perf_counter()
    grid_data = kwr.initialise_grid(config)
    t_init = time.perf_counter() - t0
    t0 = time.perf_counter()
    hydrograph = kwr.run_time_loop(grid_data, config)
    t_loop = time.perf_counter() - t0
    return hydrograph, t_init, t_loop


# ══════════════════════════════════════════════════════════════════════════════
# Pipeline
# ══════════════════════════════════════════════════════════════════════════════

def run():
    from hydroflow.utils import gpu_utils
    if not gpu_utils.cupy_available():
        print("[ERROR] CuPy not available — this pipeline requires a GPU.")
        sys.exit(1)
    import config

    # ONE folder for everything (overrides config.OUTPUT_DIR, cascades all paths).
    out_dir    = apply_output_dir(config, rcfg.OUTPUT_DIR)
    output_dir = REPO_ROOT / out_dir.rstrip("/")

    gag_files = sorted(GSSHA_DIR.glob("*.gag"))
    if not gag_files:
        print(f"No .gag files found in {GSSHA_DIR}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output folder: {out_dir}  |  method={PRECIP_METHOD}")

    # ── Stage 0: process_dem (once, shared watershed) ────────────────────────
    print("\n" + "=" * 68)
    print("  STAGE 0 — process_dem  (shared watershed, one-time)")
    print("=" * 68)
    t_dem = run_process_dem(config, out_dir)
    if t_dem:
        print(f"  process_dem finished in {hms(t_dem)}")

    summary_rows, metrics_rows, catalogue = [], [], []

    for gag_path in gag_files:
        tag = re.search(r'(\d{6}_\d{6})', gag_path.stem)
        tag = tag.group(1) if tag else gag_path.stem

        data        = parse_gag(gag_path)
        recs        = data["records"]
        interval_s  = (recs[1][0] - recs[0][0]).total_seconds() if len(recs) > 1 else 1800.0
        event_start = gag_event_start(data)        # naive datetime in local NPT
        event_end   = recs[-1][0] + timedelta(seconds=interval_s)
        sim_hours   = gag_sim_hours(data)

        # NPT → UTC for metadata and EVENT_START_UTC
        offset_h  = float(getattr(config, 'IMERG_UTC_OFFSET_HOURS', 0.0))
        start_utc = event_start - timedelta(hours=offset_h)
        end_utc   = event_end   - timedelta(hours=offset_h)

        print("\n" + "=" * 68)
        print(f"  EVENT {tag}")
        print("=" * 68)
        print(f"  Local window : {event_start:%Y-%m-%d %H:%M} → "
              f"{event_end:%Y-%m-%d %H:%M}  ({sim_hours:.1f} h)  [NPT]")
        print(f"  UTC   window : {start_utc:%Y-%m-%d %H:%M} → "
              f"{end_utc:%Y-%m-%d %H:%M}")

        # Write event metadata alongside the output hydrograph.
        meta = {
            'event_tag':        tag,
            'gag_file':         gag_path.name,
            'start_local':      event_start.strftime('%Y-%m-%d %H:%M'),
            'end_local':        event_end.strftime('%Y-%m-%d %H:%M'),
            'start_utc':        start_utc.strftime('%Y-%m-%d %H:%M'),
            'end_utc':          end_utc.strftime('%Y-%m-%d %H:%M'),
            'sim_hours':        sim_hours,
            'utc_offset_hours': offset_h,
            'precip_source':    'imerg',
        }
        with open(output_dir / f'event_meta_{tag}.json', 'w') as _f:
            json.dump(meta, _f, indent=2)
        catalogue.append(meta)

        # ── Per-event window + GPU routing (IMERG downloaded inside) ──────────
        apply_event(
            config,
            event_start.strftime("%Y-%m-%d %H:%M"),
            event_end.strftime("%Y-%m-%d %H:%M"),
            sim_hours,
            tag=tag,
        )
        hydrograph, t_init, t_loop = run_router_imerg(config)
        print(f"  init={hms(t_init)}  loop={hms(t_loop)}  "
              f"total={hms(t_init + t_loop)}")

        # Save hydrograph (tagged) into the shared folder
        hyd_csv = output_dir / f"hydrograph_{tag}.csv"
        pd.DataFrame({
            "time_s":  [h[0] for h in hydrograph],
            "time_hr": [h[0] / 3600 for h in hydrograph],
            "Q_m3s":   [h[1] for h in hydrograph],
        }).to_csv(hyd_csv, index=False)
        print(f"  Hydrograph saved → {hyd_csv.relative_to(REPO_ROOT)}")

        summary_rows.append({"event": tag, "sim_h": sim_hours,
                             "t_init": t_init, "t_loop": t_loop})

        # ── Comparison chart vs observed discharge ───────────────────────────
        disc_path = GSSHA_DIR / f"discharge_{tag}.csv"
        ts_csv    = output_dir / f"imerg_{tag}" / "timeseries.csv"
        if disc_path.exists() and ts_csv.exists():
            otl     = load_discharge_csv(disc_path, event_start)
            metrics = compute_metrics(otl, hydrograph)
            out_png = output_dir / f"comparison_{tag}.png"
            make_plot(f"{tag} (IMERG)", event_start, otl, hydrograph,
                      ts_csv, metrics, out_png)
            print(f"  NSE={metrics['nse']:.3f}  PBIAS={metrics['pbias']:.1f}%  "
                  f"Obs Qp={metrics['obs_peak_Q']:.0f}  "
                  f"Mod Qp={metrics['mod_peak_Q']:.0f} m³/s")
            metrics_rows.append({
                "event":       tag,
                "start":       event_start.strftime("%Y-%m-%d"),
                "nse":         round(metrics["nse"], 3),
                "pbias_pct":   round(metrics["pbias"], 1),
                "obs_peak_Q":  round(metrics["obs_peak_Q"], 1),
                "obs_peak_hr": round(metrics["obs_peak_s"] / 3600, 2),
                "mod_peak_Q":  round(metrics["mod_peak_Q"], 1),
                "mod_peak_hr": round(metrics["mod_peak_s"] / 3600, 2),
            })
        else:
            print("  [WARN] observed discharge or IMERG timeseries missing — "
                  "skipping chart.")

    # ── Cross-event summaries (same folder, same names as gauge) ─────────────
    print_summary_table(t_dem, summary_rows)

    timing_csv = output_dir / "timing.csv"
    pd.DataFrame([{
        "event":     r["event"],
        "sim_hours": r["sim_h"],
        "init_s":    round(r["t_init"], 2),
        "loop_s":    round(r["t_loop"], 2),
        "total_s":   round(r["t_init"] + r["t_loop"], 2),
    } for r in summary_rows]).to_csv(timing_csv, index=False)
    print(f"\n  Timing  → {timing_csv.relative_to(REPO_ROOT)}")

    if metrics_rows:
        metrics_csv = output_dir / "summary_all_floods.csv"
        pd.DataFrame(metrics_rows).to_csv(metrics_csv, index=False)
        print(f"  Metrics → {metrics_csv.relative_to(REPO_ROOT)}")

    if catalogue:
        cat_csv = output_dir / "event_catalogue.csv"
        pd.DataFrame(catalogue)[
            ['event_tag', 'gag_file', 'start_local', 'end_local',
             'start_utc', 'end_utc', 'sim_hours', 'utc_offset_hours', 'precip_source']
        ].to_csv(cat_csv, index=False)
        print(f"  Event catalogue → {cat_csv.relative_to(REPO_ROOT)}")

    print("\nAll done.")
