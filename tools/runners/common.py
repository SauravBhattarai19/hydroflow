"""
common.py
=========
Shared helpers for the flood-runner pipelines (gauge.py, imerg.py) so they
import from one place instead of one reaching into the other.

Holds: .gag parsing / OPM-CSV writing, observed-discharge loading, performance
metrics, the comparison plot, and the timing-table printer.  Nothing here knows
about the output folder — each pipeline decides where its artifacts land.
"""

import re
import csv
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Paths/constants live in runner_config.py (one place to edit).
from .runner_config import REPO_ROOT, GSSHA_DIR, OPM_DIR


# ══════════════════════════════════════════════════════════════════════════════
# Output-folder cascade
# ══════════════════════════════════════════════════════════════════════════════

def apply_output_dir(config, out_dir):
    """
    Point the in-memory model config at *out_dir* and re-derive EVERY path that
    config.py builds from OUTPUT_DIR, so overriding the folder at runtime
    cascades just like editing OUTPUT_DIR in config.py would.

    Mirrors the derivations in config.py §2/§3/§6/§9.  Returns the normalised
    out_dir string (always trailing-slash).
    """
    out_dir = str(out_dir)
    if not out_dir.endswith("/"):
        out_dir += "/"
    config.OUTPUT_DIR                  = out_dir
    config.ROUTING_DEM_PATH            = out_dir + "clipped_dem.tif"
    config.ROUTING_FLOW_DIR_PATH       = out_dir + "flow_direction.tif"
    config.ROUTING_FLOW_ACCUM_PATH     = out_dir + "clipped_flow_accumulation.tif"
    config.ROUTING_WATERSHED_MASK_PATH = out_dir + "watershed.tif"
    config.OPM_WATERSHED_GEOJSON       = out_dir + "watershed.geojson"
    config.OPM_DEFICIT_RASTER          = None   # date-stamped per event in resolve_sd_params
    config.PRECIP_IMERG_DIR            = out_dir + "imerg/"
    config.HYDROGRAPH_CSV              = out_dir + "hydrograph.csv"
    config.MASS_BALANCE_CSV            = out_dir + "mass_balance.csv"
    return out_dir


def run_process_dem(config, out_dir, skip_if_exists=False) -> float:
    """Run process_dem once into *out_dir* (shared watershed for all events).

    When *skip_if_exists* is True and the four watershed rasters are already
    present in *out_dir*, the (CPU-bound) DEM pipeline is skipped — this lets a
    parameter sweep reuse one shared, pre-seeded watershed across every config.
    """
    import os
    import time
    apply_output_dir(config, out_dir)
    os.makedirs(out_dir, exist_ok=True)
    needed = [config.ROUTING_DEM_PATH, config.ROUTING_FLOW_DIR_PATH,
              config.ROUTING_FLOW_ACCUM_PATH, config.ROUTING_WATERSHED_MASK_PATH]
    if skip_if_exists and all(os.path.exists(p) for p in needed):
        print("  [process_dem] watershed rasters already present — skipping.")
        return 0.0
    from hydroflow.core import dem_processing as process_dem
    t0 = time.perf_counter()
    process_dem.main(config)
    return time.perf_counter() - t0


# ══════════════════════════════════════════════════════════════════════════════
# Formatting
# ══════════════════════════════════════════════════════════════════════════════

def hms(seconds: float) -> str:
    """Format seconds as H:MM:SS or MM:SS.s for readability."""
    seconds = max(0.0, float(seconds))
    if seconds >= 3600:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = seconds % 60
        return f"{h}:{m:02d}:{s:04.1f}"
    elif seconds >= 60:
        m = int(seconds // 60)
        s = seconds % 60
        return f"{m}:{s:04.1f}"
    else:
        return f"{seconds:.2f}s"


# ══════════════════════════════════════════════════════════════════════════════
# .gag parsing  /  GAG → OPM conversion
# ══════════════════════════════════════════════════════════════════════════════

def parse_gag(filepath: Path) -> dict:
    gauges, records = [], []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line.startswith("COORD"):
                m = re.match(r'COORD\s+([\d.]+)\s+([\d.]+)\s+"([^"]+)"', line)
                if not m:
                    raise ValueError(f"Bad COORD: {line!r}")
                e, n, name = float(m.group(1)), float(m.group(2)), m.group(3)
                gauges.append((name.split()[0], name, e, n))
            elif line.startswith("GAGES"):
                parts = line.split()
                yr, mo, dy, hr, mn = (int(p) for p in parts[1:6])
                records.append((datetime(yr, mo, dy, hr, mn),
                                [float(v) for v in parts[6:]]))
    return {"gauges": gauges, "records": records}


def write_opm_csvs(data: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    gauges, records = data["gauges"], data["records"]
    interval_s = (records[1][0] - records[0][0]).total_seconds() if len(records) > 1 else 1800.0
    with open(out_dir / "gauges.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["gauge_id", "name", "easting_m", "northing_m"])
        for gid, name, e, n in gauges:
            w.writerow([gid, name, f"{e:.6f}", f"{n:.6f}"])
    gauge_ids = [g[0] for g in gauges]
    with open(out_dir / "timeseries.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time_s"] + gauge_ids)
        for i, (_, vals) in enumerate(records):
            w.writerow([int(round(i * interval_s))] + [f"{v:.4f}" for v in vals])


def gag_event_start(data: dict) -> datetime:
    return data["records"][0][0]


def gag_sim_hours(data: dict) -> float:
    recs = data["records"]
    interval_s = (recs[1][0] - recs[0][0]).total_seconds() if len(recs) > 1 else 1800.0
    return len(recs) * interval_s / 3600.0


# ══════════════════════════════════════════════════════════════════════════════
# Observed hydrograph + metrics
# ══════════════════════════════════════════════════════════════════════════════

def load_discharge_csv(csv_path: Path, event_start: datetime) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=["dateTime"])
    df["time_min"] = (df["dateTime"] - event_start).dt.total_seconds() / 60.0
    df = df.rename(columns={"discharge_m3s": "Q_m3s"})[["time_min", "Q_m3s"]]
    return df[df["time_min"] >= 0].reset_index(drop=True)


def compute_metrics(otl: pd.DataFrame, hydrograph: list) -> dict:
    obs_s   = otl["time_min"].values * 60.0
    obs_Q   = otl["Q_m3s"].values
    mod_s   = np.array([h[0] for h in hydrograph], dtype=float)
    mod_Q   = np.array([h[1] for h in hydrograph], dtype=float)
    mod_interp = np.interp(obs_s, mod_s, mod_Q)
    nse   = 1 - np.sum((obs_Q - mod_interp)**2) / np.sum((obs_Q - obs_Q.mean())**2)
    pbias = 100 * (mod_interp - obs_Q).sum() / obs_Q.sum()
    obs_peak_idx = obs_Q.argmax()
    mod_peak_idx = mod_Q.argmax()
    return {
        "nse":         nse,
        "pbias":       pbias,
        "obs_peak_Q":  obs_Q[obs_peak_idx],
        "obs_peak_s":  obs_s[obs_peak_idx],
        "mod_peak_Q":  mod_Q[mod_peak_idx],
        "mod_peak_s":  mod_s[mod_peak_idx],
    }


# ══════════════════════════════════════════════════════════════════════════════
# Comparison plot
# ══════════════════════════════════════════════════════════════════════════════

def make_plot(event_name: str, event_start: datetime,
              otl: pd.DataFrame, hydrograph: list,
              ts_csv: Path, metrics: dict, out_png: Path) -> None:

    mod_s = np.array([h[0] for h in hydrograph], dtype=float)
    mod_Q = np.array([h[1] for h in hydrograph], dtype=float)

    obs_dt = [event_start + timedelta(minutes=float(m)) for m in otl["time_min"]]
    mod_dt = [event_start + timedelta(seconds=float(s)) for s in mod_s]

    obs_peak_dt = event_start + timedelta(seconds=metrics["obs_peak_s"])
    mod_peak_dt = event_start + timedelta(seconds=metrics["mod_peak_s"])

    fig, (ax, ax_rain) = plt.subplots(
        2, 1, figsize=(13, 7),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True
    )

    ax.plot(obs_dt, otl["Q_m3s"], color="black", lw=1.8,
            label="Observed (GSSHA)", zorder=4)
    ax.plot(mod_dt, mod_Q, color="#1f6aa5", lw=1.8, ls="--",
            label="VSA-OPM (GPU)", zorder=3)
    ax.fill_between(mod_dt, mod_Q, alpha=0.10, color="#1f6aa5")

    ax.annotate(
        f"Obs peak\n{metrics['obs_peak_Q']:.0f} m³/s\n{obs_peak_dt.strftime('%b %d %H:%M')}",
        xy=(obs_peak_dt, metrics["obs_peak_Q"]),
        xytext=(obs_peak_dt - timedelta(hours=18), metrics["obs_peak_Q"] * 0.82),
        arrowprops=dict(arrowstyle="-|>", color="black", lw=1.2),
        fontsize=8.5, color="black"
    )
    ax.annotate(
        f"Mod peak\n{metrics['mod_peak_Q']:.0f} m³/s\n{mod_peak_dt.strftime('%b %d %H:%M')}",
        xy=(mod_peak_dt, metrics["mod_peak_Q"]),
        xytext=(mod_peak_dt + timedelta(hours=6), metrics["mod_peak_Q"] * 0.78),
        arrowprops=dict(arrowstyle="-|>", color="#1f6aa5", lw=1.2),
        fontsize=8.5, color="#1f6aa5"
    )

    stats_txt = (f"NSE   = {metrics['nse']:.3f}\n"
                 f"PBIAS = {metrics['pbias']:.1f}%\n"
                 f"Obs $Q_p$ = {metrics['obs_peak_Q']:.0f} m³/s\n"
                 f"Mod $Q_p$ = {metrics['mod_peak_Q']:.0f} m³/s")
    ax.text(0.985, 0.97, stats_txt, transform=ax.transAxes,
            va="top", ha="right", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="gray", alpha=0.85))

    ax.set_ylabel("Discharge [m³/s]", fontsize=11)
    ax.set_title(f"{event_name}  |  VSA-OPM vs Observed  |  Bagmati Basin",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10, loc="upper left")
    ax.set_ylim(bottom=0)
    ax.grid(True, ls="--", alpha=0.35)

    ts = pd.read_csv(ts_csv)
    gauge_cols = [c for c in ts.columns if c != "time_s"]
    ts["mean_mmhr"] = ts[gauge_cols].mean(axis=1) * 2.0
    ts["datetime"]  = [event_start + timedelta(seconds=float(s)) for s in ts["time_s"]]
    ax_rain.bar(ts["datetime"], ts["mean_mmhr"],
                width=timedelta(minutes=28), color="#4da6ff",
                alpha=0.7, align="edge", label="Mean areal rainfall [mm/hr]")
    ax_rain.invert_yaxis()
    ax_rain.set_ylabel("Rain [mm/hr]", fontsize=9)
    ax_rain.legend(fontsize=8, loc="lower right")
    ax_rain.grid(True, ls="--", alpha=0.3)
    ax_rain.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax_rain.xaxis.set_major_locator(mdates.DayLocator())
    fig.autofmt_xdate()
    ax_rain.set_xlabel(f"Date ({event_start.year})", fontsize=11)

    t_min = min(obs_dt[0], mod_dt[0])
    t_max = max(obs_dt[-1], mod_dt[-1])
    ax.set_xlim(t_min, t_max)

    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close(fig)
    try:
        rel = out_png.relative_to(REPO_ROOT)
    except ValueError:
        rel = out_png
    print(f"  Plot saved → {rel}")


# ══════════════════════════════════════════════════════════════════════════════
# Timing table
# ══════════════════════════════════════════════════════════════════════════════

def print_summary_table(t_dem: float, rows: list) -> None:
    W = 70
    print("\n" + "═" * W)
    print("  BENCHMARK SUMMARY  (GPU)")
    print("═" * W)
    print(f"  {'Event':<18}  {'Sim':>5}  {'Init':>9}  {'Loop':>9}  {'Total':>9}")
    print("  " + "─" * (W - 2))

    total_init = total_loop = 0.0
    for r in rows:
        total_init += r["t_init"]
        total_loop += r["t_loop"]
        total = r["t_init"] + r["t_loop"]
        print(f"  {r['event']:<18}  {r['sim_h']:>4.0f}h  "
              f"  {hms(r['t_init']):>9}  {hms(r['t_loop']):>9}  {hms(total):>9}")

    total = total_init + total_loop
    print("  " + "─" * (W - 2))
    print(f"  {'ALL EVENTS':<18}  {'':>5}  "
          f"  {hms(total_init):>9}  {hms(total_loop):>9}  {hms(total):>9}")
    print()
    print(f"  process_dem (one-time)           :  {hms(t_dem)}")
    print(f"  Total wall time (dem + routing)  :  {hms(t_dem + total)}")
    print("═" * W)
