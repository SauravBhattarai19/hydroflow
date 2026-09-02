# -*- coding: utf-8 -*-
"""
reporting.py — run outputs: hydrograph CSV, VSA partition time series and
the per-run mass-balance ledger row.
"""

import os

import numpy as np
import pandas as pd




def write_partition_series(cfg, series):
    """Write the cumulative Dunne/Horton/Impervious time series to
    {OUTPUT_DIR}/partition_{RUN_TAG}.csv (one file per event)."""
    if not series:
        return
    import csv
    tag = getattr(cfg, 'RUN_TAG', None)
    out_dir = getattr(cfg, 'OUTPUT_DIR', 'output/')
    fname = f"partition_{tag}.csv" if tag else "partition.csv"
    path = os.path.join(out_dir, fname)
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['time_hr', 'dunne_m3', 'horton_m3', 'imperv_m3'])
        for t_hr, d, h, i in series:
            w.writerow([f"{t_hr:.4f}", f"{d:.3f}", f"{h:.3f}", f"{i:.3f}"])
    print(f"  Partition time series  → {path}")


def append_mass_balance_csv(cfg, scheme, theta, rain_m3, input_m3, outflow_m3,
                             storage_m3, error_m3, rel_error, runoff_ratio,
                             partition=None, bc_m3=0.0):
    """Append one row of the mass-balance budget to MASS_BALANCE_CSV (header on create).
    Appends so successive runs at different configs accumulate for side-by-side review.
    The row also carries the run tag, the key runoff knobs, and (when available) the
    Dunne/Horton/Impervious mechanism partition so a sweep is self-describing."""
    import csv
    from datetime import datetime

    path = getattr(cfg, 'MASS_BALANCE_CSV', None) or os.path.join(
        getattr(cfg, 'OUTPUT_DIR', 'output/'), 'mass_balance.csv')
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    p = partition or {}
    header = ['timestamp', 'run_tag', 'scheme', 'theta', 'runoff_source',
              'sd_source', 'sd_max', 'ksat_scale', 'infiltration', 'impervious',
              'rain_m3', 'input_m3', 'bc_inflow_m3', 'outflow_m3', 'storage_m3',
              'error_m3', 'rel_error', 'runoff_ratio',
              'dunne_m3', 'horton_m3', 'imperv_m3',
              'dunne_frac', 'horton_frac', 'imperv_frac']
    def _f(key, fmt="{}"):
        v = p.get(key)
        return "" if v is None else fmt.format(v)
    row = [datetime.now().isoformat(timespec='seconds'),
           getattr(cfg, 'RUN_TAG', ''),
           scheme, theta, getattr(cfg, 'RUNOFF_SOURCE', 'none'),
           getattr(cfg, 'OPM_SD_SOURCE', ''),
           getattr(cfg, 'OPM_SD_MAX_INITIAL', ''),
           getattr(cfg, 'OPM_GA_KSAT_SCALE', ''),
           getattr(cfg, 'OPM_INFILTRATION', ''),
           getattr(cfg, 'IMPERVIOUS_SOURCE', ''),
           f"{rain_m3:.3f}", f"{input_m3:.3f}", f"{bc_m3:.3f}", f"{outflow_m3:.3f}",
           f"{storage_m3:.3f}", f"{error_m3:.6e}", f"{rel_error:.6e}",
           f"{runoff_ratio:.4f}",
           _f('dunne_m3', "{:.3f}"), _f('horton_m3', "{:.3f}"), _f('imperv_m3', "{:.3f}"),
           _f('dunne_frac', "{:.4f}"), _f('horton_frac', "{:.4f}"), _f('imperv_frac', "{:.4f}")]
    write_header = not os.path.exists(path)
    with open(path, 'a', newline='') as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(header)
        w.writerow(row)
    print(f"  Mass-balance row appended → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Output
# ─────────────────────────────────────────────────────────────────────────────

def save_hydrograph(hydrograph, cfg):
    """
    Save the hydrograph as a CSV with columns:
        time_s   – simulation time in seconds
        time_hr  – simulation time in hours
        Q_m3s    – discharge at the outlet [m³/s]
    Also prints peak flow and time-to-peak.
    """
    times_s  = np.array([h[0] for h in hydrograph])
    Q_values = np.array([h[1] for h in hydrograph])

    df = pd.DataFrame({
        "time_s"  : times_s,
        "time_hr" : times_s / 3600.0,
        "Q_m3s"   : Q_values,
    })

    df.to_csv(cfg.HYDROGRAPH_CSV, index=False)
    print(f"\n  Hydrograph saved → {cfg.HYDROGRAPH_CSV}")

    peak_Q  = Q_values.max()
    peak_t  = times_s[Q_values.argmax()] / 3600.0
    print(f"  Peak discharge : {peak_Q:.4f} m³/s  at t={peak_t:.2f} h")
    return df
