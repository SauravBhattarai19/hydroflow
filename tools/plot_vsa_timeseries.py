#!/usr/bin/env python
"""
plot_vsa_timeseries.py
=======================
Does the VSA (saturated) area grow monotonically to a peak and just stay
saturated, or does it actually recede between rain bursts (as Darcy's-law
drainage should make it do)?

Tracks the CURRENT vsa_mask fraction (not a cumulative "ever" mask) at every
timestep for the physically-consistent combined config (vsa+horton+imperv),
alongside the rainfall hyetograph, for event 202409_202409 (two rain bursts).

Writes: outputs collection/combinations_100m/figures/mechanism_spatial/vsa_area_timeseries.png
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
import os
os.chdir(REPO_ROOT)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
from tools.runners.common import apply_output_dir
from tools.runners.runner_config import OPM_DIR

EVENT   = "202409_202409"
LEAF    = REPO_ROOT / "outputs collection/combinations_100m/chan_off/diffusive/vsa+horton+imperv"
OUT_DIR = REPO_ROOT / "outputs collection/combinations_100m/figures/mechanism_spatial"
DT      = 300.0


def asnumpy(x):
    return x if isinstance(x, np.ndarray) else x.get()


def main():
    cat = pd.read_csv(LEAF / "event_catalogue.csv")
    row = cat[cat.event_tag == EVENT].iloc[0]

    apply_output_dir(config, str(LEAF) + "/")
    config.RUNOFF_MECHANISMS      = ['vsa', 'horton', 'impervious']
    config.OPM_INFILTRATION       = 'green_ampt'
    config.IMPERVIOUS_SOURCE      = 'lcz'
    config.OPM_SD_REDUCER         = 'max'
    config.CHANNEL_ROUTING        = False
    config.ROUTING_SCHEME         = 'diffusive'
    config.DIFFUSION_THETA        = 1.0
    config.BACKEND                = 'gpu'
    config.PRECIP_METHOD          = 'thiessen'
    config.PRECIP_GAUGE_FILE      = str((OPM_DIR / EVENT / "gauges.csv").relative_to(REPO_ROOT))
    config.PRECIP_TIMESERIES_FILE = str((OPM_DIR / EVENT / "timeseries.csv").relative_to(REPO_ROOT))
    config.TOTAL_SIMULATION_TIME_HOURS = float(row.sim_hours)
    config.EVENT_START_UTC        = row.start_utc
    config.RUN_TAG                = EVENT

    from vsa_opm.core.routing import router as kwr
    grid_data = kwr.initialise_grid(config)
    runoff_engine = grid_data["runoff_engine"]
    precip_engine = grid_data["precip_engine"]
    n_cells = grid_data["n_cells"]

    T = config.TOTAL_SIMULATION_TIME_HOURS * 3600.0
    t_hist, vsa_frac_hist, rain_hist = [], [], []
    t = 0.0
    while t < T:
        rain_1d = precip_engine.get_field_1d(t)
        _ = runoff_engine.get_effective_1d(t, rain_1d)
        vsa_frac_hist.append(float(asnumpy(runoff_engine._vsa_mask).sum()) / n_cells)
        rain_hist.append(float(asnumpy(rain_1d).mean()) * 1000.0 * 3600.0)  # mm/hr areal mean
        t_hist.append(t / 3600.0)
        runoff_engine.update_state(rain_1d, DT)
        t += DT

    t_hist = np.array(t_hist); vsa_frac_hist = np.array(vsa_frac_hist) * 100; rain_hist = np.array(rain_hist)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True,
                                    gridspec_kw={"height_ratios": [1, 2]})
    ax1.bar(t_hist, rain_hist, width=DT/3600, color="#4da6ff", align="edge")
    ax1.invert_yaxis()
    ax1.set_ylabel("Rain\n(mm/hr)")
    ax1.set_title(f"VSA (saturated) area over time vs rainfall -- {EVENT}\n"
                  "does it recede between the two rain bursts, or only ratchet up?", fontweight="bold")

    ax2.plot(t_hist, vsa_frac_hist, color="#3b6ea5", lw=2)
    ax2.fill_between(t_hist, vsa_frac_hist, color="#3b6ea5", alpha=0.15)
    ax2.set_ylabel("VSA area\n(% of watershed, RIGHT NOW)")
    ax2.set_xlabel("hours since event start")
    ax2.grid(True, ls="--", alpha=0.3)
    peak_t = t_hist[vsa_frac_hist.argmax()]
    ax2.annotate(f"peak {vsa_frac_hist.max():.1f}% @ t={peak_t:.0f}h",
                 (peak_t, vsa_frac_hist.max()), textcoords="offset points",
                 xytext=(0, 8), ha="center", fontsize=9)
    end_val = vsa_frac_hist[-1]
    ax2.annotate(f"end of sim: {end_val:.1f}%", (t_hist[-1], end_val),
                 textcoords="offset points", xytext=(-60, -14), fontsize=9)

    fig.tight_layout()
    out = OUT_DIR / "vsa_area_timeseries.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print(f"peak VSA fraction: {vsa_frac_hist.max():.2f}% at t={peak_t:.1f}h")
    print(f"VSA fraction at end of sim: {end_val:.2f}%")
    print(f"min VSA fraction after the peak (checking for recession): "
          f"{vsa_frac_hist[t_hist > peak_t].min():.2f}%" if (t_hist > peak_t).any() else "n/a")
    print(f"-> {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
