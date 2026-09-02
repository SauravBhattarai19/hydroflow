#!/usr/bin/env python
"""
plot_mechanism_spatial.py
==========================
Spatial diagnostic: WHERE does each runoff mechanism (Dunne/VSA vs Horton vs
impervious) actually fire, and does the VSA saturated area overlap the cells
that already produced Horton (infiltration-excess) runoff earlier in the storm?

Motivation
----------
master_summary.csv shows VSA-only performs far worse than Horton-only, and in
the combined run Horton supplies ~80% of runoff volume vs ~15-20% for Dunne
(F12/F14 in tools/plot_combinations.py).  That's a *volume* answer.  This
script answers the *spatial* question directly from the runoff engine's own
per-cell state (vsa.py: a cell is Dunne-active iff it's inside `_vsa_mask`,
Horton-active iff outside it with infiltration exceeded -- mutually exclusive
at any instant, but a cell can visit both regimes over a storm as the VSA
source area expands).

It does NOT run the router/routing -- only the runoff-generation engines are
stepped (no hydraulics needed for this question), so it's cheap: reuses the
already-initialised grid + already-cached rasters + already-converted gauge
CSVs from the 202409_202409 leaf of outputs collection/combinations_100m/.

Writes
------
  <out>/mechanism_dominant_map.png   static: majority mechanism per cell (by
                                      cumulative volume) -- the headline figure
  <out>/mechanism_expansion.gif      animated: VSA envelope growing over the
                                      cumulative Horton-active footprint
  <out>/mechanism_overlap_bar.png    ever-Horton / ever-VSA / ever-both venn-style bar

Usage
-----
  python tools/plot_mechanism_spatial.py
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
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import matplotlib.animation as animation

import config  # legacy root scenario module (mutable, same one tools/ uses)
from tools.runners.common import apply_output_dir
from tools.runners.runner_config import OPM_DIR

EVENT   = "202409_202409"           # largest VSA volume share of the 4 floods
LEAF    = REPO_ROOT / "outputs collection/combinations_100m/chan_off/diffusive/vsa+horton+imperv"
OUT_DIR = REPO_ROOT / "outputs collection/combinations_100m/figures/mechanism_spatial"
DT      = 300.0                      # 5 min -- coarser than production (1 s), fine for
                                      # a runoff-generation-only diagnostic (no routing)
SNAPSHOT_HR = 2.0                    # one frame every 2 simulated hours


def asnumpy(x):
    if isinstance(x, np.ndarray):
        return x
    return x.get()   # cupy array


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

    from hydroflow.core.routing import router as kwr
    print(f"Initialising grid for {EVENT} (runoff-generation only, no routing)...")
    grid_data = kwr.initialise_grid(config)

    runoff_engine = grid_data["runoff_engine"]
    precip_engine = grid_data["precip_engine"]
    n_cells       = grid_data["n_cells"]
    s_rows        = asnumpy(grid_data["s_rows"])
    s_cols        = asnumpy(grid_data["s_cols"])
    nrows, ncols  = grid_data["nrows"], grid_data["ncols"]
    cell_area     = grid_data["cell_area"]

    def to_grid(arr_1d, fill=np.nan):
        g = np.full((nrows, ncols), fill, dtype=float)
        g[s_rows, s_cols] = asnumpy(arr_1d)
        return g

    ws_mask_grid = to_grid(np.ones(n_cells), fill=0.0).astype(bool)

    T = config.TOTAL_SIMULATION_TIME_HOURS * 3600.0
    cum_dunne  = np.zeros(n_cells)
    cum_horton = np.zeros(n_cells)
    cum_imperv = np.zeros(n_cells)
    ever_vsa    = np.zeros(n_cells, dtype=bool)
    ever_horton = np.zeros(n_cells, dtype=bool)

    frames = []   # (t_hr, ever_vsa_grid, ever_horton_grid)
    t = 0.0
    next_snap = 0.0
    n_steps = int(T / DT) + 1
    print(f"Stepping runoff engine: {n_steps} steps of dt={DT:.0f}s over {T/3600:.1f}h...")
    step = 0
    while t < T:
        rain_1d = precip_engine.get_field_1d(t)
        _ = runoff_engine.get_effective_1d(t, rain_1d)   # populates _last_*_rate

        cum_dunne  += asnumpy(runoff_engine._last_dunne_rate)  * cell_area * DT
        cum_horton += asnumpy(runoff_engine._last_horton_rate) * cell_area * DT
        cum_imperv += asnumpy(runoff_engine._last_imperv_rate) * cell_area * DT
        ever_vsa    |= asnumpy(runoff_engine._vsa_mask)
        ever_horton |= (asnumpy(runoff_engine._last_horton_rate) > 0.0)

        runoff_engine.update_state(rain_1d, DT)
        t += DT
        step += 1

        if t >= next_snap:
            frames.append((t / 3600.0, to_grid(ever_vsa.astype(float), 0.0).astype(bool),
                           to_grid(ever_horton.astype(float), 0.0).astype(bool)))
            next_snap += SNAPSHOT_HR * 3600.0

    print(f"Done: {step} steps, {len(frames)} snapshot frames.")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Static: dominant mechanism per cell (by cumulative volume) ───────────
    dom = np.full(n_cells, -1, dtype=int)   # -1=dry, 0=dunne, 1=horton, 2=imperv
    vol = np.stack([cum_dunne, cum_horton, cum_imperv], axis=0)
    active = vol.sum(axis=0) > 0
    dom[active] = vol[:, active].argmax(axis=0)
    dom_grid = to_grid(dom.astype(float), fill=-2.0)

    colors = ["#dddddd", "#3b6ea5", "#e08214", "#7a7a7a"]  # dry, dunne, horton, imperv
    cmap = ListedColormap(colors)
    fig, ax = plt.subplots(figsize=(9, 8))
    plot_grid = np.where(ws_mask_grid, dom_grid, np.nan)
    im = ax.imshow(np.where(plot_grid == -1, 0, np.where(plot_grid < 0, np.nan, plot_grid + 1)),
                   cmap=cmap, vmin=0, vmax=3)
    ax.set_title(f"Dominant runoff-generating mechanism per cell\n"
                 f"event {EVENT} -- by cumulative storm volume", fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])
    handles = [Patch(fc=colors[0], label="dry / no runoff"),
               Patch(fc=colors[1], label="Dunne (VSA saturation-excess) dominant"),
               Patch(fc=colors[2], label="Horton (infiltration-excess) dominant"),
               Patch(fc=colors[3], label="Impervious dominant")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=2, fontsize=9)
    fig.tight_layout()
    out1 = OUT_DIR / "mechanism_dominant_map.png"
    fig.savefig(out1, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out1.relative_to(REPO_ROOT)}")

    # ── Overlap bar: ever-Horton / ever-VSA / ever-both ───────────────────────
    only_h = (ever_horton & ~ever_vsa).sum()
    only_v = (ever_vsa & ~ever_horton).sum()
    both    = (ever_vsa & ever_horton).sum()
    neither = n_cells - only_h - only_v - both
    fig, ax = plt.subplots(figsize=(7, 5))
    cats = ["Horton\nonly, ever", "VSA (Dunne)\nonly, ever", "Both\n(VSA cell that was\nHorton before saturating)", "Neither"]
    vals = [only_h, only_v, both, neither]
    pct = [100 * v / n_cells for v in vals]
    bars = ax.bar(cats, pct, color=["#e08214", "#3b6ea5", "#8a4fae", "#dddddd"], edgecolor="k")
    for b, p, v in zip(bars, pct, vals):
        ax.text(b.get_x() + b.get_width() / 2, p + 1, f"{p:.1f}%\n({v:,} cells)",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Share of watershed cells (%)")
    ax.set_title(f"Cell-level mechanism footprint over the storm -- {EVENT}\n"
                 "'Both' = cell generated Horton runoff before it saturated into the VSA", fontsize=10)
    ax.set_ylim(0, max(pct) * 1.3)
    fig.tight_layout()
    out2 = OUT_DIR / "mechanism_overlap_bar.png"
    fig.savefig(out2, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out2.relative_to(REPO_ROOT)}")
    print(f"     Horton-only ever: {pct[0]:.1f}% | VSA-only ever: {pct[1]:.1f}% | "
          f"Both: {pct[2]:.1f}% | Neither: {pct[3]:.1f}%")

    # ── Animated GIF: VSA envelope expanding over the ever-Horton footprint ──
    fig, ax = plt.subplots(figsize=(8, 7))

    def render(i):
        ax.clear()
        t_hr, vsa_g, hor_g = frames[i]
        cat_g = np.zeros_like(dom_grid)
        cat_g[~ws_mask_grid] = np.nan
        cat_g[ws_mask_grid & ~hor_g & ~vsa_g] = 0
        cat_g[ws_mask_grid & hor_g & ~vsa_g] = 1
        cat_g[ws_mask_grid & vsa_g & ~hor_g] = 2
        cat_g[ws_mask_grid & vsa_g & hor_g] = 3
        ax.imshow(cat_g, cmap=ListedColormap(["#f2f2f2", "#e08214", "#3b6ea5", "#8a4fae"]),
                  vmin=0, vmax=3)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{EVENT}  |  t = {t_hr:5.1f} h since event start\n"
                     "orange=ever-Horton  blue=VSA-saturated now  purple=both", fontweight="bold")
        return []

    ani = animation.FuncAnimation(fig, render, frames=len(frames), blit=False)
    out3 = OUT_DIR / "mechanism_expansion.gif"
    ani.save(out3, writer="pillow", fps=4)
    plt.close(fig)
    print(f"  -> {out3.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
