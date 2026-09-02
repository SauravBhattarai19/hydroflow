#!/usr/bin/env python
"""
plot_mechanism_independence.py
===============================
Sharper version of the VSA/Horton question: not "how do they split volume when
run TOGETHER" (plot_mechanism_spatial.py), but --

    Run VSA alone (RUNOFF_MECHANISMS=['vsa']) and Horton alone
    (RUNOFF_MECHANISMS=['horton']) as two INDEPENDENT simulations.  Does
    Horton-alone, run without VSA ever claiming a cell as saturated, already
    generate comparable runoff on the SAME cells VSA-alone activates?

This matters because master_summary.csv shows 'horton' alone (NSE 0.815,
chan_off/diffusive) beats 'vsa+horton' (0.812) and 'horton+imperv' (0.808) at
the basin outlet -- so at the outlet, Horton alone looks sufficient. This
script checks whether that's because Horton is independently reproducing the
same cell-level saturation-like behaviour VSA produces (true redundancy), or
whether VSA's cells are just a small enough share of total volume that their
absence barely moves the outlet metric even though they're doing something
locally distinct (weak-outlet-sensitivity, not redundancy).

Physical hinge: with RUNOFF_MECHANISMS=['horton'] only, `_vsa_mask` is
permanently all-False (vsa.py:263-269) so pervious_frac == excess_frac
everywhere -- pure Green-Ampt, no topographic saturation test at all. Green-
Ampt's infiltration capacity decays as f_p = Ksat*(1+psi*dtheta0/F) -> Ksat as
cumulative infiltration F grows, so excess_frac -> 1 - Ksat/intensity for
sustained intense rain REGARDLESS of upslope contributing area. With this
basin's very low Ksat (~0.4-5 mm/hr per the Green-Ampt log line), that limit
can be reached almost anywhere given enough intense rain -- which would look
like redundancy from a totally different physical cause (local infiltration
exhaustion, not topographic convergence).

Writes (outputs collection/combinations_100m/figures/mechanism_spatial/)
  independence_scatter.png   per-cell depth: VSA-alone vs Horton-alone
  independence_map.png       V-only / H-only / both / neither, over VSA's own
                              footprint
  independence_summary.txt   the headline numbers
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

import config
from tools.runners.common import apply_output_dir
from tools.runners.runner_config import OPM_DIR

EVENT   = "202409_202409"
LEAF    = REPO_ROOT / "outputs collection/combinations_100m/chan_off/diffusive/vsa+horton+imperv"
OUT_DIR = REPO_ROOT / "outputs collection/combinations_100m/figures/mechanism_spatial"
DT      = 300.0
ACTIVE_MM = 1.0     # a cell counts as "active" if it sheds > 1 mm over the storm


def asnumpy(x):
    return x if isinstance(x, np.ndarray) else x.get()


def base_cfg():
    cat = pd.read_csv(LEAF / "event_catalogue.csv")
    row = cat[cat.event_tag == EVENT].iloc[0]
    apply_output_dir(config, str(LEAF) + "/")
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
    return row


def run_single_mechanism(mech: str):
    """Run the runoff engine ALONE with RUNOFF_MECHANISMS=[mech]; return
    (cum_depth_mm (n_cells,), grid_data) -- grid_data reused for reshaping."""
    row = base_cfg()
    config.RUNOFF_MECHANISMS = [mech]
    config.OPM_INFILTRATION  = 'green_ampt' if mech == 'horton' else 'none'
    config.IMPERVIOUS_SOURCE = 'none'
    config.RUN_TAG            = f"{EVENT}_{mech}alone"

    from vsa_opm.core.routing import router as kwr
    print(f"\n=== {mech.upper()}-ONLY (independent run) ===")
    grid_data = kwr.initialise_grid(config)
    runoff_engine = grid_data["runoff_engine"]
    precip_engine = grid_data["precip_engine"]
    n_cells   = grid_data["n_cells"]
    cell_area = grid_data["cell_area"]

    T = config.TOTAL_SIMULATION_TIME_HOURS * 3600.0
    cum_vol = np.zeros(n_cells)
    t = 0.0
    while t < T:
        rain_1d = precip_engine.get_field_1d(t)
        _ = runoff_engine.get_effective_1d(t, rain_1d)
        if mech == 'vsa':
            cum_vol += asnumpy(runoff_engine._last_dunne_rate) * cell_area * DT
        else:
            cum_vol += asnumpy(runoff_engine._last_horton_rate) * cell_area * DT
        runoff_engine.update_state(rain_1d, DT)
        t += DT

    cum_depth_mm = cum_vol / cell_area * 1000.0
    return cum_depth_mm, grid_data


def to_grid(arr_1d, s_rows, s_cols, nrows, ncols, fill=np.nan):
    g = np.full((nrows, ncols), fill, dtype=float)
    g[s_rows, s_cols] = arr_1d
    return g


def main():
    depth_vsa, gd = run_single_mechanism('vsa')
    depth_hor, _  = run_single_mechanism('horton')

    s_rows, s_cols = asnumpy(gd["s_rows"]), asnumpy(gd["s_cols"])
    nrows, ncols   = gd["nrows"], gd["ncols"]
    ws_mask_grid   = to_grid(np.ones(len(depth_vsa)), s_rows, s_cols, nrows, ncols, 0.0).astype(bool)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    V = depth_vsa > ACTIVE_MM
    H = depth_hor > ACTIVE_MM
    n = len(depth_vsa)

    v_only = V & ~H
    both   = V & H
    h_only = H & ~V
    neither = ~V & ~H

    vol_vsa_in_V   = depth_vsa[V].sum()
    vol_hor_in_V   = depth_hor[V].sum()      # what horton-alone would have produced on VSA's own cells

    lines = []
    lines.append(f"Event: {EVENT}  (independent single-mechanism runs, {ACTIVE_MM} mm activity threshold)")
    lines.append(f"Watershed cells: {n:,}")
    lines.append("")
    lines.append(f"VSA-alone active cells   : {V.sum():,}  ({100*V.mean():.1f}% of watershed)")
    lines.append(f"Horton-alone active cells: {H.sum():,}  ({100*H.mean():.1f}% of watershed)")
    lines.append("")
    lines.append(f"Of VSA-alone's {V.sum():,} active cells:")
    lines.append(f"  also active in Horton-alone (\"both\")   : {both.sum():,}  "
                 f"({100*both.sum()/max(V.sum(),1):.1f}% of VSA's footprint)")
    lines.append(f"  NOT active in Horton-alone (\"VSA-only\") : {v_only.sum():,}  "
                 f"({100*v_only.sum()/max(V.sum(),1):.1f}% of VSA's footprint)")
    lines.append("")
    lines.append("Magnitude check on VSA's own footprint (does Horton-alone match the DEPTH, not just fire at all?):")
    lines.append(f"  total VSA-alone depth on V cells    : {vol_vsa_in_V:,.0f} mm-cells")
    lines.append(f"  total Horton-alone depth on V cells : {vol_hor_in_V:,.0f} mm-cells "
                 f"({100*vol_hor_in_V/max(vol_vsa_in_V,1e-9):.1f}% of VSA-alone's own depth there)")
    lines.append("")
    lines.append(f"Horton-alone footprint is {H.sum()/max(V.sum(),1):.1f}x larger in area than VSA-alone's footprint.")

    summary = "\n".join(lines)
    print("\n" + summary)
    (OUT_DIR / "independence_summary.txt").write_text(summary + "\n")

    # ── Scatter: per-cell depth, VSA-alone vs Horton-alone ────────────────────
    fig, ax = plt.subplots(figsize=(7, 7))
    show = V | H
    ax.scatter(depth_vsa[show & ~V], depth_hor[show & ~V], s=3, alpha=0.15, color="grey", label="Horton-only cell")
    ax.scatter(depth_vsa[v_only], depth_hor[v_only], s=6, alpha=0.5, color="#3b6ea5", label="VSA-only (not matched by Horton)")
    ax.scatter(depth_vsa[both], depth_hor[both], s=6, alpha=0.5, color="#8a4fae", label="Both (VSA cell, Horton-alone also fires)")
    lim = max(depth_vsa[show].max(), depth_hor[show].max()) * 1.05
    ax.plot([0, lim], [0, lim], "k--", lw=1, label="1:1")
    ax.set_xlabel("VSA-alone cumulative depth (mm)")
    ax.set_ylabel("Horton-alone cumulative depth (mm)")
    ax.set_title(f"Per-cell runoff depth: independent VSA-only vs Horton-only runs\n{EVENT}")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, ls="--", alpha=0.3)
    fig.tight_layout()
    out1 = OUT_DIR / "independence_scatter.png"
    fig.savefig(out1, dpi=180)
    plt.close(fig)
    print(f"  -> {out1.relative_to(REPO_ROOT)}")

    # ── Map: VSA's footprint, split by whether Horton-alone matches it ────────
    cat_grid = np.full((nrows, ncols), -1.0)
    cat_grid[ws_mask_grid] = 0
    cat_g_1d = np.zeros(n)
    cat_g_1d[h_only] = 1
    cat_g_1d[v_only] = 2
    cat_g_1d[both]   = 3
    cat_grid = to_grid(cat_g_1d, s_rows, s_cols, nrows, ncols, fill=-1.0)
    cat_grid = np.where(ws_mask_grid, cat_grid, np.nan)

    colors = ["#eeeeee", "#e08214", "#3b6ea5", "#8a4fae"]
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.imshow(cat_grid, cmap=ListedColormap(colors), vmin=0, vmax=3)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"Independent VSA-only vs Horton-only footprints -- {EVENT}\n"
                 "does Horton, run ALONE, already cover VSA's cells?", fontweight="bold")
    handles = [Patch(fc=colors[0], label="neither active"),
               Patch(fc=colors[1], label="Horton-alone only"),
               Patch(fc=colors[2], label="VSA-alone only (Horton-alone misses it)"),
               Patch(fc=colors[3], label="both (Horton-alone matches VSA's cell)")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=2, fontsize=9)
    fig.tight_layout()
    out2 = OUT_DIR / "independence_map.png"
    fig.savefig(out2, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out2.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
