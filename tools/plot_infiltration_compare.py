#!/usr/bin/env python
"""
plot_infiltration_compare.py
=============================
Dedicated figure set for the OPM_INFILTRATION ablation ('green_ampt' sandbox-
recharge cap vs 'none' / uncapped) that sits alongside the main channel x
scheme x mechanisms sweep (tools/plot_combinations.py, which only reads the
'green_ampt' rows of master_summary.csv).

Why a separate script: the interesting comparison here is specifically about
the VSA sandbox physics, and specifically for the mechanism subsets that
actually exercise it ('vsa', 'vsa+impervious', 'vsa+horton', 'vsa+horton+
impervious'). Subsets that never select 'vsa' are — correctly — unaffected
by this axis (the sandbox cap has nothing to disaggregate), so they're
excluded from these figures rather than diluting them.

Outputs → outputs collection/combinations_100m/figures/infiltration_compare/
    G01_nse_by_mechanism.png     grouped bars, green_ampt vs none, per mechanism
    G02_pbias_by_mechanism.png   same for PBIAS
    G03_hydrograph_overlay.png   obs vs green_ampt vs none, 3 mechanisms, one flood

Also includes a second comparison: each mechanism run in ITS OWN best-performing
infiltration setting — Horton alone capped (green_ampt, its natural physics)
vs VSA alone uncapped (none, its best-case footprint) — and the +impervious
counterparts, so the "GSSHA-like mechanism" and the "new model" are each given
their fairest shot before being set against each other.
    G04_nse_pure_mechanism.png / G05_pbias_pure_mechanism.png
    G06_hydrograph_pure_mechanism.png
"""

import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tools.runners.common import load_discharge_csv
from tools.runners.runner_config import GSSHA_DIR

ROOT    = REPO_ROOT / "outputs collection/combinations_100m"
FIG_DIR = ROOT / "figures" / "infiltration_compare"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Only mechanism subsets that include 'vsa' — the sandbox cap is a no-op otherwise.
VSA_MECHS  = ["vsa", "vsa+imperv", "vsa+horton", "vsa+horton+imperv"]
MECH_LABEL = {
    "vsa":               "VSA only\n(the new model)",
    "vsa+imperv":        "VSA + impervious",
    "vsa+horton":        "VSA + Horton",
    "vsa+horton+imperv": "VSA+Horton+Imperv\n(production)",
}
INFILT_COLOR = {"green_ampt": "#1f6aa5", "none": "#d1495b"}
INFILT_LABEL = {"green_ampt": "capped (green_ampt)", "none": "uncapped (none)"}

# 202409_202409's observed record cuts off mid-storm (test_data/gssha_format/
# discharge_202409_202409.csv only has 319 rows vs ~576 for the other 3 events)
# — use a flood with a complete observed record for hydrograph overlays instead.
HYDRO_EVENT = "202407_202407"


def load_master_all():
    df = pd.read_csv(ROOT / "master_summary.csv")
    for c in ("channel", "scheme", "mechanisms", "infiltration", "event"):
        df[c] = df[c].astype(str)
    return df


def load_master():
    return load_master_all()[load_master_all()["mechanisms"].isin(VSA_MECHS)].copy()


# "Pure mechanism, best-case infiltration" pairs: Horton is naturally green_ampt
# (that IS its infiltration-excess physics); VSA is given its best-case setting
# (uncapped) found in the G01 comparison. (mech, infilt, family_label) tuples.
PURE_PAIRS = [
    ("horton",        "green_ampt", "vsa",        "none"),
    ("horton+imperv", "green_ampt", "vsa+imperv", "none"),
]
PAIR_GROUP_LABEL = {"horton": "Alone", "horton+imperv": "+ impervious"}
FAMILY_COLOR = {"horton": "#e67e22", "vsa": "#2a9d8f"}
FAMILY_LABEL = {"horton": "Horton only (capped, its own physics)",
                "vsa":    "VSA only (uncapped, best case)"}


# ══════════════════════════════════════════════════════════════════════════════
# G01 / G02 — grouped bars, mean over channel x scheme x event
# ══════════════════════════════════════════════════════════════════════════════
def bar_compare(df, metric, ylabel, title, out_png, hline=None):
    g = df.groupby(["mechanisms", "infiltration"])[metric].mean().unstack("infiltration")
    g = g.reindex(VSA_MECHS)

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(g))
    w = 0.35
    for i, infilt in enumerate(["green_ampt", "none"]):
        ax.bar(x + (i - 0.5) * w, g[infilt], width=w,
               color=INFILT_COLOR[infilt], label=INFILT_LABEL[infilt])
        for xi, v in zip(x + (i - 0.5) * w, g[infilt]):
            ax.annotate(f"{v:.2f}", (xi, v), textcoords="offset points",
                        xytext=(0, 3 if v >= 0 else -12), ha="center", fontsize=8)
    if hline is not None:
        ax.axhline(hline, color="gray", ls="--", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels([MECH_LABEL[m] for m in VSA_MECHS], fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")
    ax.legend()
    ax.grid(True, axis="y", ls="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)
    print(f"  → {out_png.relative_to(REPO_ROOT)}")


# ══════════════════════════════════════════════════════════════════════════════
# G03 — hydrograph overlay: obs vs green_ampt vs none, for one flood
# ══════════════════════════════════════════════════════════════════════════════
def leaf_path(channel, scheme, mech, infilt):
    p = ROOT / f"chan_{channel}" / scheme / mech
    return p if infilt == "green_ampt" else p / "infilt_none"


def hydro_overlay(channel, scheme, event, out_png):
    ec = pd.read_csv(leaf_path(channel, scheme, VSA_MECHS[0], "green_ampt") / "event_catalogue.csv")
    row = ec[ec["event_tag"] == event].iloc[0]
    event_start = datetime.strptime(row["start_local"], "%Y-%m-%d %H:%M")
    disc_path = GSSHA_DIR / f"discharge_{event}.csv"
    otl = load_discharge_csv(disc_path, event_start)

    fig, axes = plt.subplots(len(VSA_MECHS), 1, figsize=(11, 3.1 * len(VSA_MECHS)),
                              sharex=True)
    for ax, mech in zip(axes, VSA_MECHS):
        ax.plot(otl["time_min"] / 60.0, otl["Q_m3s"], color="black", lw=1.6,
                label="Observed")
        for infilt in ("green_ampt", "none"):
            hp = leaf_path(channel, scheme, mech, infilt) / f"hydrograph_{event}.csv"
            if not hp.exists():
                continue
            h = pd.read_csv(hp)
            ax.plot(h["time_hr"], h["Q_m3s"], color=INFILT_COLOR[infilt], lw=1.5,
                    ls="--", label=f"VSA-OPM ({INFILT_LABEL[infilt]})")
        ax.set_ylabel("Q [m³/s]", fontsize=9)
        ax.set_title(MECH_LABEL[mech].replace("\n", " "), fontsize=10, fontweight="bold")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, ls="--", alpha=0.3)
    axes[-1].set_xlabel("Time [hr]")
    fig.suptitle(f"Sandbox recharge cap ablation — event {event}  "
                 f"(chan_{channel}/{scheme})", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_png, dpi=160)
    plt.close(fig)
    print(f"  → {out_png.relative_to(REPO_ROOT)}")


# ══════════════════════════════════════════════════════════════════════════════
# G04 / G05 — Horton(capped) vs VSA(uncapped), alone and +impervious
# ══════════════════════════════════════════════════════════════════════════════
def bar_compare_pure(df_all, metric, ylabel, title, out_png, hline=None):
    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(PURE_PAIRS))
    w = 0.35
    for side, fam in enumerate(("horton", "vsa")):
        vals = []
        for h_mech, h_inf, v_mech, v_inf in PURE_PAIRS:
            mech, inf = (h_mech, h_inf) if fam == "horton" else (v_mech, v_inf)
            sub = df_all[(df_all.mechanisms == mech) & (df_all.infiltration == inf)]
            vals.append(sub[metric].mean())
        xi = x + (side - 0.5) * w
        ax.bar(xi, vals, width=w, color=FAMILY_COLOR[fam], label=FAMILY_LABEL[fam])
        for xx, v in zip(xi, vals):
            ax.annotate(f"{v:.2f}", (xx, v), textcoords="offset points",
                        xytext=(0, 3 if v >= 0 else -12), ha="center", fontsize=8)
    if hline is not None:
        ax.axhline(hline, color="gray", ls="--", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels([PAIR_GROUP_LABEL[h] for h, *_ in PURE_PAIRS], fontsize=10)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")
    ax.legend(fontsize=8.5)
    ax.grid(True, axis="y", ls="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)
    print(f"  → {out_png.relative_to(REPO_ROOT)}")


# ══════════════════════════════════════════════════════════════════════════════
# G06 — hydrograph overlay for the same pure-mechanism pairs
# ══════════════════════════════════════════════════════════════════════════════
def hydro_overlay_pure(channel, scheme, event, out_png):
    ec = pd.read_csv(leaf_path(channel, scheme, "horton", "green_ampt") / "event_catalogue.csv")
    row = ec[ec["event_tag"] == event].iloc[0]
    event_start = datetime.strptime(row["start_local"], "%Y-%m-%d %H:%M")
    disc_path = GSSHA_DIR / f"discharge_{event}.csv"
    otl = load_discharge_csv(disc_path, event_start)

    fig, axes = plt.subplots(len(PURE_PAIRS), 1, figsize=(11, 3.6 * len(PURE_PAIRS)),
                              sharex=True)
    for ax, (h_mech, h_inf, v_mech, v_inf) in zip(axes, PURE_PAIRS):
        ax.plot(otl["time_min"] / 60.0, otl["Q_m3s"], color="black", lw=1.6,
                label="Observed")
        for fam, mech, inf in (("horton", h_mech, h_inf), ("vsa", v_mech, v_inf)):
            hp = leaf_path(channel, scheme, mech, inf) / f"hydrograph_{event}.csv"
            h = pd.read_csv(hp)
            ax.plot(h["time_hr"], h["Q_m3s"], color=FAMILY_COLOR[fam], lw=1.6,
                    ls="--", label=FAMILY_LABEL[fam])
        ax.set_ylabel("Q [m³/s]", fontsize=9)
        ax.set_title(PAIR_GROUP_LABEL[h_mech], fontsize=10, fontweight="bold")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, ls="--", alpha=0.3)
    axes[-1].set_xlabel("Time [hr]")
    fig.suptitle(f"Horton (capped) vs VSA (uncapped) — event {event}  "
                 f"(chan_{channel}/{scheme})", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_png, dpi=160)
    plt.close(fig)
    print(f"  → {out_png.relative_to(REPO_ROOT)}")


def main():
    df = load_master()
    print(f"Loaded {len(df)} rows across {df['mechanisms'].nunique()} VSA-containing mechanisms.")

    bar_compare(df, "nse", "Mean NSE (4 floods)",
                "Sandbox recharge cap: NSE by mechanism",
                FIG_DIR / "G01_nse_by_mechanism.png", hline=0.0)
    bar_compare(df, "pbias_pct", "Mean PBIAS [%] (4 floods)",
                "Sandbox recharge cap: PBIAS by mechanism",
                FIG_DIR / "G02_pbias_by_mechanism.png", hline=0.0)

    # Best-performing channel/scheme combo from the main sweep (chan_off/diffusive).
    hydro_overlay("off", "diffusive", HYDRO_EVENT,
                  FIG_DIR / "G03_hydrograph_overlay.png")

    df_all = load_master_all()
    bar_compare_pure(df_all, "nse", "Mean NSE (4 floods)",
                      "Horton (capped) vs VSA (uncapped) — each at its best",
                      FIG_DIR / "G04_nse_pure_mechanism.png", hline=0.0)
    bar_compare_pure(df_all, "pbias_pct", "Mean PBIAS [%] (4 floods)",
                      "Horton (capped) vs VSA (uncapped) — each at its best",
                      FIG_DIR / "G05_pbias_pure_mechanism.png", hline=0.0)
    hydro_overlay_pure("off", "diffusive", HYDRO_EVENT,
                        FIG_DIR / "G06_hydrograph_pure_mechanism.png")

    print("\nDone.")


if __name__ == "__main__":
    main()
