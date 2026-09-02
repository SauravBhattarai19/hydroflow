#!/usr/bin/env python
"""
plot_vsa_sweep_results.py
==========================
Analysis + figures for the VSA parameter sweep produced by
``tools/run_vsa_param_sweep.py`` (246 configs x 4 floods = 984 rows).

Sweep axes
----------
  scheme       : kinematic / diffusive / muskingum
  mechanisms   : vsa / vsa+imperv / vsa+horton / horton
  sd_min       : 0.1 / 0.001 / 0.00001            (vsa-containing mechanisms only)
  ksat         : 4.4 / 0.44 / 0.044  m/day        (vsa-containing mechanisms only)
  sd_max       : 0.1 / 0.3 / 0.5     m            (vsa-containing mechanisms only)

Reads
-----
  outputs collection/vsa_param_sweep_100m/master_summary.csv

Writes
------
  outputs collection/vsa_param_sweep_100m/figures/config_ranking.csv
  outputs collection/vsa_param_sweep_100m/figures/*.png

Usage
-----
  python tools/plot_vsa_sweep_results.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
ROOT      = REPO_ROOT / "outputs collection" / "vsa_param_sweep_100m"
FIG_DIR   = ROOT / "figures"

MECH_ORDER = ["vsa", "vsa+imperv", "vsa+horton", "horton"]
MECH_LABEL = {"vsa": "VSA only", "vsa+imperv": "VSA+Imperv",
              "vsa+horton": "VSA+Horton", "horton": "Horton only"}
MECH_COLOR = dict(zip(MECH_ORDER, ["#3b6ea5", "#5aa469", "#c2453c", "#e08214"]))
SCHEME_ORDER = ["kinematic", "diffusive", "muskingum"]
SCHEME_COLOR = {"kinematic": "#c2453c", "diffusive": "#3b6ea5", "muskingum": "#5aa469"}
SCHEME_MARKER = {"kinematic": "o", "diffusive": "^", "muskingum": "s"}
VSA_MECHS = ["vsa", "vsa+imperv", "vsa+horton"]   # mechanisms where sd_min/ksat/sd_max are active


def load():
    p = ROOT / "master_summary.csv"
    if not p.exists():
        sys.exit(f"[ERROR] {p} not found. Run tools/run_vsa_param_sweep.py --aggregate first.")
    df = pd.read_csv(p)
    df["mechanisms"] = df["mechanisms"].astype(str)
    df["scheme"] = df["scheme"].astype(str)
    return df


def per_config_means(df):
    """One row per config = mean across the 4 floods, sorted by NSE (best first)."""
    key = ["scheme", "mechanisms", "sd_min", "ksat", "sd_max"]
    num = [c for c in df.select_dtypes("number").columns if c not in ("sd_min", "ksat", "sd_max")]
    g = df.groupby(key, dropna=False)[num].mean().reset_index()
    return g.sort_values("nse", ascending=False).reset_index(drop=True)


def config_label(row):
    lbl = f"{row.scheme[:4]}/{MECH_LABEL[row.mechanisms]}"
    if row.mechanisms in VSA_MECHS:
        lbl += f"\nsdmin={row.sd_min:g} ksat={row.ksat:g} sdmax={row.sd_max:g}"
    return lbl


# ── Figure 1: best/worst configs ──────────────────────────────────────────────
def fig_best_worst(g):
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    for ax, sub, title, color_by_rank in [
        (axes[0], g.head(15), "Top 15 configs by mean NSE", False),
        (axes[1], g.tail(10).iloc[::-1], "Bottom 10 configs by mean NSE", False),
    ]:
        colors = [MECH_COLOR[m] for m in sub.mechanisms]
        y = np.arange(len(sub))
        ax.barh(y, sub.nse, color=colors, edgecolor="k", linewidth=0.5)
        ax.set_yticks(y)
        ax.set_yticklabels([config_label(r) for r in sub.itertuples()], fontsize=7.5)
        ax.invert_yaxis()
        ax.axvline(0, color="k", linewidth=0.8)
        ax.set_xlabel("mean NSE (4 floods)")
        ax.set_title(title, fontsize=11)
        ax.grid(axis="x", alpha=0.3)
    handles = [plt.Rectangle((0, 0), 1, 1, color=MECH_COLOR[m]) for m in MECH_ORDER]
    fig.legend(handles, [MECH_LABEL[m] for m in MECH_ORDER], loc="lower center",
               ncol=4, fontsize=9, frameon=False)
    fig.suptitle("VSA parameter sweep — best & worst configurations", fontsize=13)
    fig.tight_layout(rect=[0, 0.06, 1, 0.96])
    fig.savefig(FIG_DIR / "fig1_best_worst_configs.png", dpi=180)
    plt.close(fig)


# ── Figure 2: NSE by mechanism x scheme ───────────────────────────────────────
def fig_mechanism_scheme(g):
    fig, ax = plt.subplots(figsize=(9, 6))
    width = 0.25
    x = np.arange(len(MECH_ORDER))
    for i, scheme in enumerate(SCHEME_ORDER):
        vals, errs = [], []
        for m in MECH_ORDER:
            sub = g[(g.mechanisms == m) & (g.scheme == scheme)]["nse"]
            vals.append(sub.mean())
            errs.append(sub.std() if len(sub) > 1 else 0)
        ax.bar(x + (i - 1) * width, vals, width, yerr=errs, capsize=3,
               color=SCHEME_COLOR[scheme], label=scheme, edgecolor="k", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels([MECH_LABEL[m] for m in MECH_ORDER])
    ax.axhline(0, color="k", linewidth=0.8)
    ax.set_ylabel("mean NSE across configs (± std)")
    ax.set_title("NSE by mechanism subset and routing scheme")
    ax.legend(title="scheme")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig2_nse_by_mechanism_scheme.png", dpi=180)
    plt.close(fig)


# ── Figures 3-5: sensitivity to sd_min / ksat / sd_max ───────────────────────
def fig_sensitivity_curve(g, param, xlabel, fname, logx=False):
    sub = g[g.mechanisms.isin(VSA_MECHS)]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for ax, scheme in zip(axes, SCHEME_ORDER):
        s = sub[sub.scheme == scheme]
        for m in VSA_MECHS:
            sm = s[s.mechanisms == m].groupby(param)["nse"].agg(["mean", "std"]).reset_index()
            ax.errorbar(sm[param], sm["mean"], yerr=sm["std"], marker="o", capsize=3,
                        color=MECH_COLOR[m], label=MECH_LABEL[m], linewidth=1.8)
        if logx:
            ax.set_xscale("log")
        ax.axhline(0, color="k", linewidth=0.6)
        ax.set_title(scheme)
        ax.set_xlabel(xlabel)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("mean NSE (± std across other axes)")
    axes[0].legend(fontsize=8)
    fig.suptitle(f"Sensitivity of NSE to {xlabel}", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(FIG_DIR / fname, dpi=180)
    plt.close(fig)


# ── Figure 6: overall sensitivity ranking ─────────────────────────────────────
def fig_sensitivity_ranking(g):
    effects = {}
    # mechanism & scheme apply to every config
    effects["mechanism"] = g.groupby("mechanisms")["nse"].mean()
    effects["scheme"] = g.groupby("scheme")["nse"].mean()
    # sd_min / ksat / sd_max only meaningful where 'vsa' is active
    vsa_g = g[g.mechanisms.isin(VSA_MECHS)]
    effects["sd_min"] = vsa_g.groupby("sd_min")["nse"].mean()
    effects["ksat"] = vsa_g.groupby("ksat")["nse"].mean()
    effects["sd_max"] = vsa_g.groupby("sd_max")["nse"].mean()

    ranges = {k: v.max() - v.min() for k, v in effects.items()}
    order = sorted(ranges, key=ranges.get, reverse=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#3b6ea5" if k in ("mechanism", "scheme") else "#c2453c" for k in order]
    ax.barh(order, [ranges[k] for k in order], color=colors, edgecolor="k", linewidth=0.5)
    ax.invert_yaxis()
    ax.set_xlabel("range of mean NSE across the parameter's levels\n(swing = sensitivity)")
    ax.set_title("Sensitivity ranking: which knob moves NSE the most?")
    ax.grid(axis="x", alpha=0.3)
    for i, k in enumerate(order):
        ax.text(ranges[k] + 0.01, i, f"{ranges[k]:.3f}", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig6_sensitivity_ranking.png", dpi=180)
    plt.close(fig)
    return ranges, order, effects


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = load()
    g = per_config_means(df)
    g.to_csv(FIG_DIR / "config_ranking.csv", index=False)

    fig_best_worst(g)
    fig_mechanism_scheme(g)
    fig_sensitivity_curve(g, "sd_min", "SD_min (m)", "fig3_sensitivity_sdmin.png", logx=True)
    fig_sensitivity_curve(g, "ksat", "OPM_K_SAT (m/day)", "fig4_sensitivity_ksat.png", logx=True)
    fig_sensitivity_curve(g, "sd_max", "SD_max (m)", "fig5_sensitivity_sdmax.png")
    ranges, order, effects = fig_sensitivity_ranking(g)

    print(f"\nWrote {FIG_DIR}\n")
    print("Best config:")
    print(g.iloc[0][["scheme", "mechanisms", "sd_min", "ksat", "sd_max", "nse", "pbias_pct"]])
    print("\nSensitivity ranking (swing in mean NSE, most -> least sensitive):")
    for k in order:
        print(f"  {k:10s} swing={ranges[k]:.4f}")


if __name__ == "__main__":
    main()
