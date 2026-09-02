#!/usr/bin/env python
"""
plot_combinations.py
====================
Presentation figure suite for the full-factorial scenario sweep produced by
``tools/run_combinations.py`` (28 configs x 4 floods).

Sweep axes
----------
  channel     : off / on         (rectangular channel hydraulics vs wide sheet)
  scheme      : kinematic / diffusive
  mechanisms  : the 7 non-empty subsets of {vsa, horton, imperv}

Reads
-----
  outputs collection/combinations_100m/master_summary.csv     (112 rows, joined)
  outputs collection/combinations_100m/<leaf>/hydrograph_<event>.csv
  outputs collection/combinations_100m/<leaf>/partition_<event>.csv
  test_data/gssha_format/discharge_<event>.csv                (observed ground truth)
where <leaf> = chan_{off,on}/{scheme}/{mechanisms}.

Writes
------
  outputs collection/combinations_100m/figures/slides/*.png   (16:9, large fonts)
  outputs collection/combinations_100m/figures/paper/*.png    (compact, high DPI)
  outputs collection/combinations_100m/figures/config_ranking.csv

Every figure is written once and rendered in both styles.  The observed hydrograph
is always overlaid as ground truth; the best config (highest mean NSE across the 4
floods) anchors the showcase and is highlighted throughout.

Usage
-----
  python tools/plot_combinations.py                 # all figures, both styles
  python tools/plot_combinations.py --style slides  # slides | paper | both (default both)
  python tools/plot_combinations.py --only F08 F11  # subset by figure id
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
from matplotlib.lines import Line2D

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
ROOT      = REPO_ROOT / "outputs collection" / "combinations_100m"
GSSHA_DIR = REPO_ROOT / "test_data" / "gssha_format"
FIG_ROOT  = ROOT / "figures"

EVENTS = ["202308_202308", "202407_202407", "202407_202408", "202409_202409"]

# ── Encodings (consistent across the whole suite) ─────────────────────────────
MECH_ORDER = ["vsa", "horton", "imperv", "vsa+horton", "vsa+imperv",
              "horton+imperv", "vsa+horton+imperv"]
MECH_LABEL = {
    "vsa": "VSA only", "horton": "Horton only", "imperv": "Imperv only",
    "vsa+horton": "VSA+Horton", "vsa+imperv": "VSA+Imperv",
    "horton+imperv": "Horton+Imperv", "vsa+horton+imperv": "All (VSA+Horton+Imp)",
}
MECH_COLOR = {m: c for m, c in zip(MECH_ORDER, plt.get_cmap("tab10").colors)}
SCHEME_MARKER = {"kinematic": "o", "diffusive": "^"}
SCHEME_COLOR  = {"kinematic": "#c2453c", "diffusive": "#3b6ea5"}
CHAN_LABEL    = {"off": "sheet", "on": "channel"}

PART_KEYS   = ["dunne", "horton", "imperv"]
PART_COLORS = {"dunne": "#3b6ea5", "horton": "#e08214", "imperv": "#7a7a7a"}
PART_LABEL  = {"dunne": "Dunne (saturation-excess)",
               "horton": "Horton (infiltration-excess)",
               "imperv": "Impervious (urban)"}

# ── Style presets (looped over so each figure renders twice) ──────────────────
STYLES = {
    "slides": dict(scale=1.30, dpi=150, base_fs=15, line=2.2, obs_line=3.0),
    "paper":  dict(scale=1.00, dpi=220, base_fs=10, line=1.5, obs_line=2.2),
}
_ACTIVE = {}   # mutable holder for the style currently being rendered


# ════════════════════════════════════════════════════════════════════════════
# Data access
# ════════════════════════════════════════════════════════════════════════════
def load_master() -> pd.DataFrame:
    p = ROOT / "master_summary.csv"
    if not p.exists():
        sys.exit(f"[ERROR] {p} not found. Run tools/run_combinations.py --aggregate first.")
    df = pd.read_csv(p)
    df["channel"]    = df["channel"].astype(str)
    df["scheme"]     = df["scheme"].astype(str)
    df["mechanisms"] = df["mechanisms"].astype(str)
    df["event"]      = df["event"].astype(str)
    if "infiltration" in df.columns:
        # leaf_path() below resolves to the no-suffix (OPM_INFILTRATION='green_ampt')
        # path only; the 'none' variant lives one level deeper (.../infilt_none/) and
        # is a separate ablation axis (see tools/run_combinations.py), not part of
        # this figure set — keep these figures apples-to-apples with the original
        # channel x scheme x mechanisms sweep.
        df = df[df["infiltration"] == "green_ampt"].drop(columns=["infiltration"])
    if "sd_reducer" in df.columns:
        # Same reasoning as infiltration above — 'divide' is a separate ablation
        # axis (.../sd_divide/), not part of this figure set.
        df = df[df["sd_reducer"] == "max"].drop(columns=["sd_reducer"])
    return df


def leaf_path(channel, scheme, mech) -> Path:
    return ROOT / f"chan_{channel}" / scheme / mech


def cfg_key(row) -> tuple:
    return (row["channel"], row["scheme"], row["mechanisms"])


def cfg_label(channel, scheme, mech) -> str:
    return f"{CHAN_LABEL[channel]}|{scheme[:4]}|{MECH_LABEL[mech]}"


def per_config_means(df) -> pd.DataFrame:
    """One row per config = mean of the 4 floods, sorted by NSE (best first)."""
    num = df.select_dtypes("number").columns
    g = (df.groupby(["channel", "scheme", "mechanisms"])[list(num)]
           .mean().reset_index())
    g["mech_rank"] = g["mechanisms"].map({m: i for i, m in enumerate(MECH_ORDER)})
    return g.sort_values("nse", ascending=False).reset_index(drop=True)


def best_config(df) -> dict:
    g = per_config_means(df)
    b = g.iloc[0]
    return dict(channel=b.channel, scheme=b.scheme, mechanisms=b.mechanisms,
                nse=b.nse, pbias=b.pbias_pct)


def event_starts() -> dict:
    """tag -> event start datetime, from any leaf's event_catalogue.csv."""
    for cat in ROOT.glob("chan_*/*/*/event_catalogue.csv"):
        d = pd.read_csv(cat)
        return {str(r.event_tag): pd.to_datetime(r.start_local) for _, r in d.iterrows()}
    return {}


def event_dates() -> dict:
    """tag -> human date (YYYY-MM-DD) from any summary_all_floods.csv."""
    for sp in ROOT.glob("chan_*/*/*/summary_all_floods.csv"):
        d = pd.read_csv(sp)
        if "start" in d.columns:
            return {str(r.event): str(r.start) for _, r in d.iterrows()}
    return {}


def modelled(channel, scheme, mech, event):
    p = leaf_path(channel, scheme, mech) / f"hydrograph_{event}.csv"
    if not p.exists():
        return None
    d = pd.read_csv(p)
    return d["time_hr"].values, d["Q_m3s"].values


def observed(event, starts):
    disc = GSSHA_DIR / f"discharge_{event}.csv"
    if not disc.exists() or event not in starts:
        return None
    d = pd.read_csv(disc, parse_dates=["dateTime"])
    t_hr = (d["dateTime"] - starts[event]).dt.total_seconds() / 3600.0
    m = t_hr >= 0
    return t_hr[m].values, d["discharge_m3s"][m].values


def partition(channel, scheme, mech, event):
    p = leaf_path(channel, scheme, mech) / f"partition_{event}.csv"
    if not p.exists():
        return None
    return pd.read_csv(p)


# ════════════════════════════════════════════════════════════════════════════
# Render helpers
# ════════════════════════════════════════════════════════════════════════════
def _fig(w, h):
    s = _ACTIVE["scale"]
    plt.rcParams.update({"font.size": _ACTIVE["base_fs"],
                         "axes.titlesize": _ACTIVE["base_fs"] + 1,
                         "axes.titleweight": "bold"})
    return plt.subplots(figsize=(w * s, h * s))


def _fign(nr, nc, w, h, **kw):
    s = _ACTIVE["scale"]
    plt.rcParams.update({"font.size": _ACTIVE["base_fs"],
                         "axes.titlesize": _ACTIVE["base_fs"] + 1,
                         "axes.titleweight": "bold"})
    return plt.subplots(nr, nc, figsize=(w * s, h * s), **kw)


def _save(fig, name, rect=None):
    out_dir = FIG_ROOT / _ACTIVE["name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=rect) if rect else fig.tight_layout()
    out = out_dir / name
    fig.savefig(out, dpi=_ACTIVE["dpi"], bbox_inches="tight")
    plt.close(fig)
    print(f"    → {out.relative_to(REPO_ROOT)}")


def _event_title(ev, dates):
    return f"{dates.get(ev, ev)}  ({ev})"


# ════════════════════════════════════════════════════════════════════════════
# GROUP 1 — whole-sweep overview
# ════════════════════════════════════════════════════════════════════════════
def _heatmap_matrix(df, value_col):
    """Return (labels[28], mat[28x5]) sorted channel→scheme→mechanism; last col=mean."""
    rows, labels = [], []
    for ch in ["off", "on"]:
        for sc in ["kinematic", "diffusive"]:
            for mech in MECH_ORDER:
                sub = df[(df.channel == ch) & (df.scheme == sc) & (df.mechanisms == mech)]
                if sub.empty:
                    continue
                vals = [sub[sub.event == ev][value_col].mean() for ev in EVENTS]
                rows.append(vals + [np.nanmean(vals)])
                labels.append(cfg_label(ch, sc, mech))
    return labels, np.array(rows)


def _heatmap(df, value_col, title, cmap, vlim, fmt, name, best):
    labels, mat = _heatmap_matrix(df, value_col)
    best_lab = cfg_label(best["channel"], best["scheme"], best["mechanisms"])
    dates = event_dates()
    cols = [dates.get(e, e)[5:] if dates.get(e) else e for e in EVENTS] + ["MEAN"]
    fig, ax = _fig(10, 11)
    im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=vlim[0], vmax=vlim[1])
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=30, ha="right")
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=_ACTIVE["base_fs"] - 3)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if np.isnan(v):
                continue
            ax.text(j, i, fmt.format(v), ha="center", va="center",
                    fontsize=_ACTIVE["base_fs"] - 4,
                    color="white" if abs((v - vlim[0]) / (vlim[1] - vlim[0]) - 0.5) > 0.32 else "black")
    # highlight best row + separators between channel/scheme blocks
    if best_lab in labels:
        bi = labels.index(best_lab)
        ax.add_patch(Rectangle((-0.5, bi - 0.5), len(cols), 1, fill=False,
                               edgecolor="lime", lw=3))
    for k in (7, 14, 21):
        ax.axhline(k - 0.5, color="k", lw=1.2)
    ax.axvline(3.5, color="k", lw=1.2)   # separate MEAN column
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, label=value_col)
    ax.set_title(title)
    _save(fig, name, rect=(0, 0, 1, 0.98))


def F01_nse_heatmap(df, best):
    _heatmap(df, "nse", "Model skill (NSE) — all 28 configs x 4 floods\n"
             "green box = best mean-NSE config", "RdYlGn", (-0.5, 1.0), "{:.2f}",
             "F01_nse_heatmap.png", best)


def F02_pbias_heatmap(df, best):
    _heatmap(df, "pbias_pct", "Percent bias (PBIAS %) — 0 is unbiased "
             "(negative = model too low)", "RdBu", (-100, 100), "{:.0f}",
             "F02_pbias_heatmap.png", best)


def F03_peak_heatmap(df, best):
    d = df.copy()
    d["peak_err_pct"] = (d["mod_peak_Q"] / d["obs_peak_Q"] - 1.0) * 100
    d["peak_time_err"] = d["mod_peak_hr"] - d["obs_peak_hr"]
    dates = event_dates()
    cols = [dates.get(e, e)[5:] if dates.get(e) else e for e in EVENTS] + ["MEAN"]
    best_lab = cfg_label(best["channel"], best["scheme"], best["mechanisms"])
    fig, axes = _fign(1, 2, 17, 11)
    specs = [("peak_err_pct", "Peak magnitude error (%)", "RdBu", (-120, 120), "{:.0f}"),
             ("peak_time_err", "Peak timing error (h)  +late / -early", "PuOr", (-8, 8), "{:.1f}")]
    for ax, (col, ttl, cmap, vlim, fmt) in zip(axes, specs):
        labels, mat = _heatmap_matrix(d, col)
        im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=vlim[0], vmax=vlim[1])
        ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=30, ha="right")
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels if ax is axes[0] else [], fontsize=_ACTIVE["base_fs"] - 4)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                if not np.isnan(mat[i, j]):
                    ax.text(j, i, fmt.format(mat[i, j]), ha="center", va="center",
                            fontsize=_ACTIVE["base_fs"] - 5)
        if best_lab in labels:
            bi = labels.index(best_lab)
            ax.add_patch(Rectangle((-0.5, bi - 0.5), len(cols), 1, fill=False,
                                   edgecolor="lime", lw=3))
        for k in (7, 14, 21):
            ax.axhline(k - 0.5, color="k", lw=1.0)
        ax.axvline(3.5, color="k", lw=1.0)
        ax.set_title(ttl)
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    fig.suptitle("Peak error across the sweep", fontsize=_ACTIVE["base_fs"] + 3, fontweight="bold")
    _save(fig, "F03_peak_heatmap.png", rect=(0, 0, 1, 0.97))


# ════════════════════════════════════════════════════════════════════════════
# GROUP 2 — ranking & trade-offs
# ════════════════════════════════════════════════════════════════════════════
def F04_skill_ranking(df, best):
    g = per_config_means(df).sort_values("nse").reset_index(drop=True)  # best at top after barh
    labels = [cfg_label(r.channel, r.scheme, r.mechanisms) for _, r in g.iterrows()]
    y = np.arange(len(g))
    colors = [SCHEME_COLOR[r.scheme] for _, r in g.iterrows()]
    hatch  = ["///" if r.channel == "on" else "" for _, r in g.iterrows()]
    fig, ax = _fig(11, 11)
    bars = ax.barh(y, g["nse"].values, color=colors, edgecolor="k", lw=0.6)
    for b, h in zip(bars, hatch):
        b.set_hatch(h)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=_ACTIVE["base_fs"] - 3)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("Mean NSE across 4 floods")
    ax.set_title("Config ranking by mean NSE (best on top)")
    # mark the winner
    ax.get_yticklabels()[-1].set_color("green")
    ax.get_yticklabels()[-1].set_fontweight("bold")
    handles = [Patch(fc=SCHEME_COLOR["kinematic"], label="kinematic"),
               Patch(fc=SCHEME_COLOR["diffusive"], label="diffusive"),
               Patch(fc="white", ec="k", hatch="///", label="channel ON"),
               Patch(fc="white", ec="k", label="channel OFF (sheet)")]
    ax.legend(handles=handles, loc="lower right", fontsize=_ACTIVE["base_fs"] - 3, frameon=True)
    _save(fig, "F04_skill_ranking.png")


def F05_nse_vs_pbias(df, best):
    g = per_config_means(df)
    fig, ax = _fig(11, 8)
    for _, r in g.iterrows():
        ax.scatter(r.pbias_pct, r.nse, s=140, marker=SCHEME_MARKER[r.scheme],
                   c=[MECH_COLOR[r.mechanisms]],
                   edgecolors="k" if r.channel == "on" else "none",
                   linewidths=1.6, alpha=0.9, zorder=3)
    bl = g[(g.channel == best["channel"]) & (g.scheme == best["scheme"]) &
           (g.mechanisms == best["mechanisms"])].iloc[0]
    ax.annotate("BEST", (bl.pbias_pct, bl.nse), textcoords="offset points",
                xytext=(8, 8), fontweight="bold", color="green")
    ax.axvline(0, color="grey", ls="--", lw=1)
    ax.axhspan(0.7, 1.0, color="green", alpha=0.06)
    ax.set_xlabel("PBIAS (%)   0 = unbiased"); ax.set_ylabel("NSE   1 = perfect")
    ax.set_title("Accuracy trade-off (mean of 4 floods)\nideal = top, centred on 0")
    ax.grid(True, ls="--", alpha=0.3)
    mech_h = [Patch(fc=MECH_COLOR[m], label=MECH_LABEL[m]) for m in MECH_ORDER]
    sch_h  = [Line2D([], [], marker=SCHEME_MARKER[s], color="grey", ls="none",
                     label=s, ms=10) for s in ["kinematic", "diffusive"]]
    ch_h   = [Line2D([], [], marker="o", mfc="grey", mec="k", mew=1.6, ls="none",
                     label="channel ON", ms=10),
              Line2D([], [], marker="o", mfc="grey", mec="none", ls="none",
                     label="channel OFF", ms=10)]
    ax.legend(handles=mech_h + sch_h + ch_h, fontsize=_ACTIVE["base_fs"] - 4,
              loc="lower left", ncol=2, frameon=True)
    _save(fig, "F05_nse_vs_pbias.png")


def F06_peak_scatter(df, best):
    fig, ax = _fig(9, 9)
    for _, r in df.iterrows():
        ax.scatter(r.obs_peak_Q, r.mod_peak_Q, s=60, marker=SCHEME_MARKER[r.scheme],
                   c=[MECH_COLOR[r.mechanisms]],
                   edgecolors="k" if r.channel == "on" else "none",
                   linewidths=0.8, alpha=0.7)
    lim = max(df.obs_peak_Q.max(), df.mod_peak_Q.max()) * 1.05
    ax.plot([0, lim], [0, lim], "k--", lw=1.2, label="1:1")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel("Observed peak Q (m³/s)"); ax.set_ylabel("Modeled peak Q (m³/s)")
    ax.set_title("Peak discharge: modeled vs observed (all 112 runs)")
    mech_h = [Patch(fc=MECH_COLOR[m], label=MECH_LABEL[m]) for m in MECH_ORDER]
    ax.legend(handles=mech_h + [Line2D([], [], color="k", ls="--", label="1:1")],
              fontsize=_ACTIVE["base_fs"] - 4, loc="upper left")
    ax.grid(True, ls="--", alpha=0.3)
    _save(fig, "F06_peak_scatter.png")


# ════════════════════════════════════════════════════════════════════════════
# GROUP 3 — factor main effects
# ════════════════════════════════════════════════════════════════════════════
def F07_factor_effects(df, best):
    d = df.copy()
    d["peak_time_err"] = d["mod_peak_hr"] - d["obs_peak_hr"]
    factors = [("channel", ["off", "on"], lambda v: CHAN_LABEL[v]),
               ("scheme", ["kinematic", "diffusive"], lambda v: v),
               ("mechanisms", MECH_ORDER, lambda v: MECH_LABEL[v])]
    metrics = [("nse", "NSE"), ("peak_time_err", "Peak timing error (h)")]
    fig, axes = _fign(2, 3, 17, 10)
    for ri, (mcol, mlab) in enumerate(metrics):
        for ci, (fcol, order, lab) in enumerate(factors):
            ax = axes[ri, ci]
            data = [d[d[fcol] == lv][mcol].dropna().values for lv in order]
            bp = ax.boxplot(data, patch_artist=True, showmeans=True,
                            medianprops=dict(color="k"))
            for patch, lv in zip(bp["boxes"], order):
                col = (MECH_COLOR[lv] if fcol == "mechanisms"
                       else SCHEME_COLOR.get(lv, "#88a") if fcol == "scheme" else "#6a9fb5")
                patch.set_facecolor(col); patch.set_alpha(0.75)
            ax.set_xticklabels([lab(v) for v in order],
                               rotation=30 if fcol == "mechanisms" else 0, ha="right",
                               fontsize=_ACTIVE["base_fs"] - 4)
            if mcol == "peak_time_err":
                ax.axhline(0, color="grey", ls="--", lw=1)
            if ci == 0:
                ax.set_ylabel(mlab)
            if ri == 0:
                ax.set_title(f"by {fcol}")
            ax.grid(True, axis="y", ls="--", alpha=0.3)
    fig.suptitle("Main effects — how much each design axis moves skill & timing",
                 fontsize=_ACTIVE["base_fs"] + 3, fontweight="bold")
    _save(fig, "F07_factor_effects.png", rect=(0, 0, 1, 0.96))


# ════════════════════════════════════════════════════════════════════════════
# GROUP 4 — hydrograph comparisons (observed always overlaid)
# ════════════════════════════════════════════════════════════════════════════
def _overlay(scenarios, title, name, colors=None):
    """scenarios = list of ((channel,scheme,mech), label, color)."""
    starts = event_starts(); dates = event_dates()
    fig, axes = _fign(2, 2, 15, 9)
    for ax, ev in zip(axes.ravel(), EVENTS):
        obs = observed(ev, starts)
        if obs is not None:
            ax.plot(obs[0], obs[1], color="black", lw=_ACTIVE["obs_line"],
                    label="Observed", zorder=10)
        for (ch, sc, mech), lab, col in scenarios:
            md = modelled(ch, sc, mech, ev)
            if md is not None:
                ax.plot(md[0], md[1], lw=_ACTIVE["line"], color=col, label=lab, alpha=0.9)
        ax.set_title(_event_title(ev, dates))
        ax.set_xlabel("hours since event start"); ax.set_ylabel("Discharge (m³/s)")
        ax.set_ylim(bottom=0); ax.grid(True, ls="--", alpha=0.3)
    axes.ravel()[0].legend(fontsize=_ACTIVE["base_fs"] - 4, loc="upper right")
    fig.suptitle(title, fontsize=_ACTIVE["base_fs"] + 3, fontweight="bold")
    _save(fig, name, rect=(0, 0, 1, 0.96))


def F08_hydro_mechanism(df, best):
    ch, sc = best["channel"], best["scheme"]
    scen = [((ch, sc, m), MECH_LABEL[m], MECH_COLOR[m]) for m in MECH_ORDER]
    _overlay(scen, f"Runoff mechanism comparison  (channel {ch}, {sc})\n"
             "observed = black; all 7 mechanism combinations", "F08_hydro_mechanism.png")


def F09_hydro_channel(df, best):
    sc, mech = best["scheme"], "vsa+horton+imperv"
    scen = [(("off", sc, mech), "Channel OFF (wide sheet)", "#c2453c"),
            (("on",  sc, mech), "Channel ON (rectangular)", "#3b6ea5")]
    _overlay(scen, f"Channel routing on/off  (full physics, {sc})\n"
             "channel hydraulics sharpen & advance the peak", "F09_hydro_channel.png")


def F10_hydro_scheme(df, best):
    ch, mech = best["channel"], "vsa+horton+imperv"
    scen = [((ch, "kinematic", mech), "Kinematic wave", SCHEME_COLOR["kinematic"]),
            ((ch, "diffusive", mech), "Diffusive wave (θ=1)", SCHEME_COLOR["diffusive"])]
    _overlay(scen, f"Routing scheme comparison  (full physics, channel {ch})\n"
             "diffusion adds attenuation", "F10_hydro_scheme.png")


def F11_best_showcase(df, best):
    ch, sc, mech = best["channel"], best["scheme"], best["mechanisms"]
    starts = event_starts(); dates = event_dates()
    fig, axes = _fign(2, 2, 15, 9)
    for ax, ev in zip(axes.ravel(), EVENTS):
        obs = observed(ev, starts)
        if obs is not None:
            ax.plot(obs[0], obs[1], color="black", lw=_ACTIVE["obs_line"],
                    label="Observed", zorder=10)
        md = modelled(ch, sc, mech, ev)
        if md is not None:
            ax.plot(md[0], md[1], lw=_ACTIVE["line"] + 0.6, color="#c2453c",
                    label="Model (best)")
            ax.fill_between(md[0], md[1], color="#c2453c", alpha=0.12)
        row = df[(df.channel == ch) & (df.scheme == sc) &
                 (df.mechanisms == mech) & (df.event == ev)]
        if not row.empty:
            r = row.iloc[0]
            txt = (f"NSE {r.nse:.2f}\nPBIAS {r.pbias_pct:+.0f}%\n"
                   f"Qp obs {r.obs_peak_Q:.0f} / mod {r.mod_peak_Q:.0f}\n"
                   f"tp obs {r.obs_peak_hr:.1f} / mod {r.mod_peak_hr:.1f} h")
            ax.text(0.97, 0.95, txt, transform=ax.transAxes, ha="right", va="top",
                    fontsize=_ACTIVE["base_fs"] - 3,
                    bbox=dict(boxstyle="round", fc="white", ec="grey", alpha=0.85))
        ax.set_title(_event_title(ev, dates))
        ax.set_xlabel("hours since event start"); ax.set_ylabel("Discharge (m³/s)")
        ax.set_ylim(bottom=0); ax.grid(True, ls="--", alpha=0.3)
    axes.ravel()[0].legend(fontsize=_ACTIVE["base_fs"] - 3, loc="upper left")
    fig.suptitle(f"Best model — {cfg_label(ch, sc, mech)}   "
                 f"(mean NSE {best['nse']:.2f})",
                 fontsize=_ACTIVE["base_fs"] + 3, fontweight="bold")
    _save(fig, "F11_best_showcase.png", rect=(0, 0, 1, 0.96))


# ════════════════════════════════════════════════════════════════════════════
# GROUP 5 — runoff-generation physics
# ════════════════════════════════════════════════════════════════════════════
def F12_partition_bars(df, best):
    """Stacked Dunne/Horton/Imperv fraction per mechanism subset — one panel per event."""
    dates = event_dates()
    fig, axes = _fign(2, 2, 15, 9)
    for ax, ev in zip(axes.ravel(), EVENTS):
        d = df[df.event == ev]
        # collapse over channel+scheme (fractions are set by runoff generation)
        g = d.groupby("mechanisms")[["dunne_frac", "horton_frac", "imperv_frac"]].mean()
        g = g.reindex(MECH_ORDER).dropna(how="all")
        x = np.arange(len(g)); bot = np.zeros(len(g))
        for key in PART_KEYS:
            v = g[f"{key}_frac"].values * 100
            ax.bar(x, v, bottom=bot, color=PART_COLORS[key], edgecolor="white",
                   label=PART_LABEL[key])
            bot += v
        ax.set_xticks(x)
        ax.set_xticklabels([MECH_LABEL[m] for m in g.index], rotation=35, ha="right",
                           fontsize=_ACTIVE["base_fs"] - 4)
        ax.set_ylabel("Share of runoff (%)"); ax.set_ylim(0, 100)
        ax.set_title(_event_title(ev, dates))
    axes.ravel()[0].legend(fontsize=_ACTIVE["base_fs"] - 4, loc="upper left")
    fig.suptitle("Runoff partition by active mechanism set (per flood)",
                 fontsize=_ACTIVE["base_fs"] + 3, fontweight="bold")
    _save(fig, "F12_partition_bars.png", rect=(0, 0, 1, 0.96))


def F13_runoff_ratio(df, best):
    dates = event_dates()
    g = df.groupby(["mechanisms", "event"]).runoff_ratio.mean().unstack("event")
    g = g.reindex(MECH_ORDER).dropna(how="all")
    x = np.arange(len(g)); n = len(EVENTS); w = 0.8 / n
    fig, ax = _fig(12, 7)
    for i, ev in enumerate(EVENTS):
        if ev in g.columns:
            ax.bar(x + (i - (n - 1) / 2) * w, g[ev].values, w,
                   label=dates.get(ev, ev)[:10])
    ax.set_xticks(x); ax.set_xticklabels([MECH_LABEL[m] for m in g.index],
                                         rotation=30, ha="right")
    ax.set_ylabel("Runoff ratio (routed out / rain)")
    ax.set_title("Runoff ratio by mechanism set × flood")
    ax.legend(fontsize=_ACTIVE["base_fs"] - 3); ax.grid(True, axis="y", ls="--", alpha=0.3)
    _save(fig, "F13_runoff_ratio.png")


def F14_partition_cumulative(df, best):
    """Cumulative Dunne/Horton/Imperv volume vs time for the best full-mechanism config."""
    ch, sc = best["channel"], best["scheme"]
    mech = "vsa+horton+imperv"
    dates = event_dates()
    fig, axes = _fign(2, 2, 15, 9)
    for ax, ev in zip(axes.ravel(), EVENTS):
        p = partition(ch, sc, mech, ev)
        if p is None:
            ax.set_title(f"{ev} (no data)"); continue
        t = p["time_hr"].values
        d, h, im = p["dunne_m3"].values, p["horton_m3"].values, p["imperv_m3"].values
        ax.stackplot(t, d / 1e6, h / 1e6, im / 1e6,
                     colors=[PART_COLORS["dunne"], PART_COLORS["horton"], PART_COLORS["imperv"]],
                     labels=[PART_LABEL["dunne"], PART_LABEL["horton"], PART_LABEL["imperv"]])
        ax.set_title(_event_title(ev, dates))
        ax.set_xlabel("hours since event start")
        ax.set_ylabel("Cumulative runoff (×10⁶ m³)")
        ax.grid(True, ls="--", alpha=0.3)
    axes.ravel()[0].legend(fontsize=_ACTIVE["base_fs"] - 4, loc="upper left")
    fig.suptitle(f"Runoff generation over time — full physics ({CHAN_LABEL[ch]}, {sc})\n"
                 "impervious responds first → Horton → Dunne",
                 fontsize=_ACTIVE["base_fs"] + 3, fontweight="bold")
    _save(fig, "F14_partition_cumulative.png", rect=(0, 0, 1, 0.95))


# ════════════════════════════════════════════════════════════════════════════
# Driver
# ════════════════════════════════════════════════════════════════════════════
FIGURES = {
    "F01": F01_nse_heatmap, "F02": F02_pbias_heatmap, "F03": F03_peak_heatmap,
    "F04": F04_skill_ranking, "F05": F05_nse_vs_pbias, "F06": F06_peak_scatter,
    "F07": F07_factor_effects,
    "F08": F08_hydro_mechanism, "F09": F09_hydro_channel, "F10": F10_hydro_scheme,
    "F11": F11_best_showcase,
    "F12": F12_partition_bars, "F13": F13_runoff_ratio, "F14": F14_partition_cumulative,
}


def write_ranking_csv(df):
    g = per_config_means(df).drop(columns=["mech_rank"], errors="ignore")
    keep = ["channel", "scheme", "mechanisms", "nse", "pbias_pct",
            "obs_peak_Q", "mod_peak_Q", "obs_peak_hr", "mod_peak_hr",
            "runoff_ratio", "dunne_frac", "horton_frac", "imperv_frac"]
    g = g[[c for c in keep if c in g.columns]].round(3)
    FIG_ROOT.mkdir(parents=True, exist_ok=True)
    out = FIG_ROOT / "config_ranking.csv"
    g.to_csv(out, index=False)
    print(f"  ranking table ({len(g)} configs) → {out.relative_to(REPO_ROOT)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--style", choices=["slides", "paper", "both"], default="both")
    ap.add_argument("--only", nargs="+", metavar="Fxx",
                    help="subset of figure ids (e.g. F08 F11)")
    args = ap.parse_args()

    df = load_master()
    best = best_config(df)
    print(f"Loaded {len(df)} rows | best config = "
          f"chan_{best['channel']}/{best['scheme']}/{best['mechanisms']} "
          f"(mean NSE {best['nse']:.3f})")

    write_ranking_csv(df)

    ids = args.only if args.only else list(FIGURES.keys())
    ids = [i.upper() for i in ids]
    bad = [i for i in ids if i not in FIGURES]
    if bad:
        sys.exit(f"[ERROR] unknown figure id(s): {bad}. Valid: {list(FIGURES)}")

    styles = ["slides", "paper"] if args.style == "both" else [args.style]
    for st in styles:
        _ACTIVE.clear(); _ACTIVE.update(STYLES[st]); _ACTIVE["name"] = st
        print(f"\n[{st}]  →  {(FIG_ROOT / st).relative_to(REPO_ROOT)}")
        for i in ids:
            try:
                FIGURES[i](df, best)
            except Exception as e:                       # keep going on one bad leaf
                print(f"    [WARN] {i} failed: {e}")
    print("\nDone.")


if __name__ == "__main__":
    main()
