#!/usr/bin/env python
"""
plot_channel_mask.py
====================
Explainer visuals for the channel mask + Strahler-order width assignment used by
CHANNEL_ROUTING.  Built from the REAL cached grid of the combinations sweep.

Outputs (into outputs collection/combinations_100m/figures/channel/):
  channel_mask_orders.png     – watershed: Strahler order & assigned width (2 panels)
  channel_mask_threshold.gif  – animation: mask grows as the faccum threshold loosens
  channel_xsection.png        – sheet vs channel vs over-wide cross-section schematic

Run:  conda run -n opm python tools/plot_channel_mask.py
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Rectangle, Patch
import matplotlib.animation as animation

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

LEAF = "outputs collection/combinations_100m/chan_on/diffusive/vsa+horton+imperv"
OUT  = REPO_ROOT / "outputs collection/combinations_100m/figures/channel"
OUT.mkdir(parents=True, exist_ok=True)

ORDER_COLORS = ["#c9e8ff", "#9ecae1", "#6baed6", "#4292c6",
                "#2171b5", "#08519c", "#e6550d", "#a63603"]  # 1..8 (7,8 warm)


# ── Build the real grid once ──────────────────────────────────────────────────
def build():
    import config
    from tools.runners.common import apply_output_dir
    from hydroflow.core.routing import router as kwr
    from hydroflow.core import routing as ru

    apply_output_dir(config, LEAF)
    config.BACKEND = "cpu"
    config.CHANNEL_ROUTING = True
    gd = kwr.initialise_grid(config)

    to_np = lambda a: a.get() if hasattr(a, "get") else np.asarray(a)
    s_rows = to_np(gd["s_rows"]).astype(int)
    s_cols = to_np(gd["s_cols"]).astype(int)
    faccum = to_np(gd["faccum_1d"]).astype(float)
    ds_idx = to_np(gd["ds_idx"]).astype(int)
    n      = int(gd["n_cells"])
    csz    = float(gd["cell_size"])
    order  = ru.compute_strahler_order(ds_idx, n).astype(int)

    wtab = config.CHANNEL_WIDTH_BY_ORDER
    max_o = max(wtab)
    width = np.array([wtab.get(min(o, max_o), csz) for o in order], float)

    return dict(s_rows=s_rows, s_cols=s_cols, faccum=faccum, order=order,
                width=width, n=n, csz=csz, wtab=wtab, max_o=max_o)


def to_2d(g, vals, rmin, cmin, H, W, fill=np.nan):
    a = np.full((H, W), fill, float)
    a[g["s_rows"] - rmin, g["s_cols"] - cmin] = vals
    return a


# ── Figure 1: order + width maps ──────────────────────────────────────────────
def fig_orders(g, bbox):
    rmin, rmax, cmin, cmax, H, W = bbox
    thr = max(1, g["n"] // 100)
    mask = g["faccum"] > thr
    order2d = to_2d(g, np.where(mask, g["order"], np.nan), rmin, cmin, H, W)
    width2d = to_2d(g, np.where(mask, g["width"], np.nan), rmin, cmin, H, W)
    extent2d = to_2d(g, np.ones(g["n"]), rmin, cmin, H, W)

    cmap = ListedColormap(ORDER_COLORS)
    norm = BoundaryNorm(np.arange(0.5, 9.5, 1), cmap.N)

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    for ax in axes:
        ax.imshow(extent2d, cmap=ListedColormap(["#eeeeee"]), interpolation="none")
        ax.set_xticks([]); ax.set_yticks([])
    im0 = axes[0].imshow(order2d, cmap=cmap, norm=norm, interpolation="none")
    axes[0].set_title(f"Channel mask — Strahler order\n"
                      f"{int(mask.sum()):,} of {g['n']:,} cells (faccum > {thr})",
                      fontweight="bold")
    cb0 = fig.colorbar(im0, ax=axes[0], fraction=0.045, pad=0.02, ticks=range(1, 9))
    cb0.set_label("Strahler order")

    im1 = axes[1].imshow(width2d, cmap="plasma", interpolation="none")
    axes[1].set_title("Assigned channel width B [m]\n(cell size = "
                      f"{g['csz']:.0f} m)", fontweight="bold")
    fig.colorbar(im1, ax=axes[1], fraction=0.045, pad=0.02, label="width B (m)")

    fig.suptitle("How the channel mask works — high-flow-accumulation cells become "
                 "confined rivers", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT / "channel_mask_orders.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {(OUT / 'channel_mask_orders.png').relative_to(REPO_ROOT)}")


# ── Animation: mask grows as the threshold loosens ───────────────────────────
def gif_threshold(g, bbox):
    rmin, rmax, cmin, cmax, H, W = bbox
    fa = g["faccum"]
    order2d  = to_2d(g, g["order"], rmin, cmin, H, W)
    faccum2d = to_2d(g, fa, rmin, cmin, H, W)
    extent2d = to_2d(g, np.ones(g["n"]), rmin, cmin, H, W)

    # frames: top-% from very strict (0.1%) to loose (12%), then hold
    top_pcts = np.concatenate([np.linspace(0.1, 12, 44), np.full(6, 12)])
    cmap = ListedColormap(ORDER_COLORS)
    norm = BoundaryNorm(np.arange(0.5, 9.5, 1), cmap.N)

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(extent2d, cmap=ListedColormap(["#eeeeee"]), interpolation="none")
    im = ax.imshow(np.full_like(order2d, np.nan), cmap=cmap, norm=norm,
                   interpolation="none")
    ax.set_xticks([]); ax.set_yticks([])
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02, ticks=range(1, 9))
    cb.set_label("Strahler order")
    title = ax.set_title("", fontweight="bold", fontsize=13)

    def update(k):
        top = top_pcts[k]
        thr = np.percentile(fa, 100 - top)
        m = faccum2d > thr
        disp = np.where(m, order2d, np.nan)
        im.set_data(disp)
        n_ch = int((g["faccum"] > thr).sum())
        oo = g["order"][g["faccum"] > thr]
        orng = f"{oo.min()}–{oo.max()}" if oo.size else "—"
        title.set_text(f"Channel mask = top {top:.1f}% by flow accumulation\n"
                       f"{n_ch:,} channel cells   |   Strahler orders {orng}   "
                       f"(faccum > {thr:.0f})")
        return im, title

    ani = animation.FuncAnimation(fig, update, frames=len(top_pcts),
                                  blit=False, interval=140)
    out = OUT / "channel_mask_threshold.gif"
    ani.save(out, writer=animation.PillowWriter(fps=7), dpi=90)
    plt.close(fig)
    print(f"  → {out.relative_to(REPO_ROOT)}")


# ── Schematic: sheet vs channel vs over-wide cross-section ────────────────────
def fig_xsection(g):
    csz = g["csz"]
    V_per_len = 50.0            # fixed cross-section area [m²] for a fair comparison
    cases = [("Wide sheet", csz, "#6baed6"),
             ("Confined channel (order 6, B=28 m)", 28.0, "#08519c"),
             ("Over-wide B=300 m  (> cell!)", 300.0, "#a63603")]

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
    for ax, (name, B, col) in zip(axes, cases):
        depth = V_per_len / B
        # draw the 100 m cell footprint
        ax.add_patch(Rectangle((0, -0.2), csz, 0.2, fc="#dddddd", ec="k", lw=1.2))
        ax.text(csz / 2, -0.32, f"DEM cell = {csz:.0f} m", ha="center", va="top",
                fontsize=10)
        # draw the water rectangle (width B, height depth), centred
        x0 = (csz - B) / 2
        ax.add_patch(Rectangle((x0, 0), B, depth, fc=col, ec="k", lw=1.2, alpha=0.85))
        ax.plot([x0, x0 + B], [depth, depth], color=col, lw=2)
        # cell boundary lines
        ax.axvline(0, color="k", ls=":", lw=1)
        ax.axvline(csz, color="k", ls=":", lw=1)
        over = B > csz
        ax.set_title(name + ("\n⚠ exceeds cell width" if over else ""),
                     fontweight="bold", color="#a63603" if over else "k",
                     fontsize=11)
        ax.text(csz / 2, depth + 0.12,
                f"B = {B:.0f} m   depth = {depth:.2f} m\n"
                f"R = A/P = {(B*depth)/(B+2*depth):.2f} m",
                ha="center", va="bottom", fontsize=10)
        lo = min(x0, 0) - 20; hi = max(x0 + B, csz) + 20
        ax.set_xlim(lo, hi); ax.set_ylim(-0.5, 2.6)
        ax.set_xlabel("width (m)"); ax.set_ylabel("depth (m)")
        ax.set_aspect("auto")
    fig.suptitle("Same water volume, three widths — why B must stay ≤ cell size\n"
                 "(confining to B<cell → deeper, faster;  B>cell → shallower than a "
                 "sheet AND spills outside the cell)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(OUT / "channel_xsection.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {(OUT / 'channel_xsection.png').relative_to(REPO_ROOT)}")


def gif_strahler(g, bbox):
    """Animate the Strahler COMPUTATION on the STREAM NETWORK: reveal stream
    cells in upstream-first order (ascending drainage area = a valid topological
    order), coloured by final order.  You watch the order-1/2 tributary tips
    light up first and the order CLIMB at every confluence down to the outlet.

    Hillslope cells (order 1-2, ~80% of the basin) are hidden so the merging is
    legible — the visible tips are still fed by them, just off-screen."""
    rmin, rmax, cmin, cmax, H, W = bbox
    order = g["order"].astype(int)
    fa    = g["faccum"]
    n     = g["n"]

    # Stream network = top ~4% by drainage area (a fuller tree than the 1% mask).
    thr    = np.percentile(fa, 96)
    stream = fa > thr
    idx    = np.where(stream)[0]

    # upstream-first reveal: order stream cells by ascending drainage area
    order_s = np.argsort(fa[idx], kind="stable")     # positions into idx
    nframes = 60
    appear_full = np.full(n, np.inf)
    appear_full[idx[order_s]] = (np.arange(idx.size) * nframes) // idx.size

    order2d  = to_2d(g, np.where(stream, order, np.nan), rmin, cmin, H, W)
    appear2d = to_2d(g, appear_full, rmin, cmin, H, W, fill=np.inf)
    net2d    = to_2d(g, np.where(stream, 1.0, np.nan), rmin, cmin, H, W)
    extent2d = to_2d(g, np.ones(n), rmin, cmin, H, W)

    cmap = ListedColormap(ORDER_COLORS)
    norm = BoundaryNorm(np.arange(0.5, 9.5, 1), cmap.N)

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(extent2d, cmap=ListedColormap(["#f2f2f2"]), interpolation="none")
    ax.imshow(net2d, cmap=ListedColormap(["#d9d9d9"]), interpolation="none")  # faint full tree
    im = ax.imshow(np.full_like(order2d, np.nan), cmap=cmap, norm=norm,
                   interpolation="none")
    ax.set_xticks([]); ax.set_yticks([])
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02, ticks=range(1, 9))
    cb.set_label("Strahler order")
    title = ax.set_title("", fontweight="bold", fontsize=13)

    frames = list(range(nframes)) + [nframes - 1] * 8
    def update(k):
        im.set_data(np.where(appear2d <= k, order2d, np.nan))
        rev = appear_full <= k
        frac = 100.0 * rev.sum() / idx.size
        cur_max = int(order[rev].max()) if rev.any() else 0
        title.set_text("Strahler order — computed upstream-first (tips → outlet)\n"
                       f"{frac:3.0f}% of stream cells done   |   "
                       f"highest order so far: {cur_max}")
        return im, title

    ani = animation.FuncAnimation(fig, update, frames=frames, blit=False,
                                  interval=120)
    out = OUT / "strahler_computation.gif"
    ani.save(out, writer=animation.PillowWriter(fps=8), dpi=90)
    plt.close(fig)
    print(f"  → {out.relative_to(REPO_ROOT)}")


# ── Single clean 1%-threshold network (presentation slide) ────────────────────
def fig_1pct(g, bbox, frac_pct=1.0):
    rmin, rmax, cmin, cmax, H, W = bbox
    thr  = max(1, int(round(g["n"] * frac_pct / 100.0)))   # CHANNEL_FACCUM_THRESHOLD
    mask = g["faccum"] > thr
    order2d  = to_2d(g, np.where(mask, g["order"], np.nan), rmin, cmin, H, W)
    extent2d = to_2d(g, np.ones(g["n"]), rmin, cmin, H, W)
    cmap = ListedColormap(ORDER_COLORS)
    norm = BoundaryNorm(np.arange(0.5, 9.5, 1), cmap.N)

    fig, ax = plt.subplots(figsize=(9, 10))
    ax.imshow(extent2d, cmap=ListedColormap(["#eeeeee"]), interpolation="none")
    im = ax.imshow(order2d, cmap=cmap, norm=norm, interpolation="none")
    ax.set_xticks([]); ax.set_yticks([])
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02, ticks=range(1, 9))
    cb.set_label("Strahler order")
    oo, ww = g["order"][mask], g["width"][mask]
    box = (f"CHANNEL_FACCUM_THRESHOLD = {thr}  (= {frac_pct:g}% of basin area)\n"
           f"channel cells: {int(mask.sum()):,} / {g['n']:,}  ({100*mask.mean():.1f}%)\n"
           f"Strahler orders: {oo.min()}–{oo.max()}\n"
           f"widths: {ww.min():.0f}–{ww.max():.0f} m   (cell = {g['csz']:.0f} m)")
    ax.text(0.02, 0.02, box, transform=ax.transAxes, va="bottom", ha="left",
            fontsize=11, family="monospace",
            bbox=dict(boxstyle="round", fc="white", ec="grey", alpha=0.92))
    ax.set_title(f"Channel network — drainage-area threshold {frac_pct:g}%",
                 fontweight="bold", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT / "channel_1pct.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {(OUT / 'channel_1pct.png').relative_to(REPO_ROOT)}")


# ── Threshold comparison: 0.5% / 1% / 5% (how the network changes) ────────────
def fig_threshold_compare(g, bbox, tops=(0.25, 1.0, 5.0)):
    rmin, rmax, cmin, cmax, H, W = bbox
    extent2d = to_2d(g, np.ones(g["n"]), rmin, cmin, H, W)
    cmap = ListedColormap(ORDER_COLORS)
    norm = BoundaryNorm(np.arange(0.5, 9.5, 1), cmap.N)
    fig, axes = plt.subplots(1, len(tops), figsize=(6 * len(tops), 7))
    im = None
    for ax, top in zip(axes, tops):
        thr  = max(1, int(round(g["n"] * top / 100.0)))
        mask = g["faccum"] > thr
        o2d  = to_2d(g, np.where(mask, g["order"], np.nan), rmin, cmin, H, W)
        ax.imshow(extent2d, cmap=ListedColormap(["#eeeeee"]), interpolation="none")
        im = ax.imshow(o2d, cmap=cmap, norm=norm, interpolation="none")
        ax.set_xticks([]); ax.set_yticks([])
        oo, ww = g["order"][mask], g["width"][mask]
        ax.set_title(f"threshold {top:g}%   (faccum > {thr})\n"
                     f"{int(mask.sum()):,} cells · orders {oo.min()}–{oo.max()} · "
                     f"B {ww.min():.0f}–{ww.max():.0f} m", fontsize=11, fontweight="bold")
    cb = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02, ticks=range(1, 9))
    cb.set_label("Strahler order")
    fig.suptitle("How the flow-accumulation threshold sets the river network\n"
                 "↑ threshold → only the largest rivers (higher order, wider, fewer) · "
                 "↓ threshold → finer tributaries included (lower order)",
                 fontsize=13, fontweight="bold")
    fig.savefig(OUT / "channel_threshold_compare.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {(OUT / 'channel_threshold_compare.png').relative_to(REPO_ROOT)}")


# ── Stepped GIF cycling the specific thresholds ──────────────────────────────
def gif_threshold_steps(g, bbox, tops=(0.25, 1.0, 2.0, 5.0), holds=12):
    rmin, rmax, cmin, cmax, H, W = bbox
    order2d  = to_2d(g, g["order"].astype(float), rmin, cmin, H, W)
    faccum2d = to_2d(g, g["faccum"], rmin, cmin, H, W)
    extent2d = to_2d(g, np.ones(g["n"]), rmin, cmin, H, W)
    cmap = ListedColormap(ORDER_COLORS)
    norm = BoundaryNorm(np.arange(0.5, 9.5, 1), cmap.N)

    seq = [t for t in tops for _ in range(holds)]
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(extent2d, cmap=ListedColormap(["#eeeeee"]), interpolation="none")
    im = ax.imshow(np.full_like(order2d, np.nan), cmap=cmap, norm=norm,
                   interpolation="none")
    ax.set_xticks([]); ax.set_yticks([])
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02, ticks=range(1, 9))
    cb.set_label("Strahler order")
    title = ax.set_title("", fontweight="bold", fontsize=13)

    def update(k):
        top = seq[k]
        thr = max(1, int(round(g["n"] * top / 100.0)))
        m = g["faccum"] > thr
        im.set_data(np.where(faccum2d > thr, order2d, np.nan))
        oo = g["order"][m]
        title.set_text(f"CHANNEL_FACCUM_THRESHOLD = {top:g}% of basin area\n"
                       f"{int(m.sum()):,} channel cells   |   Strahler orders "
                       f"{oo.min()}–{oo.max()}   (faccum > {thr})")
        return im, title

    ani = animation.FuncAnimation(fig, update, frames=len(seq), blit=False, interval=90)
    out = OUT / "channel_threshold_steps.gif"
    ani.save(out, writer=animation.PillowWriter(fps=10), dpi=90)
    plt.close(fig)
    print(f"  → {out.relative_to(REPO_ROOT)}")


# ── How depth is provided: depth = V / store_area ────────────────────────────
def fig_depth(g):
    csz = g["csz"]
    widths = [("Wide sheet", csz, "#6baed6"),
              ("Order 5  (B=18 m)", 18, "#4292c6"),
              ("Order 6  (B=28 m)", 28, "#08519c"),
              ("Order 7  (B=45 m)", 45, "#e6550d"),
              ("Order 8  (B=70 m)", 70, "#a63603")]
    L = csz                                   # reach length through the cell ≈ cell size
    V = np.linspace(0, 20000, 300)            # stored water volume in the cell [m³]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.2))

    # Panel A: depth-from-volume curves
    for lab, B, col in widths:
        ax1.plot(V, V / (B * L), color=col, lw=2.4, label=lab)
    Vref = 8000
    ax1.axvline(Vref, color="grey", ls="--", lw=1)
    ax1.set_xlabel("water stored in the cell,  V  [m³]")
    ax1.set_ylabel("flow depth  h  [m]")
    ax1.set_title(r"$h = \dfrac{V}{\mathrm{store\_area}}$,   "
                  r"store_area $= B\cdot L$  (channel)  or  $A_{cell}$  (sheet)",
                  fontsize=12)
    ax1.legend(frameon=False); ax1.grid(True, ls="--", alpha=0.3); ax1.set_ylim(0, 8)

    # Panel B: cross-sections at Vref → same volume, different depths
    ax2.add_patch(Rectangle((-csz/2, -0.25), csz, 0.25, fc="#dddddd", ec="k", lw=1))
    ax2.text(0, -0.42, f"one DEM cell ({csz:.0f} m wide)", ha="center", va="top", fontsize=9)
    for lab, B, col in widths:
        h = Vref / (B * L)
        ax2.add_patch(Rectangle((-B/2, 0), B, h, fc=col, ec="k", lw=1, alpha=0.8))
        ax2.text(0, h + 0.08, f"{lab.split()[0]} {lab.split()[1] if 'sheet' not in lab else ''}"
                              f"\nB={B:.0f} m → h={h:.2f} m",
                 ha="center", va="bottom", fontsize=8.5, color=col, fontweight="bold")
    ax2.axvline(-csz/2, color="k", ls=":", lw=1); ax2.axvline(csz/2, color="k", ls=":", lw=1)
    ax2.set_xlim(-csz/2 - 15, csz/2 + 15); ax2.set_ylim(-0.6, 5.5)
    ax2.set_xlabel("width (m)"); ax2.set_ylabel("depth h (m)")
    ax2.set_title(f"Same stored volume (V = {Vref:,} m³) → narrower B → deeper h",
                  fontsize=12)

    fig.suptitle("How flow depth is provided — confinement makes channel cells deeper "
                 "(→ higher Manning velocity & wave celerity)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT / "channel_depth.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {(OUT / 'channel_depth.png').relative_to(REPO_ROOT)}")


def gif_order_to_channels(g, bbox, final_pct=1.0):
    """Three-act narrative GIF:
      ① Strahler order on EVERY pixel — the full field, orders 1..max (NOT capped
         at 8; the colormap auto-sizes to the basin's true max order).
      ② Raise the flow-accumulation threshold — hillslopes (low order / low faccum)
         fade to grey as the cut climbs.
      ③ What remains is the channel network, coloured by order (+ widths)."""
    rmin, rmax, cmin, cmax, H, W = bbox
    order = g["order"].astype(int)
    fa    = g["faccum"]
    n     = g["n"]
    max_o = int(order.max())

    # Colormap that is NOT limited to 8: reuse the 8 house colours if the basin
    # tops out ≤8, else sample a continuous ramp so orders 9,10,… get their own hue.
    if max_o <= len(ORDER_COLORS):
        colors = ORDER_COLORS[:max_o]
    else:
        base = plt.get_cmap("turbo")
        colors = [base(i / (max_o - 1)) for i in range(max_o)]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(0.5, max_o + 1.5, 1), cmap.N)

    order2d  = to_2d(g, order.astype(float), rmin, cmin, H, W)
    faccum2d = to_2d(g, fa, rmin, cmin, H, W)
    extent2d = to_2d(g, np.ones(n), rmin, cmin, H, W)
    wtab = g["wtab"]; max_key = max(wtab)

    final_thr = max(1, int(round(n * final_pct / 100.0)))
    plan  = [(0.0, "all")] * 10
    plan += [(t, "rise") for t in np.logspace(0, np.log10(final_thr), 34)]
    plan += [(float(final_thr), "final")] * 14

    fig, ax = plt.subplots(figsize=(9, 9.4))
    ax.imshow(extent2d, cmap=ListedColormap(["#eeeeee"]), interpolation="none")
    im = ax.imshow(order2d, cmap=cmap, norm=norm, interpolation="none")
    ax.set_xticks([]); ax.set_yticks([])
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02, ticks=range(1, max_o + 1))
    cb.set_label(f"Strahler order  (this basin 1–{max_o}; unbounded in general)")
    title = ax.set_title("", fontweight="bold", fontsize=12)

    def update(k):
        thr, phase = plan[k]
        im.set_data(np.where(faccum2d > thr, order2d, np.nan))
        mask = fa > thr
        oo = order[mask]
        rng = f"{oo.min()}–{oo.max()}" if oo.size else "—"
        if phase == "all":
            title.set_text("① Strahler order on EVERY pixel\n"
                           f"all {n:,} cells · orders 1–{max_o} · order 1 = hillslopes (68%)")
        elif phase == "rise":
            title.set_text("② Limiting by flow accumulation\n"
                           f"keep faccum > {thr:.0f}  →  {int(mask.sum()):,} cells · orders {rng}")
        else:
            ww = g["width"][mask]
            title.set_text("③ These are the channels + their orders\n"
                           f"faccum > {thr:.0f} ({final_pct:g}%) · {int(mask.sum()):,} cells · "
                           f"orders {rng} · B {ww.min():.0f}–{ww.max():.0f} m")
        return im, title

    ani = animation.FuncAnimation(fig, update, frames=len(plan), blit=False, interval=110)
    out = OUT / "order_to_channels.gif"
    ani.save(out, writer=animation.PillowWriter(fps=9), dpi=90)
    plt.close(fig)
    print(f"  → {out.relative_to(REPO_ROOT)}")


def fig_full_vs_masked(g, bbox, frac_pct=0.25):
    """Left: Strahler order on EVERY pixel (the full computation, order 1 = hillslopes).
    Right: the SAME orders, but only channel cells (faccum > threshold) are shown —
    the threshold hides the low orders, it does not change any cell's order."""
    rmin, rmax, cmin, cmax, H, W = bbox
    thr  = max(1, int(round(g["n"] * frac_pct / 100.0)))
    mask = g["faccum"] > thr
    all2d    = to_2d(g, g["order"].astype(float), rmin, cmin, H, W)
    msk2d    = to_2d(g, np.where(mask, g["order"], np.nan), rmin, cmin, H, W)
    extent2d = to_2d(g, np.ones(g["n"]), rmin, cmin, H, W)
    cmap = ListedColormap(ORDER_COLORS)
    norm = BoundaryNorm(np.arange(0.5, 9.5, 1), cmap.N)

    fig, axes = plt.subplots(1, 2, figsize=(15, 8))
    for ax in axes:
        ax.imshow(extent2d, cmap=ListedColormap(["#f2f2f2"]), interpolation="none")
        ax.set_xticks([]); ax.set_yticks([])
    im = axes[0].imshow(all2d, cmap=cmap, norm=norm, interpolation="none")
    o = g["order"]
    axes[0].set_title("Strahler order on EVERY pixel\n"
                      f"computed over all {g['n']:,} cells → orders {o.min()}–{o.max()}  "
                      f"(order 1 = 68% of basin = hillslopes)", fontsize=11, fontweight="bold")
    axes[1].imshow(msk2d, cmap=cmap, norm=norm, interpolation="none")
    oo = g["order"][mask]
    axes[1].set_title(f"Only channel cells shown  (faccum > {thr}, {frac_pct:g}%)\n"
                      f"same orders — low ones just hidden → displayed range {oo.min()}–{oo.max()}",
                      fontsize=11, fontweight="bold")
    cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02, ticks=range(1, 9))
    cb.set_label("Strahler order")
    fig.suptitle("The threshold does NOT change any order — it only chooses which pixels "
                 "are drawn as 'channel'", fontsize=13, fontweight="bold")
    fig.savefig(OUT / "strahler_full_vs_masked.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {(OUT / 'strahler_full_vs_masked.png').relative_to(REPO_ROOT)}")


def main():
    g = build()
    pad = 8
    rmin = max(0, g["s_rows"].min() - pad); rmax = g["s_rows"].max() + pad + 1
    cmin = max(0, g["s_cols"].min() - pad); cmax = g["s_cols"].max() + pad + 1
    bbox = (rmin, rmax, cmin, cmax, rmax - rmin, cmax - cmin)
    print("Rendering channel-mask explainers:")
    fig_orders(g, bbox)
    fig_xsection(g)
    gif_threshold(g, bbox)
    gif_strahler(g, bbox)
    print("Done.")


if __name__ == "__main__":
    main()
