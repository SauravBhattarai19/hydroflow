#!/usr/bin/env python
"""
run_vsa_param_sweep.py
=======================
VSA-sandbox PARAMETER sensitivity sweep for the 100 m OPM model.  Companion to
tools/run_combinations.py (which sweeps channel/scheme/mechanism/infiltration
axes); this script instead holds channel + infiltration fixed and sweeps the
three OPM sandbox scalars — SD_min, OPM_K_SAT, SD_max — across routing scheme
and a focused set of runoff-mechanism subsets.

Axes (edit the lists below to add/trim)
---------------------------------------
    SCHEMES      = ['kinematic', 'diffusive', 'muskingum']
    MECH_SUBSETS = ['vsa'], ['vsa+impervious'], ['vsa+horton'], ['horton']
    SD_MIN_OPTIONS  = [0.1, 0.001, 0.00001]                 m  (OPM_SD_MIN — trimmed
                       from an original 5-value list to 3, still spanning the
                       full 5-order-of-magnitude range)
    KSAT_OPTIONS    = [4.4, 0.44, 0.044]                    m/day  (OPM_K_SAT, the
                       sandbox's lateral Darcy conductivity — NOT the Green-Ampt
                       vertical Ksat used by Horton.  Trimmed from an original
                       4-value list; the calibrated default 44.0 is excluded here)
    SD_MAX_OPTIONS  = [0.1, 0.3, 0.5]                       m  (OPM_SD_MAX_INITIAL —
                       trimmed from an original 5-value list to 3)

Fixed for every run in this study:
    CHANNEL_ROUTING = True           (channel cross-section routing always on)
    OPM_SD_SOURCE   = 'manual'       (forced — otherwise SERVES/GEE would silently
                                       override SD_max/phi and the SD_max axis above
                                       would have zero effect on the result)
    OPM_INFILTRATION = 'green_ampt'  (sandbox recharge cap on; also what makes
                                       Horton's own infiltration-excess active)

Why 'horton'-only skips the SD_min/Ksat/SD_max cross-product
--------------------------------------------------------------
When 'vsa' is absent from RUNOFF_MECHANISMS the OPM sandbox is never built
(hydroflow/core/runoff/vsa.py: `if not self._vsa_on: ... return`) — the VSA mask
stays permanently empty.  SD_min, OPM_K_SAT and SD_max only affect that sandbox,
so for the 'horton'-only subset all 27 (3×3×3) combinations would produce an
IDENTICAL result.  This script runs 'horton'-only once per routing scheme
instead of 27 duplicate times.

    3 vsa-containing subsets × 3(SD_min) × 3(Ksat) × 3(SD_max) × 3(scheme) = 243
  + 1 horton-only subset     × 3(scheme)                                  =   3
    ────────────────────────────────────────────────────────────────────────
    246 configs × 4 floods = 984 router runs.

At ~5 min/config (4 floods, observed on this basin's 100 m grid on GPU) that's
roughly 20 hours of serial wall time, or ~7 hours split across 3 concurrent
shards (--shard I/N below) — run it under tmux/nohup and let it resume across
sessions (already-complete configs are skipped; see --force).

Output tree (under outputs collection/vsa_param_sweep_100m/)
--------------------------------------------------------------
    _shared/                                       cached rasters (seeded once)
    kinematic/vsa/sdmin_0.001/ksat_44.0/sdmax_0.1/   ← one config = 4 floods
        hydrograph_<event>.csv  (×4)
        comparison_<event>.png  (×4)
        summary_all_floods.csv
        mass_balance.csv
        partition_<event>.csv
    kinematic/horton/                                ← no sdmin/ksat/sdmax suffix
    ...
    master_summary.csv             ← EVERY (config × flood) row, one table
    run_log.csv                    ← per-config status + wall time

Usage
-----
    python tools/run_vsa_param_sweep.py                 # run everything (resumable)
    python tools/run_vsa_param_sweep.py --list          # list configs, run nothing
    python tools/run_vsa_param_sweep.py --force         # re-run completed configs too
    python tools/run_vsa_param_sweep.py --aggregate     # only rebuild master_summary.csv
    python tools/run_vsa_param_sweep.py kinematic vsa+horton   # only matching leaf paths

Running several configs at once (GPU has headroom — one config uses ~30% of
the card, so 3 concurrent processes fit comfortably)
------------------------------------------------------------------------------
Each config writes to its own leaf folder and is independently resumable, so
N processes can each take a disjoint round-robin slice of the sweep with
--shard I/N (I = 0..N-1).  Launch each in its own tmux window/session:

    python tools/run_vsa_param_sweep.py --shard 0/3
    python tools/run_vsa_param_sweep.py --shard 1/3
    python tools/run_vsa_param_sweep.py --shard 2/3

Each shard writes its own run_log_shard{I}of{N}.csv (a shared run_log.csv
would have the last-finishing shard clobber the others' history); all shards
read/write the SAME master_summary.csv, which is safe because aggregate()
does an atomic temp-file + os.replace rather than truncating in place.
"""

import os
import sys
import time
import shutil
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
os.chdir(REPO_ROOT)                      # model uses paths relative to repo root
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
from tools.runners import gauge


# ══════════════════════════════════════════════════════════════════════════════
# Axes — EDIT HERE to change what gets run
# ══════════════════════════════════════════════════════════════════════════════
SCHEMES = ['kinematic', 'diffusive', 'muskingum']   # routing scheme

# Only these 4 mechanism subsets (not the full 7 — see run_combinations.py for that).
MECH_SUBSETS = [
    ['vsa'],
    ['vsa', 'impervious'],
    ['vsa', 'horton'],
    ['horton'],
]

# (value, folder-label) pairs — explicit labels avoid float-formatting surprises
# in path names (e.g. 1e-05 vs 0.00001) and keep the tree human-sortable.
SD_MIN_OPTIONS = [
    (0.1,     '0.1'),
    (0.001,   '0.001'),
    (0.00001, '0.00001'),
]
KSAT_OPTIONS = [
    (4.4,   '4.4'),
    (0.44,  '0.44'),
    (0.044, '0.044'),
]
SD_MAX_OPTIONS = [
    (0.1, '0.1'),
    (0.3, '0.3'),
    (0.5, '0.5'),
]

# Fixed knobs (held constant across every combo in this study).
DIFFUSION_THETA  = 1.0            # only used when scheme == 'diffusive'
CHANNEL_ROUTING  = True           # "chan on" for every run — not a swept axis here
OPM_SD_SOURCE    = 'manual'       # forced so OPM_SD_MAX_INITIAL/OPM_SD_MIN actually apply
OPM_INFILTRATION = 'green_ampt'   # sandbox recharge cap on; enables Horton's f_p too


# ══════════════════════════════════════════════════════════════════════════════
# Paths
# ══════════════════════════════════════════════════════════════════════════════
ROOT   = "outputs collection/vsa_param_sweep_100m"
SHARED = REPO_ROOT / ROOT / "_shared"

# Reuse the already-seeded shared cache from the channel/scheme/mechanism sweep
# (same watershed, same 4 flood events) — no DEM re-processing, no GEE re-download.
BASELINE_SRC = REPO_ROOT / "outputs collection/combinations_100m/_shared"

SEED_FILES = [
    "clipped_dem.tif", "flow_direction.tif", "clipped_flow_accumulation.tif",
    "watershed.tif", "watershed.geojson",
    "ksat_hihydro.tif", "lulc_mannings_lcz.tif", "texture_sandclay.tif",
]


# ══════════════════════════════════════════════════════════════════════════════
# Config enumeration + naming
# ══════════════════════════════════════════════════════════════════════════════
_MECH_TOKEN = {'vsa': 'vsa', 'horton': 'horton', 'impervious': 'imperv'}
_CANON      = ['vsa', 'horton', 'impervious']   # canonical order for names


def _mech_name(mset):
    return '+'.join(_MECH_TOKEN[m] for m in _CANON if m in mset)


def _leaf(scheme, mset, sd_min_lbl=None, ksat_lbl=None, sd_max_lbl=None):
    base = f"{scheme}/{_mech_name(mset)}"
    if 'vsa' in mset:
        base = f"{base}/sdmin_{sd_min_lbl}/ksat_{ksat_lbl}/sdmax_{sd_max_lbl}"
    return base


def _base_overrides(scheme, mset):
    return {
        'CHANNEL_ROUTING':   CHANNEL_ROUTING,
        'ROUTING_SCHEME':    scheme,
        'DIFFUSION_THETA':   DIFFUSION_THETA,
        'RUNOFF_MECHANISMS': list(mset),
        'OPM_SD_SOURCE':     OPM_SD_SOURCE,
        'OPM_INFILTRATION':  OPM_INFILTRATION,
        'IMPERVIOUS_SOURCE': 'lcz' if 'impervious' in mset else 'none',
    }


def all_configs():
    """Return [(leaf_path, overrides_dict, meta_dict), ...] for the full product."""
    configs = []
    for scheme in SCHEMES:
        for mset in MECH_SUBSETS:
            if 'vsa' not in mset:
                # 'horton'-only: SD_min/Ksat/SD_max are no-ops (sandbox never
                # initializes without 'vsa') — one run per scheme, no suffix.
                leaf = _leaf(scheme, mset)
                overrides = _base_overrides(scheme, mset)
                meta = {
                    'scheme': scheme, 'mechanisms': _mech_name(mset),
                    'sd_min': None, 'ksat': None, 'sd_max': None,
                }
                configs.append((leaf, overrides, meta))
                continue

            for sd_min_val, sd_min_lbl in SD_MIN_OPTIONS:
                for ksat_val, ksat_lbl in KSAT_OPTIONS:
                    for sd_max_val, sd_max_lbl in SD_MAX_OPTIONS:
                        leaf = _leaf(scheme, mset, sd_min_lbl, ksat_lbl, sd_max_lbl)
                        overrides = _base_overrides(scheme, mset)
                        overrides.update({
                            'OPM_SD_MIN':         sd_min_val,
                            'OPM_K_SAT':          ksat_val,
                            'OPM_SD_MAX_INITIAL': sd_max_val,
                        })
                        meta = {
                            'scheme': scheme, 'mechanisms': _mech_name(mset),
                            'sd_min': sd_min_val, 'ksat': ksat_val, 'sd_max': sd_max_val,
                        }
                        configs.append((leaf, overrides, meta))
    return configs


# ══════════════════════════════════════════════════════════════════════════════
# Raster seeding (no DEM re-processing, no GEE re-download)
# ══════════════════════════════════════════════════════════════════════════════
def _shared_raster_list():
    return [BASELINE_SRC / f for f in SEED_FILES if (BASELINE_SRC / f).exists()]


def seed_shared():
    SHARED.mkdir(parents=True, exist_ok=True)

    already = all((SHARED / f).exists() for f in SEED_FILES)
    if already:
        print(f"  _shared already seeded ({len(SEED_FILES)} static rasters) "
              f"→ {SHARED.relative_to(REPO_ROOT)}")
        return

    src = _shared_raster_list()
    if not src:
        sys.exit(f"[ERROR] No cached rasters found in {BASELINE_SRC}.\n"
                 f"        Run tools/run_combinations.py once first (it seeds "
                 f"the watershed + GEE Ksat/LCZ/texture rasters this sweep reuses).")
    for f in src:
        dst = SHARED / f.name
        if not dst.exists():
            shutil.copy2(f, dst)
    print(f"  _shared seeded with {len(src)} rasters → {SHARED.relative_to(REPO_ROOT)}")


def seed_leaf(leaf_path: Path):
    leaf_path.mkdir(parents=True, exist_ok=True)
    for f in SHARED.iterdir():
        dst = leaf_path / f.name
        if not dst.exists():
            shutil.copy2(f, dst)


def is_done(leaf_path: Path) -> bool:
    """A config is 'done' once the gauge pipeline wrote its grand summary."""
    return (leaf_path / "summary_all_floods.csv").exists()


# ══════════════════════════════════════════════════════════════════════════════
# Master aggregation: every (config × flood) row in one table
# ══════════════════════════════════════════════════════════════════════════════
def aggregate():
    """Walk every leaf, join its per-flood metrics with the mass-balance partition,
    and write ROOT/master_summary.csv."""
    rows = []
    for leaf, _ov, meta in all_configs():
        leaf_path = REPO_ROOT / ROOT / leaf
        summ = leaf_path / "summary_all_floods.csv"
        if not summ.exists():
            continue
        df = pd.read_csv(summ)

        mb_path = leaf_path / "mass_balance.csv"
        mb = None
        if mb_path.exists():
            mb = pd.read_csv(mb_path).drop_duplicates('run_tag', keep='last') \
                   .set_index('run_tag')

        for _, r in df.iterrows():
            ev  = str(r['event'])
            row = {
                'scheme':      meta['scheme'],
                'mechanisms':  meta['mechanisms'],
                'sd_min':      meta['sd_min'],
                'ksat':        meta['ksat'],
                'sd_max':      meta['sd_max'],
                'event':       ev,
                'nse':         r.get('nse'),
                'pbias_pct':   r.get('pbias_pct'),
                'obs_peak_Q':  r.get('obs_peak_Q'),
                'mod_peak_Q':  r.get('mod_peak_Q'),
                'obs_peak_hr': r.get('obs_peak_hr'),
                'mod_peak_hr': r.get('mod_peak_hr'),
            }
            if mb is not None and ev in mb.index:
                m = mb.loc[ev]
                for col in ('runoff_ratio', 'dunne_frac', 'horton_frac',
                            'imperv_frac', 'rel_error'):
                    if col in mb.columns:
                        row[col] = m[col]
            rows.append(row)

    if not rows:
        print("  [aggregate] no completed configs yet — nothing to write.")
        return
    out = REPO_ROOT / ROOT / "master_summary.csv"
    cols = ['scheme', 'mechanisms', 'sd_min', 'ksat', 'sd_max', 'event',
            'nse', 'pbias_pct', 'obs_peak_Q', 'mod_peak_Q', 'obs_peak_hr', 'mod_peak_hr',
            'runoff_ratio', 'dunne_frac', 'horton_frac', 'imperv_frac', 'rel_error']
    df = pd.DataFrame(rows)
    df = df[[c for c in cols if c in df.columns]]
    # Atomic write (temp + os.replace) — several shards may call aggregate()
    # around the same time when run in parallel; a plain to_csv() truncates
    # the file first and could race with another shard's read/write.
    tmp = out.with_suffix(".csv.tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, out)
    print(f"\n  master_summary.csv → {out.relative_to(REPO_ROOT)}  ({len(df)} rows)")


# ══════════════════════════════════════════════════════════════════════════════
# Driver
# ══════════════════════════════════════════════════════════════════════════════
def run_study(filters=None, force=False, shard=None):
    """
    shard : (idx, total) or None.  When set, this process only runs the
    configs at positions idx, idx+total, idx+2*total, ... of the (filtered)
    list — a round-robin split so N processes (e.g. N tmux panes) can each
    take a disjoint 1/N slice of the sweep and run concurrently.  Safe because
    every config writes to its OWN leaf folder (seed_leaf/is_done are per-leaf)
    and GPU headroom allows several runs at once (~30% VRAM/compute each).
    """
    seed_shared()
    configs = all_configs()
    selected = [(leaf, ov, meta) for leaf, ov, meta in configs
                if not filters or all(f in leaf for f in filters)]
    if shard is not None:
        idx, total = shard
        selected = selected[idx::total]

    shard_tag = f"  [shard {shard[0]}/{shard[1]}]" if shard else ""
    print(f"\n{'='*70}\n  VSA PARAMETER SWEEP — {len(selected)} configs × 4 floods{shard_tag}"
          f"\n  axes: scheme{SCHEMES}  mechanisms={[_mech_name(m) for m in MECH_SUBSETS]}"
          f"\n        sd_min={[v for v, _ in SD_MIN_OPTIONS]}"
          f"  ksat={[v for v, _ in KSAT_OPTIONS]}"
          f"  sd_max={[v for v, _ in SD_MAX_OPTIONS]}"
          f"\n  (fixed: CHANNEL_ROUTING={CHANNEL_ROUTING}  OPM_SD_SOURCE={OPM_SD_SOURCE!r}"
          f"  OPM_INFILTRATION={OPM_INFILTRATION!r})"
          f"\n{'='*70}")

    log = []
    for i, (leaf, overrides, meta) in enumerate(selected, 1):
        leaf_path = REPO_ROOT / ROOT / leaf
        print(f"\n\n########## [{i}/{len(selected)}]  {leaf} ##########")
        if is_done(leaf_path) and not force:
            print("  already complete — skipping (use --force to re-run).")
            log.append((leaf, "skip", 0.0))
            continue
        key = {k: overrides[k] for k in
               ('ROUTING_SCHEME', 'RUNOFF_MECHANISMS',
                'OPM_SD_MIN', 'OPM_K_SAT', 'OPM_SD_MAX_INITIAL')
               if k in overrides}
        print(f"  overrides: {key}")
        seed_leaf(leaf_path)
        t0 = time.time()
        try:
            gauge.run(output_dir=f"{ROOT}/{leaf}/",
                      overrides=overrides,
                      skip_process_dem=True)
            status = "ok"
        except Exception as exc:                       # keep the sweep going
            status = f"FAIL: {exc}"
            traceback.print_exc()
        log.append((leaf, status, round(time.time() - t0, 1)))

    # ── per-config status log ────────────────────────────────────────────────
    # Shard-specific filename: concurrent shards must NOT share one run_log.csv
    # (each process only knows its own slice, so a shared file would have the
    # last-finishing shard's write clobber the others' history).
    (REPO_ROOT / ROOT).mkdir(parents=True, exist_ok=True)
    log_name = f"run_log_shard{shard[0]}of{shard[1]}.csv" if shard else "run_log.csv"
    pd.DataFrame(log, columns=['config', 'status', 'seconds']).to_csv(
        REPO_ROOT / ROOT / log_name, index=False)

    print(f"\n\n{'='*70}\n  SWEEP COMPLETE\n{'='*70}")
    for leaf, status, secs in log:
        print(f"  {status:>6}  {secs:8.1f}s  {leaf}")
    n_fail = sum(1 for _, s, _ in log if s.startswith("FAIL"))
    if n_fail:
        print(f"\n  [WARN] {n_fail} config(s) failed — see tracebacks above.")

    aggregate()
    print(f"\n  Done.  Re-run any time to resume; --aggregate rebuilds the summary.")


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--list" in args:
        for leaf, ov, _ in all_configs():
            tag = {k: ov[k] for k in
                   ('ROUTING_SCHEME', 'RUNOFF_MECHANISMS',
                    'OPM_SD_MIN', 'OPM_K_SAT', 'OPM_SD_MAX_INITIAL')
                   if k in ov}
            print(f"  {leaf:62s}  {tag}")
        print(f"\n  {len(all_configs())} configs × 4 floods.")
        sys.exit(0)
    if "--aggregate" in args:
        aggregate()
        sys.exit(0)
    force = "--force" in args

    # --shard I/N  or  --shard=I/N  — run only every Nth config, offset by I
    # (see run_study's docstring on shard= for why this is parallel-safe).
    shard = None
    filtered_args = []
    skip_next = False
    for i, a in enumerate(args):
        if skip_next:
            skip_next = False
            continue
        if a == "--shard":
            idx_s, total_s = args[i + 1].split("/")
            shard = (int(idx_s), int(total_s))
            skip_next = True
        elif a.startswith("--shard="):
            idx_s, total_s = a.split("=", 1)[1].split("/")
            shard = (int(idx_s), int(total_s))
        elif not a.startswith("--"):
            filtered_args.append(a)
    filters = filtered_args or None
    run_study(filters=filters, force=force, shard=shard)
