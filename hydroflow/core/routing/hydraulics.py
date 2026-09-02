# -*- coding: utf-8 -*-
"""
hydraulics.py — per-step flow kernels shared by both compute backends.

Manning velocity/discharge (sheet and confined rectangular channel),
the diffusive-wave (CASC2D/GSSHA-style water-surface-slope) discharge,
the volume flux limiter and the legacy uniform-rainfall array builder.
All array functions are backend-agnostic (NumPy or CuPy via the xp arg).
"""

import numpy as np




# ---------------------------------------------------------------------------
# 5.  Manning's equation (vectorised over all active cells)
# ---------------------------------------------------------------------------

def mannings_velocity(depth, slope, n):
    """
    V = (1/n) * depth^(2/3) * slope^(1/2)    [m/s]

    Parameters are 1-D arrays (one value per active cell).
    """
    return (1.0 / n) * (depth ** (2.0 / 3.0)) * (slope ** 0.5)


def cell_discharge(depth, velocity, cell_size):
    """
    Q = V * width * depth   [m³/s]
    Assumes wide rectangular cross-section → width ≈ cell_size.
    """
    return velocity * cell_size * depth


def mannings_discharge(depth, slope, n, width, chan_mask, cell_size, xp):
    """
    Kinematic Manning discharge with an optional CONFINED rectangular channel
    section on channel cells.

    Overland cells (``chan_mask`` False) use the wide-channel shortcut R ≈ depth
    and a flow width of ``cell_size`` — reproducing ``mannings_velocity`` followed
    by ``cell_discharge`` *bit-for-bit* (same arithmetic, same grouping).  Channel
    cells use a true rectangular cross-section of width ``B`` (≪ cell_size):

        A   = B · h                       (cross-section area)
        P   = B + 2·h                     (wetted perimeter)
        R   = A / P = B·h/(B+2h)          (hydraulic radius; < h when B is finite)
        Q   = (1/n) · R^(2/3) · √S · A    [m³/s]

    Confining the flow to ``B`` instead of spreading it across the whole DEM cell
    makes channel cells run deeper and faster (correct celerity / attenuation),
    which the wide-sheet ``R ≈ depth`` assumption cannot capture.

    Parameters
    ----------
    depth     : (n,) array – flow depth [m] (channel cells: depth over B·L footprint)
    slope     : (n,) array – friction slope [m/m]
    n         : scalar or (n,) array – Manning's n
    width     : (n,) array – flow width [m]: cell_size overland, B on channel cells
    chan_mask : (n,) bool array – True where the rectangular section applies
    cell_size : float – DEM cell size [m] (overland width)
    xp        : array module (numpy or cupy)

    Returns
    -------
    Q    : (n,) array [m³/s]  – Manning discharge (pre flux-limiter)
    A_xs : (n,) array [m²]    – flow cross-section area; celerity uses c = 5/3·Q/A_xs
    """
    # Overland (wide sheet): identical arithmetic to the original kinematic path.
    velocity   = (1.0 / n) * (depth ** (2.0 / 3.0)) * (slope ** 0.5)
    Q_overland = velocity * cell_size * depth
    A_overland = depth * cell_size

    # Channel (rectangular): true hydraulic radius R = A/P.
    A_chan = depth * width
    R_chan = A_chan / (width + 2.0 * depth)
    Q_chan = (1.0 / n) * (R_chan ** (2.0 / 3.0)) * (slope ** 0.5) * A_chan

    Q    = xp.where(chan_mask, Q_chan, Q_overland)
    A_xs = xp.where(chan_mask, A_chan, A_overland)
    return Q, A_xs


def diffusive_wave_discharge(depth, dem, dist, slope_bnd, n, ds_safe, valid_ds,
                             theta, cell_size, xp, min_depth, width, chan_mask):
    """
    CASC2D / GSSHA-style diffusion-wave cell discharge [m³/s].

    Replaces the pure-kinematic ``mannings_velocity``→``cell_discharge`` pair when
    ``ROUTING_SCHEME='diffusive'``.  The friction slope becomes the *water-surface*
    slope along the D8 flow path, which lets the wave attenuate (peak flattening) and
    slow under an adverse gradient — physics the bed-slope kinematic wave cannot capture.

        S_w    = slope_bnd  +  θ · (h_i − h_ds)/dist               (water-surface slope)
        S_eff  = max(S_w, 0)                                        (adverse grad → no flow)
        h_hb   = max(WSE_i, WSE_ds) − max(z_i, z_ds)               (depth over higher bed)
        h_flow = max( (1−θ)·h_i + θ·h_hb , min_depth )            (kinematic↔diffusion blend)
        Q      = (1/n) · h_flow^(5/3) · S_eff^(1/2) · cell_size

    θ blends BOTH the slope and the conveyance depth between the two coherent endpoints:
    θ=0 → own depth + bed slope = the kinematic scheme *exactly*; θ=1 → flow-depth-over-the-
    higher-bed + water-surface slope = the full CASC2D/GSSHA-style diffusion wave.  The bed
    term is the existing ``slope_bnd`` (= ``slope_1d`` = (z_i−z_ds)/dist, already floored at
    MIN_SLOPE with watershed-boundary handling), so steep cells get the true bed slope while
    flat cells keep draining; the depth-gradient term can still drive S_w below zero → the
    clamp reproduces backwater slowdown.  Cells with no valid downstream neighbour
    (``~valid_ds`` — the outlet and any cell draining off-mask) keep ``slope_bnd`` and their
    own depth, i.e. free outflow identical to the kinematic scheme.

    All arithmetic is via ``xp`` (NumPy or CuPy) so the helper runs on CPU and GPU alike.

    Parameters
    ----------
    depth     : (n,) array  – current flow depth per cell [m]
    dem       : (n,) array  – bed elevation per cell [m]
    dist      : (n,) array  – flow-path length to the downstream cell [m] (dx or dx·√2)
    slope_bnd : (n,) array  – bed slope (used only for ~valid_ds free-outflow cells)
    n         : scalar or (n,) array – Manning's n
    ds_safe   : (n,) int array – downstream index, clamped to 0 where invalid
    valid_ds  : (n,) bool array – True where the cell has a real downstream neighbour
    theta     : float – diffusion weight θ∈[0,1]
    cell_size : float – flow width ≈ cell size [m]
    xp        : array module (numpy or cupy)
    min_depth : float – wet/dry conveyance-depth floor [m]
    width     : (n,) array – flow width [m]: cell_size overland, channel width B on
                             channel cells (CONFINED rectangular conveyance).
    chan_mask : (n,) bool array – True where the rectangular section R=A/P applies.

    Returns
    -------
    Q_out  : (n,) array  [m³/s]  (NOT yet flux-limited — caller applies the CFL limiter)
    A_xs   : (n,) array  [m²]    conveyance cross-section area; celerity denominator
                                 (c = 5/3·Q/A_xs).  Overland: h_flow·cell_size.
    S_eff  : (n,) array  [m/m]   effective (clamped) friction slope used for Q.
    """
    depth_ds = depth[ds_safe]
    dem_ds   = dem[ds_safe]

    # Water-surface slope along the flow path: floored bed slope + θ depth-gradient term.
    S_w = slope_bnd + theta * (depth - depth_ds) / dist
    # Free-outflow cells (no downstream) fall back to the kinematic bed slope.
    S_eff = xp.where(valid_ds, S_w, slope_bnd)
    S_eff = xp.maximum(S_eff, 0.0)                       # adverse gradient → no discharge

    # Conveyance depth: blend own depth (kinematic) with flow-depth-over-the-higher-bed
    # (LISFLOOD-FP diffusion-wave convention) by the SAME θ as the slope, so the two terms
    # stay a coherent pair — θ=0 → own depth + bed slope = kinematic exactly; θ=1 → higher-
    # bed depth + water-surface slope = full diffusion wave.  In the normal downhill case
    # the higher-bed depth already equals the upstream cell's own depth.
    wse      = dem + depth
    wse_ds   = wse[ds_safe]
    h_higher = xp.maximum(wse, wse_ds) - xp.maximum(dem, dem_ds)
    h_flow   = (1.0 - theta) * depth + theta * h_higher
    h_flow   = xp.where(valid_ds, h_flow, depth)          # free-outflow cells: own depth
    h_flow   = xp.maximum(h_flow, min_depth)

    # Conveyance discharge.  Overland (chan_mask False): wide sheet R≈h_flow,
    # width=cell_size — identical arithmetic to the original diffusive path.
    # Channel: confined rectangular section, true hydraulic radius R=A/P.
    Q_overland = (1.0 / n) * (h_flow ** (5.0 / 3.0)) * (S_eff ** 0.5) * cell_size
    A_overland = h_flow * cell_size

    A_chan = h_flow * width
    R_chan = A_chan / (width + 2.0 * h_flow)
    Q_chan = (1.0 / n) * (R_chan ** (2.0 / 3.0)) * (S_eff ** 0.5) * A_chan

    Q    = xp.where(chan_mask, Q_chan, Q_overland)
    A_xs = xp.where(chan_mask, A_chan, A_overland)
    # A_xs exposed so callers use the correct celerity denominator (c=5/3·Q/A_xs);
    # S_eff exposed for diagnostics.
    return Q, A_xs, S_eff


def normal_depth(Q_ref, slope, n, width, chan_mask, xp, iters=3):
    """
    Manning normal depth, matching the conveyance convention of ``mannings_discharge``.

    Given a reference discharge ``Q_ref`` [m³/s], solve Manning's equation for the
    flow depth ``h`` [m].  Overland cells (``chan_mask`` False) use the wide-sheet
    shortcut ``R ≈ h`` (``width = cell_size``), giving the exact closed form

        h = (n·Q/(B·√S))^(3/5)  ,   A_xs = B·h

    Confined channel cells (``chan_mask`` True, ``width = B ≪ cell_size``) use the
    true rectangular hydraulic radius ``R = A/P`` (``A = B·h``, ``P = B + 2h``),
    refined from the wide-sheet guess by a few Newton iterations — the same split
    ``mannings_discharge`` makes between overland and channel cells.

    Used by the Muskingum–Cunge scheme to obtain the reference flow area (hence
    celerity ``c = 5/3·Q_ref/A_xs``) from a rating rather than from a stored volume.

    Parameters
    ----------
    Q_ref     : (n,) array – reference discharge [m³/s] (negatives floored to 0)
    slope     : (n,) array – bed slope [m/m] (already floored at MIN_SLOPE)
    n         : scalar or (n,) array – Manning's n
    width     : (n,) array – section width B [m] (cell_size overland, B on channels)
    chan_mask : (n,) bool array – True where the rectangular R=A/P section applies
    xp        : array module (numpy or cupy)
    iters     : int – Newton iterations for channel cells (3 is ample)

    Returns
    -------
    h    : (n,) array [m]  – normal (flow) depth
    A_xs : (n,) array [m²] – flow cross-section area B·h  (celerity denominator)
    """
    sqrtS = xp.sqrt(slope)
    Qpos  = xp.maximum(Q_ref, 0.0)
    # Wide-sheet closed form (R ≈ h) — exact overland, Newton seed for channels.
    h   = (n * Qpos / (width * sqrtS + 1e-30)) ** 0.6
    ref = chan_mask & (Qpos > 1e-12)        # only refine wet channel cells
    for _ in range(iters):
        A  = width * h
        P  = width + 2.0 * h
        Qp = (1.0 / n) * sqrtS * A ** (5.0 / 3.0) / P ** (2.0 / 3.0)
        dQ = ((1.0 / n) * sqrtS * A ** (2.0 / 3.0) * P ** (-5.0 / 3.0)
              * ((5.0 / 3.0) * width * P - (4.0 / 3.0) * A))
        step = xp.where(ref, (Qp - Qpos) / xp.maximum(dQ, 1e-30), 0.0)
        h    = xp.maximum(h - step, 0.0)
    A_xs = width * h
    return h, A_xs


def muskingum_cunge_step(I2, I1, O1, Q_L, c, A_xs, width, slope, dist, dt, xp):
    """
    One variable-parameter Muskingum–Cunge (Ponce–Yevjevich) update per cell.

    Each D8 cell is treated as a reach of length ``dist``.  The outflow at the new
    time is

        O₂ = C0·I₂ + C1·I₁ + C2·O₁ + C3·Q_L          (C0 + C1 + C2 = 1)

    with the Cunge coefficients written in Courant / grid-Peclet form so that the
    scheme's *numerical* diffusion equals the *physical* hydraulic diffusivity
    D = Q/(2·B·S₀) of the diffusion-wave equation — grid-independent, physically
    correct attenuation from a kinematic-wave scheme:

        Cr = c·dt/dist                     (Courant number, c = 5/3·V)
        Dg = q/(S₀·c·dist) = (3/5)·(A_xs/B)/(S₀·dist)   (grid Peclet; = 1 − 2X)
        C0 = (−1 + Cr + Dg)/denom ,  C1 = (1 + Cr − Dg)/denom
        C2 = ( 1 − Cr + Dg)/denom ,  C3 = (2·Cr)/denom ,  denom = 1 + Cr + Dg

    ``Dg`` is evaluated as ``(3/5)·h_eff/(S₀·dist)`` with ``h_eff = A_xs/B`` — this
    is identically ``q/(S₀·c·dist)`` (since ``q/c = 3/5·h_eff``) but stays finite as
    ``Q_ref → 0`` (dry cells give ``Cr = Dg = 0`` cleanly instead of 0/0).

    ``C0`` is intentionally *not* clamped to ≥0 — a negative ``C0`` is the faithful
    MC representation of the wave and only produces a tiny, mass-conserving dip; the
    final ``O₂`` is floored at 0 as a physical safeguard (backflow is unphysical).
    Mass is conserved by the caller's volume ledger (every ``O₂·dt`` scattered
    downstream), independent of this shape function, so the floor is safe.

    Parameters
    ----------
    I2, I1 : (n,) arrays – inflow rate this / previous step [m³/s]
    O1     : (n,) array  – outflow rate previous step [m³/s]
    Q_L    : (n,) array  – lateral inflow rate (effective runoff) [m³/s]
    c      : (n,) array  – kinematic celerity 5/3·V [m/s]
    A_xs   : (n,) array  – reference flow area [m²]
    width  : (n,) array  – section width B [m]
    slope  : (n,) array  – bed slope [m/m]
    dist   : (n,) array  – reach length Δx [m]
    dt     : float       – time step [s]
    xp     : array module (numpy or cupy)

    Returns
    -------
    O2       : (n,) array [m³/s] – outflow this step (floored at 0)
    neg_frac : 0-d array         – fraction of cells whose raw O₂ was negative
                                   (floored) — a resolution diagnostic
    """
    Cr    = c * dt / dist
    h_eff = A_xs / xp.maximum(width, 1e-30)
    Dg    = 0.6 * h_eff / (slope * dist)
    denom = 1.0 + Cr + Dg
    C0 = (-1.0 + Cr + Dg) / denom
    C1 = ( 1.0 + Cr - Dg) / denom
    C2 = ( 1.0 - Cr + Dg) / denom
    C3 = ( 2.0 * Cr)       / denom
    O2 = C0 * I2 + C1 * I1 + C2 * O1 + C3 * Q_L
    neg      = O2 < 0.0
    neg_frac = neg.sum() / xp.maximum(O2.size, 1)
    O2 = xp.maximum(O2, 0.0)
    return O2, neg_frac


def flux_limiter(Q_out, volume, dt):
    """
    Volume-conservative CFL limiter.

    Caps Q_out so that a cell can never drain more water than it currently
    stores in a single time step:

        Q_out_limited = min(Q_out, volume / dt)

    This prevents the positive-feedback runaway that occurs in the explicit
    kinematic-wave scheme when the Courant number C = V * dt / dx > 1.
    The fix is mass-conservative: the downstream cell simply receives less
    inflow, which is physically correct (there is no more water to give).

    Parameters
    ----------
    Q_out  : 1-D float array  – Manning's discharge [m³/s] for each cell
    volume : 1-D float array  – current stored volume [m³] for each cell
    dt     : float            – time step [s]

    Returns
    -------
    Q_out_limited : 1-D float array  [m³/s]
    """
    return np.minimum(Q_out, np.maximum(volume, 0.0) / dt)


# ---------------------------------------------------------------------------
# 6.  Rainfall array builder
# ---------------------------------------------------------------------------

def build_rainfall_array(shape, intensity_mm_hr, duration_hours, dt_seconds, t_seconds):
    """
    Return a 2-D rainfall array (m/s) for the current simulation time.

    For a spatially uniform event:
        - intensity_mm_hr converted to m/s = intensity / (1000 * 3600)
        - Applied only while t_seconds < duration_hours * 3600

    The function signature accepts `shape` so it can later be replaced by a
    spatially variable (e.g., radar) array without changing the router logic.

    Parameters
    ----------
    shape           : (nrows, ncols) of the grid
    intensity_mm_hr : uniform rainfall rate [mm/hr]
    duration_hours  : rainfall duration [hr]
    dt_seconds      : time step [s]  (unused here; kept for API consistency)
    t_seconds       : current simulation time [s]

    Returns
    -------
    rain_ms : 2-D float64 array  (m/s)
    """
    rain_ms_value = (intensity_mm_hr / (1000.0 * 3600.0)
                     if t_seconds < duration_hours * 3600.0
                     else 0.0)
    return np.full(shape, rain_ms_value, dtype=np.float64)
