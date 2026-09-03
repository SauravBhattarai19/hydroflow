# -*- coding: utf-8 -*-
"""
terrain_rules.py
=================
Generate a spatially-varying parameter raster (currently: Manning's n) from a
DEM by applying a user-supplied elevation rule.

There's no dedicated ``MANNINGS_N_SOURCE='elevation'`` config path — the
existing ``'raster'`` source (``hydroflow.core.routing.surface.resolve_mannings_n``)
already reprojects/resamples an arbitrary GeoTIFF onto the routing grid via
``align_raster_to_dem`` and falls back bad values to ``MANNINGS_N``, so
generating a Manning's-n GeoTIFF here and pointing ``MANNINGS_N_RASTER_PATH``
at it reuses that path unchanged: ``MANNINGS_N_SOURCE="raster"``.
"""

import os

import numpy as np


_SANE_N_RANGE = (0.005, 1.0)


def mannings_n_from_dem(dem_path, rule, output_path, *, nodata_n=None):
    """
    Write a Manning's-n GeoTIFF derived from a DEM's elevation.

    Parameters
    ----------
    dem_path : str
        Path to a DEM GeoTIFF (e.g. a pipeline result's ``clipped_dem``).
        The output raster shares its profile/transform/CRS/shape, so if this
        is already the routing grid's own DEM the result needs no further
        alignment (``MANNINGS_N_SOURCE='raster'`` would realign it anyway if
        it weren't).
    rule : callable | list[tuple] | dict
        - callable: ``f(elev_array) -> n_array``, same shape, vectorized.
        - list of ascending ``(upper_bound_elev, n)`` tuples, first match
          wins, e.g.
          ``[(1000, 0.03), (1500, 0.06), (2000, 0.10), (float("inf"), 0.15)]``
        - dict ``{(min_elev, max_elev): n}`` — half-open ``[min, max)`` bins.
    output_path : str
        Where to write the n-raster.
    nodata_n : float, optional
        Value for DEM-nodata cells / elevations unmatched by a dict rule.
        Defaults to the mean of the assigned values.

    Returns
    -------
    str
        *output_path* (so this chains: ``cfg.MANNINGS_N_RASTER_PATH =
        mannings_n_from_dem(...)``).
    """
    import rasterio

    if not os.path.isfile(dem_path):
        raise FileNotFoundError(f"DEM not found: {dem_path}")

    with rasterio.open(dem_path) as src:
        elev = src.read(1).astype(np.float64)
        profile = src.profile.copy()
        nodata = src.nodata

    valid = np.isfinite(elev)
    if nodata is not None:
        valid &= (elev != nodata)

    n_arr = np.full(elev.shape, np.nan, dtype=np.float64)

    if callable(rule):
        n_arr[valid] = np.asarray(rule(elev), dtype=np.float64)[valid]
    elif isinstance(rule, dict):
        for (lo, hi), n_val in rule.items():
            mask = valid & (elev >= lo) & (elev < hi)
            n_arr[mask] = float(n_val)
    else:
        breakpoints = sorted(rule, key=lambda pair: pair[0])
        assigned = np.zeros(elev.shape, dtype=bool)
        for upper, n_val in breakpoints:
            mask = valid & ~assigned & (elev <= upper)
            n_arr[mask] = float(n_val)
            assigned |= mask

    assigned_mask = valid & np.isfinite(n_arr)
    if not assigned_mask.any():
        raise ValueError(
            "mannings_n_from_dem: rule produced no assigned values — check "
            "that the rule's elevation range overlaps the DEM's actual range "
            f"({np.nanmin(elev[valid]):.1f} to {np.nanmax(elev[valid]):.1f})."
        )

    fill = float(nodata_n) if nodata_n is not None else float(n_arr[assigned_mask].mean())
    n_arr[~assigned_mask] = fill

    lo_ok, hi_ok = _SANE_N_RANGE
    out_of_range = n_arr[assigned_mask][
        (n_arr[assigned_mask] < lo_ok) | (n_arr[assigned_mask] > hi_ok)
    ]
    if out_of_range.size:
        print(f"  [WARN] mannings_n_from_dem: {out_of_range.size} cell(s) have "
             f"n outside the typical [{lo_ok}, {hi_ok}] range — check your rule.")

    profile.update(dtype="float32", count=1, nodata=None)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(n_arr.astype(np.float32), 1)

    print(f"  Manning's n raster written -> {output_path}  "
         f"(range=[{n_arr.min():.4f}, {n_arr.max():.4f}], mean={n_arr.mean():.4f})")
    return output_path
