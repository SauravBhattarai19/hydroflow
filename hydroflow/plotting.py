# -*- coding: utf-8 -*-
"""
plotting.py
===========
Small, good-enough plotting helpers for common pipeline outputs, so a run's
results can be visualized in one call instead of hand-writing
matplotlib/rasterio/geopandas boilerplate every time.

Every function accepts *flexible input*: a path (to a file or an OUTPUT_DIR),
a pandas DataFrame (where applicable), a run_pipeline()/stage result dict, or
a Config object (its OUTPUT_DIR / derived-path attributes are used). Every
function returns ``(fig, ax)`` and accepts an optional ``ax=`` to draw into
caller-provided axes, matplotlib convention.

matplotlib/rasterio/geopandas/pandas are hard dependencies of hydroflow, so
these imports are unconditional — no optional-dependency handling needed.
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _get_fig_ax(ax):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.figure
    return fig, ax


def _resolve_df(source, *, df_key, csv_key, cfg_attr, default_filename):
    """Resolve *source* to a DataFrame: DataFrame passthrough, dict (prefers
    an already-loaded DataFrame under df_key, else reads csv_key), a
    Config-like object (uses cfg_attr, falling back to OUTPUT_DIR/default),
    or a path (file, or a directory to join with default_filename)."""
    if isinstance(source, pd.DataFrame):
        return source
    if isinstance(source, dict):
        if df_key in source and isinstance(source[df_key], pd.DataFrame):
            return source[df_key]
        if csv_key in source:
            return pd.read_csv(source[csv_key])
        raise KeyError(
            f"dict source has neither '{df_key}' nor '{csv_key}' key; "
            f"pass a path, DataFrame, or Config object instead."
        )
    if hasattr(source, cfg_attr) or hasattr(source, "OUTPUT_DIR"):
        path = getattr(source, cfg_attr, None) or os.path.join(
            getattr(source, "OUTPUT_DIR", "."), default_filename)
        return pd.read_csv(path)
    path = str(source)
    if os.path.isdir(path):
        path = os.path.join(path, default_filename)
    return pd.read_csv(path)


def _resolve_path(source, *, key, cfg_attr, default_filename):
    """Resolve *source* to a file path: dict (looks up key), a Config-like
    object (uses cfg_attr, falling back to OUTPUT_DIR/default_filename), or
    a path (file, or a directory to join with default_filename)."""
    if isinstance(source, dict):
        if key in source:
            return source[key]
        raise KeyError(f"dict source has no '{key}' key.")
    if hasattr(source, cfg_attr):
        v = getattr(source, cfg_attr)
        if v:
            return v
    if hasattr(source, "OUTPUT_DIR"):
        return os.path.join(source.OUTPUT_DIR, default_filename)
    path = str(source)
    if os.path.isdir(path):
        return os.path.join(path, default_filename)
    return path


def plot_hydrograph(source, ax=None, *, label=None, annotate_peak=True,
                    color=None, **kwargs):
    """
    Plot outlet discharge Q(t) from a hydrograph.

    Parameters
    ----------
    source : str | os.PathLike | pandas.DataFrame | dict | Config
        Path to hydrograph.csv (or an OUTPUT_DIR containing it), an
        already-loaded DataFrame (time_hr, Q_m3s columns), a run_pipeline()
        result dict (uses 'hydrograph_df' if present, else 'hydrograph_csv'),
        or a Config object (uses HYDROGRAPH_CSV).
    ax : matplotlib.axes.Axes, optional
    label : str, optional legend label
    annotate_peak : bool, default True — mark peak Q with a dot + label
    color : str, optional line color

    Returns
    -------
    (fig, ax)
    """
    df = _resolve_df(source, df_key="hydrograph_df", csv_key="hydrograph_csv",
                     cfg_attr="HYDROGRAPH_CSV", default_filename="hydrograph.csv")
    fig, ax = _get_fig_ax(ax)
    ax.plot(df["time_hr"], df["Q_m3s"], label=label, color=color, **kwargs)

    if annotate_peak and len(df):
        i = df["Q_m3s"].idxmax()
        peak_t, peak_q = df["time_hr"].loc[i], df["Q_m3s"].loc[i]
        ax.plot(peak_t, peak_q, "o", color=color or "crimson", zorder=5)
        ax.annotate(f"peak {peak_q:.2f} m³/s\n@ t={peak_t:.2f} h",
                   (peak_t, peak_q), textcoords="offset points",
                   xytext=(10, -28), fontsize=9)

    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Discharge Q (m³/s)")
    ax.set_title("Outlet hydrograph")
    if label:
        ax.legend()
    return fig, ax


def plot_raster(source, ax=None, *, cmap="viridis", label=None,
                hillshade=False, **kwargs):
    """
    Generic single-band GeoTIFF viewer (DEM, Manning's n raster, flow
    accumulation, ...).

    Parameters
    ----------
    source : str | os.PathLike
        Path to a single-band GeoTIFF.
    ax : matplotlib.axes.Axes, optional
    cmap : str, colormap (ignored when hillshade=True)
    label : str, optional colorbar label
    hillshade : bool, default False — render as a shaded-relief RGB (no
        colorbar) instead of a flat colormap; intended for DEMs.

    Returns
    -------
    (fig, ax)
    """
    import rasterio
    from rasterio.plot import plotting_extent

    path = str(source)
    with rasterio.open(path) as src:
        arr = src.read(1, masked=True)
        ext = plotting_extent(src)

    fig, ax = _get_fig_ax(ax)

    if hillshade:
        from matplotlib.colors import LightSource
        ls = LightSource(azdeg=315, altdeg=45)
        filled = arr.filled(np.nanmin(arr) if arr.count() else 0.0)
        rgb = ls.shade(filled, cmap=plt.get_cmap(cmap), vert_exag=2,
                       blend_mode="soft")
        ax.imshow(rgb, extent=ext)
    else:
        im = ax.imshow(arr, extent=ext, cmap=cmap, **kwargs)
        fig.colorbar(im, ax=ax, label=label or "")

    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    return fig, ax


def plot_watershed(source, ax=None, *, cmap="terrain", boundary_color="crimson",
                   hillshade=True, **kwargs):
    """
    Plot the clipped DEM (optionally hillshaded) with the delineated
    watershed boundary overlaid.

    Parameters
    ----------
    source : dict | Config | str | os.PathLike
        A run_pipeline()/stage_process_dem result dict (uses 'clipped_dem'
        and 'watershed_geojson'), a Config object (uses ROUTING_DEM_PATH and
        OPM_WATERSHED_GEOJSON), or an OUTPUT_DIR path.
    ax : matplotlib.axes.Axes, optional
    cmap : str, DEM colormap (ignored when hillshade=True)
    boundary_color : str, watershed boundary line color
    hillshade : bool, default True

    Returns
    -------
    (fig, ax)
    """
    import geopandas as gpd

    dem_path = _resolve_path(source, key="clipped_dem",
                             cfg_attr="ROUTING_DEM_PATH",
                             default_filename="clipped_dem.tif")
    geojson_path = _resolve_path(source, key="watershed_geojson",
                                 cfg_attr="OPM_WATERSHED_GEOJSON",
                                 default_filename="watershed.geojson")

    fig, ax = plot_raster(dem_path, ax=ax, cmap=cmap, hillshade=hillshade, **kwargs)
    gpd.read_file(geojson_path).boundary.plot(ax=ax, color=boundary_color,
                                              linewidth=1.5)
    ax.set_title("Watershed")
    return fig, ax


def plot_mass_balance(source, ax=None, *, run_tag=None):
    """
    Bar chart of a run's mass-balance closure: total input (rainfall-derived
    runoff + boundary-condition inflow) vs outflow vs storage vs closure
    error, read from mass_balance.csv.

    Parameters
    ----------
    source : str | os.PathLike | pandas.DataFrame | dict | Config
        Path to mass_balance.csv (or an OUTPUT_DIR containing it), an
        already-loaded DataFrame, a run_pipeline() result dict (uses
        'mass_balance_csv'), or a Config object (uses MASS_BALANCE_CSV).
    ax : matplotlib.axes.Axes, optional
    run_tag : str, optional — select a specific RUN_TAG row; default: last row.

    Returns
    -------
    (fig, ax)
    """
    df = _resolve_df(source, df_key="mass_balance_df", csv_key="mass_balance_csv",
                     cfg_attr="MASS_BALANCE_CSV",
                     default_filename="mass_balance.csv")
    row = df[df["run_tag"] == run_tag].iloc[-1] if run_tag else df.iloc[-1]

    total_in = float(row["input_m3"]) + float(row["bc_inflow_m3"])
    values = [total_in, float(row["outflow_m3"]), float(row["storage_m3"]),
             float(row["error_m3"])]
    labels = ["Input\n(runoff+BC)", "Outflow", "Storage", "Error"]
    colors = ["#3b6ea5", "#e08214", "#7a7a7a", "#c0392b"]

    fig, ax = _get_fig_ax(ax)
    ax.bar(labels, values, color=colors)
    ax.set_ylabel("Volume (m³)")
    ax.set_title(f"Mass balance (rel. error {float(row['rel_error']):.2e})")
    return fig, ax
