# -*- coding: utf-8 -*-
"""
notebook_map.py
================
Jupyter-only interactive bounding-box picker, for drawing ``DEM_BOUNDS_WGS84``
on a map instead of typing coordinates by hand. Not usable from a plain
``.py`` script or the CLI — it needs a live ipywidgets/ipyleaflet frontend.

Requires the ``notebook`` extra: ``pip install hydroflow[notebook]``
(geemap, ipyleaflet, ipywidgets) — kept separate from the ``gee`` extra so
headless/script users doing scripted GEE downloads aren't forced into the
widget stack.

Usage (in a Jupyter cell)::

    from hydroflow.utils.notebook_map import pick_bounds_map, get_drawn_bounds

    m = pick_bounds_map(center=(27.7, 85.3), zoom=9)
    m   # display the map, draw a rectangle with the toolbar

    bounds = get_drawn_bounds(m)   # after drawing
    # -> (min_lon, min_lat, max_lon, max_lat), feed straight into
    #    Config(DEM_BOUNDS_WGS84=bounds, ...)
"""


def _require_geemap():
    try:
        import geemap
        return geemap
    except ImportError as exc:
        raise ImportError(
            "Interactive map picking needs the notebook extra: "
            "pip install hydroflow[notebook]"
        ) from exc


def pick_bounds_map(center=(27.7, 85.3), zoom=9, **kwargs):
    """
    Create an interactive map with a rectangle draw control, for picking a
    ``DEM_BOUNDS_WGS84`` box in a Jupyter notebook.

    Parameters
    ----------
    center : (lat, lon), initial map center.
    zoom : int, initial zoom level.
    **kwargs
        Passed through to ``geemap.Map()``. ``ee_initialize`` defaults to
        False here (drawing a box doesn't need Earth Engine credentials);
        pass ``ee_initialize=True`` if you also want to overlay EE layers.

    Returns
    -------
    geemap.Map
        Display it as a cell's last expression (or ``display(m)``), draw a
        rectangle using the toolbar, then call ``get_drawn_bounds(m)``.

    Raises
    ------
    ImportError
        If geemap/ipyleaflet/ipywidgets aren't installed
        (``pip install hydroflow[notebook]``).
    """
    geemap = _require_geemap()
    kwargs.setdefault("ee_initialize", False)
    m = geemap.Map(center=center, zoom=zoom, **kwargs)
    return m


def get_drawn_bounds(map_obj):
    """
    Extract ``(min_lon, min_lat, max_lon, max_lat)`` from the last rectangle
    drawn on *map_obj* (as created by :func:`pick_bounds_map`).

    Parameters
    ----------
    map_obj : geemap.Map
        A map returned by :func:`pick_bounds_map`, after the user has drawn
        a rectangle using its draw toolbar.

    Returns
    -------
    (min_lon, min_lat, max_lon, max_lat) : tuple of float
        Directly usable as ``Config(DEM_BOUNDS_WGS84=...)``.

    Raises
    ------
    RuntimeError
        No shape has been drawn yet on *map_obj*.
    """
    features = getattr(map_obj, "draw_features", None) or []
    if not features:
        raise RuntimeError(
            "No shape has been drawn yet — draw a rectangle on the map's "
            "toolbar first, then call get_drawn_bounds(map_obj)."
        )
    last = features[-1]
    feature = last.__geo_interface__ if hasattr(last, "__geo_interface__") else last
    return _bounds_from_geojson_feature(feature)


def _bounds_from_geojson_feature(feature):
    """
    Pure function: a GeoJSON Feature/geometry dict (Polygon coordinates, as
    produced by ipyleaflet's DrawControl) -> (min_lon, min_lat, max_lon,
    max_lat). Factored out so it's testable without constructing any real
    map/widget.
    """
    geometry = feature.get("geometry", feature)
    if geometry.get("type") != "Polygon":
        raise RuntimeError(
            f"Expected a drawn rectangle (Polygon), got '{geometry.get('type')}' "
            "— use the rectangle tool on the map's draw toolbar."
        )
    coords = geometry["coordinates"][0]
    lons = [pt[0] for pt in coords]
    lats = [pt[1] for pt in coords]
    return (min(lons), min(lats), max(lons), max(lats))
