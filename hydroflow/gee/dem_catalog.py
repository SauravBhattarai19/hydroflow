# -*- coding: utf-8 -*-
"""
dem_catalog.py
===============
STAC-lite catalog of DEM datasets reachable through Google Earth Engine,
curated for watershed/hydrology work.

Listing the catalog (`list_dems`, `describe_dems`, `get_dem_info`) is plain
metadata with no dependency on `earthengine-api` — safe to import and call
even without `pip install hydroflow[gee]`, so users can browse options before
deciding whether to install the extra. Only the actual download
(`hydroflow.gee.dem_gee.download_dem`) needs GEE auth.

Each entry approximates a STAC collection record (id, title, description,
spatial extent bbox) trimmed to what a DEM download needs: the GEE asset id,
whether it's a single Image or an ImageCollection (needs `.mosaic()`), the
band to read, and the native pixel size.
"""

# Coverage bboxes are the datasets' documented nominal extents, used only as
# an advisory check (download still proceeds; see check_coverage). Some
# datasets (e.g. usgs_3dep_1m) have real coverage that is patchy/tiled even
# inside their bbox.
DEM_CATALOG = {
    "nasadem": {
        "title": "NASADEM (void-filled SRTM)",
        "gee_id": "NASA/NASADEM_HGT/001",
        "gee_type": "Image",
        "band": "elevation",
        "resolution_m": 30,
        "bbox": (-180.0, -56.0, 180.0, 60.0),
        "description": (
            "NASA's void-filled reprocessing of SRTM (2000), merged with "
            "ASTER GDEM and other sources to fill gaps. Good general-purpose "
            "default; covers 56°S to 60°N."
        ),
    },
    "srtm": {
        "title": "SRTM GL1 v3",
        "gee_id": "USGS/SRTMGL1_003",
        "gee_type": "Image",
        "band": "elevation",
        "resolution_m": 30,
        "bbox": (-180.0, -56.0, 180.0, 60.0),
        "description": (
            "Original SRTM v3 (not void-filled). Same coverage as NASADEM; "
            "kept for comparison / reproducing older studies."
        ),
    },
    "merit": {
        "title": "MERIT DEM (hydrologically conditioned)",
        "gee_id": "MERIT/DEM/v1_0_3",
        "gee_type": "Image",
        "band": "dem",
        "resolution_m": 92,
        "bbox": (-180.0, -60.0, 180.0, 90.0),
        "description": (
            "Yamazaki et al. (2017) — SRTM/AW3D-based DEM with multi-error "
            "removal (stripe noise, absolute bias, tree-height bias, speckle). "
            "Recommended when flow-routing/delineation quality matters more "
            "than raw resolution."
        ),
    },
    "alos": {
        "title": "ALOS World 3D (AW3D30) v4.1",
        "gee_id": "JAXA/ALOS/AW3D30/V4_1",
        "gee_type": "ImageCollection",
        "band": "DSM",
        "resolution_m": 30,
        "bbox": (-180.0, -82.0, 180.0, 82.0),
        "description": (
            "JAXA optical stereo-derived global DSM (surface height, "
            "including canopy/buildings — not bare-earth). Useful where "
            "SRTM-era radar DEMs show known artifacts."
        ),
    },
    "copernicus_glo30": {
        "title": "Copernicus DEM GLO-30 (2024-1)",
        "gee_id": "COPERNICUS/DEM/GLO30_2024_1",
        "gee_type": "ImageCollection",
        "band": "DEM",
        "resolution_m": 30,
        "bbox": (-180.0, -90.0, 180.0, 90.0),
        "description": (
            "TanDEM-X-derived global DSM (ESA/Copernicus). Newest and most "
            "consistent near-global coverage of the surface-model options."
        ),
    },
    "usgs_3dep_1m": {
        "title": "USGS 3DEP 1m (lidar)",
        "gee_id": "USGS/3DEP/1m",
        "gee_type": "ImageCollection",
        "band": "elevation",
        "resolution_m": 1,
        "bbox": (-179.0, 17.5, -64.0, 71.5),
        "description": (
            "US lidar-derived bare-earth DEM, ~1m. United States (incl. "
            "Alaska/Hawaii/territories) only, and real coverage is patchy/"
            "tiled rather than wall-to-wall — a request inside this bbox can "
            "still fail if the area wasn't surveyed. Best for small US basins "
            "needing fine detail."
        ),
    },
    "gmted2010": {
        "title": "GMTED2010 (coarse global fallback)",
        "gee_id": "USGS/GMTED2010_FULL",
        "gee_type": "Image",
        "band": "be75",
        "resolution_m": 232,
        "bbox": (-180.0, -90.0, 180.0, 84.0),
        "description": (
            "~7.5 arc-second (~232m) global composite. Much coarser than the "
            "other options; useful only as a last-resort fallback or for "
            "very large / continental-scale domains."
        ),
    },
}


def list_dems():
    """Return the DEM catalog as a list of dicts (STAC-lite entries).

    Each dict has an ``'id'`` key (the catalog key, e.g. ``'nasadem'``) in
    addition to ``title``, ``gee_id``, ``gee_type``, ``band``,
    ``resolution_m``, ``bbox`` and ``description``.
    """
    return [dict(id=key, **meta) for key, meta in DEM_CATALOG.items()]


def describe_dems():
    """Human-readable table of the catalog (used by `hydroflow list-dems`)."""
    lines = [
        "Available DEM sources (Google Earth Engine — requires "
        "'pip install hydroflow[gee]' + authentication to download):",
        "",
    ]
    for key, meta in DEM_CATALOG.items():
        lon0, lat0, lon1, lat1 = meta["bbox"]
        lines.append(f"  {key}  —  {meta['title']}")
        lines.append(f"      gee_id: {meta['gee_id']}  "
                     f"(band: {meta['band']}, {meta['gee_type']})")
        lines.append(f"      resolution: ~{meta['resolution_m']} m   "
                     f"coverage: lon [{lon0}, {lon1}]  lat [{lat0}, {lat1}]")
        lines.append(f"      {meta['description']}")
        lines.append("")
    return "\n".join(lines)


def get_dem_info(name):
    """Look up one catalog entry by key.

    Returns a dict with an ``'id'`` key added. Raises KeyError (listing the
    valid keys) if *name* isn't in the catalog.
    """
    key = str(name).strip().lower()
    if key not in DEM_CATALOG:
        raise KeyError(
            f"Unknown DEM source '{name}'. Available: {', '.join(DEM_CATALOG)}. "
            "See hydroflow.describe_available_dems() for details."
        )
    return dict(id=key, **DEM_CATALOG[key])


def check_coverage(name, bbox_wgs84):
    """True if *bbox_wgs84* (min_lon, min_lat, max_lon, max_lat) falls inside
    the catalog entry's advertised coverage bbox.

    Advisory only — some datasets (e.g. usgs_3dep_1m) have real coverage
    that is patchier than their bounding box suggests.
    """
    info = get_dem_info(name)
    min_lon, min_lat, max_lon, max_lat = bbox_wgs84
    c_lon0, c_lat0, c_lon1, c_lat1 = info["bbox"]
    return (min_lon >= c_lon0 and max_lon <= c_lon1
            and min_lat >= c_lat0 and max_lat <= c_lat1)
