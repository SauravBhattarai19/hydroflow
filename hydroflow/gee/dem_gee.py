"""
dem_gee.py
==========
DEM downloader on Google Earth Engine, for basins that don't yet have a local
DEM_PATH (every other GEE-backed input — SERVES, HiHydroSoil, SoilGrids,
LULC/LCZ, IMERG — is pulled by the rest of vsa_opm.gee once a DEM and
watershed boundary exist; this is the one missing piece needed to bootstrap a
brand-new basin from scratch).

GEE dataset
-----------
    NASA/NASADEM_HGT/001   band 'elevation'   — void-filled SRTM, ~30m native

Downsamples with a real area-average (reduceResolution mean), not a naive
reproject/resample, so the 100m output isn't just nearest/bilinear-picked
30m pixels.

Large bounding boxes exceed GEE's ~50MB getDownloadURL request-size cap, so
big requests are automatically split into a lon/lat tile grid and mosaicked
locally with rasterio.merge (same idea as imerg_gee.py's date-chunking, just
spatial instead of temporal).

Authentication
--------------
    Same as the rest of vsa_opm.gee: vsa_opm.gee.auth.authenticate(project).
"""

import os
import math
import logging

logger = logging.getLogger(__name__)

from .auth import authenticate as _authenticate  # noqa: E402

try:
    import ee
    GEE_AVAILABLE = True
except ImportError:
    GEE_AVAILABLE = False

# GEE's synchronous getDownloadURL request-size cap is 50,331,648 bytes; its
# internal size estimate for a single float band has run empirically at
# ~9.5 bytes/pixel (GEO_TIFF format overhead included) — stay well under that.
_MAX_PIXELS_PER_TILE = 3_500_000


def _download_one(img, geometry, target_crs_epsg, scale_m, output_path):
    import urllib.request
    url = img.clip(geometry).getDownloadURL({
        'region': geometry,
        'crs': target_crs_epsg,
        'scale': scale_m,
        'format': 'GEO_TIFF',
    })
    urllib.request.urlretrieve(url, output_path)


def download_dem(bbox_wgs84, target_crs_epsg, scale_m, output_path,
                  project=None, dataset='NASA/NASADEM_HGT/001', band='elevation'):
    """
    Download a DEM from GEE, reprojected + area-averaged to (target_crs_epsg,
    scale_m), clipped to bbox_wgs84.

    Parameters
    ----------
    bbox_wgs84 : (min_lon, min_lat, max_lon, max_lat)
    target_crs_epsg : str, e.g. 'EPSG:32616'
    scale_m : float, output pixel size in metres
    output_path : str

    Caches to *output_path*; skips download if the file already exists.
    Large boxes are auto-tiled and mosaicked. Returns the output path on
    success, or None on failure.
    """
    if os.path.isfile(output_path):
        logger.info("DEM raster cached: %s", output_path)
        return output_path

    if not GEE_AVAILABLE:
        logger.warning("earthengine-api not installed")
        return None

    if not _authenticate(project):
        return None

    try:
        import tempfile
        import numpy as np
        import rasterio
        from rasterio.merge import merge as rio_merge

        min_lon, min_lat, max_lon, max_lat = bbox_wgs84

        # Rough pixel-count estimate (equirectangular approximation is fine —
        # only used to decide how finely to tile).
        mid_lat_rad = math.radians((min_lat + max_lat) / 2.0)
        m_per_deg_lon = 111_320.0 * math.cos(mid_lat_rad)
        m_per_deg_lat = 110_540.0
        width_px = ((max_lon - min_lon) * m_per_deg_lon) / scale_m
        height_px = ((max_lat - min_lat) * m_per_deg_lat) / scale_m
        total_px = width_px * height_px

        n_tiles = max(1, math.ceil(total_px / _MAX_PIXELS_PER_TILE))
        side = max(1, math.ceil(math.sqrt(n_tiles)))
        nx, ny = side, side

        img = ee.Image(dataset).select(band)
        img = (img.reduceResolution(reducer=ee.Reducer.mean(), maxPixels=1024)
                  .reproject(crs=target_crs_epsg, scale=scale_m))

        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

        if nx == 1 and ny == 1:
            geometry = ee.Geometry.Rectangle(list(bbox_wgs84), 'EPSG:4326', geodesic=False)
            _download_one(img, geometry, target_crs_epsg, scale_m, output_path)
            logger.info("DEM downloaded: %s", output_path)
            return output_path

        logger.info("Bounding box too large for one request (~%.1fM px) — "
                    "tiling %dx%d and mosaicking.", total_px / 1e6, nx, ny)
        lon_edges = np.linspace(min_lon, max_lon, nx + 1)
        lat_edges = np.linspace(min_lat, max_lat, ny + 1)

        tile_paths = []
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(nx):
                for j in range(ny):
                    tile_bbox = (lon_edges[i], lat_edges[j], lon_edges[i + 1], lat_edges[j + 1])
                    tile_path = os.path.join(tmpdir, f"tile_{i}_{j}.tif")
                    geometry = ee.Geometry.Rectangle(list(tile_bbox), 'EPSG:4326', geodesic=False)
                    _download_one(img, geometry, target_crs_epsg, scale_m, tile_path)
                    tile_paths.append(tile_path)
                    logger.info("  tile %d/%d downloaded", len(tile_paths), nx * ny)

            srcs = [rasterio.open(p) for p in tile_paths]
            mosaic, out_transform = rio_merge(srcs)
            profile = srcs[0].profile.copy()
            profile.update(height=mosaic.shape[1], width=mosaic.shape[2],
                           transform=out_transform)
            for s in srcs:
                s.close()
            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write(mosaic)

        logger.info("DEM downloaded (mosaicked from %d tiles): %s", nx * ny, output_path)
        return output_path

    except Exception as exc:
        logger.warning("GEE DEM download failed: %s", exc)
        return None
