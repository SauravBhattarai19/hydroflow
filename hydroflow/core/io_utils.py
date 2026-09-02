# -*- coding: utf-8 -*-
"""
io_utils.py — shared raster I/O helpers.

Sampling and grid-alignment utilities used by the routing surface fields and
the runoff soil-parameter resolution.
"""

import numpy as np
import rasterio




# ---------------------------------------------------------------------------
# 7.  Raster alignment
# ---------------------------------------------------------------------------

def align_raster_to_dem(src_path, dem_path, resampling='nearest'):
    """
    Reproject/resample *src_path* to exactly match the routing DEM grid.

    Returns a 2-D numpy array with the same (height, width) as the DEM.
    """
    from rasterio.warp import reproject, Resampling

    METHODS = {
        'nearest':  Resampling.nearest,
        'bilinear': Resampling.bilinear,
    }

    with rasterio.open(dem_path) as dem:
        dst_crs       = dem.crs
        dst_transform = dem.transform
        dst_shape     = (dem.height, dem.width)

    with rasterio.open(src_path) as src:
        dst_array = np.empty(dst_shape, dtype=src.dtypes[0])
        reproject(
            source=rasterio.band(src, 1),
            destination=dst_array,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=METHODS.get(resampling, Resampling.nearest),
        )
    return dst_array


def raster_band_1d(raster_path, s_rows, s_cols):
    """
    Read band 1 of a routing-grid-aligned raster into a per-cell (n_cells,)
    array (used for the SERVES deficit and the HiHydroSoil Ksat rasters).

    Returns float64 with nodata/out-of-grid cells set to NaN, or None if the
    raster grid does not match the routing grid.
    """
    with rasterio.open(raster_path) as src:
        arr2d  = src.read(1).astype(np.float64)
        nodata = src.nodata
    if nodata is not None:
        arr2d[arr2d == nodata] = np.nan

    _to_np = lambda a: a.get() if hasattr(a, 'get') else np.asarray(a)
    sr = _to_np(s_rows); sc = _to_np(s_cols)
    if sr.max() >= arr2d.shape[0] or sc.max() >= arr2d.shape[1]:
        return None
    return arr2d[sr, sc]
