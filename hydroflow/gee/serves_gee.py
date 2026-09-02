"""
serves_gee.py
=============
SERVES + LULC + SoilGrids + HiHydroSoil pipeline on Google Earth Engine
for OPM parameters.

Computes SD_max_initial from the SERVES soil-moisture deficit formula:
    SM_deficit = (porosity − θ_SERVES) × Z_r
where
    porosity   = wcsat from HiHydroSoil v2.0 (saturated water content)
    θ_SERVES   = SERVES volumetric soil moisture [WP, FC]
    Z_r        = root zone depth [m] from ESA WorldCover LULC + lookup CSV
    SD_max     = max(SM_deficit) across the watershed

Also derives drainable porosity φ = mean(porosity − FC),
and K_sat from HiHydroSoil v2.0.

GEE datasets
------------
    ESA/WorldCover/v200/2021                              — LULC 10m
    LANDSAT/LC08/C02/T1_L2  +  LANDSAT/LC09/C02/T1_L2    — NDVI 30m
    COPERNICUS/S2_SR_HARMONIZED                           — NDVI 10m (alt)
    MODIS/061/MOD13A2                                     — NDVI 1km (alt)
    ISRIC/SoilGrids250m/v2_0/wv0033                       — field capacity
    ISRIC/SoilGrids250m/v2_0/wv1500                       — wilting point
    projects/sat-io/open-datasets/HiHydroSoilv2_0/wcsat   — porosity 250m

Authentication
--------------
    Priority: GOOGLE_APPLICATION_CREDENTIALS → GEE_SERVICE_ACCOUNT_KEY
              → ee.Initialize() default → ee.Authenticate()
"""

import os
import math
import logging

import pandas as pd

logger = logging.getLogger(__name__)

from .auth import authenticate as _authenticate  # noqa: E402

try:
    import ee
    GEE_AVAILABLE = True
except ImportError:
    GEE_AVAILABLE = False

_DEPTH_BANDS = {
    'b0':   'val_0_5cm_mean',
    'b10':  'val_5_15cm_mean',
    'b30':  'val_15_30cm_mean',
    'b60':  'val_30_60cm_mean',
    'b100': 'val_60_100cm_mean',
    'b200': 'val_100_200cm_mean',
}

# SoilGrids texture (projects/soilgrids-isric/{sand,clay,silt}_mean) band suffixes.
_TEXTURE_DEPTH = {
    'b0':   '0-5cm_mean',
    'b10':  '5-15cm_mean',
    'b30':  '15-30cm_mean',
    'b60':  '30-60cm_mean',
    'b100': '60-100cm_mean',
    'b200': '100-200cm_mean',
}

# SERVES coefficients (matches serves.js CONFIG)
_NDVI_COEFFICIENT = 1.33
_NDVI_INTERCEPT = -0.049
_LANDSAT_SCALE = 0.0000275
_LANDSAT_OFFSET = -0.2

# GEE ImageCollection ID for WUDAPT Local Climate Zones (100 m global).
_LCZ_COLLECTION = "RUB/RUBCLIM/LCZ/global_lcz_map/latest"


def _download_aligned_image(img, dem_path, geometry, output_path, n_bands=1,
                            max_pixels_per_tile_per_band=4_000_000):
    """
    Download *img*, pixel-aligned to *dem_path* (same crs/crs_transform/
    dimensions convention every function in this module uses), clipped to
    *geometry*.  Auto-tiles + mosaics when the request would exceed GEE's
    ~50MB ``getDownloadURL`` cap (empirically ~9-9.5 bytes/pixel/band for
    GEO_TIFF exports) — small requests take the original single-call path
    unchanged (byte-identical behaviour for existing basins), larger ones
    split into a grid of ``crs_transform``-shifted sub-tiles and mosaic with
    ``rasterio.merge``.  Mirrors the tiling ``dem_gee.py`` already does for
    ``region``+``scale`` downloads, adapted here for the
    ``crs_transform``+``dimensions`` alignment mode this module's ancillary
    rasters use instead.

    Returns *output_path* on success, None on failure.
    """
    import math
    import tempfile
    import urllib.request
    import rasterio
    from rasterio.merge import merge as rio_merge

    with rasterio.open(dem_path) as dem:
        crs = str(dem.crs)
        t = dem.transform
        width, height = dem.width, dem.height

    total_px = width * height
    max_px_per_tile = max(1, max_pixels_per_tile_per_band // max(1, n_bands))
    n_tiles = max(1, math.ceil(total_px / max_px_per_tile))
    side = max(1, math.ceil(math.sqrt(n_tiles)))
    nx, ny = side, side

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    if nx == 1 and ny == 1:
        crs_transform = [t.a, t.b, t.c, t.d, t.e, t.f]
        url = img.clip(geometry).getDownloadURL({
            'crs': crs, 'crs_transform': crs_transform,
            'dimensions': [width, height], 'format': 'GEO_TIFF',
        })
        urllib.request.urlretrieve(url, output_path)
        return output_path

    logger.info("Request too large for one GEE download (~%.1fM px) — "
               "tiling %dx%d and mosaicking.", total_px / 1e6, nx, ny)
    tile_w = math.ceil(width / nx)
    tile_h = math.ceil(height / ny)

    tile_paths = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for ix in range(nx):
            for iy in range(ny):
                col0, row0 = ix * tile_w, iy * tile_h
                w = min(tile_w, width - col0)
                h = min(tile_h, height - row0)
                if w <= 0 or h <= 0:
                    continue
                tile_transform = [
                    t.a, t.b, t.c + col0 * t.a + row0 * t.b,
                    t.d, t.e, t.f + col0 * t.d + row0 * t.e,
                ]
                tile_path = os.path.join(tmpdir, f"tile_{ix}_{iy}.tif")
                url = img.clip(geometry).getDownloadURL({
                    'crs': crs, 'crs_transform': tile_transform,
                    'dimensions': [w, h], 'format': 'GEO_TIFF',
                })
                urllib.request.urlretrieve(url, tile_path)
                tile_paths.append(tile_path)
                logger.info("  tile %d/%d downloaded (%dx%d)", len(tile_paths), nx * ny, w, h)

        srcs = [rasterio.open(p) for p in tile_paths]
        mosaic, out_transform = rio_merge(srcs)
        profile = srcs[0].profile.copy()
        profile.update(height=mosaic.shape[1], width=mosaic.shape[2],
                       transform=out_transform, count=mosaic.shape[0])
        for s in srcs:
            s.close()
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(mosaic)

    logger.info("Downloaded+mosaicked (%d tiles): %s", len(tile_paths), output_path)
    return output_path


def _get_land_cover_image(lulc_source, geometry=None):
    """
    Raw land cover class image for the given source (band renamed to 'Map').

    Call only after GEE has been authenticated (inside a try block that
    follows _authenticate()).

    For LCZ the collection has 6 regional tiles; filterBounds+mosaic selects
    the tile(s) covering the watershed.  'LCZ_Filter' is the smoothed band
    (preferred over raw 'LCZ' for hydraulic parameter mapping).
    """
    if lulc_source == 'lcz':
        col = ee.ImageCollection(_LCZ_COLLECTION)
        if geometry is not None:
            col = col.filterBounds(geometry)
        return col.mosaic().select('LCZ_Filter').rename('Map')
    return ee.Image('ESA/WorldCover/v200/2021').select('Map')


def _get_root_depth_image(lulc_source, lut_path, geometry=None):
    """
    GEE image with 'root_depth' band [m], remapped from land cover class codes.

    Reads class_code → root_zone_depth_m from *lut_path*.
    Call only after GEE authentication.
    """
    lut = pd.read_csv(lut_path)
    from_codes = lut['class_code'].tolist()
    to_depths  = lut['root_zone_depth_m'].tolist()
    lc_img = _get_land_cover_image(lulc_source, geometry=geometry)
    return lc_img.remap(from_codes, to_depths).rename('root_depth')




def _load_watershed_geometry(geojson_path, simplify_tolerance_m=None):
    """Load watershed GeoJSON → EPSG:4326 → ee.Geometry."""
    import geopandas as gpd

    gdf = gpd.read_file(geojson_path)
    gdf_4326 = gdf.to_crs("EPSG:4326")
    dissolved = gdf_4326.dissolve()
    geom = dissolved.geometry.iloc[0]

    if simplify_tolerance_m is not None and simplify_tolerance_m > 0:
        centroid = geom.centroid
        deg_per_m = 1.0 / (111320.0 * math.cos(math.radians(centroid.y)))
        tol_deg = simplify_tolerance_m * deg_per_m
        geom = geom.simplify(tol_deg, preserve_topology=True)

    if geom.geom_type == 'MultiPolygon':
        coords = [list(p.exterior.coords) for p in geom.geoms]
        return ee.Geometry.MultiPolygon(coords)
    else:
        coords = list(geom.exterior.coords)
        return ee.Geometry.Polygon(coords)


# ── NDVI retrieval (mirrors serves.js Sections 3–5) ─────────────────────────

def _get_ndvi_landsat(geometry, target_date, search_window):
    """Landsat 8/9 NDVI composite up to search_window days before target_date."""
    target = ee.Date(target_date)
    start = target.advance(-search_window, 'day')
    end = target.advance(1, 'day')   # backward-only: no post-event scenes

    def _mask_and_ndvi(image):
        qa = image.select('QA_PIXEL')
        mask = (qa.bitwiseAnd(1 << 1).eq(0)
                .And(qa.bitwiseAnd(1 << 3).eq(0))
                .And(qa.bitwiseAnd(1 << 4).eq(0))
                .And(qa.bitwiseAnd(1 << 5).eq(0)))
        nir = image.select('SR_B5').multiply(_LANDSAT_SCALE).add(_LANDSAT_OFFSET)
        red = image.select('SR_B4').multiply(_LANDSAT_SCALE).add(_LANDSAT_OFFSET)
        ndvi = nir.subtract(red).divide(nir.add(red)).clamp(-1, 1).rename('NDVI')
        return ndvi.updateMask(mask)

    l8 = (ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
          .filterBounds(geometry).filterDate(start, end)
          .filter(ee.Filter.lt('CLOUD_COVER', 100))
          .map(_mask_and_ndvi))
    l9 = (ee.ImageCollection('LANDSAT/LC09/C02/T1_L2')
          .filterBounds(geometry).filterDate(start, end)
          .filter(ee.Filter.lt('CLOUD_COVER', 100))
          .map(_mask_and_ndvi))

    return l8.merge(l9).median().clip(geometry)


def _get_ndvi_sentinel2(geometry, target_date, search_window):
    """Sentinel-2 NDVI composite up to search_window days before target_date."""
    target = ee.Date(target_date)
    start = target.advance(-search_window, 'day')
    end = target.advance(1, 'day')   # backward-only: no post-event scenes

    def _mask_and_ndvi(image):
        scl = image.select('SCL')
        mask = scl.eq(4).Or(scl.eq(5)).Or(scl.eq(6)).Or(scl.eq(7))
        nir = image.select('B8').divide(10000)
        red = image.select('B4').divide(10000)
        ndvi = nir.subtract(red).divide(nir.add(red)).clamp(-1, 1).rename('NDVI')
        return ndvi.updateMask(mask)

    s2 = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
          .filterBounds(geometry).filterDate(start, end)
          .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 100))
          .map(_mask_and_ndvi))

    return s2.median().clip(geometry)


def _get_ndvi_modis(geometry, target_date, search_window=16):
    """MODIS NDVI closest 16-day composite up to search_window*2 (min 32) days before target."""
    target   = ee.Date(target_date)
    lookback = max(search_window * 2, 32)   # guarantee ≥1 MODIS 16-day composite
    modis = (ee.ImageCollection('MODIS/061/MOD13A2')
             .filterBounds(geometry)
             .filterDate(target.advance(-lookback, 'day'), target.advance(1, 'day')))

    closest = ee.Image(
        modis.map(lambda img: img.set(
            'date_diff',
            ee.Date(img.get('system:time_start')).difference(target, 'day').abs()
        )).sort('date_diff').first()
    )
    return closest.select('NDVI').multiply(0.0001).rename('NDVI').clip(geometry)


def _get_ndvi(geometry, target_date, satellite, search_window):
    """Dispatch to the appropriate NDVI retriever."""
    if satellite == 'sentinel2':
        return _get_ndvi_sentinel2(geometry, target_date, search_window)
    elif satellite == 'modis':
        return _get_ndvi_modis(geometry, target_date, search_window)
    else:
        return _get_ndvi_landsat(geometry, target_date, search_window)


# ── LULC / LCZ raster download ─────────────────────────────────────────────

def download_lulc_raster(dem_path, watershed_geojson_path, output_path,
                         project=None):
    """
    Download ESA WorldCover 2021 from GEE, pixel-aligned to the routing DEM.

    Uses ``crs_transform`` + ``dimensions`` from the DEM so the output
    raster is on the exact same grid — no rasterio reproject needed.

    Caches to *output_path*; skips download if the file already exists.

    Returns the output path on success, or None on failure.
    """
    if os.path.isfile(output_path):
        logger.info("LULC raster cached: %s", output_path)
        return output_path

    if not GEE_AVAILABLE:
        logger.warning("earthengine-api not installed")
        return None

    if not _authenticate(project):
        return None

    try:
        geometry = _load_watershed_geometry(watershed_geojson_path)
        lulc = ee.Image('ESA/WorldCover/v200/2021').select('Map')

        result = _download_aligned_image(lulc, dem_path, geometry, output_path, n_bands=1)
        logger.info("LULC raster downloaded: %s", output_path)
        return result

    except Exception as exc:
        logger.warning("GEE LULC download failed: %s", exc)
        return None


def download_lcz_raster(dem_path, watershed_geojson_path, output_path,
                        project=None):
    """
    Download WUDAPT Local Climate Zones from GEE (RUB/RUBCLIM/LCZ/global_lcz_map/latest),
    pixel-aligned to the routing DEM.

    Class codes 1–10 are built-up LCZ types; 11–17 are natural types (A–G).
    Caches to *output_path*; skips download if the file already exists.

    Returns the output path on success, or None on failure.
    """
    if os.path.isfile(output_path):
        logger.info("LCZ raster cached: %s", output_path)
        return output_path

    if not GEE_AVAILABLE:
        logger.warning("earthengine-api not installed")
        return None

    if not _authenticate(project):
        return None

    try:
        geometry = _load_watershed_geometry(watershed_geojson_path)
        lcz = (ee.ImageCollection(_LCZ_COLLECTION)
               .filterBounds(geometry)
               .mosaic()
               .select('LCZ_Filter'))

        result = _download_aligned_image(lcz, dem_path, geometry, output_path, n_bands=1)
        logger.info("LCZ raster downloaded: %s", output_path)
        return result

    except Exception as exc:
        logger.warning("GEE LCZ download failed: %s", exc)
        return None


def download_deficit_raster(dem_path, watershed_geojson_path, output_path,
                            lookup_csv_path, target_date,
                            satellite='landsat', search_window=16,
                            soil_depth_band='b30', project=None,
                            lulc_source='worldcover'):
    """
    Download the SM-deficit raster ``(porosity − θ_SERVES) × Z_r`` from GEE,
    pixel-aligned to the routing DEM.  Cached to *output_path*.

    Returns the output path on success, or None on failure.
    """
    if os.path.isfile(output_path):
        logger.info("Deficit raster cached: %s", output_path)
        return output_path

    if not GEE_AVAILABLE:
        logger.warning("earthengine-api not installed")
        return None

    if not _authenticate(project):
        return None

    try:
        geometry = _load_watershed_geometry(watershed_geojson_path)

        band = _DEPTH_BANDS.get(soil_depth_band, 'val_15_30cm_mean')
        root_depth = _get_root_depth_image(lulc_source, lookup_csv_path,
                                           geometry=geometry)
        ndvi = _get_ndvi(geometry, target_date, satellite, search_window)

        # NDVI coverage check (backward window only — post-storm scenes excluded)
        _cov = ndvi.mask().rename('cov').reduceRegion(
            reducer=ee.Reducer.mean(), geometry=geometry,
            scale=30.0, bestEffort=True, maxPixels=int(1e8),
        ).getInfo().get('cov')
        if _cov is not None:
            _pct = 100.0 * float(_cov)
            print(f"  NDVI coverage   |  {_pct:.0f}%  "
                  f"(backward window={search_window} d  satellite={satellite})")
            if _pct < 70.0:
                print(f"  [WARN] NDVI coverage {_pct:.0f}% < 70% — deficit raster "
                      f"may be unreliable. Consider increasing "
                      f"SERVES_SEARCH_WINDOW beyond {search_window} days.")

        fc = ee.Image('ISRIC/SoilGrids250m/v2_0/wv0033').select(band) \
            .rename('fc')
        wp = ee.Image('ISRIC/SoilGrids250m/v2_0/wv1500').select(band) \
            .rename('wp')

        wcsat_col = ee.ImageCollection(
            "projects/sat-io/open-datasets/HiHydroSoilv2_0/wcsat"
        )
        porosity = wcsat_col.mosaic().multiply(0.0001).rename('porosity')

        et_frac = ndvi.multiply(_NDVI_COEFFICIENT).add(_NDVI_INTERCEPT) \
            .clamp(0, 1)
        paw = fc.subtract(wp)
        theta = et_frac.multiply(paw).add(wp)
        theta = theta.where(theta.lt(wp), wp)
        theta = theta.where(theta.gt(fc), fc)

        deficit = porosity.subtract(theta).max(0) \
            .multiply(root_depth).rename('deficit')

        result = _download_aligned_image(deficit, dem_path, geometry, output_path, n_bands=1)
        logger.info("Deficit raster downloaded: %s", output_path)
        return result

    except Exception as exc:
        logger.warning("GEE deficit raster download failed: %s", exc)
        return None


def download_ksat_raster(dem_path, watershed_geojson_path, output_path,
                         project=None):
    """
    Download the HiHydroSoil v2.0 vertical saturated hydraulic conductivity
    (Ksat) raster, pixel-aligned to the routing DEM.  Cached to *output_path*.

    Output units: **mm/hr** (human-readable; range ~0–625).

    HiHydroSoil v2.0 stores every layer as int = float × 10000, so the physical
    value (cm/day) is raw × 0.0001 (verified against the dataset docs — same
    factor used for `wcsat`).  cm/day → mm/hr is × (10/24).  This Ksat is the
    *vertical* surface conductivity for Green-Ampt infiltration — NOT the lateral
    transmissivity OPM_K_SAT that drives the sandbox Darcy drainage.

    Returns the output path on success, or None on failure.
    """
    if os.path.isfile(output_path):
        logger.info("Ksat raster cached: %s", output_path)
        return output_path

    if not GEE_AVAILABLE:
        logger.warning("earthengine-api not installed")
        return None

    if not _authenticate(project):
        return None

    try:
        geometry = _load_watershed_geometry(watershed_geojson_path)

        ksat_col = ee.ImageCollection(
            "projects/sat-io/open-datasets/HiHydroSoilv2_0/ksat"
        )
        # raw × 0.0001 → cm/day ;  × (10/24) → mm/hr
        ksat_mmhr = ksat_col.mosaic().multiply(0.0001) \
            .multiply(10.0 / 24.0).rename('ksat_mmhr')

        result = _download_aligned_image(ksat_mmhr, dem_path, geometry, output_path, n_bands=1)
        logger.info("Ksat raster downloaded: %s", output_path)
        return result

    except Exception as exc:
        logger.warning("GEE Ksat raster download failed: %s", exc)
        return None


def download_texture_raster(dem_path, watershed_geojson_path, output_path,
                            soil_depth_band='b30', project=None):
    """
    Download a 2-band SoilGrids texture raster, pixel-aligned to the routing DEM:
        band 1 = sand %,  band 2 = clay %   (stored g/kg → ×0.1 → %).
    Used to derive the Green-Ampt wetting-front suction ψ per cell (USDA texture
    class → Rawls 1983 table), classified client-side.  Cached to *output_path*.

    Returns the output path on success, or None on failure.
    """
    if os.path.isfile(output_path):
        logger.info("Texture raster cached: %s", output_path)
        return output_path

    if not GEE_AVAILABLE:
        logger.warning("earthengine-api not installed")
        return None

    if not _authenticate(project):
        return None

    try:
        geometry = _load_watershed_geometry(watershed_geojson_path)
        suf = _TEXTURE_DEPTH.get(soil_depth_band, '15-30cm_mean')
        sand = ee.Image('projects/soilgrids-isric/sand_mean') \
            .select(f'sand_{suf}').multiply(0.1).rename('sand_pct')
        clay = ee.Image('projects/soilgrids-isric/clay_mean') \
            .select(f'clay_{suf}').multiply(0.1).rename('clay_pct')
        img = sand.addBands(clay)

        result = _download_aligned_image(img, dem_path, geometry, output_path, n_bands=2)
        logger.info("Texture raster downloaded: %s", output_path)
        return result

    except Exception as exc:
        logger.warning("GEE texture raster download failed: %s", exc)
        return None


# ── Main pipeline ────────────────────────────────────────────────────────────

def compute_opm_params(
    watershed_geojson_path,
    cell_size,
    lookup_csv_path,
    target_date,
    satellite='landsat',
    search_window=16,
    soil_depth_band='b30',
    project=None,
    gauge_csv_path=None,
    target_crs='EPSG:32645',
    sd_reducer='mean',
    lulc_source='worldcover',
):
    """
    Single GEE pipeline: SERVES + LULC + SoilGrids → OPM parameters.

    SM_deficit = (porosity − θ_SERVES) × Z_r
    SD_max     = max(SM_deficit) over the watershed
    φ          = mean(porosity − FC)

    When gauge_csv_path is provided, also computes per-polygon max(deficit)
    using Voronoi (Thiessen) polygons built from gauge locations.

    Returns
    -------
    dict or None
        {sd_max, sd_min, phi, sd_max_per_polygon (if gauges), ...}
        sd_min is a fixed OPM floor of 0.001 m, not a SERVES-derived value.
    """
    if not GEE_AVAILABLE:
        logger.warning("earthengine-api not installed")
        return None

    if project and not os.environ.get('GEE_PROJECT'):
        os.environ['GEE_PROJECT'] = project

    if not _authenticate(project):
        return None

    try:
        band = _DEPTH_BANDS.get(soil_depth_band, 'val_15_30cm_mean')

        geometry = _load_watershed_geometry(
            watershed_geojson_path,
            simplify_tolerance_m=cell_size,
        )

        # ── Scale: floor at Landsat 30 m (used for all reduceRegion calls) ──
        scale = max(float(cell_size), 30.0)

        # ── Land cover → root zone depth Z_r (WorldCover or LCZ) ─────────
        lulc = _get_land_cover_image(lulc_source, geometry=geometry)
        root_depth = _get_root_depth_image(lulc_source, lookup_csv_path,
                                           geometry=geometry)

        # ── NDVI ─────────────────────────────────────────────────────────
        ndvi = _get_ndvi(geometry, target_date, satellite, search_window)

        # NDVI coverage check (backward window only — post-storm scenes excluded)
        _cov = ndvi.mask().rename('cov').reduceRegion(
            reducer=ee.Reducer.mean(), geometry=geometry,
            scale=scale, bestEffort=True, maxPixels=int(1e8),
        ).getInfo().get('cov')
        if _cov is not None:
            _pct = 100.0 * float(_cov)
            print(f"  NDVI coverage   |  {_pct:.0f}%  "
                  f"(backward window={search_window} d  satellite={satellite})")
            if _pct < 70.0:
                print(f"  [WARN] NDVI coverage {_pct:.0f}% < 70% — SD_max estimate "
                      f"may be unreliable. Consider increasing "
                      f"SERVES_SEARCH_WINDOW beyond {search_window} days.")

        # ── Soil properties from SoilGrids ───────────────────────────────
        fc = ee.Image('ISRIC/SoilGrids250m/v2_0/wv0033').select(band) \
            .rename('fc')
        wp = ee.Image('ISRIC/SoilGrids250m/v2_0/wv1500').select(band) \
            .rename('wp')

        # ── Porosity from HiHydroSoil v2.0 wcsat ────────────────────────
        wcsat_col = ee.ImageCollection(
            "projects/sat-io/open-datasets/HiHydroSoilv2_0/wcsat"
        )
        porosity = wcsat_col.mosaic().multiply(0.0001).rename('porosity')

        # ── SERVES θ (mirrors serves.js:452–466) ────────────────────────
        et_frac = ndvi.multiply(_NDVI_COEFFICIENT).add(_NDVI_INTERCEPT) \
            .clamp(0, 1).rename('et_frac')

        paw = fc.subtract(wp)   # plant available water
        theta = et_frac.multiply(paw).add(wp).rename('theta')
        theta = theta.where(theta.lt(wp), wp)
        theta = theta.where(theta.gt(fc), fc)

        # Handle water bodies: assign θ = FC (serves.js:518)
        water = ee.Image("JRC/GSW1_4/GlobalSurfaceWater") \
            .select('occurrence').gt(0)
        theta = theta.where(water, fc)

        # Handle negative NDVI: assign θ = FC (serves.js:522)
        neg_ndvi = ndvi.lt(0)
        theta = theta.where(neg_ndvi, fc)

        # ── SM_deficit = (porosity − θ) × Z_r ───────────────────────────
        deficit = porosity.subtract(theta).max(0) \
            .multiply(root_depth).rename('deficit')

        # ── Reduce over watershed ────────────────────────────────────────
        _reducer = ee.Reducer.mean() if sd_reducer == 'mean' else ee.Reducer.max()
        deficit_stats = deficit.reduceRegion(
            reducer=_reducer,
            geometry=geometry,
            scale=scale,
            bestEffort=True,
            maxPixels=int(1e9),
        ).getInfo()

        sd_max_raw = deficit_stats.get('deficit')
        if sd_max_raw is None:
            logger.warning(
                "SERVES deficit returned null — check NDVI coverage for "
                "target date %s", target_date
            )
            return None

        sd_max = float(sd_max_raw)
        sd_min = 0.001

        if sd_max <= 0:
            logger.warning(
                "Computed SD_max=%.4f m is non-positive; "
                "soil may be fully saturated at target date.", sd_max
            )
            return None

        # ── θ diagnostics ────────────────────────────────────────────────
        theta_stats = theta.reduceRegion(
            reducer=ee.Reducer.minMax(),
            geometry=geometry,
            scale=scale,
            bestEffort=True,
            maxPixels=int(1e9),
        ).getInfo()

        # ── Root depth diagnostics (scale: 10 m for WorldCover, 100 m for LCZ) ─
        _lc_scale = 100.0 if lulc_source == 'lcz' else 10.0
        root_stats = root_depth.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=geometry,
            scale=_lc_scale,
            bestEffort=True,
            maxPixels=int(1e9),
        ).getInfo()

        # ── Land cover class distribution ─────────────────────────────────
        lulc_hist = lulc.reduceRegion(
            reducer=ee.Reducer.frequencyHistogram(),
            geometry=geometry,
            scale=_lc_scale,
            bestEffort=True,
            maxPixels=int(1e9),
        ).getInfo()
        lulc_fractions = lulc_hist.get('Map', lulc_hist.get('remapped', {}))

        # ── φ = porosity − FC (porosity already computed above) ──────────
        phi_img = porosity.subtract(fc).rename('phi')

        phi_stats = phi_img.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=geometry,
            scale=max(float(cell_size), 250.0),
            bestEffort=True,
            maxPixels=int(1e9),
        ).getInfo()

        phi_mean = float(phi_stats.get('phi', 0.10))
        if phi_mean <= 0:
            logger.warning(
                "Computed φ=%.4f is non-positive; using default 0.10", phi_mean
            )
            phi_mean = 0.10

        # NOTE: per-zone SD is no longer reduced here over Voronoi polygons.
        # That rebuilt the precipitation partition independently and dropped
        # zones whose station sits outside the watershed.  The engine now reduces
        # the deficit raster (download_deficit_raster) per zone using the exact
        # rainfall partition (cell_polygon), so the two can never disagree.

        result = {
            'sd_max':              sd_max,
            'sd_min':              sd_min,
            'phi':                 phi_mean,
            'theta_min':           float(theta_stats.get('theta_min', 0)),
            'theta_max':           float(theta_stats.get('theta_max', 0)),
            'root_depth_mean':     float(root_stats.get('root_depth', 0)),
            'lulc_fractions':      lulc_fractions,
            'target_date':         target_date,
            'satellite':           satellite,
            'soil_depth_band':     soil_depth_band,
            'source':              f'serves_{lulc_source}_gee',
        }
        return result

    except Exception as exc:
        logger.warning("GEE SERVES/LULC query failed: %s", exc)
        return None
