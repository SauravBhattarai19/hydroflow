import os
import sys
import rasterio
import rasterio.features
from rasterio.warp import calculate_default_transform, reproject, Resampling
from pyproj import CRS, Transformer
import numpy as np
import geopandas as gpd
from shapely.geometry import shape

# Monkey-patch np.in1d for pysheds compatibility with numpy 2.0+
if not hasattr(np, 'in1d'):
    np.in1d = np.isin

import warnings


def _import_pysheds_grid():
    """Import pysheds' Grid, tolerating a too-old numba on the host Python.

    pysheds JIT-compiles its kernels with numba.  Some bundled interpreters ship
    a numba that is too old for their own Python — e.g. QGIS-LTR on macOS bundles
    numba 0.50.1, which cannot compile the ``LIST_EXTEND`` bytecode emitted by
    Python 3.9, so ``from pysheds.grid import Grid`` raises at import time.  When
    that happens we switch numba to pure-Python mode (``DISABLE_JIT``) and retry:
    pysheds then runs interpreted — slower, but with identical results — instead
    of crashing the DEM stage.  On a healthy environment the first import
    succeeds and JIT stays on, so nothing changes.
    """
    try:
        from pysheds.grid import Grid
        return Grid
    except Exception as exc:  # noqa: BLE001 — numba compile errors, etc.
        try:
            import numba
            numba.config.DISABLE_JIT = True
        except Exception as numba_exc:  # noqa: BLE001
            # `import numba` itself failing (as opposed to numba importing fine
            # but pysheds' @njit compile step choking on it) is usually a
            # compiled-extension/numpy ABI mismatch (e.g. an OS-packaged numba
            # built against a different numpy than the one actually on
            # sys.path) rather than a "too old to JIT this bytecode" problem —
            # DISABLE_JIT can't fix that, so say so plainly instead of a bare
            # re-raise of an opaque SystemError.
            raise RuntimeError(
                "pysheds could not be imported "
                f"({type(exc).__name__}: {exc}), and numba itself could not be "
                f"imported either to try the pure-Python fallback "
                f"({type(numba_exc).__name__}: {numba_exc}). This combination "
                "usually means the installed numba's compiled extension does "
                "not match the installed numpy (e.g. an OS/apt-packaged numba "
                "mixed with a separately pip-installed numpy). Try: "
                "pip install --upgrade numba"
            ) from numba_exc
        # Drop the half-imported pysheds modules so its @njit decorators re-run
        # in disabled mode on the retry below.
        for _m in [m for m in list(sys.modules)
                   if m == "pysheds" or m.startswith("pysheds.")]:
            del sys.modules[_m]
        pyver = f"{sys.version_info.major}.{sys.version_info.minor}"
        nbver = getattr(numba, "__version__", "?")
        try:
            from pysheds.grid import Grid
            print(f"[WARN] This interpreter's numba ({nbver}) cannot JIT-compile "
                  f"pysheds on Python {pyver} ({type(exc).__name__}).  "
                  f"Running pysheds in pure-Python compatibility mode — results are "
                  f"identical, only slower (a one-time DEM step).")
            return Grid
        except Exception as exc2:  # noqa: BLE001
            raise RuntimeError(
                "pysheds still failed to import even after disabling numba JIT "
                f"({type(exc2).__name__}: {exc2}). If this is the same error as "
                "before disabling JIT, numba's compiled internals are likely "
                "broken/mismatched with numpy regardless of the JIT setting — "
                "try: pip install --upgrade numba"
            ) from exc2


_Grid = None


def _get_pysheds_grid():
    """Lazily import pysheds' Grid the first time the pysheds engine is
    actually used.

    Deliberately NOT imported at module load time: on an environment where
    pysheds/numba are broken (see ``_import_pysheds_grid``'s ABI-mismatch
    branch above — encountered for real during this feature's own testing,
    not hypothetical), an eager import here would make the *entire*
    ``dem_processing`` module fail to import — including the unrelated
    ``pyflwdir`` engine, which doesn't need pysheds at all.  Since pyflwdir is
    now the plugin's default engine, a broken pysheds must not block it.
    """
    global _Grid
    if _Grid is None:
        _Grid = _import_pysheds_grid()
    return _Grid

# Suppress numba warnings from pysheds
warnings.filterwarnings("ignore", message="The TBB threading layer requires TBB version")


def _import_pyflwdir():
    """Import pyflwdir, tolerating a too-old numba on the host Python.

    pyflwdir JIT-compiles its D8/accumulation kernels with numba (its own
    ``@njit`` decorators, e.g. in ``pyflwdir.core_d8``) — the same numba
    runtime pysheds uses, so the same failure mode is possible in principle
    (e.g. QGIS-LTR bundling a numba too old for its bundled Python).  Unlike
    ``_import_pysheds_grid`` above, pyflwdir does not itself expose a documented
    "disable JIT" switch, but ``numba.config.DISABLE_JIT`` is a global flag on
    the numba runtime itself, not something owned by pysheds — it applies to
    *any* package's ``@njit`` functions imported afterwards, so the same
    retry-in-pure-Python trick has been verified to work for pyflwdir too
    (import pyflwdir fresh after flipping the flag).  If pyflwdir still fails
    after that retry, we give up with a clear, actionable error rather than a
    raw numba traceback.  ``pyflwdir`` is the QGIS plugin's default engine
    (see ``vsa_opm.config.OpmConfig.DELINEATION_ENGINE``), so this path IS on
    the critical path for most users — it must degrade to a clear message,
    not a bare traceback, and must not be masked by an unrelated pysheds
    failure (pysheds' own ``Grid`` is imported lazily elsewhere in this module
    for exactly that reason).
    """
    try:
        import pyflwdir
        return pyflwdir
    except Exception as exc:  # noqa: BLE001 — numba compile errors, etc.
        try:
            import numba
            numba.config.DISABLE_JIT = True
        except Exception as numba_exc:  # noqa: BLE001
            # `import numba` itself failing (vs. numba importing fine but
            # pyflwdir's @njit compile choking on it) usually means a
            # compiled-extension/numpy ABI mismatch (e.g. an OS/apt-packaged
            # numba built against a different numpy than the one actually on
            # sys.path) — DISABLE_JIT can't fix that, so say so plainly.
            raise RuntimeError(
                "DELINEATION_ENGINE='pyflwdir' failed to import pyflwdir "
                f"({type(exc).__name__}: {exc}), and numba itself could not "
                f"be imported either to try the pure-Python fallback "
                f"({type(numba_exc).__name__}: {numba_exc}). This combination "
                "usually means the installed numba's compiled extension does "
                "not match the installed numpy (e.g. an OS/apt-packaged numba "
                "mixed with a separately pip-installed numpy). Try: "
                "pip install --upgrade numba"
            ) from numba_exc
        for _m in [m for m in list(sys.modules)
                   if m == "pyflwdir" or m.startswith("pyflwdir.")]:
            del sys.modules[_m]
        pyver = f"{sys.version_info.major}.{sys.version_info.minor}"
        nbver = getattr(numba, "__version__", "?")
        try:
            import pyflwdir
            print(f"[WARN] This interpreter's numba ({nbver}) could not JIT-compile "
                  f"pyflwdir on Python {pyver} ({type(exc).__name__}). "
                  f"Running pyflwdir in pure-Python compatibility mode — slower, "
                  f"same results.")
            return pyflwdir
        except Exception as exc2:  # noqa: BLE001
            raise RuntimeError(
                "DELINEATION_ENGINE='pyflwdir' could not be imported even after "
                "disabling numba JIT (NUMBA_DISABLE_JIT). This usually means the "
                "installed numba is incompatible with this Python interpreter "
                "(common in QGIS-bundled Python on some platforms). Either "
                "upgrade/reinstall numba+pyflwdir for this interpreter, or use "
                "DELINEATION_ENGINE='pysheds' (the default) instead. "
                f"Original import error: {type(exc).__name__}: {exc}. "
                f"Retry error: {type(exc2).__name__}: {exc2}"
            ) from exc2

def reproject_dem(input_dem_path, output_dem_path, target_crs_epsg):
    """
    Reprojects a DEM to a target CRS.
    """
    print(f"Reprojecting DEM: {input_dem_path} to {target_crs_epsg}")
    with rasterio.open(input_dem_path) as src:
        # Determine the target CRS
        target_crs = CRS(target_crs_epsg)

        # Calculate the transform and dimensions for the reprojected DEM
        transform, width, height = calculate_default_transform(
            src.crs, target_crs, src.width, src.height, *src.bounds
        )

        profile = src.profile
        profile.update({
            'crs': target_crs,
            'transform': transform,
            'width': width,
            'height': height,
            'nodata': src.nodata if src.nodata is not None else -9999 # Ensure nodata is set
        })

        with rasterio.open(output_dem_path, 'w', **profile) as dst:
            reproject(
                source=rasterio.band(src, 1),
                destination=rasterio.band(dst, 1),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=target_crs,
                resampling=Resampling.bilinear,
                num_threads=os.cpu_count()
            )
    print(f"Reprojected DEM saved to: {output_dem_path}")
    return output_dem_path

def _terrain(grid, dem, profile, output_dir):
    """
    Outlet-independent terrain analysis: fill sinks, resolve flats, flow
    direction and flow accumulation.  Writes filled_dem.tif, inflated_dem.tif,
    flow_direction.tif and flow_accumulation.tif.

    Returns (filled_dem, inflated_dem, flow_direction, flow_accumulation).

    This is the first half of the DEM stage — it needs no pour point, so the
    plugin can run it, show the stream network, and let the user pick an outlet
    against it before delineation (see ``analyze_terrain``).
    """
    # 1. Fill sinks
    print("Filling sinks...")
    filled_dem = grid.fill_depressions(dem)
    # Resolve flats to ensure all areas drain
    inflated_dem = grid.resolve_flats(filled_dem)

    with rasterio.open(os.path.join(output_dir,"filled_dem.tif"), 'w', **profile) as dst:
        dst.write(np.asarray(filled_dem), 1)
    print(f"Filled DEM saved to: {os.path.join(output_dir,'filled_dem.tif')}")

    # Save the hydrologically conditioned DEM (fill + resolve_flats).
    #
    # Why float32 matters for INTEGER source DEMs (e.g. SRTM int16):
    #   resolve_flats adds sub-metre increments (e.g. +0.001 m) to flat cells
    #   so the D8 algorithm can assign unambiguous flow directions.  If the file
    #   is written back as int16, those increments round to zero and the flat
    #   areas look slope-less again in the router → water pools → delayed surge.
    #   Saving as float32 preserves the increments exactly.
    #
    # If the source DEM is already floating-point the dtype is kept as-is;
    # the float32 upgrade only activates for integer-typed inputs.
    src_dtype = profile.get('dtype', 'float32')
    is_int_dem = np.dtype(src_dtype).kind in ('i', 'u')   # signed/unsigned int
    save_dtype = 'float32' if is_int_dem else src_dtype
    inflated_arr = np.asarray(inflated_dem).astype(save_dtype)
    # Replace any NaN or original nodata with a safe sentinel
    orig_nodata = profile.get('nodata')
    if orig_nodata is not None:
        inflated_arr[inflated_arr == np.array(orig_nodata, dtype=save_dtype)] = -9999.0
    inflated_arr[~np.isfinite(inflated_arr)] = -9999.0
    inflated_profile = profile.copy()
    inflated_profile.update(dtype=save_dtype, nodata=-9999.0)
    with rasterio.open(os.path.join(output_dir,"inflated_dem.tif"), 'w', **inflated_profile) as dst:
        dst.write(inflated_arr, 1)
    print(f"Inflated DEM (float32) saved to: {os.path.join(output_dir,'inflated_dem.tif')}")

    # 2. Flow direction
    print("Calculating flow direction...")
    # Specify directional mapping (N, NE, E, SE, S, SW, W, NW)
    dirmap = (64, 128, 1, 2, 4, 8, 16, 32)
    flow_direction = grid.flowdir(inflated_dem, dirmap=dirmap)

    fd_profile = profile.copy()
    fd_profile.update(dtype=flow_direction.dtype, nodata=None)
    with rasterio.open(os.path.join(output_dir,"flow_direction.tif"), 'w', **fd_profile) as dst:
        dst.write(np.asarray(flow_direction), 1)
    print(f"Flow direction saved to: {os.path.join(output_dir,'flow_direction.tif')}")

    # 3. Flow accumulation (contributing area)
    print("Calculating flow accumulation (contributing area)...")
    flow_accumulation = grid.accumulation(flow_direction, dirmap=dirmap)

    fa_profile = profile.copy()
    fa_profile.update(dtype=flow_accumulation.dtype, nodata=None)
    with rasterio.open(os.path.join(output_dir,"flow_accumulation.tif"), 'w', **fa_profile) as dst:
        dst.write(np.asarray(flow_accumulation), 1)
    print(f"Flow accumulation saved to: {os.path.join(output_dir,'flow_accumulation.tif')}")

    return filled_dem, inflated_dem, flow_direction, flow_accumulation


# ── D8 encoding note (pyflwdir path) ────────────────────────────────────────
# pyflwdir.core_d8._ds = [[32, 64, 128], [16, 0, 1], [8, 4, 2]] indexed by
# [dr+1, dc+1] (dr,dc = downstream row/col delta) is, cell-for-cell, the exact
# same per-direction CODE MEANING as pysheds' dirmap (N=64, NE=128, E=1, SE=2,
# S=4, SW=8, W=16, NW=32) used above and in vsa_opm/core/routing/terrain.py's
# D8_MOVE table.  Verified by reading pyflwdir's source directly (_ds matrix)
# and confirmed empirically with an unambiguous monotonic-cone synthetic DEM
# (8/8 directions match) — so a code of e.g. 64 always means "flows to the
# cell to the north" on both engines, and no remapping table is needed;
# ``flw.to_array(ftype="d8")`` / the ``d8`` array from fill_depressions() can
# be written straight to flow_direction.tif and terrain.py's D8_MOVE table
# will decode it correctly regardless of which engine produced it.
#
# IMPORTANT CAVEAT (verified this session, not merely assumed): the two
# engines' fill/direction ALGORITHMS pick different directions on the same
# terrain, even where neither's fill actually changed the elevation —
# pysheds selects the neighbor with the steepest SLOPE (elevation drop /
# distance, so diagonal neighbors are penalized by the extra sqrt(2)
# distance), matching the classic ESRI/O'Callaghan & Mark convention. pyflwdir
# (Wang & Liu 2006, the algorithm dem.fill_depressions/from_dem implements)
# selects the neighbor with the steepest elevation DROP with no diagonal-
# distance penalty. On a controlled monotonic-cone test DEM with zero filling
# needed, pysheds matched slope-based steepest descent 100% of cells while
# pyflwdir matched drop-based steepest descent 100% of cells — i.e. each
# engine is internally consistent and correctly implements its own documented
# algorithm, but the two will disagree on a meaningful fraction of individual
# cells (observed ~40% disagreement on steep terrain in the dem_250.tif smoke
# test) purely from this tie-breaking-rule difference, not from a bug. This is
# in addition to the (expected, already documented) differences from the two
# engines' distinct depression-filling strategies. Net effect: whole-basin
# aggregates (delineated area, hydrograph shape) have been validated to agree
# closely between engines (see task session notes: Pearl River +2.6%/-0.7%
# area error vs. USGS gauges; dem_250.tif smoke test: 12,331 vs 12,340 km²,
# both single-component/fully-connected watersheds) — but do NOT expect
# per-cell flow_direction.tif equality between engines, even away from flats
# or reservoirs. Flag this for anyone doing cell-level D8 comparisons.
_PYSHEDS_DIRMAP = (64, 128, 1, 2, 4, 8, 16, 32)  # (N, NE, E, SE, S, SW, W, NW)


def _terrain_pyflwdir(dem_path, profile, output_dir):
    """
    pyflwdir equivalent of ``_terrain``: priority-flood depression filling
    (via ``pyflwdir.from_dem``) instead of pysheds' fill_depressions +
    resolve_flats, D8 flow direction and flow accumulation (cell counts).

    Writes the same four files ``_terrain`` writes (filled_dem.tif,
    inflated_dem.tif, flow_direction.tif, flow_accumulation.tif) with the same
    dtypes/profiles/conventions, so downstream code (terrain.py::load_rasters,
    the router, the QGIS plugin) works unchanged regardless of which engine
    produced them.

    There is no pysheds-equivalent "filled then separately resolve_flats"
    split for pyflwdir's priority-flood algorithm — filled_dem.tif and
    inflated_dem.tif both hold ``pyflwdir``'s single conditioned DEM.  That is
    expected: the whole point of this engine is a different (correct, for
    reservoir/flat-dominated basins) conditioning strategy.

    Returns (conditioned_dem, flow_direction_array, flow_accumulation_array,
    flw) — ``flw`` (the pyflwdir.FlwdirRaster) is returned too since
    ``_delineate_pyflwdir`` needs it for basin delineation; it cannot be
    reconstructed cheaply from the raster files alone.
    """
    pyflwdir = _import_pyflwdir()

    with rasterio.open(dem_path) as src:
        elevtn = src.read(1).astype(np.float64)
        nodata = src.nodata
        transform = src.transform

    if nodata is None:
        nodata = -9999.0
        elevtn = np.where(~np.isfinite(elevtn), nodata, elevtn)

    print("Running pyflwdir priority-flood depression fill + D8 "
          "(pyflwdir.dem.fill_depressions, the same routine pyflwdir.from_dem "
          "uses internally — called directly here so we get both the "
          "conditioned DEM and the D8 array from a single fill pass)...")
    conditioned_dem, d8 = pyflwdir.dem.fill_depressions(
        elevtn, outlets="edge", nodata=nodata,
    )
    flw = pyflwdir.from_array(
        d8, ftype="d8", check_ftype=False, transform=transform, latlon=False,
    )
    n_resolved = len(flw.idxs_seq)
    print(f"  idxs_seq coverage: {n_resolved}/{flw.size} "
          f"({100.0 * n_resolved / flw.size:.1f}%)")
    conditioned_dem = conditioned_dem.astype(np.float64)

    # filled_dem.tif: kept as the source dtype like pysheds' filled_dem.tif.
    with rasterio.open(os.path.join(output_dir, "filled_dem.tif"), 'w', **profile) as dst:
        dst.write(conditioned_dem.astype(profile['dtype']), 1)
    print(f"Filled DEM (pyflwdir-conditioned) saved to: "
          f"{os.path.join(output_dir, 'filled_dem.tif')}")

    # inflated_dem.tif: float32 (or source float dtype), -9999 sentinel — same
    # convention _terrain uses, so clip_dem_by_watershed() downstream is
    # unaffected by which engine produced it.
    src_dtype = profile.get('dtype', 'float32')
    is_int_dem = np.dtype(src_dtype).kind in ('i', 'u')
    save_dtype = 'float32' if is_int_dem else src_dtype
    inflated_arr = conditioned_dem.astype(save_dtype)
    inflated_arr[~np.isfinite(inflated_arr)] = -9999.0
    inflated_arr[inflated_arr == np.array(nodata, dtype=save_dtype)] = -9999.0
    inflated_profile = profile.copy()
    inflated_profile.update(dtype=save_dtype, nodata=-9999.0)
    with rasterio.open(os.path.join(output_dir, "inflated_dem.tif"), 'w', **inflated_profile) as dst:
        dst.write(inflated_arr, 1)
    print(f"Inflated DEM (float32) saved to: {os.path.join(output_dir, 'inflated_dem.tif')}")

    # D8 flow direction — same per-cell code convention as pysheds (verified
    # above), so no remap needed.  Use the `d8` array from fill_depressions()
    # directly (it's what `flw` was built from) rather than round-tripping
    # through flw.to_array(), which is equivalent but redundant.
    flow_direction = d8
    fd_profile = profile.copy()
    fd_profile.update(dtype=flow_direction.dtype, nodata=None)
    with rasterio.open(os.path.join(output_dir, "flow_direction.tif"), 'w', **fd_profile) as dst:
        dst.write(np.asarray(flow_direction), 1)
    print(f"Flow direction saved to: {os.path.join(output_dir, 'flow_direction.tif')}")

    # Flow accumulation — CELL COUNTS, not km^2.  vsa.py multiplies by cell
    # area itself downstream (upslope_area = faccum_1d * cell_area), so this
    # must match pysheds' grid.accumulation() convention exactly.
    flow_accumulation = flw.upstream_area(unit="cell")
    fa_profile = profile.copy()
    fa_profile.update(dtype=flow_accumulation.dtype, nodata=None)
    with rasterio.open(os.path.join(output_dir, "flow_accumulation.tif"), 'w', **fa_profile) as dst:
        dst.write(np.asarray(flow_accumulation), 1)
    print(f"Flow accumulation saved to: {os.path.join(output_dir, 'flow_accumulation.tif')}")

    return conditioned_dem, flow_direction, flow_accumulation, flw


def _stream_mask(flow_accumulation):
    """The stream cells used both for outlet snapping and for the display layer.

    Top 1 % of cells by accumulation, minimum 1.  A fixed value (e.g. 1000)
    fails on coarse or small DEMs where the total cell count is below the
    threshold, leaving the stream mask empty.
    """
    n_cells_total = int(flow_accumulation.size)
    stream_threshold = max(1, n_cells_total // 100)
    return flow_accumulation > stream_threshold, stream_threshold, n_cells_total


def _delineate_pyflwdir(flw, conditioned_dem, flow_direction, flow_accumulation,
                         profile, output_point_latlon, target_crs_epsg, output_dir,
                         crop_buffer_cells=20):
    """
    pyflwdir equivalent of ``_delineate``: transform the pour point, snap it to
    the stream network (nearest stream cell, matching pysheds' snap_to_mask
    semantics rather than pyflwdir's own downstream-trace ``snap()``), delineate
    the basin, and write watershed.tif / watershed.geojson plus already-clipped
    clipped_dem.tif / clipped_flow_accumulation.tif tightly cropped to the
    watershed bounding box (+ ``crop_buffer_cells`` cell buffer).  Mirrors the
    validated Pearl-River scratch script's export step, and avoids shipping a
    full-DEM-extent raster for basins much smaller than their source DEM tile
    (e.g. large GEE-downloaded ancillary rasters keyed off this raster's
    extent).

    The pysheds path's ``_delineate`` + ``clip_dem_by_watershed`` /
    ``clip_flow_accumulation_by_watershed`` are untouched by this function and
    keep writing full-extent watershed.tif; the crop-to-bbox behaviour here is
    pyflwdir-path-only, per the task spec.
    """
    src_crs_latlon = CRS("EPSG:4326")
    target_crs = CRS(target_crs_epsg)
    transformer = Transformer.from_crs(src_crs_latlon, target_crs, always_xy=True)
    output_point_x, output_point_y = transformer.transform(
        output_point_latlon[1], output_point_latlon[0])
    print(f"Output point (lat/lon): {output_point_latlon}")
    print(f"Output point (projected): ({output_point_x}, {output_point_y})")

    transform = profile['transform']
    height, width = flow_accumulation.shape
    bounds = rasterio.transform.array_bounds(height, width, transform)  # (l, b, r, t)
    if not (bounds[0] <= output_point_x <= bounds[2] and bounds[1] <= output_point_y <= bounds[3]):
        print("Warning: Output point is outside the DEM's bounds. Watershed delineation might fail or be empty.")

    print("Snapping outlet point to nearest stream cell...")
    streams, stream_threshold, n_cells_total = _stream_mask(flow_accumulation)
    print(f"  Stream threshold: {stream_threshold} cells  (1 % of {n_cells_total} total)")

    # Nearest stream cell to the raw click point — same semantics as pysheds'
    # grid.snap_to_mask (nearest-in-any-direction), not pyflwdir's own
    # FlwdirRaster.snap() (which traces strictly downstream from a start
    # point and would silently relocate a slightly-off-channel click).
    col_click, row_click = ~transform * (output_point_x, output_point_y)
    row_click, col_click = int(round(row_click)), int(round(col_click))
    stream_rows, stream_cols = np.where(np.asarray(streams))
    if stream_rows.size == 0:
        raise RuntimeError("No stream cells found above threshold — cannot snap outlet.")
    dist2 = (stream_rows - row_click) ** 2 + (stream_cols - col_click) ** 2
    nearest = np.argmin(dist2)
    snap_row, snap_col = int(stream_rows[nearest]), int(stream_cols[nearest])
    snap_x, snap_y = transform * (snap_col + 0.5, snap_row + 0.5)
    print(f"Output point (projected): ({output_point_x:.2f}, {output_point_y:.2f})")
    print(f"Snapped outlet (projected): ({snap_x:.2f}, {snap_y:.2f})")
    print(f"Snapped cell accumulation: {flow_accumulation[snap_row, snap_col]}")

    print("Delineating watershed...")
    idx0 = snap_row * width + snap_col
    basin = flw.basins(idxs=np.array([idx0], dtype=flw.idxs_ds.dtype))
    watershed = (np.asarray(basin) > 0)

    watershed_uint8 = watershed.astype('uint8')
    ws_profile = profile.copy()
    ws_profile.update(dtype=watershed_uint8.dtype, nodata=0)
    with rasterio.open(os.path.join(output_dir, "watershed.tif"), 'w', **ws_profile) as dst:
        dst.write(watershed_uint8, 1)
    print(f"Watershed saved to: {os.path.join(output_dir, 'watershed.tif')}")

    print("Exporting watershed to GeoJSON...")
    shapes = rasterio.features.shapes(watershed_uint8, transform=transform)
    polygons = [shape(geom) for geom, val in shapes if val == 1]
    if polygons:
        gdf = gpd.GeoDataFrame({'geometry': polygons}, crs=target_crs_epsg)
        gdf.to_file(os.path.join(output_dir, 'watershed.geojson'), driver='GeoJSON')
        print(f"Watershed vector saved to: {os.path.join(output_dir, 'watershed.geojson')}")

    # ── Tight crop to watershed bbox (+ buffer) ─────────────────────────────
    rows, cols = np.where(watershed)
    r0 = max(0, int(rows.min()) - crop_buffer_cells)
    r1 = min(height, int(rows.max()) + crop_buffer_cells + 1)
    c0 = max(0, int(cols.min()) - crop_buffer_cells)
    c1 = min(width, int(cols.max()) + crop_buffer_cells + 1)
    print(f"Cropping pyflwdir outputs to watershed bbox rows[{r0}:{r1}] cols[{c0}:{c1}] "
          f"({r1 - r0}x{c1 - c0} px, buffer={crop_buffer_cells} cells)...")

    crop_transform = rasterio.transform.from_origin(
        transform.c + c0 * transform.a, transform.f + r0 * transform.e,
        transform.a, -transform.e)

    ws_crop = watershed[r0:r1, c0:c1]
    dem_crop = conditioned_dem[r0:r1, c0:c1].astype('float32')
    dem_crop = np.where(ws_crop, dem_crop, -9999.0).astype('float32')
    fa_crop = flow_accumulation[r0:r1, c0:c1].astype('float32')
    fa_crop = np.where(ws_crop, fa_crop, -1.0).astype('float32')

    clipped_dem_profile = profile.copy()
    clipped_dem_profile.update(height=dem_crop.shape[0], width=dem_crop.shape[1],
                                transform=crop_transform, dtype='float32', nodata=-9999.0)
    clipped_dem_path = os.path.join(output_dir, "clipped_dem.tif")
    with rasterio.open(clipped_dem_path, 'w', **clipped_dem_profile) as dst:
        dst.write(dem_crop, 1)
    print(f"Clipped DEM saved to: {clipped_dem_path}")

    clipped_fa_profile = profile.copy()
    clipped_fa_profile.update(height=fa_crop.shape[0], width=fa_crop.shape[1],
                               transform=crop_transform, dtype='float32', nodata=-1.0)
    clipped_fa_path = os.path.join(output_dir, "clipped_flow_accumulation.tif")
    with rasterio.open(clipped_fa_path, 'w', **clipped_fa_profile) as dst:
        dst.write(fa_crop, 1)
    print(f"Clipped flow accumulation saved to: {clipped_fa_path}")

    return watershed, clipped_dem_path, clipped_fa_path


def _delineate(grid, flow_direction, flow_accumulation, profile,
               output_point_latlon, target_crs_epsg, output_dir):
    """
    Outlet-dependent watershed delineation: transform the pour point, snap it to
    the stream network, run the catchment, and write watershed.tif /
    watershed.geojson.  Returns the watershed raster.

    This is the second half of the DEM stage; it consumes the terrain products
    from ``_terrain`` (in-memory in the all-in-one path, or reloaded from disk in
    ``delineate_from_outlet``).
    """
    dirmap = (64, 128, 1, 2, 4, 8, 16, 32)

    # Transform output point to DEM's CRS
    src_crs_latlon = CRS("EPSG:4326") # Assuming input point is always lat/lon
    target_crs = CRS(target_crs_epsg)
    transformer = Transformer.from_crs(src_crs_latlon, target_crs, always_xy=True)

    # Convert lat/lon to projected coordinates
    output_point_x, output_point_y = transformer.transform(output_point_latlon[1], output_point_latlon[0])
    print(f"Output point (lat/lon): {output_point_latlon}")
    print(f"Output point (projected): ({output_point_x}, {output_point_y})")

    # Check bounds
    bounds = grid.bbox
    print(f"DEM Bounds (projected): {bounds}")
    if not (bounds[0] <= output_point_x <= bounds[2] and bounds[1] <= output_point_y <= bounds[3]):
        print("Warning: Output point is outside the DEM's bounds. Watershed delineation might fail or be empty.")

    print("Snapping outlet point to nearest stream cell...")
    streams, stream_threshold, n_cells_total = _stream_mask(flow_accumulation)
    print(f"  Stream threshold: {stream_threshold} cells  (1 % of {n_cells_total} total)")

    # Use pysheds built-in snap_to_mask: works in projected coordinate space
    # Returns (x, y) i.e. (easting, northing) of the snapped cell
    snap_x, snap_y = grid.snap_to_mask(streams, (output_point_x, output_point_y))
    print(f"Output point (projected): ({output_point_x:.2f}, {output_point_y:.2f})")
    print(f"Snapped outlet (projected): ({snap_x:.2f}, {snap_y:.2f})")
    print(f"Snapped cell accumulation: {flow_accumulation[grid.nearest_cell(snap_x, snap_y)[1], grid.nearest_cell(snap_x, snap_y)[0]]}")

    print("Delineating watershed...")
    # Use xytype='coordinate' so pysheds handles the col/row conversion internally
    watershed = grid.catchment(x=snap_x, y=snap_y, fdir=flow_direction, dirmap=dirmap, xytype='coordinate')

    ws_profile = profile.copy()
    watershed_uint8 = np.asarray(watershed).astype('uint8')
    ws_profile.update(dtype=watershed_uint8.dtype, nodata=0)
    with rasterio.open(os.path.join(output_dir,"watershed.tif"), 'w', **ws_profile) as dst:
        dst.write(watershed_uint8, 1)
    print(f"Watershed saved to: {os.path.join(output_dir,'watershed.tif')}")

    print("Exporting watershed to GeoJSON...")
    # Generate vector polygons from the raster
    shapes = rasterio.features.shapes(watershed_uint8, transform=profile['transform'])
    polygons = []
    for geom, val in shapes:
        if val == 1:
            polygons.append(shape(geom))

    if polygons:
        gdf = gpd.GeoDataFrame({'geometry': polygons}, crs=target_crs_epsg)
        gdf.to_file(os.path.join(output_dir,'watershed.geojson'), driver='GeoJSON')
        print(f"Watershed vector saved to: {os.path.join(output_dir,'watershed.geojson')}")

    return watershed


def _write_streams_geojson(flow_accumulation, profile, target_crs_epsg, output_dir):
    """Vectorise the stream mask to streams.geojson for on-canvas outlet picking.

    Additive display aid only — uses the exact same stream threshold as outlet
    snapping (``_stream_mask``) so what the user sees is what the snap targets.
    Does not touch any existing output file.
    """
    streams, stream_threshold, n_cells_total = _stream_mask(flow_accumulation)
    print(f"Vectorising streams for display (threshold {stream_threshold} cells "
          f"= 1 % of {n_cells_total})...")
    stream_uint8 = np.asarray(streams).astype('uint8')
    shapes = rasterio.features.shapes(stream_uint8, transform=profile['transform'])
    polygons = [shape(geom) for geom, val in shapes if val == 1]
    out_path = os.path.join(output_dir, "streams.geojson")
    if polygons:
        gdf = gpd.GeoDataFrame({'geometry': polygons}, crs=target_crs_epsg)
        gdf.to_file(out_path, driver='GeoJSON')
        print(f"Stream network saved to: {out_path}")
        return out_path
    print("No stream cells above threshold — streams.geojson not written.")
    return None


def perform_hydrological_analysis(dem_path, output_point_latlon, target_crs_epsg, output_dir,
                                   engine='pysheds'):
    """
    Performs hydrological analysis: fill, flow direction, flow accumulation,
    and watershed delineation.

    Behaviour is unchanged from before the terrain/delineation split — it simply
    runs the terrain step then the delineation step in the same order, writing
    the same files.  Kept as the single-shot entry point for the CLI / API /
    batch path.

    Parameters
    ----------
    engine : {'pysheds', 'pyflwdir'}
        Delineation engine.  Defaults to 'pysheds' — the original engine — so
        existing callers are unaffected unless they opt in.  See
        vsa_opm.config.OpmConfig.DELINEATION_ENGINE.
    """
    print(f"Starting hydrological analysis on: {dem_path}  (engine={engine})")

    if engine == 'pyflwdir':
        with rasterio.open(dem_path) as src:
            profile = src.profile.copy()
        conditioned_dem, flow_direction, flow_accumulation, flw = _terrain_pyflwdir(
            dem_path, profile, output_dir)
        watershed, _clipped_dem_path, _clipped_fa_path = _delineate_pyflwdir(
            flw, conditioned_dem, flow_direction, flow_accumulation, profile,
            output_point_latlon, target_crs_epsg, output_dir)
        return conditioned_dem, watershed, flow_accumulation, profile

    if engine != 'pysheds':
        raise ValueError(f"Unknown DELINEATION_ENGINE '{engine}' (expected 'pysheds' or 'pyflwdir')")

    grid = _get_pysheds_grid().from_raster(dem_path)
    dem = grid.read_raster(dem_path).astype(np.float64)

    # Read profile for saving later
    with rasterio.open(dem_path) as src:
        profile = src.profile.copy()

    filled_dem, inflated_dem, flow_direction, flow_accumulation = _terrain(
        grid, dem, profile, output_dir)

    watershed = _delineate(grid, flow_direction, flow_accumulation, profile,
                           output_point_latlon, target_crs_epsg, output_dir)

    return filled_dem, watershed, flow_accumulation, profile


def analyze_terrain(dem_path, target_crs_epsg, output_dir, engine='pysheds'):
    """
    Phase 1 of the DEM stage for interactive use: reproject the DEM and run the
    outlet-independent terrain analysis, then vectorise the stream network so a
    user can pick the outlet against it on the map canvas.

    Writes reprojected_dem.tif, filled_dem.tif, inflated_dem.tif,
    flow_direction.tif, flow_accumulation.tif and streams.geojson.  No pour point
    required.  Returns a dict of output paths.

    Parameters
    ----------
    engine : {'pysheds', 'pyflwdir'}
        Delineation engine.  Defaults to 'pysheds' so existing callers (CLI,
        vsa_opm.pipeline, the QGIS plugin) are unaffected unless they opt in.
        'pyflwdir' uses priority-flood depression filling, which correctly
        resolves flow across large flat reservoirs/lakes that defeat pysheds'
        fill_depressions + resolve_flats (see module docstring / dem_processing
        session notes).
    """
    os.makedirs(output_dir, exist_ok=True)
    if not os.path.exists(dem_path):
        raise FileNotFoundError(
            f"DEM file not found at {dem_path}. Please provide a valid DEM_PATH.")

    # Reproject DEM (same first step as main())
    reprojected_dem_path = os.path.join(output_dir, "reprojected_dem.tif")
    reproject_dem(dem_path, reprojected_dem_path, target_crs_epsg)

    with rasterio.open(reprojected_dem_path) as src:
        profile = src.profile.copy()

    if engine == 'pyflwdir':
        _conditioned, _fdir, flow_accumulation, _flw = _terrain_pyflwdir(
            reprojected_dem_path, profile, output_dir)
    elif engine == 'pysheds':
        grid = _get_pysheds_grid().from_raster(reprojected_dem_path)
        dem = grid.read_raster(reprojected_dem_path).astype(np.float64)
        _filled, _inflated, _fdir, flow_accumulation = _terrain(grid, dem, profile, output_dir)
    else:
        raise ValueError(f"Unknown DELINEATION_ENGINE '{engine}' (expected 'pysheds' or 'pyflwdir')")

    streams_path = _write_streams_geojson(flow_accumulation, profile, target_crs_epsg, output_dir)

    print("\n--- Terrain analysis complete ---")
    print("Pick an outlet on the stream network, then delineate the watershed.")
    return {
        "reprojected_dem":   reprojected_dem_path,
        "filled_dem":        os.path.join(output_dir, "filled_dem.tif"),
        "inflated_dem":      os.path.join(output_dir, "inflated_dem.tif"),
        "flow_direction":    os.path.join(output_dir, "flow_direction.tif"),
        "flow_accumulation": os.path.join(output_dir, "flow_accumulation.tif"),
        "streams":           streams_path,
    }


def delineate_from_outlet(output_dir, output_point_latlon, target_crs_epsg, engine='pysheds'):
    """
    Phase 2 of the DEM stage for interactive use: given the terrain products
    already written by ``analyze_terrain``, snap the picked outlet to the stream
    network, delineate the watershed and clip the DEM / flow accumulation to it.

    Reloads flow_direction, flow_accumulation and the reprojected-DEM profile
    from disk (the same rasters ``analyze_terrain`` wrote), so the result is
    identical to the all-in-one ``perform_hydrological_analysis`` path.  Returns a
    dict of output paths.

    Parameters
    ----------
    engine : {'pysheds', 'pyflwdir'}
        Must match the engine passed to the preceding ``analyze_terrain`` call
        (the on-disk rasters were produced by that engine).  Defaults to
        'pysheds'.
    """
    reprojected_dem_path = os.path.join(output_dir, "reprojected_dem.tif")
    fdir_path = os.path.join(output_dir, "flow_direction.tif")
    facc_path = os.path.join(output_dir, "flow_accumulation.tif")
    inflated_dem_path = os.path.join(output_dir, "inflated_dem.tif")
    for p in (reprojected_dem_path, fdir_path, facc_path, inflated_dem_path):
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Missing terrain product {p}. Run analyze_terrain first.")

    with rasterio.open(reprojected_dem_path) as src:
        profile = src.profile.copy()

    if engine == 'pyflwdir':
        pyflwdir = _import_pyflwdir()
        with rasterio.open(fdir_path) as src:
            flow_direction = src.read(1)
        with rasterio.open(facc_path) as src:
            flow_accumulation = src.read(1).astype(np.float64)
        with rasterio.open(inflated_dem_path) as src:
            conditioned_dem = src.read(1).astype(np.float64)
        # Reconstruct the FlwdirRaster from the saved D8 raster — cheaper than
        # re-running the depression fill, and this is the documented way to
        # get an "actionable" pyflwdir object back from a flow-direction array.
        flw = pyflwdir.from_array(
            flow_direction, ftype="d8", check_ftype=False,
            transform=profile['transform'], latlon=False,
        )
        watershed, clipped_dem_path, clipped_fa_path = _delineate_pyflwdir(
            flw, conditioned_dem, flow_direction, flow_accumulation, profile,
            output_point_latlon, target_crs_epsg, output_dir)

        print("\n--- Watershed delineation complete ---")
        return {
            "watershed_tif":     os.path.join(output_dir, "watershed.tif"),
            "watershed_geojson": os.path.join(output_dir, "watershed.geojson"),
            "clipped_dem":       clipped_dem_path,
            "clipped_flow_accumulation": clipped_fa_path,
        }

    if engine != 'pysheds':
        raise ValueError(f"Unknown DELINEATION_ENGINE '{engine}' (expected 'pysheds' or 'pyflwdir')")

    # Reload the grid and terrain rasters from disk.
    grid = _get_pysheds_grid().from_raster(fdir_path)
    flow_direction = grid.read_raster(fdir_path)
    flow_accumulation = grid.read_raster(facc_path)

    watershed = _delineate(grid, flow_direction, flow_accumulation, profile,
                           output_point_latlon, target_crs_epsg, output_dir)

    # Clip DEM by watershed — use the inflated (fill + resolve_flats, float32)
    # DEM so slopes preserve the sub-metre gradients from resolve_flats.
    clipped_dem_path = os.path.join(output_dir, "clipped_dem.tif")
    clip_dem_by_watershed(inflated_dem_path, watershed, clipped_dem_path,
                          profile, nodata_fill=-9999.0)

    clipped_fa_path = os.path.join(output_dir, "clipped_flow_accumulation.tif")
    clip_flow_accumulation_by_watershed(flow_accumulation, watershed, clipped_fa_path, profile)

    print("\n--- Watershed delineation complete ---")
    return {
        "watershed_tif":     os.path.join(output_dir, "watershed.tif"),
        "watershed_geojson": os.path.join(output_dir, "watershed.geojson"),
        "clipped_dem":       clipped_dem_path,
        "clipped_flow_accumulation": clipped_fa_path,
    }

def clip_dem_by_watershed(original_dem_path, watershed_raster, output_clipped_dem_path,
                          dem_profile, nodata_fill=None):
    """
    Clips the original DEM by the delineated watershed boundary.

    nodata_fill : value written to outside-watershed pixels.  Defaults to
                  dem_profile['nodata'].  Pass explicitly when the source file
                  uses a different nodata convention (e.g. float32 with -9999).
    """
    print(f"Clipping DEM: {original_dem_path} by watershed...")
    if nodata_fill is None:
        nodata_fill = dem_profile['nodata']

    with rasterio.open(original_dem_path) as src:
        dem_data = src.read(1)
        clipped_dem_data = np.where(np.asarray(watershed_raster) > 0, dem_data, nodata_fill)

        clipped_profile = dem_profile.copy()
        clipped_profile.update(
            dtype=clipped_dem_data.dtype,
            nodata=nodata_fill,
        )

        with rasterio.open(output_clipped_dem_path, 'w', **clipped_profile) as dst:
            dst.write(clipped_dem_data, 1)
    print(f"Clipped DEM saved to: {output_clipped_dem_path}")
    return output_clipped_dem_path

def clip_flow_accumulation_by_watershed(flow_accumulation, watershed_raster, output_path, dem_profile):
    """
    Clips the flow accumulation raster by the delineated watershed boundary.
    Pixels outside the watershed are set to nodata (-1).
    The output represents contributing area (number of upstream cells) within
    the watershed only — i.e. the spatially distributed contributing area.
    """
    print("Clipping flow accumulation by watershed...")
    fa_array = np.asarray(flow_accumulation).astype(np.float32)
    ws_mask  = np.asarray(watershed_raster) > 0
    nodata_val = -1.0
    clipped_fa = np.where(ws_mask, fa_array, nodata_val)

    fa_profile = dem_profile.copy()
    fa_profile.update(dtype='float32', nodata=nodata_val)

    with rasterio.open(output_path, 'w', **fa_profile) as dst:
        dst.write(clipped_fa, 1)
    print(f"Clipped flow accumulation saved to: {output_path}")
    return output_path

def main(cfg):
    """
    Run the full DEM pre-processing pipeline.

    Parameters
    ----------
    cfg : object
        Any object exposing DEM_PATH, TARGET_CRS_EPSG, OUTPUT_POINT and
        OUTPUT_DIR attributes (vsa_opm.config.OpmConfig or equivalent).
        Optionally exposes DELINEATION_ENGINE ('pysheds' default, or
        'pyflwdir'); missing attribute defaults to 'pysheds' so callers built
        against older OpmConfig objects are unaffected.
    """
    dem_path            = cfg.DEM_PATH
    target_crs_epsg     = cfg.TARGET_CRS_EPSG
    output_point_latlon = cfg.OUTPUT_POINT
    output_dir          = cfg.OUTPUT_DIR
    engine              = getattr(cfg, 'DELINEATION_ENGINE', 'pysheds')

    os.makedirs(output_dir, exist_ok=True)
    if not os.path.exists(dem_path):
        raise FileNotFoundError(
            f"DEM file not found at {dem_path}. Please provide a valid DEM_PATH.")

    # Reproject DEM
    reprojected_dem_path = os.path.join(output_dir, "reprojected_dem.tif")
    reprojected_dem_path = reproject_dem(dem_path, reprojected_dem_path, target_crs_epsg)

    # Perform hydrological analysis
    filled_dem, watershed, flow_accumulation, dem_profile = perform_hydrological_analysis(
        reprojected_dem_path, output_point_latlon, target_crs_epsg, output_dir, engine=engine)

    if engine == 'pyflwdir':
        # _delineate_pyflwdir already wrote the tightly-cropped clipped_dem.tif
        # and clipped_flow_accumulation.tif (see its docstring for why this
        # engine's clip step differs from pysheds' full-extent clip below) —
        # re-clipping here would overwrite the crop with a full-extent version
        # and defeat the point of it.
        pass
    else:
        # Clip DEM by watershed — use the inflated (fill + resolve_flats, float32) DEM
        # so that slopes preserve the sub-metre gradients from resolve_flats.
        # The integer filled_dem rounds those increments to zero, collapsing flat
        # areas to slope=0 and causing a delayed drainage surge in the router.
        inflated_dem_path = os.path.join(output_dir, "inflated_dem.tif")
        clipped_dem_path  = os.path.join(output_dir, "clipped_dem.tif")
        clip_dem_by_watershed(inflated_dem_path, watershed, clipped_dem_path,
                              dem_profile, nodata_fill=-9999.0)

        # Clip flow accumulation (contributing area) by watershed
        clipped_fa_path = os.path.join(output_dir, "clipped_flow_accumulation.tif")
        clip_flow_accumulation_by_watershed(flow_accumulation, watershed, clipped_fa_path, dem_profile)

    print("""
--- Processing Complete ---""")
    print(f"All output files are saved in the '{output_dir}' directory.")
    print("Files include: filled_dem.tif, flow_direction.tif, flow_accumulation.tif, watershed.tif, clipped_dem.tif, clipped_flow_accumulation.tif")
