# -*- coding: utf-8 -*-
"""
alg_process_dem.py
==================
QGIS Processing Algorithm: DEM Pre-processing.

Wraps process_dem.main() — reprojects DEM, fills sinks, computes flow
direction and accumulation, delineates watershed.

Inputs
------
  INPUT_DEM       : raster layer or file path
  TARGET_CRS      : target coordinate reference system
  OUTLET_LAT      : outlet latitude  (WGS-84 decimal degrees)
  OUTLET_LON      : outlet longitude (WGS-84 decimal degrees)
  OUTPUT_DIR      : folder for output rasters

Outputs
-------
  WATERSHED_TIF   : output/watershed.tif
  CLIPPED_DEM     : output/clipped_dem.tif
  FLOW_DIRECTION  : output/flow_direction.tif
  FLOW_ACCUM      : output/clipped_flow_accumulation.tif
"""

import os
import sys

from qgis.core import (
    QgsProcessingAlgorithm,
    QgsProcessingParameterRasterLayer,
    QgsProcessingParameterCrs,
    QgsProcessingParameterNumber,
    QgsProcessingParameterEnum,
    QgsProcessingParameterFolderDestination,
    QgsProcessingParameterRasterDestination,
    QgsProcessingOutputRasterLayer,
    QgsProcessingOutputVectorLayer,
    QgsProcessingContext,
    QgsProcessingFeedback,
)

from ..bridge import ensure_core


class ProcessDemAlgorithm(QgsProcessingAlgorithm):
    """QGIS Processing wrapper for vsa_opm.core.process_dem.main()."""

    # Parameter IDs
    INPUT_DEM = "INPUT_DEM"
    TARGET_CRS = "TARGET_CRS"
    ENGINE = "ENGINE"
    OUTLET_LAT = "OUTLET_LAT"
    OUTLET_LON = "OUTLET_LON"
    OUTPUT_DIR = "OUTPUT_DIR"

    # Literal vsa_opm.config.OpmConfig.DELINEATION_ENGINE values, in dropdown
    # order — index 0 (pyflwdir) is the default, matching the interactive
    # Tab-1 dialog (see qgis_plugin/ui/tab_dem.py).  pyflwdir (priority-flood
    # fill) correctly routes flow across large flat reservoirs/lakes that can
    # silently disconnect the upstream basin with pysheds (the legacy engine,
    # kept for basins already validated against it).
    _ENGINE_OPTIONS = ["pyflwdir", "pysheds"]

    def createInstance(self):  # noqa: N802
        return ProcessDemAlgorithm()

    def name(self):
        return "process_dem"

    def displayName(self):  # noqa: N802
        return "1. DEM Pre-processing"

    def group(self):
        return "VSA-OPM Pipeline"

    def groupId(self):  # noqa: N802
        return "vsaopm_pipeline"

    def shortHelpString(self):  # noqa: N802
        return (
            "Reprojects the DEM to the target CRS, fills sinks, computes D8 "
            "flow direction and accumulation, snaps the outlet point to the "
            "nearest stream cell, and delineates the watershed.\n\n"
            "Delineation engine: pyflwdir (default, priority-flood fill — "
            "correctly routes flow across large flat reservoirs/lakes) or "
            "pysheds (legacy engine).\n\n"
            "Outputs: watershed.tif, clipped_dem.tif, flow_direction.tif, "
            "clipped_flow_accumulation.tif, watershed.geojson."
        )

    def initAlgorithm(self, config=None):  # noqa: N802
        self.addParameter(
            QgsProcessingParameterRasterLayer(
                self.INPUT_DEM, "DEM raster"
            )
        )
        self.addParameter(
            QgsProcessingParameterCrs(
                self.TARGET_CRS, "Target CRS", defaultValue="EPSG:32645"
            )
        )
        self.addParameter(
            QgsProcessingParameterEnum(
                self.ENGINE, "Delineation engine",
                options=self._ENGINE_OPTIONS, defaultValue=0  # pyflwdir
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.OUTLET_LAT, "Outlet latitude (WGS-84)",
                type=QgsProcessingParameterNumber.Double,
                defaultValue=27.632222, minValue=-90.0, maxValue=90.0
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.OUTLET_LON, "Outlet longitude (WGS-84)",
                type=QgsProcessingParameterNumber.Double,
                defaultValue=85.293333, minValue=-180.0, maxValue=180.0
            )
        )
        self.addParameter(
            QgsProcessingParameterFolderDestination(
                self.OUTPUT_DIR, "Output directory"
            )
        )

    def processAlgorithm(self, parameters, context, feedback):  # noqa: N802
        # QGIS Processing sets sys.stderr/stdout to None;
        # numpy's C extension needs them writable during import.
        import io
        if sys.stderr is None:
            sys.stderr = io.StringIO()
        if sys.stdout is None:
            sys.stdout = io.StringIO()

        ensure_core()

        from vsa_opm.config import OpmConfig
        from vsa_opm.core import dem_processing as pd_mod

        # Build config
        dem_layer = self.parameterAsRasterLayer(parameters, self.INPUT_DEM, context)
        dem_path = dem_layer.source() if dem_layer else parameters[self.INPUT_DEM]
        crs = self.parameterAsCrs(parameters, self.TARGET_CRS, context)
        engine = self._ENGINE_OPTIONS[
            self.parameterAsEnum(parameters, self.ENGINE, context)
        ]
        lat = self.parameterAsDouble(parameters, self.OUTLET_LAT, context)
        lon = self.parameterAsDouble(parameters, self.OUTLET_LON, context)
        out_dir = self.parameterAsString(parameters, self.OUTPUT_DIR, context)

        os.makedirs(out_dir, exist_ok=True)

        cfg = OpmConfig(
            DEM_PATH=dem_path,
            TARGET_CRS_EPSG=crs.authid(),
            OUTPUT_POINT=(lat, lon),
            OUTPUT_DIR=out_dir,
            DELINEATION_ENGINE=engine,
        )
        cfg.update_output_paths()

        feedback.setProgress(5)
        feedback.pushInfo(f"Running DEM pre-processing (engine={engine}) …")

        pd_mod.main(cfg)

        feedback.setProgress(95)

        # ── Auto-load output layers into the QGIS map canvas ─────────────────
        outputs = {
            "WATERSHED_TIF":  (os.path.join(out_dir, "watershed.tif"),        "Watershed",             "raster"),
            "CLIPPED_DEM":    (os.path.join(out_dir, "clipped_dem.tif"),       "Clipped DEM",           "raster"),
            "FLOW_DIRECTION": (os.path.join(out_dir, "flow_direction.tif"),    "Flow Direction",        "raster"),
            "FLOW_ACCUM":     (os.path.join(out_dir, "clipped_flow_accumulation.tif"), "Flow Accumulation", "raster"),
        }

        # Also load watershed GeoJSON vector if it was created
        geojson_path = os.path.join(out_dir, "watershed.geojson")
        if os.path.exists(geojson_path):
            outputs["WATERSHED_VEC"] = (geojson_path, "Watershed Boundary", "vector")

        from qgis.core import QgsProcessingContext, QgsProject

        results = {}
        for key, (path, name, layer_type) in outputs.items():
            if not os.path.exists(path):
                feedback.pushInfo(f"  Skipping {name} — file not found: {path}")
                continue

            feedback.pushInfo(f"  Loading layer: {name}")
            details = QgsProcessingContext.LayerDetails(name, QgsProject.instance(), key)
            context.addLayerToLoadOnCompletion(path, details)
            results[key] = path

        feedback.setProgress(100)
        feedback.pushInfo(
            f"\nDEM Pre-processing complete. {len(results)} layers added to the map."
        )
        return results
