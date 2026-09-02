# -*- coding: utf-8 -*-
"""
tab_dem.py
==========
Tab 1 — DEM & Watershed Setup (guided two-step workflow).

Step 1 — Digital Elevation Model
    DEM file + target CRS, then [🏔 Analyze terrain].  This runs the
    outlet-independent terrain analysis (fill → flow direction → flow
    accumulation) and draws the stream network on the map canvas.

Step 2 — Watershed Outlet
    With the streams visible, pick the outlet on the map (or type lat/lon),
    then [💧 Delineate watershed].

The tab itself does no threading or Earth Engine work — it only emits
``request_analyze_terrain`` / ``request_delineate``; the main dialog runs the
background worker and loads the resulting layers.  Earth Engine / event settings
now live on the Precipitation and Runoff tabs (they are not needed for DEM
preprocessing at all).
"""

import os

from qgis.PyQt.QtWidgets import (
    QWidget, QFormLayout, QGroupBox, QVBoxLayout, QHBoxLayout,
    QPushButton, QDoubleSpinBox, QLabel, QComboBox,
)
from qgis.PyQt.QtCore import pyqtSignal
from qgis.gui import QgsFileWidget, QgsProjectionSelectionWidget, QgsMapToolEmitPoint
from qgis.core import QgsCoordinateReferenceSystem, QgsPointXY


class TabDem(QWidget):
    """DEM & Watershed configuration tab (guided two-step)."""

    # Emitted when the user picks a point on the map
    outlet_picked = pyqtSignal(float, float)   # lat, lon
    # Emitted when the user asks to run a DEM step (handled by the main dialog)
    request_analyze_terrain = pyqtSignal()
    request_delineate = pyqtSignal()

    def __init__(self, iface, parent=None):
        super().__init__(parent)
        self._iface = iface
        self._map_tool = None    # QgsMapToolEmitPoint instance
        self._prev_tool = None   # map tool active before picking
        self._terrain_ready = False
        self._build_ui()

    # ── UI construction ────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        # ── Step 1 · DEM Input ───────────────────────────────────────────────
        grp_dem = QGroupBox("1 · Digital Elevation Model")
        form_dem = QFormLayout(grp_dem)

        self.dem_widget = QgsFileWidget()
        self.dem_widget.setStorageMode(QgsFileWidget.GetFile)
        self.dem_widget.setFilter("GeoTIFF (*.tif *.tiff);;All files (*)")
        self.dem_widget.setDialogTitle("Select DEM raster")
        self.dem_widget.fileChanged.connect(self._invalidate_terrain)
        form_dem.addRow("DEM file:", self.dem_widget)

        self.crs_widget = QgsProjectionSelectionWidget()
        self.crs_widget.setCrs(QgsCoordinateReferenceSystem("EPSG:32645"))
        self.crs_widget.crsChanged.connect(self._invalidate_terrain)
        form_dem.addRow("Target CRS:", self.crs_widget)

        self.engine_combo = QComboBox()
        # (display label, underlying vsa_opm.config.OpmConfig.DELINEATION_ENGINE value)
        self._ENGINE_CHOICES = [
            ("pyflwdir  (recommended)", "pyflwdir"),
            ("pysheds  (legacy)", "pysheds"),
        ]
        for label, _value in self._ENGINE_CHOICES:
            self.engine_combo.addItem(label)
        self.engine_combo.setCurrentIndex(0)   # pyflwdir by default
        self.engine_combo.setToolTip(
            "Delineation engine used by both steps below.\n\n"
            "pyflwdir (default): priority-flood depression filling. Correctly "
            "routes flow across large flat areas — reservoirs, lakes, wide "
            "floodplains — that can silently disconnect the upstream basin "
            "with the legacy engine.\n\n"
            "pysheds (legacy): the original engine. Kept for basins already "
            "validated against it, or if pyflwdir cannot be installed on this "
            "system."
        )
        self.engine_combo.currentIndexChanged.connect(self._invalidate_terrain)
        form_dem.addRow("Delineation engine:", self.engine_combo)

        self.analyze_btn = QPushButton("🏔  Analyze terrain")
        self.analyze_btn.setToolTip(
            "Reproject the DEM and compute flow direction / flow accumulation,\n"
            "then draw the stream network on the map so you can pick an outlet.\n"
            "No pour point or Earth Engine account is needed for this step."
        )
        self.analyze_btn.clicked.connect(self.request_analyze_terrain.emit)
        analyze_row = QHBoxLayout()
        analyze_row.addStretch()
        analyze_row.addWidget(self.analyze_btn)
        form_dem.addRow(analyze_row)

        root.addWidget(grp_dem)

        # ── Step 2 · Outlet Point ────────────────────────────────────────────
        grp_outlet = QGroupBox("2 · Watershed Outlet")
        v_outlet = QVBoxLayout(grp_outlet)

        self._outlet_hint = QLabel(
            "Run “Analyze terrain” first — then pick your outlet on a stream."
        )
        self._outlet_hint.setWordWrap(True)
        self._outlet_hint.setStyleSheet("color: #666;")
        v_outlet.addWidget(self._outlet_hint)

        coord_row = QHBoxLayout()

        self.lat_spin = QDoubleSpinBox()
        self.lat_spin.setRange(-90.0, 90.0)
        self.lat_spin.setDecimals(6)
        self.lat_spin.setValue(27.632222)
        self.lat_spin.setSuffix("°  (lat)")

        self.lon_spin = QDoubleSpinBox()
        self.lon_spin.setRange(-180.0, 180.0)
        self.lon_spin.setDecimals(6)
        self.lon_spin.setValue(85.293333)
        self.lon_spin.setSuffix("°  (lon)")

        self.pick_btn = QPushButton("📍  Pick on map")
        self.pick_btn.setToolTip(
            "Click on the QGIS map canvas to set the outlet point.\n"
            "The map must be in a geographic CRS (EPSG:4326) or the\n"
            "coordinates will be reprojected automatically."
        )
        self.pick_btn.setCheckable(True)
        self.pick_btn.clicked.connect(self._toggle_map_pick)

        coord_row.addWidget(QLabel("Latitude:"))
        coord_row.addWidget(self.lat_spin)
        coord_row.addSpacing(10)
        coord_row.addWidget(QLabel("Longitude:"))
        coord_row.addWidget(self.lon_spin)
        coord_row.addSpacing(10)
        coord_row.addWidget(self.pick_btn)
        coord_row.addStretch()
        v_outlet.addLayout(coord_row)

        self.delineate_btn = QPushButton("💧  Delineate watershed")
        self.delineate_btn.setToolTip(
            "Snap the outlet to the stream network and delineate the\n"
            "contributing watershed, then clip the DEM to it."
        )
        self.delineate_btn.setEnabled(False)   # enabled once terrain is analyzed
        self.delineate_btn.clicked.connect(self.request_delineate.emit)
        delin_row = QHBoxLayout()
        delin_row.addStretch()
        delin_row.addWidget(self.delineate_btn)
        v_outlet.addLayout(delin_row)

        root.addWidget(grp_outlet)

        # ── Output Directory ─────────────────────────────────────────────────
        grp_out = QGroupBox("Output")
        form_out = QFormLayout(grp_out)

        self.output_dir_widget = QgsFileWidget()
        self.output_dir_widget.setStorageMode(QgsFileWidget.GetDirectory)
        self.output_dir_widget.setDialogTitle("Select output directory")
        # Default to a writable location under the user's home folder — never
        # inside the plugin/Program Files tree (which may be read-only).
        default_out = os.path.join(os.path.expanduser("~"), "VSA-OPM", "output")
        self.output_dir_widget.setFilePath(default_out)
        form_out.addRow("Output directory:", self.output_dir_widget)

        root.addWidget(grp_out)
        root.addStretch()

    # ── Guided-workflow state (driven by the main dialog) ──────────────────────

    def set_terrain_ready(self, ready: bool):
        """Enable outlet picking / delineation once terrain analysis has run."""
        self._terrain_ready = bool(ready)
        self.delineate_btn.setEnabled(self._terrain_ready)
        self.pick_btn.setEnabled(self._terrain_ready)
        if ready:
            self._outlet_hint.setText(
                "Streams are on the map — pick your outlet on a stream, "
                "then “Delineate watershed”."
            )
            self._outlet_hint.setStyleSheet("color: #1B6CA8;")
        else:
            self._outlet_hint.setText(
                "Run “Analyze terrain” first — then pick your outlet on a stream."
            )
            self._outlet_hint.setStyleSheet("color: #666;")

    def set_busy(self, busy: bool):
        """Disable the step buttons while a background DEM step is running."""
        self.analyze_btn.setEnabled(not busy)
        if busy:
            self.delineate_btn.setEnabled(False)
        else:
            self.delineate_btn.setEnabled(self._terrain_ready)

    def activate_pick(self):
        """Programmatically switch the map into outlet-pick mode."""
        if not self.pick_btn.isChecked():
            self.pick_btn.setChecked(True)
            self._toggle_map_pick(True)

    def _invalidate_terrain(self, *args):
        """DEM or CRS changed → the analyzed terrain is stale; require re-analyze."""
        if self._terrain_ready:
            self.set_terrain_ready(False)

    # ── Map-pick tool ─────────────────────────────────────────────────────────

    def _toggle_map_pick(self, checked):
        canvas = self._iface.mapCanvas()
        if checked:
            self._prev_tool = canvas.mapTool()
            self._map_tool = QgsMapToolEmitPoint(canvas)
            self._map_tool.canvasClicked.connect(self._on_canvas_click)
            canvas.setMapTool(self._map_tool)
            self._iface.messageBar().pushMessage(
                "VSA-OPM", "Click on the map to set the outlet point.", duration=5
            )
        else:
            canvas.setMapTool(self._prev_tool)
            self._map_tool = None

    def _on_canvas_click(self, point: QgsPointXY, button):
        """Convert clicked map point to WGS-84 lat/lon."""
        from qgis.core import (
            QgsCoordinateReferenceSystem,
            QgsCoordinateTransform,
            QgsProject,
        )
        map_crs = self._iface.mapCanvas().mapSettings().destinationCrs()
        wgs84 = QgsCoordinateReferenceSystem("EPSG:4326")
        if map_crs != wgs84:
            xform = QgsCoordinateTransform(map_crs, wgs84, QgsProject.instance())
            point = xform.transform(point)

        self.lat_spin.setValue(point.y())
        self.lon_spin.setValue(point.x())
        self.outlet_picked.emit(point.y(), point.x())

        # Un-check the button and restore previous tool
        self.pick_btn.setChecked(False)
        self._toggle_map_pick(False)

    # ── Public getters ────────────────────────────────────────────────────────

    def get_dem_path(self) -> str:
        return self.dem_widget.filePath()

    def get_target_crs(self) -> str:
        return self.crs_widget.crs().authid()

    def get_delineation_engine(self) -> str:
        return self._ENGINE_CHOICES[self.engine_combo.currentIndex()][1]

    def get_outlet_point(self) -> tuple:
        return (self.lat_spin.value(), self.lon_spin.value())

    def get_output_dir(self) -> str:
        return self.output_dir_widget.filePath()

    # ── Populate from config ──────────────────────────────────────────────────

    def apply_config(self, cfg):
        """Populate widgets from an OpmConfig object."""
        if cfg.DEM_PATH:
            self.dem_widget.setFilePath(cfg.DEM_PATH)
        self.crs_widget.setCrs(QgsCoordinateReferenceSystem(cfg.TARGET_CRS_EPSG))
        lat, lon = cfg.OUTPUT_POINT
        self.lat_spin.setValue(lat)
        self.lon_spin.setValue(lon)
        if cfg.OUTPUT_DIR:
            self.output_dir_widget.setFilePath(cfg.OUTPUT_DIR)
        engine = getattr(cfg, "DELINEATION_ENGINE", "pyflwdir")
        for i, (_label, value) in enumerate(self._ENGINE_CHOICES):
            if value == engine:
                self.engine_combo.setCurrentIndex(i)
                break

    def write_to_config(self, cfg):
        """Write widget values into an OpmConfig object."""
        cfg.DEM_PATH = self.get_dem_path()
        cfg.TARGET_CRS_EPSG = self.get_target_crs()
        cfg.OUTPUT_POINT = self.get_outlet_point()
        cfg.OUTPUT_DIR = self.get_output_dir()
        cfg.DELINEATION_ENGINE = self.get_delineation_engine()
        cfg.update_output_paths()
