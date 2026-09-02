# -*- coding: utf-8 -*-
"""
main_dialog.py
==============
OpmMainDialog — the primary 5-tab QDialog for the VSA-OPM plugin.

Layout
------
  ┌──────────────────────────────────────────────┐
  │  [Tab 1: DEM] [Tab 2: Precip] [Tab 3: Runoff] │
  │  [Tab 4: Routing] [Tab 5: Results]            │
  ├──────────────────────────────────────────────┤
  │  Stage checkboxes:  ☑ DEM   ☑ Routing  ☐ VSA│
  ├──────────────────────────────────────────────┤
  │  Progress bar  ███████░░░░░  57 %            │
  ├──────────────────────────────────────────────┤
  │  Log panel (QPlainTextEdit, read-only)       │
  ├──────────────────────────────────────────────┤
  │  [Run]  [Cancel]  [Save Config]  [Close]     │
  └──────────────────────────────────────────────┘
"""

import os

from qgis.PyQt.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTabWidget,
    QPushButton, QProgressBar, QPlainTextEdit,
    QGroupBox, QCheckBox, QMessageBox,
    QSizePolicy,
)
from qgis.PyQt.QtCore import Qt

from ..bridge.config_bridge import OpmConfig
from ..bridge.runner import OpmWorker, DemStepWorker
from ..bridge.dependencies import missing as missing_deps
from .dependency_dialog import DependencyDialog
from .layer_utils import add_raster, add_vector, zoom_to_layer
from .tab_dem import TabDem
from .tab_precip import TabPrecip
from .tab_runoff import TabRunoff
from .tab_routing import TabRouting
from .tab_results import TabResults


class OpmMainDialog(QDialog):
    """
    Main VSA-OPM modelling dialog.

    Parameters
    ----------
    iface : QgisInterface
    parent : QWidget, optional
    """

    def __init__(self, iface, parent=None):
        super().__init__(parent)
        self._iface = iface
        self._worker = None
        self._dem_worker = None   # background worker for the guided DEM steps

        self.setWindowTitle("VSA-OPM Hydrological Model")
        self.setMinimumSize(780, 680)
        self.resize(900, 760)
        self.setWindowFlags(self.windowFlags() | Qt.WindowMaximizeButtonHint)

        self._build_ui()
        self._connect_signals()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        # ── Tab widget ────────────────────────────────────────────────────────
        self.tabs = QTabWidget()

        self.tab_dem = TabDem(self._iface)
        self.tab_precip = TabPrecip()
        self.tab_runoff = TabRunoff()
        self.tab_routing = TabRouting()
        self.tab_results = TabResults(self._iface)

        self.tabs.addTab(self.tab_dem,     "1 · DEM & Watershed")
        self.tabs.addTab(self.tab_precip,  "2 · Precipitation")
        self.tabs.addTab(self.tab_runoff,  "3 · Runoff")
        self.tabs.addTab(self.tab_routing, "4 · Routing")
        self.tabs.addTab(self.tab_results, "5 · Results")

        root.addWidget(self.tabs, stretch=3)

        # ── Stage selection ───────────────────────────────────────────────────
        grp_stages = QGroupBox("Pipeline Stages to Run")
        h_stages = QHBoxLayout(grp_stages)

        self.chk_dem = QCheckBox("DEM Pre-processing")
        self.chk_dem.setChecked(True)
        self.chk_dem.setToolTip(
            "Run process_dem.py: reproject, fill sinks, flow direction,\n"
            "flow accumulation, watershed delineation."
        )

        self.chk_routing = QCheckBox("Kinematic-Wave Routing")
        self.chk_routing.setChecked(True)
        self.chk_routing.setToolTip(
            "Run kinematic_wave_router.py: initialise grid, time loop,\n"
            "save hydrograph CSV."
        )

        self.chk_vsa = QCheckBox("Standalone VSA-OPM")
        self.chk_vsa.setChecked(False)
        self.chk_vsa.setToolTip(
            "Run vsa_opm.py standalone to generate vsa_opm_results.csv\n"
            "(useful for inspecting OPM dynamics without the full router)."
        )

        h_stages.addWidget(self.chk_dem)
        h_stages.addWidget(self.chk_routing)
        h_stages.addWidget(self.chk_vsa)
        h_stages.addStretch()

        root.addWidget(grp_stages)

        # ── Progress bar ──────────────────────────────────────────────────────
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        root.addWidget(self.progress_bar)

        # ── Log panel ─────────────────────────────────────────────────────────
        self.log_panel = QPlainTextEdit()
        self.log_panel.setReadOnly(True)
        self.log_panel.setMaximumBlockCount(5000)   # keep last 5000 lines
        self.log_panel.setPlaceholderText("Model output will appear here during a run …")
        monofont = self.log_panel.font()
        monofont.setFamily("Monospace")
        monofont.setPointSize(9)
        self.log_panel.setFont(monofont)
        self.log_panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        root.addWidget(self.log_panel, stretch=2)

        # ── Button row ────────────────────────────────────────────────────────
        btn_row = QHBoxLayout()

        self.run_btn = QPushButton("▶  Run")
        self.run_btn.setDefault(True)
        self.run_btn.setMinimumWidth(100)
        self.run_btn.setStyleSheet(
            "QPushButton { background-color: #2E86AB; color: white; "
            "font-weight: bold; padding: 6px 16px; border-radius: 4px; }"
            "QPushButton:hover { background-color: #1B6CA8; }"
            "QPushButton:disabled { background-color: #aaa; }"
        )

        self.cancel_btn = QPushButton("⏹  Cancel")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.setMinimumWidth(90)

        self.save_cfg_btn = QPushButton("💾  Save Config")
        self.save_cfg_btn.setToolTip(
            "Write current UI settings to a Python config file\n"
            "compatible with config.py format."
        )

        self.deps_btn = QPushButton("🔧  Dependencies")
        self.deps_btn.setToolTip(
            "Check / install the Python packages the model needs\n"
            "(rasterio, pysheds, …) into QGIS's own Python."
        )

        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.close)

        btn_row.addWidget(self.run_btn)
        btn_row.addWidget(self.cancel_btn)
        btn_row.addWidget(self.save_cfg_btn)
        btn_row.addWidget(self.deps_btn)
        btn_row.addStretch()
        btn_row.addWidget(self.close_btn)

        root.addLayout(btn_row)

    # ── Signal connections ────────────────────────────────────────────────────

    def _connect_signals(self):
        self.run_btn.clicked.connect(self._on_run)
        self.cancel_btn.clicked.connect(self._on_cancel)
        self.save_cfg_btn.clicked.connect(self._on_save_config)
        self.deps_btn.clicked.connect(self._open_dependencies)

        # Guided DEM workflow (Tab 1) — the tab only requests a step; the dialog
        # runs it off-thread and loads the resulting layers.
        self.tab_dem.request_analyze_terrain.connect(self._on_analyze_terrain)
        self.tab_dem.request_delineate.connect(self._on_delineate)

        # Keep the Earth Engine project id in sync across the Precip & Runoff tabs
        # (one config value, shown in two places).
        self.tab_precip.gee_project.textChanged.connect(self._sync_gee_project)
        self.tab_runoff._gee_project.textChanged.connect(self._sync_gee_project)

    def _sync_gee_project(self, text):
        """Mirror the GEE project id between the Precip and Runoff tabs."""
        for widget in (self.tab_precip.gee_project, self.tab_runoff._gee_project):
            if widget.text() != text:
                widget.blockSignals(True)
                widget.setText(text)
                widget.blockSignals(False)

    # ── Guided DEM workflow ────────────────────────────────────────────────────

    def _on_analyze_terrain(self):
        """Run outlet-independent terrain analysis, then draw streams on the map."""
        if not self._deps_ok():
            return
        dem = self.tab_dem.get_dem_path()
        out = self.tab_dem.get_output_dir()
        if not dem or not os.path.exists(dem):
            QMessageBox.warning(self, "No DEM", "Select a valid DEM file first.")
            return
        if not out:
            QMessageBox.warning(self, "No output directory",
                                "Choose an output directory first.")
            return
        params = {
            "dem_path": dem,
            "target_crs_epsg": self.tab_dem.get_target_crs(),
            "output_dir": out,
            "engine": self.tab_dem.get_delineation_engine(),
        }
        self._start_dem_step("analyze_terrain", params)

    def _on_delineate(self):
        """Snap the picked outlet to the stream network and delineate the watershed."""
        if not self._deps_ok():
            return
        out = self.tab_dem.get_output_dir()
        if not out or not os.path.exists(os.path.join(out, "flow_direction.tif")):
            QMessageBox.warning(self, "Analyze terrain first",
                                "Run “Analyze terrain” before delineating the watershed.")
            return
        lat, lon = self.tab_dem.get_outlet_point()
        params = {
            "output_dir": out,
            "output_point_latlon": (lat, lon),
            "target_crs_epsg": self.tab_dem.get_target_crs(),
            "engine": self.tab_dem.get_delineation_engine(),
        }
        self._start_dem_step("delineate", params)

    def _start_dem_step(self, task, params):
        """Spawn the background DEM-step worker and route its signals."""
        self.log_panel.clear()
        self.run_btn.setEnabled(False)
        self.tab_dem.set_busy(True)
        self.progress_bar.setRange(0, 0)   # indeterminate/busy

        self._dem_worker = DemStepWorker(task, params, parent=self)
        self._dem_worker.log.connect(self._append_log)
        self._dem_worker.finished.connect(self._on_dem_step_finished)
        self._dem_worker.error.connect(self._on_dem_step_error)
        self._dem_worker.start()

    def _end_dem_step(self):
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.run_btn.setEnabled(True)
        self.tab_dem.set_busy(False)

    def _on_dem_step_finished(self, result: dict):
        self._end_dem_step()
        task = result.get("task")
        if task == "analyze_terrain":
            dem = add_raster(result.get("reprojected_dem"), "DEM (reprojected)")
            add_raster(result.get("flow_accumulation"), "Flow accumulation")
            streams = add_vector(result.get("streams"), "Streams")
            zoom_to_layer(self._iface, streams or dem)
            self.tab_dem.set_terrain_ready(True)
            self.tab_dem.activate_pick()
            self._append_log("\n✅  Terrain ready — pick your outlet on a stream.")
            self._iface.messageBar().pushInfo(
                "VSA-OPM", "Streams drawn — pick your outlet on a stream, then Delineate.")
        elif task == "delineate":
            ws = add_vector(result.get("watershed_geojson"), "Watershed boundary")
            zoom_to_layer(self._iface, ws)
            self._append_log("\n✅  Watershed delineated.")
            self._iface.messageBar().pushSuccess(
                "VSA-OPM", "Watershed delineated — you can now run Routing.")

    def _on_dem_step_error(self, message: str):
        self._end_dem_step()
        self._append_log(f"\n❌  ERROR: {message}")
        self._iface.messageBar().pushCritical("VSA-OPM", message.splitlines()[0])
        QMessageBox.critical(self, "VSA-OPM — DEM Step Error", message)

    def _deps_ok(self) -> bool:
        """Guard: ensure required Python packages are installed; offer the installer."""
        miss = missing_deps(include_optional=False)
        if not miss:
            return True
        names = ", ".join(m[1] for m in miss)
        resp = QMessageBox.question(
            self, "Missing Python packages",
            f"The model needs these packages, which are not installed in "
            f"QGIS's Python:\n\n    {names}\n\n"
            "Open the Dependencies manager to install them now?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes,
        )
        if resp == QMessageBox.Yes:
            self._open_dependencies()
        return False

    # ── Run / Cancel ──────────────────────────────────────────────────────────

    def _on_run(self):
        """Validate config, build stages list, spawn worker thread."""
        # ── Dependency guard ──────────────────────────────────────────────────
        # Catch missing packages up-front so users get a guided installer instead
        # of a cryptic "ModuleNotFoundError: No module named 'rasterio'".
        if not self._deps_ok():
            return

        cfg = self._collect_config()
        if cfg is None:
            return   # validation failed; user already shown a message

        stages = []
        if self.chk_dem.isChecked():
            stages.append("process_dem")
        if self.chk_routing.isChecked():
            stages.append("routing")
        if self.chk_vsa.isChecked():
            stages.append("vsa_opm")

        if not stages:
            QMessageBox.warning(self, "No stages selected",
                                "Please tick at least one pipeline stage to run.")
            return

        # ── GPU VRAM check ────────────────────────────────────────────────────
        if cfg.BACKEND == "gpu":
            self._check_gpu_vram()

        # ── Start worker ──────────────────────────────────────────────────────
        self.log_panel.clear()
        self.progress_bar.setValue(0)
        self.run_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)

        self._worker = OpmWorker(cfg, stages, parent=self)
        self._worker.progress.connect(self.progress_bar.setValue)
        self._worker.log.connect(self._append_log)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_cancel(self):
        if self._worker and self._worker.isRunning():
            self._worker.cancel()
        self.cancel_btn.setEnabled(False)

    # ── Worker slots ──────────────────────────────────────────────────────────

    def _append_log(self, text: str):
        self.log_panel.appendPlainText(text)
        # Auto-scroll to bottom
        sb = self.log_panel.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _on_finished(self, result: dict):
        self.run_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.progress_bar.setValue(100)
        self._append_log("\n✅  Pipeline finished successfully.")

        # Push to results tab
        if result.get("hydrograph_df") is not None or result.get("watershed_tif"):
            self.tab_results.update_results(result)
            self.tabs.setCurrentWidget(self.tab_results)

        # QGIS message bar
        self._iface.messageBar().pushSuccess(
            "VSA-OPM", "Model run complete.  Check the Results tab."
        )

    def _on_error(self, message: str):
        self.run_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self._append_log(f"\n❌  ERROR: {message}")
        self._iface.messageBar().pushCritical("VSA-OPM", message.splitlines()[0])
        QMessageBox.critical(self, "VSA-OPM — Run Error", message)

    # ── Config helpers ────────────────────────────────────────────────────────

    def _collect_config(self) -> OpmConfig:
        """
        Build an OpmConfig from all UI tabs.
        Returns None if validation fails (and shows a warning to the user).
        """
        cfg = OpmConfig()
        self.tab_dem.write_to_config(cfg)
        self.tab_precip.write_to_config(cfg)
        self.tab_runoff.write_to_config(cfg)
        self.tab_routing.write_to_config(cfg)

        try:
            cfg.validate()
        except ValueError as exc:
            QMessageBox.warning(self, "Configuration Error", str(exc))
            return None

        return cfg

    def _open_dependencies(self):
        """Open the dependency status/installer dialog (modal)."""
        dlg = DependencyDialog(parent=self)
        dlg.exec_()

    def _on_save_config(self):
        """Export current settings to a config.py-compatible Python file."""
        from qgis.PyQt.QtWidgets import QFileDialog
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Config", "", "Python files (*.py)"
        )
        if not path:
            return

        cfg = OpmConfig()
        self.tab_dem.write_to_config(cfg)
        self.tab_precip.write_to_config(cfg)
        self.tab_runoff.write_to_config(cfg)
        self.tab_routing.write_to_config(cfg)

        _write_config_py(cfg, path)
        self._iface.messageBar().pushSuccess(
            "VSA-OPM", f"Config saved → {path}"
        )

    # ── GPU VRAM advisory ─────────────────────────────────────────────────────

    def _check_gpu_vram(self):
        try:
            import cupy as cp
            free_b, total_b = cp.cuda.Device(0).mem_info
            free_gb = free_b / 1e9
            if free_gb < 0.5:
                QMessageBox.warning(
                    self, "Low GPU VRAM",
                    f"Only {free_gb:.2f} GB VRAM free.\n"
                    "The model may run out of memory.  "
                    "Consider switching to CPU or using float32 precision."
                )
        except Exception:  # noqa: BLE001
            pass   # CuPy not available — GPU toggle already disabled


# ── Config file writer ─────────────────────────────────────────────────────────

def _write_config_py(cfg: OpmConfig, path: str):
    """
    Write an OpmConfig to a config.py-compatible Python source file.
    The output file is valid Python and can replace config.py directly.

    Every attribute is dumped from cfg.to_dict(), so this export stays complete
    automatically as new parameters are added to OpmConfig / config.py.
    Parameters are grouped by section for readability; any attribute not
    assigned to a section falls into a final "OTHER" block.
    """
    # Section → ordered list of attribute names (mirrors config.py layout).
    sections = [
        ("1. EVENT & SCENARIO", [
            "DEM_PATH", "TARGET_CRS_EPSG", "OUTPUT_POINT", "OUTPUT_DIR",
            "EVENT_START_UTC", "TOTAL_SIMULATION_TIME_HOURS",
            "IMERG_UTC_OFFSET_HOURS", "LULC_LOOKUP_CSV", "LCZ_LOOKUP_CSV",
            "GEE_PROJECT",
        ]),
        ("2. WATERSHED PRE-PROCESSING OUTPUTS", [
            "ROUTING_DEM_PATH", "ROUTING_FLOW_DIR_PATH", "ROUTING_FLOW_ACCUM_PATH",
            "ROUTING_WATERSHED_MASK_PATH", "OPM_WATERSHED_GEOJSON",
        ]),
        ("3. PRECIPITATION", [
            "RAIN_INTENSITY_MM_HR", "RAIN_DURATION_HOURS", "PRECIP_METHOD",
            "PRECIP_GAUGE_FILE", "PRECIP_TIMESERIES_FILE", "PRECIP_IDW_POWER",
            "PRECIP_EXCLUDE_OUTSIDE_STATIONS", "IMERG_START_LOCAL",
            "IMERG_END_LOCAL", "PRECIP_IMERG_DIR", "IMERG_DATASET", "IMERG_BAND",
            "PRECIP_IMERG_FORCE_DOWNLOAD", "IMERG_BBOX_BUFFER_M",
        ]),
        ("4. RUNOFF GENERATION", [
            "RUNOFF_SOURCE", "RUNOFF_COEFFICIENT_PATH", "RUNOFF_RASTER_MANIFEST",
            "RUNOFF_CN_PATH", "RUNOFF_SCS_Ia_FACTOR",
        ]),
        ("5. OPM / VSA PARAMETERS", [
            "RUNOFF_MECHANISMS", "OPM_SD_MAX_INITIAL", "OPM_Q_MAX", "OPM_PHI",
            "OPM_K_SAT", "OPM_PER_POLYGON", "OPM_INFILTRATION",
            "OPM_GA_SUCTION_SOURCE", "OPM_GA_SUCTION_M", "OPM_GA_KSAT_SOURCE",
            "OPM_GA_KSAT_MMHR", "OPM_GA_KSAT_RASTER", "OPM_GA_KSAT_SCALE",
            "IMPERVIOUS_SOURCE", "IMPERVIOUS_RASTER_PATH", "OPM_BASEFLOW",
        ]),
        ("6. SERVES / GEE SOIL-MOISTURE DEFICIT", [
            "OPM_SD_SOURCE", "OPM_SD_REDUCER", "OPM_DEFICIT_RASTER",
            "SERVES_SATELLITE", "SERVES_SEARCH_WINDOW", "OPM_SOILGRIDS_DEPTH",
        ]),
        ("7. MANNING'S ROUGHNESS", [
            "MANNINGS_N_SOURCE", "MANNINGS_N", "MANNINGS_N_LULC_PATH",
            "MANNINGS_N_RASTER_PATH", "MANNINGS_N_CHANNEL", "CHANNEL_FACCUM_THRESHOLD",
        ]),
        ("8. GRID & NUMERICAL LIMITS", [
            "CELL_SIZE", "ROUTING_SCHEME", "DIFFUSION_THETA", "CHANNEL_ROUTING",
            "CHANNEL_WIDTH_BY_ORDER", "TIME_STEP_SECONDS", "OUTPUT_INTERVAL_SECONDS",
            "ADAPTIVE_TIMESTEP", "CFL_TARGET", "CFL_DT_MAX", "CFL_DT_MIN",
            "CFL_DT_GROW", "MIN_SLOPE", "MIN_DEPTH_M", "MAX_DEPTH_M",
        ]),
        ("9. OUTPUTS", [
            "HYDROGRAPH_CSV", "MASS_BALANCE_REPORT", "MASS_BALANCE_CSV",
        ]),
        ("10. COMPUTE BACKEND", [
            "BACKEND", "GPU_PRECISION",
        ]),
    ]

    data = cfg.to_dict()
    lines = [
        "# config.py  —  generated by the VSA-OPM QGIS Plugin\n",
        "# Values only, no logic.  This file can replace config.py directly.\n\n",
    ]

    written = set()
    for title, names in sections:
        lines.append(f"# {'='*70}\n# {title}\n# {'='*70}\n\n")
        for name in names:
            if name in data:
                lines.append(f"{name} = {data[name]!r}\n")
                written.add(name)
        lines.append("\n")

    # Any attribute not assigned to a section (future-proofing).
    leftover = [k for k in data if k not in written]
    if leftover:
        lines.append(f"# {'='*70}\n# OTHER\n# {'='*70}\n\n")
        for name in leftover:
            lines.append(f"{name} = {data[name]!r}\n")
        lines.append("\n")

    with open(path, "w", encoding="utf-8") as f:
        f.writelines(lines)
