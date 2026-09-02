# -*- coding: utf-8 -*-
"""
runner.py
=========
Background worker that runs the VSA-OPM pipeline in a QThread so the
QGIS GUI stays responsive.

Architecture
------------

    OpmWorker(QThread)
        Signals:
            progress(int)   – 0 … 100 progress percentage
            log(str)        – line of text from the model's stdout
            finished(dict)  – result dict with paths of generated files
            error(str)      – human-readable error message (run aborted)

        Slots (called from main thread):
            cancel()        – request graceful abort (sets _cancelled flag)

Stdout redirection
------------------
All existing model modules use print() for progress.  The worker redirects
sys.stdout to a _StdoutCapture object that emits the log() signal instead
of writing to the console.  The original stdout is restored on exit.

GPU memory management
---------------------
After each run, GPU memory pools are freed so QGIS's OpenGL renderer
gets the VRAM back.  This is idempotent when CuPy is not installed.

Core resolution
---------------
All science lives in the pip-installable ``vsa_opm`` package;
bridge.ensure_core() makes it importable (pip install, vendored copy in
the plugin zip, or the development checkout next to a symlinked plugin).
"""

import io
import sys
import traceback

from qgis.PyQt.QtCore import QThread, pyqtSignal

from . import ensure_core


# ── Stdout capture ─────────────────────────────────────────────────────────────

class _StdoutCapture(io.TextIOBase):
    """
    File-like object that forwards write() calls to a PyQt signal.

    Parameters
    ----------
    signal : pyqtSignal(str)
        Signal to emit for each non-empty line written.
    """

    def __init__(self, signal):
        super().__init__()
        self._signal = signal
        self._buffer = ""

    def write(self, text):
        self._buffer += text
        # Flush complete lines immediately so the log panel updates in real time
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            self._signal.emit(line)
        return len(text)

    def flush(self):
        if self._buffer:
            self._signal.emit(self._buffer)
            self._buffer = ""


# ── Worker thread ──────────────────────────────────────────────────────────────

class OpmWorker(QThread):
    """
    Runs the full VSA-OPM pipeline (or individual stages) in a background
    thread.

    Parameters
    ----------
    cfg : OpmConfig
        Configuration object built by the UI.
    stages : list of str
        Which pipeline stages to run.  Options (in order):
            'process_dem'  – DEM preprocessing (process_dem.main)
            'routing'      – kinematic-wave routing (initialise_grid + run_time_loop)
            'vsa_opm'      – standalone OPM run (vsa_opm.run_opm)
        Default: ['process_dem', 'routing']
    """

    # Qt signals (must be class-level)
    progress = pyqtSignal(int)          # 0–100
    log = pyqtSignal(str)               # one log line
    finished = pyqtSignal(dict)         # result dict
    error = pyqtSignal(str)             # error message

    def __init__(self, cfg, stages=None, parent=None):
        super().__init__(parent)
        self._cfg = cfg
        self._stages = stages or ["process_dem", "routing"]
        self._cancelled = False
        self._result = {}

    # ── Public API ────────────────────────────────────────────────────────────

    def cancel(self):
        """Request graceful cancellation (checked between stages)."""
        self._cancelled = True
        self.log.emit("[INFO] Cancellation requested — will stop after current stage.")

    # ── QThread entry point ───────────────────────────────────────────────────

    def run(self):
        """Called by QThread.start().  Runs in the worker thread."""
        ensure_core()

        # Redirect stdout so model print() calls go to the log signal
        original_stdout = sys.stdout
        sys.stdout = _StdoutCapture(self.log)

        try:
            self._run_pipeline()
        except Exception:  # noqa: BLE001
            tb = traceback.format_exc()
            self.log.emit(tb)
            self.error.emit(
                "An error occurred during the simulation.\n\n"
                f"{tb.splitlines()[-1]}"
            )
        finally:
            # Always restore stdout and free GPU memory
            sys.stdout = original_stdout
            self._release_gpu_memory()

    # ── Pipeline stages ───────────────────────────────────────────────────────

    def _run_pipeline(self):
        """Delegate to the Qt-free core pipeline (vsa_opm.pipeline)."""
        from vsa_opm.pipeline import run_pipeline

        self._result = run_pipeline(
            self._cfg,
            stages=self._stages,
            on_log=self.log.emit,
            on_progress=self.progress.emit,
            is_cancelled=lambda: self._cancelled,
        )

        if not self._cancelled:
            self.finished.emit(self._result)

    # ── GPU cleanup ──────────────────────────────────────────────────────────

    @staticmethod
    def _release_gpu_memory():
        """Free CuPy memory pools.  No-op if CuPy is not installed."""
        try:
            import cupy as cp
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
        except Exception:  # noqa: BLE001
            pass  # CuPy not available or no device — nothing to free


# ── Interactive DEM-step worker ────────────────────────────────────────────────

class DemStepWorker(QThread):
    """
    Runs a single interactive DEM step off the GUI thread so the canvas stays
    responsive while pysheds works (fill_depressions can take a while).

    Used by the guided Tab-1 workflow:
        'analyze_terrain' → dem_processing.analyze_terrain(...)   (draw streams)
        'delineate'       → dem_processing.delineate_from_outlet(...) (watershed)

    Emits the same signal shape as OpmWorker so the dialog can wire them
    interchangeably.  The result dict always carries a "task" key identifying
    which step finished, plus the core function's returned output paths.
    """

    progress = pyqtSignal(int)          # 0–100 (steps run indeterminate; kept for parity)
    log = pyqtSignal(str)               # one log line
    finished = pyqtSignal(dict)         # {"task": ..., <output paths>}
    error = pyqtSignal(str)             # error message

    def __init__(self, task, params, parent=None):
        super().__init__(parent)
        self._task = task               # "analyze_terrain" | "delineate"
        self._params = dict(params)
        self._result = {}

    def run(self):
        """Called by QThread.start().  Runs in the worker thread."""
        ensure_core()

        original_stdout = sys.stdout
        sys.stdout = _StdoutCapture(self.log)
        try:
            from vsa_opm.core import dem_processing as dp

            engine = self._params.get("engine", "pysheds")
            if self._task == "analyze_terrain":
                out = dp.analyze_terrain(
                    self._params["dem_path"],
                    self._params["target_crs_epsg"],
                    self._params["output_dir"],
                    engine=engine,
                )
            elif self._task == "delineate":
                out = dp.delineate_from_outlet(
                    self._params["output_dir"],
                    self._params["output_point_latlon"],
                    self._params["target_crs_epsg"],
                    engine=engine,
                )
            else:
                raise ValueError(f"Unknown DEM step: {self._task!r}")

            self._result = {"task": self._task, **(out or {})}
            self.finished.emit(self._result)
        except Exception:  # noqa: BLE001
            tb = traceback.format_exc()
            self.log.emit(tb)
            self.error.emit(
                f"The '{self._task}' step failed.\n\n{tb.splitlines()[-1]}"
            )
        finally:
            sys.stdout = original_stdout
            OpmWorker._release_gpu_memory()
