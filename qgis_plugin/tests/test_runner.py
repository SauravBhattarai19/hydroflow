# -*- coding: utf-8 -*-
"""
tests/test_runner.py
====================
Integration smoke-test for the OpmWorker (bridge/runner.py).

Runs a minimal CPU simulation using the actual shipped dem_500.tif and
FLOOD_03 data to verify the pipeline produces a valid hydrograph CSV.

This test does NOT require QGIS — it exercises the worker logic directly
by calling the underlying model functions (bypassing the QThread wrapper)
to avoid the need for a running Qt event loop.

Run with:
    cd /path/to/OPM
    pytest qgis_plugin/tests/test_runner.py -v
"""

import os
import sys
import pytest

# ── Add OPM root to path ──────────────────────────────────────────────────────
_OPM_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _OPM_ROOT not in sys.path:
    sys.path.insert(0, _OPM_ROOT)

from qgis_plugin.bridge.config_bridge import OpmConfig  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
def _have_output_rasters():
    """True if process_dem has already been run (output/ rasters exist)."""
    needed = [
        os.path.join(_OPM_ROOT, "output", "clipped_dem.tif"),
        os.path.join(_OPM_ROOT, "output", "flow_direction.tif"),
        os.path.join(_OPM_ROOT, "output", "clipped_flow_accumulation.tif"),
        os.path.join(_OPM_ROOT, "output", "watershed.tif"),
    ]
    return all(os.path.exists(p) for p in needed)


def _flood03_data():
    """True if FLOOD_03 gauge/timeseries CSVs are present."""
    d = os.path.join(_OPM_ROOT, "test_data", "opm_format", "FLOOD_03")
    return (
        os.path.exists(os.path.join(d, "gauges.csv")) and
        os.path.exists(os.path.join(d, "timeseries.csv"))
    )


# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.skipif(
    not _have_output_rasters(),
    reason="Output rasters not present — run process_dem first"
)
class TestRunnerCPU:
    """Direct (non-QThread) integration test of the routing stage."""

    def test_routing_produces_hydrograph_csv(self, tmp_path):
        """
        Run a 6-hour simulation (uniform rain, vsa_opm runoff, CPU) and
        verify the hydrograph CSV is written with the expected columns.
        """
        from hydroflow.core.routing import router as kwr

        cfg = OpmConfig(
            ROUTING_DEM_PATH=os.path.join(_OPM_ROOT, "output", "clipped_dem.tif"),
            ROUTING_FLOW_DIR_PATH=os.path.join(_OPM_ROOT, "output", "flow_direction.tif"),
            ROUTING_FLOW_ACCUM_PATH=os.path.join(_OPM_ROOT, "output", "clipped_flow_accumulation.tif"),
            ROUTING_WATERSHED_MASK_PATH=os.path.join(_OPM_ROOT, "output", "watershed.tif"),
            PRECIP_METHOD="uniform",
            RAIN_INTENSITY_MM_HR=20.0,
            RAIN_DURATION_HOURS=3.0,
            RUNOFF_SOURCE="none",
            MANNINGS_N=0.09,
            TIME_STEP_SECONDS=30,        # large dt → fast test
            TOTAL_SIMULATION_TIME_HOURS=6.0,
            OUTPUT_INTERVAL_SECONDS=1800,
            BACKEND="cpu",
            MIN_SLOPE=1e-4,
            MIN_DEPTH_M=1e-6,
            HYDROGRAPH_CSV=str(tmp_path / "hydrograph.csv"),
            OUTPUT_DIR=str(tmp_path),
        )

        grid_data = kwr.initialise_grid(cfg)
        hydrograph = kwr.run_time_loop(grid_data, cfg)
        df = kwr.save_hydrograph(hydrograph, cfg)

        # Assertions
        assert os.path.exists(cfg.HYDROGRAPH_CSV), "Hydrograph CSV not created"
        assert list(df.columns) == ["time_s", "time_hr", "Q_m3s"]
        assert len(df) > 0, "Hydrograph is empty"
        assert (df["Q_m3s"] >= 0).all(), "Negative discharge values found"
        assert df["Q_m3s"].max() > 0, "Peak discharge is zero (no water routed)"


@pytest.mark.skipif(
    not (_have_output_rasters() and _flood03_data()),
    reason="Output rasters or FLOOD_03 data not present"
)
class TestRunnerThiessen:
    """Test Thiessen precipitation with vsa_opm runoff."""

    def test_thiessen_vsa_opm_routing(self, tmp_path):
        from hydroflow.core.routing import router as kwr

        flood03 = os.path.join(_OPM_ROOT, "test_data", "opm_format", "FLOOD_03")
        cfg = OpmConfig(
            ROUTING_DEM_PATH=os.path.join(_OPM_ROOT, "output", "clipped_dem.tif"),
            ROUTING_FLOW_DIR_PATH=os.path.join(_OPM_ROOT, "output", "flow_direction.tif"),
            ROUTING_FLOW_ACCUM_PATH=os.path.join(_OPM_ROOT, "output", "clipped_flow_accumulation.tif"),
            ROUTING_WATERSHED_MASK_PATH=os.path.join(_OPM_ROOT, "output", "watershed.tif"),
            PRECIP_METHOD="thiessen",
            PRECIP_GAUGE_FILE=os.path.join(flood03, "gauges.csv"),
            PRECIP_TIMESERIES_FILE=os.path.join(flood03, "timeseries.csv"),
            RUNOFF_SOURCE="vsa_opm",
            OPM_SD_MAX_INITIAL=0.10,
            OPM_Q_MAX=0.50,
            OPM_PHI=0.35,
            OPM_K_SAT=44.0,
            MANNINGS_N=0.09,
            TIME_STEP_SECONDS=60,
            TOTAL_SIMULATION_TIME_HOURS=12.0,
            OUTPUT_INTERVAL_SECONDS=3600,
            BACKEND="cpu",
            MIN_SLOPE=1e-4,
            MIN_DEPTH_M=1e-6,
            HYDROGRAPH_CSV=str(tmp_path / "hydrograph_thiessen.csv"),
            OUTPUT_DIR=str(tmp_path),
        )

        grid_data = kwr.initialise_grid(cfg)
        hydrograph = kwr.run_time_loop(grid_data, cfg)
        df = kwr.save_hydrograph(hydrograph, cfg)

        assert os.path.exists(cfg.HYDROGRAPH_CSV)
        assert len(df) > 0
        assert (df["Q_m3s"] >= 0).all()
