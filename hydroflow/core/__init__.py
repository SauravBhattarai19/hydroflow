# -*- coding: utf-8 -*-
"""
vsa_opm.core — the hydrologic science.  Pure Python/NumPy (CuPy optional);
absolutely no QGIS or Qt dependencies.

Subpackages / modules
---------------------
dem_processing : DEM reprojection, sink filling, D8 flow direction /
                 accumulation, watershed delineation (pysheds).
routing        : the routing engine — terrain derivation, per-step
                 hydraulic kernels, surface parameter fields, the
                 kinematic/diffusive-wave router and run reporting.
precip         : uniform / Thiessen / IDW / IMERG precipitation engines.
runoff         : runoff generation — coefficient, raster, SCS-CN and the
                 VSA-OPM mechanics (+ Green-Ampt, impervious) with the
                 soil-parameter resolution.
opm            : standalone OPM (VSA water-balance) model runner.
io_utils       : shared raster sampling / grid-alignment helpers.
"""
