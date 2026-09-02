# -*- coding: utf-8 -*-
"""
vsa_opm.core.precip — precipitation forcing engines.

Modules
-------
engine : PrecipEngine — uniform / Thiessen / IDW / IMERG (thiessen or IDW)
         rainfall fields on the routing grid.
gpu    : PrecipEngineGPU — CuPy device-array variant (import explicitly;
         kept out of this namespace so CPU-only installs never touch CuPy).
"""

from .engine import PrecipEngine

__all__ = ["PrecipEngine"]
