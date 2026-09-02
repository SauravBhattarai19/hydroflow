"""
gpu_utils.py
============
GPU detection and array dispatch helpers.

CuPy is imported lazily so the codebase remains importable without it.
All GPU-specific logic is isolated here; other modules import only the
helper functions below.
"""

import numpy as np

# ── CuPy availability detection ───────────────────────────────────────────────
try:
    import cupy as _cupy
    import cupyx as _cupyx
    _CUPY_AVAILABLE = True
except ImportError:
    _cupy = None
    _cupyx = None
    _CUPY_AVAILABLE = False


def cupy_available() -> bool:
    """Return True if CuPy is installed and a CUDA device is reachable."""
    if not _CUPY_AVAILABLE:
        return False
    try:
        _cupy.cuda.Device(0).id   # raises RuntimeError if no device
        return True
    except Exception:
        return False


def get_xp(arr):
    """
    Return the array module (cupy or numpy) for the given array.

    Usage:
        xp = get_xp(some_array)
        result = xp.maximum(arr, 0.0)
    """
    if _CUPY_AVAILABLE and isinstance(arr, _cupy.ndarray):
        return _cupy
    return np


def to_device(arr, xp):
    """
    Transfer *arr* to the device corresponding to *xp*.

    - xp = numpy  → returns numpy array  (copies if arr is cupy)
    - xp = cupy   → returns cupy array   (copies if arr is numpy)

    Same-device arrays are returned without an extra copy (cupy.asarray
    is a no-op when the input is already on that device).
    """
    if xp is np:
        if _CUPY_AVAILABLE and isinstance(arr, _cupy.ndarray):
            return _cupy.asnumpy(arr)
        return np.asarray(arr)
    else:
        # xp is cupy
        return xp.asarray(arr)


def to_cpu(arr) -> np.ndarray:
    """Return *arr* as a NumPy array.  No-op if already NumPy."""
    if _CUPY_AVAILABLE and isinstance(arr, _cupy.ndarray):
        return _cupy.asnumpy(arr)
    return np.asarray(arr)


def scatter_add(dst, indices, src):
    """
    Atomic scatter-add:  dst[indices] += src

    Dispatches to:
      - cupyx.scatter_add  (GPU, uses CUDA atomic operations)
      - numpy.add.at       (CPU, no atomics needed — single-threaded)

    Parameters
    ----------
    dst     : 1-D array (cupy or numpy), modified in-place
    indices : 1-D integer array, same backend as dst
    src     : 1-D float array, same backend as dst
    """
    if get_xp(dst) is np:
        np.add.at(dst, indices, src)
    else:
        _cupyx.scatter_add(dst, indices, src)


def get_dtype(cfg):
    """
    Return the NumPy/CuPy dtype to use for state arrays.

    Based on cfg.GPU_PRECISION:
      'float64'  (default) → np.float64  — full double precision
      'float32'            → np.float32  — half memory/bandwidth;
                             validate model output before enabling.
    """
    precision = getattr(cfg, 'GPU_PRECISION', 'float64')
    return np.float32 if precision == 'float32' else np.float64
