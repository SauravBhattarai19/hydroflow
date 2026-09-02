# -*- coding: utf-8 -*-
"""
bridge package — glue between the QGIS plugin (UI/Processing) and the
pip-installable ``hydroflow`` core package.

The core science lives entirely in ``hydroflow`` (no QGIS imports there);
this package holds the QGIS-side wrappers (QThread worker, dependency
installer) plus :func:`ensure_core`, which makes the core importable in
every deployment mode.
"""

import os
import sys

# Plugin root is 1 level up from bridge/
_PLUGIN_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def ensure_core():
    """
    Make the ``hydroflow`` core package importable inside QGIS.

    Resolution order:
      1. Already importable (pip-installed into the QGIS interpreter).
      2. Vendored copy shipped inside the plugin zip:  <plugin>/_vendor/hydroflow
      3. Development checkout: the plugin folder is a symlink into the
         repository (install_plugin.sh), so the package sits next to it
         in the repo root.
    """
    try:
        import hydroflow  # noqa: F401
        return
    except ImportError:
        pass

    candidates = [
        os.path.join(_PLUGIN_DIR, "_vendor"),
        os.path.dirname(os.path.realpath(_PLUGIN_DIR)),   # repo root (dev symlink)
    ]
    for cand in candidates:
        if os.path.isdir(os.path.join(cand, "hydroflow")):
            if cand not in sys.path:
                sys.path.insert(0, cand)
            try:
                import hydroflow  # noqa: F401
                return
            except ImportError:
                continue

    raise ImportError(
        "The 'hydroflow' core package could not be found. Install it into the "
        "QGIS Python interpreter (pip install hydroflow) or rebuild the plugin "
        "zip so it ships a vendored copy (_vendor/hydroflow)."
    )
