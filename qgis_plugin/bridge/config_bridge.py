# -*- coding: utf-8 -*-
"""
config_bridge.py
================
Backward-compatible re-export of the core configuration object.

OpmConfig used to be defined here; it now lives in the pip-installable core
package (hydroflow.config.Config, aliased OpmConfig) so the Python API, the CLI
and the plugin all share one definition.  Existing plugin code that does

    from ..bridge.config_bridge import OpmConfig

keeps working unchanged.
"""

from . import ensure_core

ensure_core()

from hydroflow.config import OpmConfig  # noqa: E402,F401

__all__ = ["OpmConfig"]
