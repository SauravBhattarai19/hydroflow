# -*- coding: utf-8 -*-
"""
_compat.py
==========
Backward-compatibility import shim for the old package name ``vsa_opm``.

The package was renamed ``vsa_opm`` → ``hydroflow`` (it is becoming a general
hydrological + hydrodynamic model, not only VSA + OPM).  To avoid breaking
existing scripts, the QGIS plugin, and ``tools/`` / ``tests/`` — all of which do
``import vsa_opm`` or ``from vsa_opm.core.routing import router`` — this module
installs a :class:`importlib.abc.MetaPathFinder` that transparently redirects
any ``vsa_opm`` / ``vsa_opm.<sub>`` import to the matching ``hydroflow`` module.

A single :class:`DeprecationWarning` is emitted on first use.  New code should
import from ``hydroflow`` directly.
"""

import importlib
import importlib.abc
import importlib.util
import sys
import warnings

_OLD = "vsa_opm"
_NEW = "hydroflow"
_warned = False


def _warn_once():
    global _warned
    if not _warned:
        _warned = True
        warnings.warn(
            "The 'vsa_opm' package has been renamed to 'hydroflow'; "
            "'import vsa_opm' still works but is deprecated. "
            "Please update your imports to 'hydroflow'.",
            DeprecationWarning,
            stacklevel=3,
        )


class _VsaOpmAliasFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Redirect ``vsa_opm[.sub...]`` imports to ``hydroflow[.sub...]``."""

    def find_spec(self, fullname, path=None, target=None):
        if fullname != _OLD and not fullname.startswith(_OLD + "."):
            return None
        _warn_once()
        # Map the requested old name to the corresponding new dotted name and
        # let this object act as the loader (create_module returns the real
        # hydroflow module, so submodule attribute access resolves normally).
        return importlib.util.spec_from_loader(fullname, self)

    def create_module(self, spec):
        new_name = _NEW + spec.name[len(_OLD):]   # 'vsa_opm.x' → 'hydroflow.x'
        module = importlib.import_module(new_name)
        # Register the real module under the old name too, so both
        # sys.modules['vsa_opm.x'] and ['hydroflow.x'] point at one object.
        sys.modules[spec.name] = module
        return module

    def exec_module(self, module):
        # Nothing to execute — create_module already returned a fully-imported
        # module.  (Required by the Loader ABC.)
        return None


def install():
    """Register the alias finder on ``sys.meta_path`` (idempotent)."""
    if not any(isinstance(f, _VsaOpmAliasFinder) for f in sys.meta_path):
        sys.meta_path.append(_VsaOpmAliasFinder())
