# -*- coding: utf-8 -*-
"""
vsa_opm  (compatibility shim)
=============================
This package was renamed **vsa_opm → hydroflow**.  It is kept only so existing
code (`import vsa_opm`, `from vsa_opm.config import OpmConfig`,
`from vsa_opm.core.routing import router`, …) keeps working.

Importing it installs a redirect that maps every ``vsa_opm.*`` import to the
matching ``hydroflow.*`` module and emits a single ``DeprecationWarning``.
New code should ``import hydroflow`` directly.
"""

import warnings

from hydroflow._compat import install as _install

# Register the sys.meta_path finder so lazy submodule imports
# (vsa_opm.config, vsa_opm.core.routing.router, …) resolve to hydroflow.*
_install()

warnings.warn(
    "The 'vsa_opm' package has been renamed to 'hydroflow'; "
    "'import vsa_opm' still works but is deprecated. "
    "Please update your imports to 'hydroflow'.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export the public API so `from vsa_opm import OpmConfig, run_pipeline` works
# without going through the submodule finder.
from hydroflow import (  # noqa: E402,F401
    Config,
    OpmConfig,
    run_pipeline,
    DEFAULT_STAGES,
    __version__,
)

__all__ = ["Config", "OpmConfig", "run_pipeline", "DEFAULT_STAGES", "__version__"]
