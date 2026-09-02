# -*- coding: utf-8 -*-
"""
dependencies.py
===============
Detect and install the third-party Python packages the VSA-OPM model needs,
targeting the interpreter QGIS is actually running on (NOT the system Python).

Why this exists
---------------
QGIS ships its own bundled Python (OSGeo4W on Windows, a framework build on
macOS).  Running ``pip install rasterio`` in a normal terminal installs into
the *system* Python, so QGIS still can't import it.  This module locates the
QGIS interpreter and drives ``pip`` against it, so the user never has to find
the OSGeo4W Shell.

It is written to survive the usual cross-platform pitfalls without user
intervention: ``sys.executable`` pointing at the QGIS/OSGeo4W launcher instead
of a real ``python`` (probe & validate candidate interpreters), a Python that
ships without ``pip`` (bootstrap via ``ensurepip``), read-only site-packages
(retry with ``--user``), and PEP 668 "externally-managed" environments (retry
with ``--break-system-packages``).

Public API
----------
    REQUIRED, OPTIONAL          – [(import_name, pip_name, purpose), …]
    is_available(import_name)   – bool
    missing(include_optional)   – list of (import_name, pip_name, purpose)
    python_executable()         – path to the QGIS Python interpreter
    manual_command(pip_names)   – the exact command to paste into a shell
    install(pip_names, log, …)  – run pip; returns (ok: bool, returncode: int)
"""

import importlib
import os
import subprocess
import sys


# (import name, pip name, human purpose) ---------------------------------------
REQUIRED = [
    ("numpy",    "numpy",    "array math"),
    ("pandas",   "pandas",   "CSV / hydrograph I/O"),
    ("scipy",    "scipy",    "spatial weights (KDTree)"),
    ("rasterio", "rasterio", "raster read/write"),
    ("pysheds",  "pysheds",  "DEM watershed delineation (legacy engine)"),
    ("pyflwdir", "pyflwdir", "DEM watershed delineation (default engine; "
                             "handles reservoirs/flat water pysheds cannot)"),
    # Not imported by name anywhere in vsa_opm — pysheds and pyflwdir both pull
    # it in transitively for their JIT kernels.  Listed explicitly so the
    # installer can detect+fix a broken/ABI-mismatched numba (e.g. an OS/apt-
    # packaged numba built against a different numpy than the one actually on
    # sys.path — encountered for real during this feature's testing, raises a
    # SystemError on bare `import numba`, not just a slow-path JIT warning)
    # instead of the user hitting an opaque error only when they click
    # Analyze/Delineate.
    ("numba",    "numba",    "JIT-compiled delineation kernels"),
]

OPTIONAL = [
    ("matplotlib", "matplotlib",      "embedded hydrograph plot"),
    ("ee",         "earthengine-api", "IMERG / SERVES / LULC / LCZ (Earth Engine)"),
]


def is_available(import_name: str) -> bool:
    """True if ``import <import_name>`` succeeds in this interpreter."""
    try:
        importlib.import_module(import_name)
        return True
    except Exception:  # noqa: BLE001 — any import failure means "not usable"
        return False


def missing(include_optional: bool = False):
    """Return the subset of REQUIRED (+ OPTIONAL) that cannot be imported."""
    items = list(REQUIRED)
    if include_optional:
        items += OPTIONAL
    return [(mod, pip, why) for (mod, pip, why) in items if not is_available(mod)]


_CREATE_NO_WINDOW = 0x08000000 if os.name == "nt" else 0  # hide child console on Windows


def _looks_like_python(path: str) -> bool:
    """Cheap name check: does this path look like a python interpreter (not, e.g.,
    the ``qgis-bin.exe`` / ``QGIS`` app binary that ``sys.executable`` may point to)?"""
    if not path:
        return False
    name = os.path.basename(path).lower()
    return name.startswith("python") or name.startswith("pythonw")


def _validate_interpreter(path: str) -> bool:
    """True if ``path`` exists and is a runnable Python (we actually launch it).

    This is what lets us reject the QGIS app binary that some builds expose as
    ``sys.executable`` on macOS/Windows — running it would relaunch QGIS, not
    Python, so we probe with a trivial ``-c`` and require a clean exit.
    """
    if not path or not os.path.exists(path):
        return False
    try:
        proc = subprocess.run(
            [path, "-c", "import sys; sys.exit(0)"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            timeout=20, creationflags=_CREATE_NO_WINDOW,
        )
        return proc.returncode == 0
    except Exception:  # noqa: BLE001 — anything means "not a usable interpreter"
        return False


def _candidate_interpreters():
    """Ordered, de-duplicated list of paths that might be *this* environment's
    Python, derived from where the running install actually lives.

    Works across OSGeo4W (Windows), the QGIS.app framework build (macOS),
    system/conda installs (Linux) and virtualenvs, because it is driven by
    ``sysconfig`` / ``sys.prefix`` rather than hard-coded product paths.
    """
    major, minor = sys.version_info[:2]
    if os.name == "nt":
        names = ["python.exe", "python3.exe", f"python{major}.exe", f"python{major}{minor}.exe"]
    else:
        names = [f"python{major}.{minor}", "python3", f"python{major}", "python"]

    # Directories that plausibly hold this environment's interpreter.
    dirs = []
    try:
        import sysconfig
        for d in (sysconfig.get_config_var("BINDIR"), sysconfig.get_path("scripts")):
            if d:
                dirs.append(d)
    except Exception:  # noqa: BLE001
        pass
    for base in (sys.prefix, sys.exec_prefix,
                 getattr(sys, "base_prefix", ""), getattr(sys, "base_exec_prefix", "")):
        if not base:
            continue
        dirs.append(base)                       # Windows: python.exe sits in the prefix root
        dirs.append(os.path.join(base, "bin"))  # POSIX / macOS framework
        dirs.append(os.path.join(base, "Scripts"))  # Windows venv
    exe_dir = os.path.dirname(sys.executable) if sys.executable else ""
    if exe_dir:
        dirs.append(exe_dir)
        dirs.append(os.path.join(exe_dir, "bin"))

    seen, cands = set(), []
    for d in dirs:
        for n in names:
            cand = os.path.join(d, n)
            if cand not in seen:
                seen.add(cand)
                cands.append(cand)
    return cands


def python_executable() -> str:
    """
    Best-effort path to the Python interpreter QGIS runs on.

    ``sys.executable`` is unreliable inside QGIS: on Windows it is often
    ``qgis-bin.exe`` and on some macOS builds it is the ``QGIS.app`` binary —
    neither can run ``-m pip``.  Strategy:

    1. If ``sys.executable`` looks like *and* validates as a real Python, use it.
    2. Otherwise probe interpreters derived from this install's own prefixes
       (OSGeo4W, the macOS framework bin, conda/system, virtualenvs) and return
       the first one that actually runs.
    3. Fall back to ``sys.executable`` so ``manual_command`` still prints
       something the user can adapt by hand.
    """
    exe = sys.executable
    if _looks_like_python(exe) and _validate_interpreter(exe):
        return exe
    for cand in _candidate_interpreters():
        if _validate_interpreter(cand):
            return cand
    return exe


def refresh_import_paths():
    """
    Make freshly pip-installed packages importable in the RUNNING interpreter
    without restarting QGIS.

    ``pip install`` writes to a site directory that Python only scans at
    startup, so a running QGIS won't see the new packages.  We add the user
    (and, if present, global) site directories to ``sys.path`` and clear the
    import-finder caches, so a subsequent ``import rasterio`` succeeds in the
    same session.

    Returns the list of directories newly added to ``sys.path``.
    """
    import site
    import importlib

    dirs = []
    try:
        dirs.append(site.getusersitepackages())
    except Exception:  # noqa: BLE001
        pass
    try:
        dirs.extend(site.getsitepackages())
    except Exception:  # noqa: BLE001
        pass

    added = []
    for d in dirs:
        if d and os.path.isdir(d) and d not in sys.path:
            sys.path.append(d)
            added.append(d)

    importlib.invalidate_caches()
    return added


def _user_site_enabled() -> bool:
    """Whether ``pip install --user`` targets a directory this Python imports from."""
    try:
        import site
        return bool(getattr(site, "ENABLE_USER_SITE", False))
    except Exception:  # noqa: BLE001
        return False


def manual_command(pip_names) -> str:
    """The exact shell command a user can run by hand (for the fallback hint)."""
    py = python_executable()
    if " " in py:
        py = f'"{py}"'
    return f'{py} -m pip install {" ".join(pip_names)}'


def _ensure_pip(py: str, emit=None) -> bool:
    """Make sure ``py -m pip`` works; bootstrap it with ``ensurepip`` if not.

    Some minimal QGIS/framework Pythons ship without pip.  Returns True once
    ``pip`` is importable by ``py``.
    """
    def _run(cmd):
        try:
            return subprocess.run(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, timeout=300, creationflags=_CREATE_NO_WINDOW,
            )
        except Exception:  # noqa: BLE001
            return None

    r = _run([py, "-m", "pip", "--version"])
    if r is not None and r.returncode == 0:
        return True
    if emit:
        emit("[INFO] pip not available in the QGIS Python; bootstrapping with ensurepip …")
    r = _run([py, "-m", "ensurepip", "--default-pip"])
    if emit and r is not None and r.stdout:
        for ln in r.stdout.splitlines():
            emit(ln.rstrip())
    r = _run([py, "-m", "pip", "--version"])
    return bool(r is not None and r.returncode == 0)


def install(pip_names, log=None, prefer_user: bool = True, timeout: int = 1800):
    """
    Install ``pip_names`` into the QGIS interpreter.

    Strategy: try a normal install first (works on user-writable OSGeo4W
    installs); if that fails with what looks like a permission problem and a
    user-site is available, retry with ``--user``.

    Parameters
    ----------
    pip_names : list[str]
    log : callable(str) | None
        Optional line sink (e.g. a Qt signal ``.emit``) for streaming output.
    prefer_user : bool
        If True and a user-site is enabled, go straight to ``--user`` (avoids
        touching a read-only Program Files site-packages on Windows).
    timeout : int
        Seconds before the pip subprocess is abandoned.

    Returns
    -------
    (ok: bool, returncode: int)
    """
    def _emit(line):
        if log:
            log(line)

    pip_names = list(pip_names)
    if not pip_names:
        return True, 0

    py = python_executable()
    if not _ensure_pip(py, _emit):
        _emit("[ERROR] pip is unavailable in the QGIS Python and ensurepip could not bootstrap it.")
        _emit(f"[HINT] Install manually, then restart QGIS:\n    {manual_command(pip_names)}")
        return False, 1

    base = [py, "-m", "pip", "install", "--disable-pip-version-check"]
    user_ok = _user_site_enabled()

    # Ordered install strategies; more may be appended mid-loop if pip's output
    # tells us the environment needs a different flag (e.g. PEP 668).
    attempts = []
    if prefer_user and user_ok:
        attempts.append(base + ["--user"] + pip_names)
        attempts.append(base + pip_names)                 # fallback: system site
    else:
        attempts.append(base + pip_names)                 # try system site
        if user_ok:
            attempts.append(base + ["--user"] + pip_names)  # fallback: user site

    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"

    tried_break_system = False
    last_rc = 1
    i = 0
    while i < len(attempts):
        cmd = attempts[i]
        i += 1
        _emit(f"$ {' '.join(cmd)}")
        buf = []
        try:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, env=env, creationflags=_CREATE_NO_WINDOW,
            )
        except Exception as exc:  # noqa: BLE001
            _emit(f"[ERROR] Could not launch pip: {exc}")
            return False, 1

        try:
            for line in iter(proc.stdout.readline, ""):
                line = line.rstrip("\n")
                buf.append(line)
                _emit(line)
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            _emit(f"[ERROR] pip timed out after {timeout}s.")
            return False, 1

        last_rc = proc.returncode or 0
        if last_rc == 0:
            _emit("[OK] Install finished.")
            return True, 0

        blob = "\n".join(buf).lower()
        # PEP 668: interpreter refuses to install into an "externally-managed"
        # environment (Homebrew / some distro & conda Pythons).  Retry once,
        # opting in explicitly.
        if not tried_break_system and "externally-managed-environment" in blob:
            tried_break_system = True
            _emit("[WARN] Environment is externally managed; retrying with --break-system-packages …")
            retry = base + ["--break-system-packages"]
            if user_ok:
                retry += ["--user"]
            attempts.append(retry + pip_names)
            continue

        if i < len(attempts):
            _emit(f"[WARN] Attempt failed (exit {last_rc}); retrying with a different target …")

    _emit(f"[ERROR] pip failed (exit {last_rc}).")
    _emit(f"[HINT] Install manually, then restart QGIS:\n    {manual_command(pip_names)}")
    return False, last_rc
