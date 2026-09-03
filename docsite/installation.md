# Installation

hydroflow is a pure-Python package (Python 3.9+). Install it from PyPI:

```bash
pip install hydroflow
```

## Optional extras

=== "GPU"

    Adds CuPy for CUDA 12.x acceleration. The model still runs on CPU if no GPU
    is present — the GPU is used only when `BACKEND="gpu"` **and** CuPy imports.

    ```bash
    pip install "hydroflow[gpu]"
    ```

=== "Google Earth Engine"

    Adds `earthengine-api` for satellite forcing (IMERG rainfall, SERVES soil
    deficit, SoilGrids, LULC/LCZ). All GEE options are optional — scalar/manual
    settings run fully offline.

    ```bash
    pip install "hydroflow[gee]"
    ```

=== "From source"

    ```bash
    git clone https://github.com/SauravBhattarai19/hydroflow
    cd hydroflow
    pip install .            # or pip install -e . for a live checkout
    ```

## Verify

```bash
python -c "import hydroflow; print(hydroflow.__version__)"
hydroflow --help
```

## Dependencies

The core install pulls in the scientific-Python + geospatial stack: `numpy`,
`pandas`, `scipy`, `matplotlib`, `rasterio`, `geopandas`, `shapely`, `pyproj`,
`fiona`, `pysheds`, `pyflwdir`, and `pyyaml`. On most platforms `pip` installs
prebuilt wheels for the geospatial libraries — no system GDAL required.

!!! warning "Use Python 3.10–3.12"
    `fiona` and `rasterio` only ship prebuilt wheels for a handful of recent
    Python versions (currently up to 3.13). On a newer or very new Python
    (e.g. 3.14), pip silently falls back to building `fiona` from source,
    which fails unless system GDAL (`gdal-config`) is installed. If you hit
    an error mentioning `gdal-config` or `GDAL API version`, create your
    conda/venv environment with Python 3.10, 3.11, or 3.12 instead:

    ```bash
    conda create -n hydroflow-env python=3.11 -y
    conda activate hydroflow-env
    pip install hydroflow
    ```

!!! tip "Earth Engine authentication"
    GEE features (including DEM auto-download) need both an authenticated
    session and a Google Cloud project ID — see
    [Configuration → Earth Engine setup](configuration.md#earth-engine-setup-first-time)
    for the one-time signup/authenticate/`GEE_PROJECT` walkthrough.
