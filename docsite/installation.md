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

!!! tip "Earth Engine authentication"
    GEE features need a Google Cloud project. hydroflow initialises Earth Engine
    from `GOOGLE_APPLICATION_CREDENTIALS`, a `key.json` next to the package, or
    the `GEE_PROJECT` environment variable / config attribute. See the
    [Configuration](configuration.md) page.
