# API reference

The public API is small: build a [`Config`](#config), then call
[`run_pipeline`](#run_pipeline). Both are importable from the top level:

```python
from hydroflow import Config, run_pipeline, OpmConfig, DEFAULT_STAGES
```

## `run_pipeline`

::: hydroflow.run_pipeline
    options:
      heading_level: 3

## `list_available_dems` / `describe_available_dems`

Browse the DEM datasets hydroflow can auto-download from Google Earth Engine
(see [Configuration → No local DEM?](configuration.md#no-local-dem-auto-download-from-earth-engine)).
No `[gee]` install required just to list them.

::: hydroflow.list_available_dems
    options:
      heading_level: 3

::: hydroflow.describe_available_dems
    options:
      heading_level: 3

## `Config`

The single configuration object for a run. `OpmConfig` is an alias of this
class. See [Configuration](configuration.md) for the parameter groups and the
string/integer option codes.

::: hydroflow.Config
    options:
      heading_level: 3
      members:
        - from_file
        - from_dict
        - save
        - validate
        - update_output_paths
        - describe_options
        - to_dict
