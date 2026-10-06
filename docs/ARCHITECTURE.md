# s3bloom architecture

This document describes how the codebase is organised, the key data
types that travel through it, and where to make changes when adding new
functionality. It is intended for someone taking over maintenance of
the project. For the *runtime* data flow (CLI invocation → output
files), see [`workflow.md`](workflow.md). The two are complementary:
this file is about code structure, that one is about what happens at
run time.

## Reading order for a new maintainer

1. Skim [`README.md`](../README.md) to install the project and run the
   pipeline once on a small AOI. This grounds everything below.
2. Read [`workflow.md`](workflow.md) end-to-end — it explains the four
   processing stages and the science behind them.
3. Read this document for the code-level map.
4. Read source files in this order: `defaults.py` → `config.py` →
   `cli.py` → `discovery/` → `processing/` → `compositing/` →
   `export/` → `metadata/`.

## Package layout

```
src/s3bloom/
├── __init__.py             # Package version
├── defaults.py             # Tunable constants (presets, flags, colormaps)
├── config.py               # Pydantic config models (validation)
├── cli.py                  # Typer CLI; orchestrates the pipeline
│
├── discovery/              # Talk to CDSE
│   ├── search.py           #   OData catalogue query
│   └── download.py         #   OAuth2 + streaming download + extraction
│
├── processing/             # Per-pass scientific work
│   ├── reader.py           #   .SEN3 → satpy Scene
│   ├── masking.py          #   WQSF bitfield → boolean mask
│   ├── resampling.py       #   swath → projected grid
│   └── pipeline.py         #   load → mask → resample → export
│
├── compositing/            # Cross-pass aggregation
│   └── temporal.py         #   N-day rolling geometric mean
│
├── export/                 # Format-specific writers
│   ├── __init__.py         #   Dispatcher (export_dataset)
│   ├── geotiff.py          #   GeoTIFF via rioxarray
│   ├── netcdf.py           #   CF-1.8 NetCDF
│   ├── png.py              #   Quick-look maps via cartopy
│   └── naming.py           #   Filename / path conventions
│
└── metadata/               # Provenance tracking
    └── provenance.py       #   Frozen dataclass + factories
```

Two organising principles:

* **Each subpackage owns one concern**, with `__init__.py` exposing the
  public surface. Cross-package wiring lives in `processing/pipeline.py`
  and `compositing/temporal.py` — those are the only modules that touch
  every layer.
* **Configuration is centralised** in `defaults.py` (constants) and
  `config.py` (validated models). Application code never reads
  environment variables or hardcodes magic numbers.

## Dependency graph

The arrows go from caller to callee. Cycles are forbidden.

```
                    cli.py
                      │
        ┌─────────────┼─────────────┬─────────────┐
        ▼             ▼             ▼             ▼
   discovery.     processing.    compositing.    config
   {search,        pipeline       temporal       (pydantic)
    download}         │               │             ▲
        │             │               │             │
        │             ▼               │             │
        │       processing.            │           defaults
        │       {reader,               │
        │        masking,              │
        │        resampling}           │
        │             │               │
        ▼             ▼               ▼
       export.{geotiff,netcdf,png,naming}
                      │
                      ▼
                metadata.provenance
```

Notes:

* `discovery.search` imports `_create_session` from `discovery.download`
  (lazy import inside the function) so the two share retry behaviour
  without forming a top-level cycle.
* `metadata.provenance` does a deferred import of
  `s3bloom.__version__` to avoid an import cycle with the package
  ``__init__``.

## Key data types

These are the types that travel between modules — get them right and
the rest of the code reads itself.

| Type | Defined in | Carries |
|------|-----------|---------|
| `PipelineConfig` | `config.py` | The fully validated run configuration. Built once by the CLI, passed to every stage. |
| `BoundingBox` | `config.py` | AOI in WGS84 lon/lat. Has helpers to parse from CLI strings and convert to a tuple. |
| `MaskingConfig` | `config.py` | Strictness preset (or custom flag list). Use `flags_for_product(name)` for product-aware masking. |
| `ProductInfo` | `discovery/search.py` | Frozen dataclass: catalogue metadata for one CDSE product. |
| `satpy.Scene` | (satpy) | Container for a loaded `.SEN3` product. Used internally in `processing/`; never returned to the CLI. |
| `xarray.DataArray` | (xarray) | The currency of the processing/export layer. Carries the raster + CRS attributes. |
| `PassResult` | `processing/pipeline.py` | Frozen dataclass: the outputs of one pass, processed now or found on disk from an earlier run. Its arrays are read lazily from the per-pass NetCDF, never from the raw product. Hand-off from processing → compositing. |
| `Provenance` | `metadata/provenance.py` | Frozen dataclass embedded in every output file. Lineage record. |

Everything is **immutable** where possible (`frozen=True` dataclasses,
pydantic models with no late mutation). Functions take config in,
produce values out — there is no shared mutable state and no globals.

## Adding new functionality

The codebase is organised so that the most common changes are
mechanical. The patterns below are the ones to follow.

### A new bounding-box preset

Add a tuple to `BBOX_PRESETS` in `defaults.py`. The CLI picks it up
automatically through `BoundingBox.from_string` — no other change
needed.

### A new masking strictness preset

1. Add an entry to `_COMMON_FLAGS` in `defaults.py`.
2. If the preset needs different BAC processing-chain flags, add an
   entry to `_BAC_FLAGS`.
3. Tests in `tests/test_masking.py` should already cover the
   combination logic; add a case if the new preset has unusual
   structure.

### A new dataset (e.g. `iop_nn`)

1. Add a per-product algorithm-failure flag to `PRODUCT_FLAGS` in
   `defaults.py` (and add the dataset to `BAC_PRODUCTS` if it uses
   the Open-Water atmospheric correction chain).
2. Add a `COLORMAP_SETTINGS` entry — pick a `cmocean` colormap that
   matches the variable's nature, set `vmin`/`vmax` from typical
   coastal values (display only), and add the product's physical range
   in linear units to `VALID_RANGE`. Variables that the OLCI product
   stores as log10 are converted to linear on load by
   `processing/reader.py`, driven by their `units`; nothing needs to be
   declared for that.
3. Add the CF `long_name` and `standard_name` to the helpers in
   `export/netcdf.py`.

### A new output format

1. Add a new module under `export/`, exposing one
   `export_<fmt>(data, path, provenance, dataset_name)` function.
2. Add the format identifier to `FORMAT_EXTENSIONS` in
   `export/__init__.py` and the dispatch branch in `export_dataset`.
3. Add the format to the validator in `OutputConfig._validate_formats`
   in `config.py`.
4. The CLI accepts arbitrary comma-separated format strings, so no CLI
   change is needed unless you want a dedicated short flag.

### A new CDSE collection (e.g. Sentinel-3 SLSTR)

This would be a larger change because the search filter, the satpy
reader name, and the per-product flag knowledge are all currently
OLCI-L2-WFR-specific. The right structuring move is to introduce a
"collection" abstraction: a small dataclass holding the OData product-
type filter, the satpy reader name, and a callable that builds the flag
list for a given dataset. `discovery/search.py`, `processing/reader.py`
and `defaults.py` would each consume one field of that record.

### A new CLI option

1. Add a default to `defaults.py`.
2. Add a typed field on the relevant model in `config.py`, with a
   pydantic validator if the value range needs checking.
3. Surface the option on `s3bloom run` in `cli.py` and pass it through
   `_build_config`.
4. If the option affects the science (masking, projection, …),
   propagate it into `Provenance` so outputs record what was used.

## Cross-cutting conventions

* **Logging, not printing**: every module logs through a
  module-scoped `logger = logging.getLogger(__name__)`. Use
  `console.print` only at the CLI layer for user-facing status.
* **Errors are raised at the right layer**: configuration errors come
  out of pydantic validators, network errors out of `discovery/`,
  domain errors out of `processing/`. The CLI catches and formats.
* **`from __future__ import annotations` everywhere** so type hints
  read naturally without import gymnastics.
* **Heavy imports are deferred** when possible — see the lazy imports
  inside `cli.run()` and `export/__init__.py`. This keeps `--help`
  responsive and is also what makes selectively importing one
  exporter cheap.
* **Outputs are idempotent**: re-running the pipeline with the same
  parameters skips already-downloaded `.SEN3` products and, by default,
  also skips per-pass and composite exports that exist on disk and are
  current. A per-pass output is current when its provenance says
  `value_scale = linear` and names the same source product; a composite
  is current when its recorded source products match the window. Skipped
  passes still feed compositing through their NetCDF files, so a resumed
  run and a fresh run give the same composites. Pass `--overwrite` to
  force a full reprocess.
* **Every output is linear and says so.** The `s3bloom_value_scale`
  attribute / TIFF tag is the contract with downstream scripts; files
  without it predate 0.2 and hold log10 values.

## Tests

Tests live under `tests/`, one file per module (`test_<module>.py`),
and run offline on synthetic data. Network access is mocked in
`test_search.py` and `test_download.py`; `test_resampling.py` resamples
a small synthetic swath through the real satpy/pyresample code path.

One test in `test_reader.py` is marked `integration`: it loads a real
`.SEN3` product from `data/raw/` and is skipped when that product is
absent. Deselect it with `pytest -m "not integration"`.

## Things to watch out for

* **CDSE token lifetime is short** (~5–10 minutes). The `_TokenManager`
  in `discovery/download.py` refreshes proactively, but very long
  multi-product downloads will hit refresh cycles — keep the refresh
  margin (`_TOKEN_REFRESH_MARGIN_S`) generous.
* **`Authorization` is stripped on cross-domain redirects** by
  `requests`. The download flow re-applies the bearer token after the
  redirect. If you refactor that code path, preserve this behaviour.
* **`flag_meanings` / `flag_masks` length mismatches** have been
  observed in the wild; the masking layer truncates to the common
  prefix. Don't tighten that to a hard error without first verifying
  current CDSE products are clean.
* **CRS coordinates from rioxarray are not NetCDF-serializable.**
  `export/netcdf.py` strips them and writes the CRS as a CF
  grid-mapping variable (`crs`) instead; if you start using new
  accessors that attach extra coords, extend the strip list.
* **Do not pass `reader_kwargs={"unlog": True}` to satpy.** In satpy
  0.60 it makes the `wqsf` flag layer fail to load. The log10 → linear
  conversion lives in `processing/reader.py` for that reason.
* **Keep `mask_area=False` in `resample_scene`.** pyresample's default
  for swath data drops masked source pixels from the nearest-neighbour
  search, which fills masked gaps from valid neighbours.
* **`cartopy.crs.epsg(code)` doesn't cover every projection.** EPSG:3035
  has an explicit handler in `export/png.py`; add explicit handlers
  for any projection you start using regularly.
