# s3bloom

[![CI](https://github.com/anderstorstensson/s3-bloom/actions/workflows/ci.yml/badge.svg)](https://github.com/anderstorstensson/s3-bloom/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/anderstorstensson/s3-bloom/branch/main/graph/badge.svg)](https://codecov.io/gh/anderstorstensson/s3-bloom)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Sentinel-3 OLCI Level-2 phytoplankton bloom pipeline. Downloads ocean color
data from the Copernicus Data Space Ecosystem (CDSE), applies quality masking,
reprojects to a regular grid, and produces GeoTIFF, NetCDF, and PNG outputs
with multi-day composites.

Designed for monitoring phytoplankton blooms in the Baltic Sea, Kattegat, and
Skagerrak, but works for any region covered by Sentinel-3 OLCI.

## Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) for dependency management
- A free [Copernicus Data Space](https://dataspace.copernicus.eu/) account

## Installation

```bash
git clone https://github.com/anderstorstensson/s3-bloom.git && cd s3-bloom

uv venv
uv sync

# Activate
source .venv/bin/activate
```

## Configuration

Create a `.env` file in the project root:

```
CDSE_USERNAME=your_copernicus_username
CDSE_PASSWORD=your_copernicus_password
```

## Quick start

```bash
# Swedish west coast, one week in March 2024
s3bloom run --start-date 2024-03-15 --end-date 2024-03-17 --bbox swedish_west_coast

# Custom bounding box, moderate masking, multiple datasets
s3bloom run \
  --start-date 2024-03-01 --end-date 2024-03-31 \
  --bbox "7.0,57.0,12.0,59.5" \
  --masking moderate \
  --datasets "chl_nn,chl_oc4me,tsm_nn"
```

## CLI reference

### `s3bloom run`

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--start-date` | `-s` | *required* | Start date (YYYY-MM-DD) |
| `--end-date` | `-e` | *required* | End date (YYYY-MM-DD) |
| `--bbox` | `-b` | *required* | Bounding box preset or `lon_min,lat_min,lon_max,lat_max` |
| `--masking` | `-m` | `strict` | Masking strictness: `strict`, `moderate`, `relaxed` |
| `--datasets` | `-d` | `chl_nn` | Comma-separated: `chl_nn`, `chl_oc4me`, `tsm_nn` |
| `--output-dir` | `-o` | `data` | Base output directory |
| `--projection` | | `EPSG:3035` | Target CRS (ETRS89-LAEA, good for Baltic/Nordic) |
| `--resolution` | | `300` | Grid resolution in metres |
| `--composite-window` | | `3` | Rolling composite window in days |
| `--formats` | | `geotiff,netcdf,png` | Output formats |
| `--no-composites` | | off | Skip composite generation |
| `--delete-raw` | | off | Delete `.SEN3` directories after successful processing |
| `--overwrite` | | off | Re-process and overwrite outputs even if they already exist |
| `--verbose` | `-v` | off | Debug logging |

### `s3bloom list-presets`

Shows available bounding box and masking presets.

## Bounding box presets

| Name | Region | Lon | Lat |
|------|--------|-----|-----|
| `swedish_west_coast` | Bohuslän to Halland | 10.0–13.0 | 56.5–59.0 |
| `kattegat` | Kattegat strait | 10.0–13.0 | 55.5–58.0 |
| `skagerrak` | Skagerrak basin | 7.0–12.0 | 57.0–59.5 |
| `kattegat_skagerrak` | Combined region | 7.0–13.5 | 55.5–59.5 |
| `kattegat_skagerrak_extended` | Extended Kattegat–Skagerrak | 5.0–13.5 | 55.5–60.0 |
| `baltic_proper` | Central Baltic (HELCOM Baltic Proper) | 13.0–23.0 | 54.0–60.0 |
| `baltic_all` | Full Baltic + Bothnian Sea/Bay + Gulfs | 13.0–30.0 | 54.0–66.0 |

## Output structure

```
data/
├── raw/                # Downloaded .SEN3 products
├── processed/          # Per-pass outputs
│   ├── geotiff/        #   CRS-tagged rasters
│   ├── netcdf/         #   CF-compliant with full metadata
│   └── png/            #   Maps with colorbar (cmocean algae cmap)
└── composites/         # 3-day rolling geometric-mean composites
    ├── geotiff/
    ├── netcdf/
    └── png/
```

File naming:
```
s3bloom_chl_nn_pass_20240315T091500_S3A.tif          # single pass
s3bloom_chl_nn_composite3d_20240315.tif              # 3-day composite
```

## Masking

Quality masking uses the WQSF (Water Quality and Science Flags) bitfield from
each product. Flag bit positions are read from product metadata at runtime, not
hardcoded, following EUMETSAT guidance for cross-collection compatibility.

Masking is **product-aware**: different OLCI products use different atmospheric
correction paths and have different algorithm failure flags. The pipeline
automatically selects the correct flags based on the dataset, following
[EUMETSAT Matchup Protocols v8B, Appendix A](https://user.eumetsat.int/s3/eup-strapi-media/Recommendations_for_Sentinel_3_OLCI_Ocean_Colour_product_validations_in_comparison_with_in_situ_measurements_Matchup_Protocols_V8_B_e6c62ce677.pdf).

### Flag categories

| Category | Applies to | Examples |
|----------|-----------|----------|
| **Common** | All Ocean Colour products | CLOUD, CLOUD_AMBIGUOUS, CLOUD_MARGIN, INVALID, COSMETIC, SATURATED, SUSPECT, HISOLZEN, HIGHGLINT, SNOW_ICE |
| **BAC processing chain** | Open Water products only (`chl_oc4me`) | AC_FAIL, WHITECAPS, ADJAC, RWNEG_O2–O8 |
| **Product failure** | Per-product | OCNN_FAIL (`chl_nn`, `tsm_nn`, `iop_nn`), OC4ME_FAIL (`chl_oc4me`) |

### Strictness presets

The `--masking` option controls how many *common* flags are applied.
Processing-chain and product failure flags are always added automatically.

| Preset | Common flags | chl_nn total | chl_oc4me total | Use case |
|--------|-------------|:---:|:---:|----------|
| **strict** (default) | 10 (all) | 11 | 21 | Conservative; best for quantitative analysis |
| **moderate** | 7 | 8 | 10 | Balance between coverage and quality |
| **relaxed** | 4 | 5 | 5 | Maximum coverage; visual inspection only |

Run `s3bloom list-presets` to see the exact flags per product and preset.

## Datasets

| Name | Description | Atm. correction | Units in output files |
|------|-------------|:---:|-----------------------|
| `chl_nn` | Chlorophyll-a, neural network | AAC (Complex Water) | mg m⁻³ |
| `chl_oc4me` | Chlorophyll-a, OC4Me band-ratio | BAC (Open Water) | mg m⁻³ |
| `tsm_nn` | Total suspended matter, neural network | AAC (Complex Water) | g m⁻³ |

All outputs hold **linear** concentrations and carry the attribute
`s3bloom_value_scale = linear`. Files written by s3bloom before 0.2 have no
such attribute and hold log10 of the concentration; see
[docs/workflow.md](docs/workflow.md#data-units).

`chl_nn` is the default. It uses a neural network with the Alternative
Atmospheric Correction (AAC), trained for Case-2 (coastal/complex) waters,
making it better suited for the optically complex Baltic/Kattegat than the
open-ocean OC4Me algorithm which uses the Baseline Atmospheric Correction (BAC).

## Documentation

* [`docs/workflow.md`](docs/workflow.md) — step-by-step description of
  the processing pipeline, data flow, and design decisions (read this
  first).
* [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — code-level map for
  maintainers: package layout, data types, dependency graph, and how
  to add new presets, datasets, output formats, or CDSE collections.

## Development

```bash
uv sync
uv run pytest tests/ -v
```

## Acknowledgements

This pipeline builds on the open-source [Pytroll](https://pytroll.github.io/)
ecosystem for satellite data processing. In particular it relies on:

- **[satpy](https://github.com/pytroll/satpy)** — multi-mission reader and
  scene abstraction (the OLCI L2 reader and resampling glue used here).
- **[pyresample](https://github.com/pytroll/pyresample)** — swath-to-grid
  reprojection.
- **[pycoast](https://github.com/pytroll/pycoast)**,
  **[trollimage](https://github.com/pytroll/trollimage)**,
  **[trollsift](https://github.com/pytroll/trollsift)** — supporting
  imagery, coastline overlays, and filename parsing.

If you use s3bloom in published work, please also cite satpy:

> Raspaud, M. *et al.* PyTroll: An open-source, community-driven Python
> framework to process Earth observation satellite data. *Bulletin of
> the American Meteorological Society* **99**, 1329–1336 (2018).
> [doi:10.1175/BAMS-D-17-0277.1](https://doi.org/10.1175/BAMS-D-17-0277.1)

Sentinel-3 OLCI data are produced by [EUMETSAT](https://www.eumetsat.int/)
and distributed via the
[Copernicus Data Space Ecosystem](https://dataspace.copernicus.eu/) under
the Copernicus open-data licence.

## License

MIT
