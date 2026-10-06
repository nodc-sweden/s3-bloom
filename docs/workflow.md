# s3bloom processing workflow

This document describes the complete data flow from CLI invocation to final
output files. Each step maps to a specific module in `src/s3bloom/`.

## Overview

```
CLI arguments
  │
  ▼
PipelineConfig (validated)
  │
  ▼
1. Search ──── CDSE OData API ──── list[ProductInfo]
  │
  ▼
2. Download ── OData + token auth ── .SEN3 dirs in data/raw/
  │
  ▼
3. Process (per pass) ──┐
   a. Load scene        │  satpy olci_l2 reader
   b. Build mask        │  WQSF flag bitfield
   c. Apply mask        │  bad pixels → NaN
   d. Resample          │  swath → regular grid (pyresample)
   e. Export            │  GeoTIFF + NetCDF + PNG
                        ▼
               list[PassResult]
  │
  ▼
4. Composite ── geometric mean over 3-day window ── composite outputs
```

## Step 1: Product search

**Module:** `discovery/search.py`

The pipeline queries the Copernicus Data Space Ecosystem (CDSE) OData API for
Sentinel-3 OLCI Level-2 Water Full Resolution (OL_2_WFR) products that
intersect the requested bounding box and time period.

**API endpoint:** `https://catalogue.dataspace.copernicus.eu/odata/v1/Products`

The search uses an OData `$filter` with:
- `contains(Name, 'OL_2_WFR___')` — product type filter
- `ContentDate/Start ge/le` — temporal bounds
- `OData.CSC.Intersects(area=...)` — geographic intersection with a WKT polygon
  built from the bounding box

Results are sorted by sensing start time and paginated (100 products per page).

The catalogue can list one overpass several times: a near-real-time (`NR`)
product, later a non-time-critical (`NT`) one, and reprocessed copies. They
share a sensing start and would map to the same output file, so only the best
version of each overpass is kept (`NT` over `ST` over `NR`, then the latest
creation time).

**Output:** A list of `ProductInfo` dataclasses, each containing the product ID,
title, sensing time, satellite (S3A/S3B), and download size.

## Step 2: Product download

**Module:** `discovery/download.py`

### Authentication

CDSE requires an OAuth2 access token for downloads. The pipeline:
1. Reads `CDSE_USERNAME` and `CDSE_PASSWORD` from environment (loaded from `.env`)
2. POSTs to the CDSE identity provider token endpoint
3. Receives a bearer token valid for ~10 minutes

### Download process

For each product:
1. Check if the `.SEN3` directory already exists in `data/raw/` (idempotent)
2. Request the product zip via `Products({id})/$value`
3. Follow the redirect to the download server (auth header re-applied)
4. Stream to a temporary `.zip` file with progress bar
5. Extract the `.SEN3` directory
6. Delete the zip

Products are downloaded sequentially to respect CDSE rate limits.

### Disk space

Each `.SEN3` directory is ~500 MB. By default they are kept in `data/raw/` so
re-runs can skip re-downloading. Pass `--delete-raw` to remove each `.SEN3`
directory immediately after its pass is processed successfully. Failed passes
are never deleted, so they remain available for inspection or retry.

> **Note:** if `--delete-raw` was used on a previous run, the `.SEN3`
> directories are gone and will be re-downloaded on the next run, even if
> processed outputs already exist.

### .SEN3 directory structure

Each downloaded product is a directory containing ~35 NetCDF files:
```
S3A_OL_2_WFR____20240315T091500_....SEN3/
├── xfdumanifest.xml         # Product manifest
├── chl_nn.nc                # Chlorophyll (neural net)
├── chl_oc4me.nc             # Chlorophyll (OC4Me)
├── tsm_nn.nc                # Total suspended matter
├── wqsf.nc                  # Water Quality Science Flags
├── geo_coordinates.nc       # Lat/lon per pixel
├── Oa01_reflectance.nc      # Band reflectances
├── ...                      # ~30 more files
```

## Step 3: Per-pass processing

Each downloaded product goes through four sub-steps. If any step fails for a
product, processing continues with the next product.

**Skipping already-processed passes:** before loading any data, the pipeline
checks whether all expected output files (one per dataset × format combination)
already exist on disk and are current. An existing output is current when its
`s3bloom_value_scale` is `linear` and its `s3bloom_source_product` is the
product being processed; outputs from s3bloom before 0.2 (log10 values) or from
a superseded near-real-time product are reprocessed. Current passes are skipped
with a "Skipped (outputs exist)" message but still take part in compositing.
Pass `--overwrite` to force reprocessing of every pass regardless.

### 3a. Scene loading

**Module:** `processing/reader.py`

Uses satpy's `olci_l2` reader to load the requested datasets plus the WQSF
flag layer. The reader handles the multi-file `.SEN3` structure, coordinate
decoding, and metadata extraction.

The L2 chlorophyll and TSM variables are stored as log10 of the concentration.
They are converted to linear units (`10 ** value`) straight after loading, so
masking, resampling, compositing and every export work on mg m⁻³ / g m⁻³.

The `.nc` files are passed directly to satpy (the XML manifest is not used).
Files that satpy doesn't recognise (tie points, instrument data, etc.) produce
harmless warnings.

**Key detail:** The satellite identifier (S3A/S3B) and sensing time are parsed
from the directory name, not from file metadata, since the naming convention is
standardised by EUMETSAT.

### 3b. Quality masking

**Module:** `processing/masking.py`

This is the scientifically critical step. The WQSF (Water Quality Science
Flags) layer is a per-pixel bitfield where each bit indicates a quality
condition.

**How flags are read:**

Rather than hardcoding bit positions (which can change between processing
baselines), the pipeline reads `flag_meanings` and `flag_masks` attributes from
the WQSF variable's metadata. This follows EUMETSAT's recommendation for
forward compatibility.

**Masking logic:**

1. Read the flag definitions from the WQSF dataset attributes
2. Build a combined bitmask by OR-ing the masks for all requested flags
3. For each pixel: `is_bad = (wqsf_value & combined_bitmask) != 0`
4. Produce a boolean mask array (True = bad pixel)

**Masking presets:**

| Preset | # Flags | Typical masked % | Notes |
|--------|---------|------------------|-------|
| strict | 20 | 60–90% | Includes SUSPECT, HIGHGLINT, ADJAC, all RWNEG bands |
| moderate | 9 | 40–70% | Core flags only |
| relaxed | 4 | 20–50% | Clouds, invalid, saturated, snow/ice |

The strict preset is the default because for bloom monitoring, false positives
(flagging good pixels) are preferable to false negatives (keeping bad pixels
that corrupt composites).

**Prefix matching:** Flag names like `RWNEG_O2` through `RWNEG_O8` (negative
water-leaving reflectance in bands 2–8) are matched by prefix if the exact
name isn't found in the product metadata.

### 3c. Mask application

The boolean mask is applied to each requested dataset:
```python
data.where(~mask, other=NaN)
```

Bad pixels become NaN, which propagates correctly through all downstream
operations and is automatically excluded by compositing.

### 3d. Resampling

**Module:** `processing/resampling.py`

Satellite swath data has an irregular geometry (each pixel has its own lat/lon
from the satellite's viewing angle). This step reprojects the data onto a
regular rectangular grid.

**Target grid construction:**

1. The bounding box (WGS84 lon/lat) is transformed to the target projection
   using pyproj
2. A pyresample `AreaDefinition` is created with the specified resolution
3. For the default settings (swedish_west_coast, EPSG:3035, 300m), this
   produces a ~576 x 939 pixel grid

**Resampling method:** Nearest-neighbour with a 1500m radius of influence.
This is appropriate because:
- Ocean color products are discrete measurements, not continuous fields
- Interpolation would create artificial values at cloud edges
- The 300m output grid matches the native OLCI resolution

Each target pixel takes the value of its nearest swath pixel, masked or not, so
a masked pixel stays masked in the output. The radius only limits how far away
that nearest pixel may be. (Before 0.2 masked swath pixels were left out of the
search and gaps were filled from valid neighbours up to 1500m away.)

**Coordinate assignment:** After resampling, x/y coordinate arrays holding the
pixel-centre positions in the target projection are attached to the DataArray,
along with CRS metadata. The extent is not snapped to the resolution, so the
actual pixel size can differ slightly from the requested one (299.85 m for the
default grid).

### 3e. Export

Each resampled dataset is exported in all configured formats:

**GeoTIFF** (`export/geotiff.py`):
- Written via rioxarray with deflate compression
- CRS embedded in the file
- Provenance stored as TIFF tags
- Directly openable in QGIS, ArcGIS, Google Earth Engine

**NetCDF** (`export/netcdf.py`):
- CF-1.8 compliant with standard_name, units (linear, e.g. `mg m-3`) and the
  valid range of the product (0–100 mg m⁻³ for chlorophyll)
- Provenance in global attributes (prefixed `s3bloom_`), including
  `s3bloom_value_scale = linear`
- zlib compression level 4
- CRS stored as a CF grid mapping: a scalar `crs` variable referenced by the
  data variable's `grid_mapping` attribute, readable by GDAL, terra and
  rioxarray

**PNG** (`export/png.py`):
- cmocean "algae" colormap (oceanographic standard for chlorophyll)
- Linear colorbar with range 0–30 mg/m³ (appropriate for coastal
  Baltic/Kattegat)
- Title with satellite, timestamp, and masking preset
- Masked/land pixels shown in gray

## Step 4: Temporal compositing

**Module:** `compositing/temporal.py`

Individual satellite passes have large gaps due to cloud cover (often 60–90%
masked with strict settings). Compositing combines multiple passes to fill
gaps.

### Method

For each date in the time range:
1. Collect all passes within a centered N-day window (default: 3 days)
2. Stack the grids along a new `pass_idx` dimension
3. Compute the geometric mean ignoring NaN values: average `log10` of the
   concentrations, then back-transform with `10 **`
4. Export the composite in all configured formats

### Why a geometric mean

- **Suits the data**: chlorophyll is approximately log-normally distributed, so
  averaging in log space is less dominated by single high retrievals
- **Cloud-robust**: NaN-masked cloud pixels are automatically excluded
- **No weighting artefacts**: treats all valid observations equally
- **Handles overlap**: multiple S3A + S3B passes on the same day are combined
  naturally

The method is recorded in each composite as
`s3bloom_composite_method = geometric_mean`.

### Composite dates

A composite is produced for every date between the first and last pass,
regardless of whether that specific date has data. This gives continuous
daily coverage when the window captures passes from adjacent days.

**Skipping existing composites:** a composite is skipped when all its format
files exist and the source products recorded in it match the passes in the
window. If the window has gained or lost passes it is rebuilt. Pass
`--overwrite` to regenerate composites unconditionally.

**Resumed runs:** compositing reads the per-pass NetCDF files, so passes that
were processed in an earlier run are included and a resumed run gives the same
composites as a fresh one. This needs `netcdf` among `--formats`; without it
skipped passes cannot be read back, so existing composites for their dates are
kept as they are, new ones cover only the passes processed in the current run,
and the CLI prints a warning.

### File naming

```
s3bloom_chl_nn_composite3d_20240316.tif
         │         │          │
         │         │          └────────── center date
         │         └───────────────────── 3-day window
         └─────────────────────────────── dataset
```

There is one composite per dataset and date. The contributing satellites and
source products are recorded in the file's provenance.

## Data units

The OLCI L2 `chl_nn`, `chl_oc4me`, and `tsm_nn` variables are stored in the
source product as **log10** of the concentration. s3bloom converts them on
load, so every GeoTIFF, NetCDF and PNG it writes holds **linear** values
(mg m⁻³ for chlorophyll, g m⁻³ for TSM). Each file says so in the
`s3bloom_value_scale = linear` attribute (a global attribute in NetCDF, a tag
in GeoTIFF).

**Files written before s3bloom 0.2 are different.** They hold the raw log10
values although their `units` attribute says mg m⁻³, and they have no
`s3bloom_value_scale` attribute. Convert them with `10 ** value` before use:

| Value in a pre-0.2 file | Concentration |
|-------------------------|---------------|
| -1.0 | 0.1 mg m⁻³ |
| 0.0 | 1.0 mg m⁻³ |
| 0.5 | 3.2 mg m⁻³ |
| 1.0 | 10 mg m⁻³ |
| 2.0 | 100 mg m⁻³ |

Those files also have masked gaps filled from neighbouring pixels and x/y
coordinates offset by up to half a pixel (see step 3d); reprocess from the
`.SEN3` products to remove both.

## Provenance tracking

**Module:** `metadata/provenance.py`

Every output file carries an immutable provenance record containing:
- Source product name and satellite
- Sensing time
- Dataset name
- Masking preset and exact flags applied
- Target projection and resolution
- Pipeline version
- Creation timestamp
- For composites: list of source products, window size, pass count

Provenance is stored as:
- TIFF tags in GeoTIFF files
- Global attributes (prefixed `s3bloom_`) in NetCDF files
- Title metadata in PNG files

## Projection

The default projection is **EPSG:3035** (ETRS89 / LAEA Europe):
- Equal-area (preserves area measurements)
- Centred on Europe (low distortion for Baltic/Nordic region)
- Standard for EU environmental reporting (INSPIRE directive)
- Units in metres (intuitive resolution specification)

Alternative projections can be specified via `--projection`, e.g.
`EPSG:4326` (WGS84 lat/lon) or `EPSG:3006` (SWEREF99 TM, Swedish national grid).

## Memory and performance

- Products are processed **one at a time** and written to disk before the next
  product is loaded, keeping memory usage bounded
- Each pass is computed once and every format is exported from that array
- Compositing reads the per-pass NetCDF files lazily, one window at a time, so
  memory does not grow with the number of passes and raw products can be
  deleted (`--delete-raw`) as soon as a pass is written
- Downloads are sequential (max 2 concurrent) to respect CDSE rate limits
