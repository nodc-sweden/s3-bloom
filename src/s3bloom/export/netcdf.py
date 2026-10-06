"""CF-1.8 compliant NetCDF exporter.

Goals (in priority order):

1. **Self-describing**: any tool that follows the CF conventions can
   open the file and plot it without out-of-band metadata.
2. **Provenance-rich**: every output traces back to the source product,
   the masking applied, and the pipeline version.
3. **Compact**: zlib level-4 compression strikes a good balance between
   size (~3-4× smaller than uncompressed) and read latency.

CRS handling
------------
rioxarray attaches non-serializable CRS objects as coordinates which
NetCDF4 cannot write. We strip those coordinates and instead store the
CRS as a CF grid mapping: a scalar integer data variable named ``crs``
whose attributes are the CF projection parameters (from
:meth:`pyproj.CRS.to_cf`, which includes the full WKT as ``crs_wkt``)
plus the same WKT under ``spatial_ref`` for GDAL-based readers. The data
variable points at it through its ``grid_mapping`` attribute, so CF
readers (xarray with ``decode_coords="all"``, rioxarray, GDAL, terra)
recover the CRS without out-of-band metadata.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import xarray as xr
from pyproj import CRS

from s3bloom.defaults import COLORMAP_SETTINGS, VALID_RANGE
from s3bloom.metadata.provenance import Provenance

logger = logging.getLogger(__name__)

# Name of the CF grid-mapping variable holding the CRS.
_GRID_MAPPING_NAME = "crs"


def export_netcdf(
    data: xr.DataArray,
    path: Path,
    provenance: Provenance,
    dataset_name: str,
) -> Path:
    """Write *data* as a CF-1.8 compliant NetCDF file.

    Parameters
    ----------
    data : xarray.DataArray
        Resampled array on the target grid, in linear units. Values
        are written as given.
    path : pathlib.Path
        Output path.
    provenance : Provenance
        Lineage record stored as ``s3bloom_*`` global attributes,
        among them ``s3bloom_value_scale``.
    dataset_name : str
        Logical dataset name; used to look up CF ``standard_name``,
        ``long_name``, ``units`` and the valid range, and to set the
        variable name inside the file.

    Returns
    -------
    pathlib.Path
        Output path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    ds = _build_dataset(data, dataset_name, provenance)

    ds.to_netcdf(
        str(path),
        engine="netcdf4",
        encoding={
            dataset_name: {
                "dtype": "float32",
                "zlib": True,
                "complevel": 4,
                "_FillValue": np.nan,
            }
        },
    )

    logger.info("NetCDF exported: %s (%.1f MB)", path, path.stat().st_size / 1e6)
    return path


def _build_dataset(
    data: xr.DataArray,
    dataset_name: str,
    provenance: Provenance,
) -> xr.Dataset:
    """Wrap *data* in a CF-1.8 :class:`xarray.Dataset` with metadata.

    Adds ``long_name``, ``standard_name``, ``units``, axis attributes
    for the projected coordinates, a ``crs`` grid-mapping variable built
    from ``provenance.projection``, and every provenance field as a
    global attribute.

    ``units`` comes from :data:`s3bloom.defaults.COLORMAP_SETTINGS` and
    ``valid_min``/``valid_max`` from :data:`s3bloom.defaults.VALID_RANGE`
    (the physical range in linear units, not the colour-bar limits).
    Datasets without a ``VALID_RANGE`` entry get no valid-range
    attributes rather than an invented one.
    """
    color_cfg = COLORMAP_SETTINGS.get(dataset_name, {})
    valid_range = VALID_RANGE.get(dataset_name)
    valid_range_attrs = (
        {"valid_min": float(valid_range[0]), "valid_max": float(valid_range[1])}
        if valid_range is not None
        else {}
    )

    da = data.copy()

    # rioxarray/satpy attach a CRS object as a non-dim coord; netcdf4
    # can't serialize Python objects. Drop everything except the real
    # spatial coords.
    for coord_name in list(da.coords):
        if coord_name not in da.dims and coord_name not in ("x", "y"):
            da = da.drop_vars(coord_name)

    da.name = dataset_name
    da.attrs = {
        "long_name": _long_name(dataset_name),
        "units": color_cfg.get("units", ""),
        "standard_name": _standard_name(dataset_name),
        **valid_range_attrs,
        "grid_mapping": _GRID_MAPPING_NAME,
    }

    ds = da.to_dataset()

    ds.attrs = {
        "Conventions": "CF-1.8",
        "title": f"s3bloom {dataset_name} — {provenance.source_product}",
        "institution": "s3bloom pipeline",
        "source": f"Sentinel-3 {provenance.satellite} OLCI L2",
        "history": (
            f"Created {datetime.now(tz=timezone.utc).isoformat()} "
            f"by s3bloom v{provenance.pipeline_version}"
        ),
        "references": "https://sentinels.copernicus.eu/web/sentinel/missions/sentinel-3",
        **provenance.to_netcdf_attrs(),
    }

    if "x" in ds.coords:
        ds.x.attrs = {
            "standard_name": "projection_x_coordinate",
            "units": "m",
            "axis": "X",
        }
    if "y" in ds.coords:
        ds.y.attrs = {
            "standard_name": "projection_y_coordinate",
            "units": "m",
            "axis": "Y",
        }

    # Store the CRS as a CF grid-mapping variable: a scalar integer data
    # variable (not a coordinate) that the data variable references via
    # its ``grid_mapping`` attribute.
    crs = CRS.from_user_input(provenance.projection)
    ds = ds.assign(
        {
            _GRID_MAPPING_NAME: xr.DataArray(
                np.int32(0),
                attrs={**crs.to_cf(), "spatial_ref": crs.to_wkt()},
            )
        }
    )

    return ds


def _long_name(dataset_name: str) -> str:
    """CF ``long_name`` (human-readable description) for a dataset."""
    names = {
        "chl_nn": "Chlorophyll-a concentration (neural network)",
        "chl_oc4me": "Chlorophyll-a concentration (OC4Me algorithm)",
        "tsm_nn": "Total suspended matter (neural network)",
    }
    return names.get(dataset_name, dataset_name)


def _standard_name(dataset_name: str) -> str:
    """CF ``standard_name`` from the official table.

    Empty string is returned for datasets not in the lookup; CF allows
    omitting the attribute when no standard name fits.
    """
    names = {
        "chl_nn": "mass_concentration_of_chlorophyll_a_in_sea_water",
        "chl_oc4me": "mass_concentration_of_chlorophyll_a_in_sea_water",
        "tsm_nn": "mass_concentration_of_suspended_matter_in_sea_water",
    }
    return names.get(dataset_name, "")
