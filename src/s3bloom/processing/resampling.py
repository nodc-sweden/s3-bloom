"""Reproject OLCI swath data onto a regular projected grid via pyresample.

OLCI L2 products store data in a *swath* geometry: each pixel carries
its own ``(lat, lon)`` derived from the satellite's viewing angle, and
the array indices have no direct geographic meaning. This module turns
that into a regular grid with constant pixel spacing in a chosen
projection so the result is straightforward to overlay, composite and
write to GeoTIFF/NetCDF.

Why nearest-neighbour resampling
--------------------------------
Ocean-colour retrievals are discrete measurements, not samples of a
continuous field. Bilinear/cubic interpolation would create plausible-
looking but synthetic values across cloud edges and along the
coastline. Nearest-neighbour preserves the original pixel population at
the cost of a slightly blocky appearance.

Why a 1500 m radius of influence
--------------------------------
~5× the native ~300 m pixel pitch — large enough to fill the target
grid even when swath pixels are stretched at the edge of the field of
view. The radius is only the maximum distance at which the nearest
source pixel is accepted; target pixels with no source pixel inside it
(outside the swath) are set to NaN.

The radius does not by itself protect masked pixels. pyresample's
default (``mask_area=True``) drops NaN source pixels from the
neighbour search, so every masked target pixel would take the value of
the nearest *valid* source pixel up to 1500 m (~5 pixels) away, and
cloud/flag gaps would shrink or vanish. We therefore resample with
``mask_area=False``: masked source pixels stay in the search, each
target pixel takes its true nearest source pixel, and a masked source
pixel stays masked in the output.
"""

from __future__ import annotations

import logging

import numpy as np
import xarray as xr
from pyresample import create_area_def
from pyresample.geometry import AreaDefinition
from satpy import Scene

from s3bloom.config import BoundingBox, OutputConfig

logger = logging.getLogger(__name__)


def create_target_area(
    bbox: BoundingBox,
    output_config: OutputConfig,
) -> AreaDefinition:
    """Build a pyresample :class:`AreaDefinition` for the output grid.

    Parameters
    ----------
    bbox : BoundingBox
        AOI in WGS84 lon/lat. The corners are reprojected into the
        target CRS to derive the area extent in projected metres.
    output_config : OutputConfig
        Provides the target ``projection`` (CRS string) and
        ``resolution_m`` (pixel pitch in projected units, typically
        metres).

    Returns
    -------
    pyresample.geometry.AreaDefinition
        Target grid; its width/height are derived from extent ÷
        resolution. For the default ``swedish_west_coast`` AOI in
        EPSG:3035 at 300 m this is ~576 × 939 pixels.
    """
    area = create_area_def(
        area_id="s3bloom_target",
        projection=output_config.projection,
        area_extent=_bbox_to_extent(bbox, output_config.projection),
        resolution=output_config.resolution_m,
        description="s3bloom target area",
    )

    logger.info(
        "Target area: %dx%d pixels, resolution=%dm, projection=%s",
        area.width,
        area.height,
        output_config.resolution_m,
        output_config.projection,
    )
    return area


def resample_scene(
    scene: Scene,
    target_area: AreaDefinition,
    datasets: list[str],
) -> dict[str, xr.DataArray]:
    """Resample each requested dataset in *scene* onto *target_area*.

    Parameters
    ----------
    scene : satpy.Scene
        Loaded scene; masking has typically already been applied.
    target_area : pyresample.geometry.AreaDefinition
        Output grid built by :func:`create_target_area`.
    datasets : list of str
        Names of datasets to resample. Only these are resampled; other
        datasets in the scene (e.g. the ``wqsf`` flag layer) are left
        alone. Names missing from the scene are warned and skipped.

    Returns
    -------
    dict[str, xarray.DataArray]
        ``{dataset_name: resampled_dataarray}``. Each output carries
        ``x``/``y`` coordinate arrays holding the pixel-centre
        positions of *target_area* in projected metres (``y`` running
        north to south) and a ``crs`` attribute set to the WKT of
        *target_area*. Source pixels that were NaN stay NaN: a masked
        pixel is never replaced by a neighbouring valid one.
    """
    logger.info("Resampling %d datasets to target grid", len(datasets))

    # Scene.resample raises KeyError for names it does not hold, so only
    # ask for the ones that are there.
    present = [ds_name for ds_name in datasets if ds_name in scene]
    for ds_name in datasets:
        if ds_name not in present:
            logger.warning("Dataset %s not in scene; skipping resampling", ds_name)

    resampled_scene = scene.resample(
        target_area,
        datasets=present,
        resampler="nearest",
        radius_of_influence=1500,
        fill_value=np.nan,
        # Keep masked (NaN) source pixels in the neighbour search so they
        # stay masked instead of being filled from valid neighbours.
        mask_area=False,
        # Nothing to regenerate: we resample plain datasets, not
        # composites. Left on, satpy warns that the datasets we chose
        # not to resample (wqsf) "were not created".
        generate=False,
    )

    results = {}
    for ds_name in present:
        if ds_name in resampled_scene:
            da = resampled_scene[ds_name]
            results[ds_name] = _add_spatial_coords(da, target_area)
            logger.debug("Resampled %s: shape=%s", ds_name, da.shape)
        else:
            logger.warning("Dataset %s not found after resampling", ds_name)

    return results


def _bbox_to_extent(
    bbox: BoundingBox,
    projection: str,
) -> tuple[float, float, float, float]:
    """Reproject a WGS84 bounding box into ``(x_min, y_min, x_max, y_max)``.

    The two diagonally opposite corners are transformed and then sorted,
    which handles projections in which transformed coordinates may run
    in either direction (``min``/``max`` order can flip).

    Returns
    -------
    tuple[float, float, float, float]
        Area extent in the target CRS, suitable for pyresample's
        ``area_extent`` argument.
    """
    from pyproj import Transformer

    transformer = Transformer.from_crs("EPSG:4326", projection, always_xy=True)

    x_min, y_min = transformer.transform(bbox.lon_min, bbox.lat_min)
    x_max, y_max = transformer.transform(bbox.lon_max, bbox.lat_max)

    return (
        min(x_min, x_max),
        min(y_min, y_max),
        max(x_min, x_max),
        max(y_min, y_max),
    )


def _add_spatial_coords(
    data: xr.DataArray,
    area: AreaDefinition,
) -> xr.DataArray:
    """Attach projected ``x``/``y`` coordinate arrays and CRS metadata.

    Coordinates are the pixel *centres* of *area*, so the first ``x`` is
    half a pixel inside ``area_extent[0]`` and consecutive values are
    exactly one pixel apart. The y axis runs from the northernmost row
    (half a pixel below ``area_extent[3]``) down to the southernmost,
    matching the row-major top-down ordering that GeoTIFF and other
    raster formats expect.
    """
    if "y" not in data.dims or "x" not in data.dims:
        if len(data.dims) == 2:
            data = data.rename({data.dims[0]: "y", data.dims[1]: "x"})

    x_coords = np.asarray(area.projection_x_coords)
    y_coords = np.asarray(area.projection_y_coords)

    data = data.assign_coords(x=("x", x_coords), y=("y", y_coords))
    data.attrs["crs"] = area.crs.to_wkt()
    data.attrs["area_id"] = area.area_id

    return data
