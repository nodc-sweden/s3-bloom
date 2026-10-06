"""Immutable provenance records embedded in every output file.

A :class:`Provenance` instance is created once per output (per pass *and*
per dataset, or per composite) and travels through the export step,
where it is serialised into:

* TIFF tags in GeoTIFF outputs (prefix ``s3bloom_``);
* global attributes in NetCDF outputs (same prefix);
* the title text of PNG outputs.

Besides lineage, the record states how the stored numbers are to be
read: ``value_scale`` (always ``"linear"``; written as
``s3bloom_value_scale`` so that readers can tell these files from older
ones that held ``log10`` values and carry no such marker) and, for
composites, ``composite_method``.

:func:`read_output_attrs` reads the ``s3bloom_*`` attributes back from
a written NetCDF or GeoTIFF. The pipeline uses it to decide whether an
existing output is still current or has to be rebuilt;
:func:`checkable_output` picks the file to read when a product was
written in several formats.

The dataclass is ``frozen=True`` to make accidental late-stage mutation
impossible — once a record is created it represents one immutable
description of the output's lineage.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

# Prefix of every provenance attribute / TIFF tag written to an output.
ATTR_PREFIX = "s3bloom_"

# Output formats whose files carry the provenance record in a readable
# form, in the order in which :func:`checkable_output` prefers them.
RECORD_FORMATS = ("netcdf", "geotiff")

# Scale of every value the pipeline writes; see ``Provenance.value_scale``.
VALUE_SCALE = "linear"

# How :mod:`s3bloom.compositing.temporal` combines the passes of a window.
COMPOSITE_METHOD = "geometric_mean"


@dataclass(frozen=True)
class Provenance:
    """Immutable record of how one output product was produced.

    Single-pass outputs use the first nine fields; composite outputs
    additionally fill ``source_products``, ``composite_window_days``,
    ``pass_count`` and ``composite_method``.

    Attributes
    ----------
    source_product : str
        Source ``.SEN3`` directory name (passes) or a synthetic
        ``composite_YYYYMMDD`` identifier (composites).
    satellite : str
        ``"S3A"`` / ``"S3B"`` for passes; for composites the sorted
        hyphen-joined unique set, e.g. ``"S3A-S3B"``.
    sensing_time : datetime
        Pass sensing-start time, or composite centre date.
    dataset : str
        Logical dataset name (``"chl_nn"`` etc.).
    masking_preset : str
        Strictness preset name (``strict``/``moderate``/``relaxed``)
        or ``"custom"`` if explicit flags were used.
    masking_flags : tuple of str
        Exact flags applied — captured at the time of processing so
        the record reflects the actual masking, not just the preset.
    projection : str
        Target CRS as a string (``"EPSG:3035"`` or WKT).
    resolution_m : int
        Target pixel pitch in projected metres.
    pipeline_version : str
        :data:`s3bloom.__version__` at the time of creation.
    created_at : datetime
        UTC timestamp of record creation. Defaults to "now".
    source_products : tuple of str
        For composites, the contributing source product names.
    composite_window_days : int, optional
        Window length for composites; ``None`` for single passes.
    pass_count : int, optional
        Number of passes contributing to a composite; ``None`` for
        single passes.
    value_scale : str
        Scale of the stored values. Always ``"linear"`` (physical
        units such as mg m-3), never ``log10``. Machine-readable:
        downstream scripts key on the NetCDF global attribute / TIFF
        tag ``s3bloom_value_scale`` having exactly this value.
    composite_method : str
        How the passes of a composite were combined, e.g.
        ``"geometric_mean"``; empty for single passes.
    """

    source_product: str
    satellite: str
    sensing_time: datetime
    dataset: str
    masking_preset: str
    masking_flags: tuple[str, ...]
    projection: str
    resolution_m: int
    pipeline_version: str
    created_at: datetime = field(
        default_factory=lambda: datetime.now(tz=timezone.utc)
    )
    source_products: tuple[str, ...] = ()
    composite_window_days: int | None = None
    pass_count: int | None = None
    value_scale: str = VALUE_SCALE
    composite_method: str = ""

    def to_dict(self) -> dict:
        """Flatten to a string-valued dict for tag/attribute writers.

        Datetimes are serialised as ISO-8601, sequences as
        comma-separated strings, and ``None`` becomes ``""``.
        """
        return {
            "source_product": self.source_product,
            "satellite": self.satellite,
            "sensing_time": self.sensing_time.isoformat(),
            "dataset": self.dataset,
            "masking_preset": self.masking_preset,
            "masking_flags": ",".join(self.masking_flags),
            "projection": self.projection,
            "resolution_m": self.resolution_m,
            "pipeline_version": self.pipeline_version,
            "created_at": self.created_at.isoformat(),
            "source_products": ",".join(self.source_products),
            "composite_window_days": (
                str(self.composite_window_days)
                if self.composite_window_days is not None
                else ""
            ),
            "pass_count": (
                str(self.pass_count) if self.pass_count is not None else ""
            ),
            "value_scale": self.value_scale,
            "composite_method": self.composite_method,
        }

    def to_netcdf_attrs(self) -> dict[str, str]:
        """Return attributes suitable for NetCDF global attrs.

        Each non-empty field is prefixed with ``s3bloom_`` to avoid
        clashing with CF/CDM standard attribute names. The result
        always contains ``s3bloom_value_scale``; fields that only apply
        to composites (``s3bloom_composite_method`` etc.) are absent
        for single passes.
        """
        return {
            f"s3bloom_{k}": str(v) for k, v in self.to_dict().items() if v
        }


def create_pass_provenance(
    *,
    source_product: str,
    satellite: str,
    sensing_time: datetime,
    dataset: str,
    masking_preset: str,
    masking_flags: list[str],
    projection: str,
    resolution_m: int,
) -> Provenance:
    """Build a :class:`Provenance` record for a single satellite pass.

    ``value_scale`` keeps its default (``"linear"``) and
    ``composite_method`` stays empty.
    """
    from s3bloom import __version__

    return Provenance(
        source_product=source_product,
        satellite=satellite,
        sensing_time=sensing_time,
        dataset=dataset,
        masking_preset=masking_preset,
        masking_flags=tuple(masking_flags),
        projection=projection,
        resolution_m=resolution_m,
        pipeline_version=__version__,
    )


def create_composite_provenance(
    *,
    source_products: list[str],
    satellites: list[str],
    center_date: datetime,
    dataset: str,
    masking_preset: str,
    masking_flags: list[str],
    projection: str,
    resolution_m: int,
    composite_window_days: int,
) -> Provenance:
    """Build a :class:`Provenance` record for a multi-day composite.

    The synthetic ``source_product`` is ``composite_YYYYMMDD`` where
    the date is the composite centre; the actual source product names
    are stored in ``source_products``. ``composite_method`` is set to
    ``"geometric_mean"``, matching
    :mod:`s3bloom.compositing.temporal`.
    """
    from s3bloom import __version__

    unique_sats = sorted(set(satellites))
    return Provenance(
        source_product=f"composite_{center_date.strftime('%Y%m%d')}",
        satellite="-".join(unique_sats),
        sensing_time=center_date,
        dataset=dataset,
        masking_preset=masking_preset,
        masking_flags=tuple(masking_flags),
        projection=projection,
        resolution_m=resolution_m,
        pipeline_version=__version__,
        source_products=tuple(source_products),
        composite_window_days=composite_window_days,
        pass_count=len(source_products),
        composite_method=COMPOSITE_METHOD,
    )


def checkable_output(outputs: Mapping[str, Path]) -> Path | None:
    """Pick the file whose provenance record stands for a set of outputs.

    Parameters
    ----------
    outputs : Mapping[str, pathlib.Path]
        The files holding one product (one pass and dataset, or one
        composite), keyed by format name.

    Returns
    -------
    pathlib.Path or None
        The NetCDF if there is one, else the GeoTIFF, else ``None``:
        a PNG carries no readable record, so a PNG-only set of outputs
        can only be checked for existence.
    """
    for fmt in RECORD_FORMATS:
        if fmt in outputs:
            return outputs[fmt]
    return None


def read_output_attrs(path: Path) -> dict[str, str]:
    """Read the ``s3bloom_*`` attributes back from a written output.

    Parameters
    ----------
    path : pathlib.Path
        A NetCDF (``.nc``; global attributes) or GeoTIFF (``.tif``;
        dataset-level TIFF tags) written by the exporters.

    Returns
    -------
    dict[str, str]
        Every attribute whose name starts with ``s3bloom_``, e.g.
        ``s3bloom_value_scale`` or ``s3bloom_source_product``, with
        string values. Empty when the file carries no such attribute
        (outputs of versions that did not write them yet), is of
        another type (PNG), is missing, or cannot be read (e.g. it was
        truncated by an interrupted run). The file is closed again
        before returning, so the caller may overwrite it.
    """
    try:
        if path.suffix == ".nc":
            attrs = _read_netcdf_attrs(path)
        elif path.suffix == ".tif":
            attrs = _read_geotiff_tags(path)
        else:
            return {}
    except (OSError, ValueError) as exc:
        logger.debug("Could not read attributes of %s: %s", path, exc)
        return {}

    return {
        str(key): str(value)
        for key, value in attrs.items()
        if str(key).startswith(ATTR_PREFIX)
    }


def _read_netcdf_attrs(path: Path) -> dict:
    """Global attributes of the NetCDF file at *path*."""
    import xarray as xr

    with xr.open_dataset(path, engine="netcdf4") as ds:
        return dict(ds.attrs)


def _read_geotiff_tags(path: Path) -> dict:
    """Dataset-level TIFF tags of the GeoTIFF at *path*."""
    import rasterio

    with rasterio.open(path) as src:
        return dict(src.tags())
