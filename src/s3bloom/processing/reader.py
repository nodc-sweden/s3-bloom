"""Load Sentinel-3 OLCI L2 products into satpy ``Scene`` objects.

A ``.SEN3`` product is a directory of NetCDF files plus a manifest XML.
This module passes the ``.nc`` files directly to satpy's ``olci_l2``
reader; the manifest is not used because satpy reconstructs the geometry
from the per-file metadata.

The satellite identifier (S3A/S3B) and sensing time are parsed from the
directory name rather than from file metadata. This is safe because
EUMETSAT guarantees the naming convention; doing it this way avoids
having to load the heavy NetCDFs just to read two scalars.

Value scale
-----------
OLCI L2 stores chlorophyll, TSM and the IOPs as ``log10`` of the
physical quantity, and satpy hands them over unchanged with units such
as ``"lg(re mg.m-3)"``. :func:`load_scene` converts every such dataset
to **linear** values (:func:`to_linear`) right after loading, so that
masking, resampling, compositing and every exporter work on, and write,
real concentrations.

satpy's own ``reader_kwargs={"unlog": True}`` is deliberately not used:
in satpy 0.60 it makes the ``wqsf`` flag layer fail to load
(``KeyError: 'units'``), which would break quality masking.
"""

from __future__ import annotations

import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import xarray as xr
from satpy import Scene
from s3bloom._winpath import win_path

logger = logging.getLogger(__name__)

READER_NAME = "olci_l2"

# satpy/OLCI notation for "log10 of a quantity relative to <unit>",
# e.g. ``"lg(re mg.m-3)"``.
LOG10_UNITS_PREFIX = "lg("
_LOG10_REFERENCE_PREFIX = "re "

# How satpy's ``long_name`` announces ``log10`` values, e.g.
# ``"log10 scaled (Neural Net) Algal pigment concentration"``.
_LOG10_LONG_NAME_PREFIX = re.compile(r"^log10 scaled ", re.IGNORECASE)


def load_scene(
    product_path: Path,
    datasets: list[str],
) -> Scene:
    """Load a ``.SEN3`` product directory into a satpy :class:`~satpy.Scene`.

    Parameters
    ----------
    product_path : pathlib.Path
        Path to the extracted ``.SEN3`` directory.
    datasets : list of str
        Names of OLCI L2 datasets to load (e.g. ``["chl_nn"]``). Names
        not present in the product are warned about and skipped, matching
        EUMETSAT's recommendation to be permissive when collections evolve.
        The WQSF flag layer is always loaded in addition to the requested
        datasets — it is needed by the masking step.

    Returns
    -------
    satpy.Scene
        Scene with the requested datasets and ``wqsf`` available via
        ``scene[name]``. Datasets that the product stores as ``log10``
        (units starting with ``"lg("``) are returned as linear values
        with the plain unit, e.g. ``"mg.m-3"``; see :func:`to_linear`.
        ``wqsf`` is returned exactly as satpy loaded it.

    Raises
    ------
    ValueError
        If none of the requested datasets are present in the product.
    FileNotFoundError
        If the directory contains no ``.nc`` files.
    """
    logger.info("Loading scene from %s", product_path)

    filenames = _collect_filenames(product_path)
    scn = Scene(filenames=filenames, reader=READER_NAME)

    available = scn.available_dataset_names()
    to_load = []
    for ds in datasets:
        if ds in available:
            to_load.append(ds)
        else:
            logger.warning(
                "Dataset %r not available in %s. Available: %s",
                ds,
                product_path.name,
                ", ".join(sorted(available)[:20]),
            )

    if not to_load:
        raise ValueError(
            f"None of the requested datasets {datasets} are available "
            f"in {product_path.name}"
        )

    # WQSF is required for downstream quality masking, regardless of
    # which scientific datasets the caller asked for.
    scn.load([*to_load, "wqsf"])

    # Only the requested science datasets are converted; ``wqsf`` is a
    # bitfield and must reach the masking step untouched.
    for name in to_load:
        if name not in scn:
            continue
        loaded = scn[name]
        linear = to_linear(loaded)
        if linear is not loaded:
            scn[name] = linear
            logger.info(
                "Converted %s from log10 (%s) to linear (%s)",
                name,
                loaded.attrs["units"],
                linear.attrs["units"],
            )

    logger.info("Loaded datasets: %s", list(scn.keys()))
    return scn


def to_linear(data: xr.DataArray) -> xr.DataArray:
    """Convert a ``log10``-encoded DataArray to linear values.

    A dataset counts as ``log10``-encoded when its ``units`` attribute
    starts with ``"lg("`` (satpy's notation, e.g. ``"lg(re mg.m-3)"``).

    Parameters
    ----------
    data : xarray.DataArray
        Dataset as loaded by satpy. Not modified.

    Returns
    -------
    xarray.DataArray
        For ``log10``-encoded input, a new array holding ``10 ** data``
        with the same dims, coords, dtype and laziness (a dask-backed
        input stays dask-backed). Every attribute is carried over —
        including ``area``, which resampling depends on — except the
        two that describe the scale: ``units`` becomes the inner unit
        (``"lg(re mg.m-3)"`` → ``"mg.m-3"``), and a ``long_name``
        starting with ``"log10 scaled "`` (any case) loses that prefix,
        because the GeoTIFF exporter writes it as the band description.
        Other ``long_name`` values are kept unchanged. The ``encoding``
        (the product's integer packing of the ``log10`` values) is not
        carried over, as it does not describe linear data. Any other
        input is returned as is.
    """
    units = data.attrs.get("units")
    if not isinstance(units, str) or not units.startswith(LOG10_UNITS_PREFIX):
        return data

    linear_attrs = {"units": _linear_units(units)}
    long_name = data.attrs.get("long_name")
    if isinstance(long_name, str):
        linear_attrs["long_name"] = _LOG10_LONG_NAME_PREFIX.sub("", long_name)

    # Work on the underlying (dask) array: xarray arithmetic would drop
    # the attributes, and ``copy`` keeps them without touching ``data``.
    linear = data.copy(deep=False, data=10 ** data.data).drop_encoding()
    return linear.assign_attrs(**linear_attrs)


def _linear_units(log_units: str) -> str:
    """Return the unit inside an ``"lg(re <unit>)"`` string."""
    inner = log_units[len(LOG10_UNITS_PREFIX):].strip()
    if inner.endswith(")"):
        inner = inner[:-1]
    return inner.removeprefix(_LOG10_REFERENCE_PREFIX).strip()


def extract_sensing_time(product_path: Path) -> datetime:
    """Parse sensing-start time from the ``.SEN3`` directory name.

    The EUMETSAT naming convention places a 15-character timestamp
    ``YYYYMMDDTHHMMSS`` as the fifth underscore-separated field, e.g.
    ``S3A_OL_2_WFR____20240315T091500_..._..._..._.SEN3``.

    Parameters
    ----------
    product_path : pathlib.Path
        Path whose ``.name`` is the standard product directory name.

    Returns
    -------
    datetime
        UTC-aware datetime of the sensing start.

    Raises
    ------
    ValueError
        If no token of the expected shape is present in the name.
    """
    name = product_path.name
    parts = name.split("_")

    for part in parts:
        if len(part) == 15 and part[0] == "2" and "T" in part:
            try:
                return datetime.strptime(part, "%Y%m%dT%H%M%S").replace(
                    tzinfo=timezone.utc
                )
            except ValueError:
                continue

    raise ValueError(f"Could not parse sensing time from product name: {name}")


def extract_satellite(product_path: Path) -> str:
    """Return the spacecraft tag (``"S3A"``/``"S3B"``) from the product name.

    Falls back to ``"S3X"`` for unrecognised prefixes so callers do not
    have to handle ``None``.
    """
    name = product_path.name
    if name.startswith("S3A"):
        return "S3A"
    if name.startswith("S3B"):
        return "S3B"
    return "S3X"


def _collect_filenames(product_path: Path) -> list[str]:
    """Return every ``*.nc`` file in *product_path* as a list of strings.

    satpy expects a flat file list and discovers which dataset is in
    which file from the file metadata. Files that satpy doesn't
    recognise (tie points, instrument data, etc.) produce harmless
    warnings, so we feed it the entire directory.
    """
    extended = win_path(product_path)
    nc_files = sorted(
        e.path for e in os.scandir(extended) if e.name.endswith(".nc")
    )
    filenames = nc_files

    if not filenames:
        raise FileNotFoundError(
            f"No .nc or manifest files found in {product_path}"
        )

    logger.debug("Collected %d files from %s", len(filenames), product_path.name)
    return filenames
