"""Multi-day rolling-window geometric-mean composites.

Single OLCI passes typically lose 60-90 % of their ocean pixels to
clouds with strict masking. This module fills the gaps by combining
multiple passes over a centred N-day window. The per-pixel
**geometric mean** is used, skipping NaN, so that a pixel which is
cloudy in some passes but clear in others contributes only its valid
observations.

Why a NaN-skipping geometric mean
---------------------------------
* The inputs are linear concentrations, which are approximately
  log-normally distributed; the geometric mean (the mean taken in
  ``log10`` space) is the natural average and is not dominated by a
  single high value.
* It is numerically what this module has always produced: it used to
  take the plain mean of ``log10``-encoded passes, and the composite is
  now ``10 **`` that value, written in linear units.
* Simple and reproducible — no weights, no priors.
* Cloud-robust by construction.
* Composes naturally over S3A + S3B passes on the same day.

The window length comes from ``config.output.composite_window_days``.
A composite is produced for every calendar day between the earliest
and latest pass, even days without their own pass — neighbouring-day
passes keep coverage continuous.

Inputs and resumed runs
-----------------------
The passes arrive as :class:`~s3bloom.processing.pipeline.PassResult`
objects whose arrays are normally lazy views of the per-pass NetCDFs,
not arrays in memory, so only the passes of the window being built are
loaded at any time. Passes that were skipped in this run because their
outputs were current take part exactly like freshly processed ones, so
a resumed run builds the same composites as a single fresh run.

There is one composite file per dataset, window length, centre date and
format, whichever satellites contributed. It records its source
products (``s3bloom_source_products``); a composite is left alone only
while that record matches the passes of its window, and is rebuilt in
place otherwise.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import xarray as xr

from s3bloom.config import PipelineConfig
from s3bloom.export import export_dataset
from s3bloom.export.naming import composite_output_path
from s3bloom.metadata.provenance import (
    checkable_output,
    create_composite_provenance,
    read_output_attrs,
)
from s3bloom.processing.pipeline import PassResult

logger = logging.getLogger(__name__)


def create_composites(
    pass_results: list[PassResult],
    config: PipelineConfig,
) -> list[Path]:
    """Build rolling-window composites for every dataset in *pass_results*.

    For each dataset and each calendar day between the first and last
    pass, the function:

    1. Selects every pass within ``±window/2`` days of the centre date,
       in the order of their source product names.
    2. Skips the composite if it exists and was built from exactly
       these passes (see below).
    3. Stacks the passes along a new ``pass_idx`` dimension and computes
       the NaN-skipping geometric mean across it, once.
    4. Writes the computed composite in every requested output format.

    An existing composite is kept only when every expected file exists
    and the ``s3bloom_source_products`` recorded in its NetCDF (or in
    its GeoTIFF when no NetCDF is requested) are the source products of
    the window. Otherwise it is rebuilt in place, e.g. after a pass was
    added to the window or a product was replaced by a newer version.
    PNG-only composites carry no readable record and are checked for
    existence only. ``config.output.overwrite`` forces a rebuild.

    A skipped pass without ``datasets`` (no per-pass NetCDF to read it
    back from) cannot be composited. An existing composite whose window
    holds such a pass is kept as it is rather than rebuilt from the
    remaining passes; a missing one is built from the passes with data.

    Parameters
    ----------
    pass_results : list of PassResult
        Output of :func:`s3bloom.processing.pipeline.process_single_pass`
        for every pass of the run: the passes processed now and the
        ones skipped because their outputs were current. Passes whose
        ``datasets`` are empty contribute nothing. The order of the list
        does not affect the result.
    config : PipelineConfig
        Pipeline configuration; supplies the window size, output
        directories, masking metadata for provenance, and target CRS.

    Returns
    -------
    list[pathlib.Path]
        Every composite file written by this call; composites that were
        skipped as up to date are not listed. May be empty if no passes
        were provided, but the function never raises in that case.
    """
    window = config.output.composite_window_days
    by_dataset = _group_by_dataset(pass_results)

    if not by_dataset:
        logger.warning("No pass results to composite")
        return []

    output_files: list[Path] = []
    up_to_date = 0
    dataless_dates = _dataless_dates(pass_results)

    for dataset_name, date_entries in by_dataset.items():
        dates = sorted(date_entries.keys())
        logger.info(
            "Compositing %s: %d unique dates, %d-day window",
            dataset_name,
            len(dates),
            window,
        )

        for center_date in _compute_composite_dates(dates, window):
            entries = _window_entries(date_entries, center_date, window)
            if not entries:
                continue

            incomplete = any(
                abs((d - center_date).days) <= window // 2 for d in dataless_dates
            )
            written = _write_composite(
                dataset_name, center_date, entries, config, incomplete=incomplete
            )
            output_files.extend(written)
            if not written:
                up_to_date += 1

    logger.info(
        "Composites: %d file(s) written, %d composite(s) already up to date",
        len(output_files),
        up_to_date,
    )
    return output_files


def _dataless_dates(pass_results: list[PassResult]) -> set[date]:
    """Return the sensing dates of skipped passes that carry no data.

    These are passes whose outputs exist from an earlier run but could
    not be read back because no per-pass NetCDF was written.
    """
    return {
        pr.sensing_time.date()
        for pr in pass_results
        if pr.skipped and not pr.datasets
    }


def _window_entries(
    date_entries: dict[date, list[tuple[xr.DataArray, str, str]]],
    center_date: date,
    window: int,
) -> list[tuple[xr.DataArray, str, str]]:
    """Return the passes within ``±window/2`` days of *center_date*.

    The entries are ordered by source product name. A floating-point
    mean depends, in its last bit, on the order of its terms, and the
    order in which passes arrive differs between a fresh run and a
    resumed one; a fixed order makes both write the same composite.
    """
    window_start = center_date - timedelta(days=window // 2)
    window_end = center_date + timedelta(days=window // 2)

    entries = [
        entry
        for d, day_entries in date_entries.items()
        if window_start <= d <= window_end
        for entry in day_entries
    ]
    return sorted(entries, key=lambda entry: entry[2])


def _write_composite(
    dataset_name: str,
    center_date: date,
    entries: list[tuple[xr.DataArray, str, str]],
    config: PipelineConfig,
    *,
    incomplete: bool = False,
) -> list[Path]:
    """Build and export one composite unless it is already up to date.

    Parameters
    ----------
    dataset_name : str
        Dataset being composited.
    center_date : date
        Centre of the window.
    entries : list of (DataArray, satellite, source product name)
        The passes of the window, as returned by :func:`_window_entries`.
    config : PipelineConfig
        Pipeline configuration.
    incomplete : bool, optional
        ``True`` when the window also holds a skipped pass without data,
        so *entries* is only part of the window. An existing composite
        is then kept instead of being replaced by a partial one.

    Returns
    -------
    list[pathlib.Path]
        The files written, one per output format; empty when the
        existing composite was kept.
    """
    window = config.output.composite_window_days
    arrays = [entry[0] for entry in entries]
    satellites = [entry[1] for entry in entries]
    source_products = [entry[2] for entry in entries]

    center_dt = datetime(
        center_date.year,
        center_date.month,
        center_date.day,
        tzinfo=timezone.utc,
    )
    out_paths = {
        fmt: composite_output_path(
            base_dir=config.output.base_dir,
            fmt=fmt,
            dataset=dataset_name,
            center_date=center_dt,
            window_days=window,
        )
        for fmt in config.output.formats
    }

    if incomplete and all(path.exists() for path in out_paths.values()):
        logger.warning(
            "Keeping composite %s %s: its window holds skipped passes "
            "without data, so it cannot be rebuilt in full",
            dataset_name,
            center_date,
        )
        return []

    if not config.output.overwrite and _composite_is_current(
        out_paths, source_products
    ):
        logger.info(
            "Skipping composite %s %s (up to date)", dataset_name, center_date
        )
        return []

    # Computed once here, then written from memory in every format.
    # The inputs are usually lazy views of the per-pass NetCDFs.
    composite = _geometric_mean_composite(arrays).compute()

    prov = create_composite_provenance(
        source_products=source_products,
        satellites=satellites,
        center_date=center_dt,
        dataset=dataset_name,
        masking_preset=config.masking.preset,
        masking_flags=config.masking.flags_for_product(dataset_name),
        projection=config.output.projection,
        resolution_m=config.output.resolution_m,
        composite_window_days=window,
    )

    for fmt, out_path in out_paths.items():
        export_dataset(composite, out_path, fmt, prov, dataset_name)
        logger.info("Composite exported: %s", out_path)

    return list(out_paths.values())


def _composite_is_current(
    out_paths: dict[str, Path],
    source_products: list[str],
) -> bool:
    """Tell whether an existing composite was built from *source_products*.

    Parameters
    ----------
    out_paths : dict[str, pathlib.Path]
        ``{format: path}`` of every file of the composite.
    source_products : list of str
        Source product names of the passes in the window.

    Returns
    -------
    bool
        ``True`` when every file exists and the recorded
        ``s3bloom_source_products`` (read from the NetCDF, else the
        GeoTIFF) name the same products, in any order. Without a file
        to read the record from (PNG only), existence is enough.
    """
    if not out_paths or not all(path.exists() for path in out_paths.values()):
        return False

    record_path = checkable_output(out_paths)
    if record_path is None:
        return True

    recorded = read_output_attrs(record_path).get("s3bloom_source_products", "")
    return sorted(recorded.split(",")) == sorted(source_products)


def _group_by_dataset(
    pass_results: list[PassResult],
) -> dict[str, dict[date, list[tuple[xr.DataArray, str, str]]]]:
    """Reshape pass results by dataset and sensing date.

    Returns a nested dict::

        {
            dataset_name: {
                date: [(data_array, satellite, source_product_name), ...]
            }
        }

    Two passes from the same day on different spacecraft (S3A + S3B)
    naturally end up in the same date bucket and contribute equally to
    every composite that covers that day.
    """
    grouped: dict[str, dict[date, list[tuple[xr.DataArray, str, str]]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for pr in pass_results:
        d = pr.sensing_time.date()
        for ds_name, data in pr.datasets.items():
            grouped[ds_name][d].append(
                (data, pr.satellite, pr.product_path.name)
            )

    return grouped


def _compute_composite_dates(
    dates: list[date],
    window: int,
) -> list[date]:
    """Return one centre date per calendar day in the observed range.

    Even days with no pass of their own get an entry — passes from
    neighbouring days within the rolling window can still produce a
    valid composite.

    The ``window`` parameter is currently unused (the date grid is
    independent of the window length); it is kept on the signature so
    a future change can compute a different stride without breaking
    callers.
    """
    if not dates:
        return []

    all_dates = set()
    start = min(dates)
    end = max(dates)
    current = start
    while current <= end:
        all_dates.add(current)
        current += timedelta(days=1)

    return sorted(all_dates)


def _geometric_mean_composite(arrays: list[xr.DataArray]) -> xr.DataArray:
    """Geometric mean of linear DataArrays along a new dim, ignoring NaN.

    The arrays are stacked along ``pass_idx``, converted to ``log10``,
    averaged with NaN skipped, and transformed back with ``10 **``.
    Values ``<= 0`` have no logarithm and are treated like NaN, i.e.
    they do not contribute; a pixel without any positive value is NaN.

    Single-array input is short-circuited to a copy so the trivial
    one-pass-window case stays cheap. The result inherits the first
    array's attributes (CRS, units, …); these are identical across
    inputs because every array came from the same target grid.
    """
    if len(arrays) == 1:
        return arrays[0].copy()

    stacked = xr.concat(arrays, dim="pass_idx")
    # Masking non-positive values first keeps ``log10`` free of
    # divide-by-zero / invalid-value warnings.
    log_stacked = np.log10(stacked.where(stacked > 0))
    composite = 10 ** log_stacked.mean(dim="pass_idx", skipna=True)

    composite.attrs = {**arrays[0].attrs}
    return composite


