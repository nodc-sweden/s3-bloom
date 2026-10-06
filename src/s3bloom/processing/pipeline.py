"""Per-pass processing orchestrator: load → mask → resample → export.

This is the inner loop of the pipeline. Each ``.SEN3`` product passes
through this module exactly once. Failures here are caught at the CLI
level and recorded; processing of subsequent passes continues.

The :class:`PassResult` returned by :func:`process_single_pass` is the
hand-off to the compositing stage. Its arrays never depend on the raw
product: the resampled array is computed once, exported in every
format, and the result then refers to the per-pass NetCDF that was just
written, opened lazily, so that a run of any length keeps a flat memory
footprint and the ``.SEN3`` directory may be deleted right away. Only
when ``netcdf`` is not among the output formats is the computed array
itself kept in memory.

A pass whose outputs already exist and are current is not processed
again, but it is still returned (``skipped=True``), with its arrays
opened from the existing per-pass NetCDFs, so that a resumed run
composites the same passes as a fresh one.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import xarray as xr

from s3bloom.config import PipelineConfig
from s3bloom.export import export_dataset
from s3bloom.export.naming import pass_output_path
from s3bloom.metadata.provenance import (
    VALUE_SCALE,
    Provenance,
    checkable_output,
    create_pass_provenance,
    read_output_attrs,
)
from s3bloom.processing.masking import apply_mask, build_quality_mask
from s3bloom.processing.reader import (
    extract_satellite,
    extract_sensing_time,
    load_scene,
)
from s3bloom.processing.resampling import create_target_area, resample_scene

logger = logging.getLogger(__name__)

# Format whose per-pass files the compositing stage reads back.
_REOPEN_FORMAT = "netcdf"


@dataclass(frozen=True)
class PassResult:
    """Outputs and metadata for one satellite pass.

    Attributes
    ----------
    product_path : pathlib.Path
        Source ``.SEN3`` directory.
    satellite : str
        ``"S3A"``/``"S3B"``/``"S3X"``.
    sensing_time : datetime
        UTC sensing-start time (from the product directory name).
    datasets : dict[str, xarray.DataArray]
        Resampled DataArrays keyed by dataset name, for the compositing
        stage. They never read from the raw product. With ``netcdf``
        among the output formats each one is the per-pass NetCDF opened
        lazily (dask-backed; nothing is held in memory until it is
        computed); otherwise it is the computed array, held in memory.
        Empty for a skipped pass that has no per-pass NetCDF to read.
    provenance : dict[str, Provenance]
        Per-dataset provenance records of the outputs written in this
        run; empty for a skipped pass.
    output_files : list of pathlib.Path
        Every file written for this pass in this run, across all
        formats; empty for a skipped pass.
    skipped : bool
        ``True`` when the pass was not processed because its outputs
        already existed and were current.
    """

    product_path: Path
    satellite: str
    sensing_time: datetime
    datasets: dict[str, xr.DataArray]
    provenance: dict[str, Provenance]
    output_files: list[Path]
    skipped: bool = False


def process_single_pass(
    product_path: Path,
    config: PipelineConfig,
) -> PassResult:
    """Run the full per-pass pipeline on one ``.SEN3`` product.

    Steps:

    1. Read sensing-time and satellite from the directory name.
    2. Unless ``overwrite`` is set, return early when the outputs of
       this pass exist and are current (see below).
    3. Load the scene with satpy (:mod:`s3bloom.processing.reader`),
       which also converts ``log10``-encoded datasets to linear values.
    4. For every requested dataset, build a product-aware quality mask
       from WQSF and set bad pixels to NaN.
    5. Resample masked swaths onto the configured target grid, and
       check that nothing ``log10``-encoded is about to be exported.
    6. Compute each resampled dataset once, build its provenance record
       and export the computed array to every requested format.

    Existing outputs are current when every expected file exists and,
    for each dataset, the provenance record of its NetCDF (or of its
    GeoTIFF when no NetCDF is requested) has ``s3bloom_value_scale``
    equal to ``"linear"`` and ``s3bloom_source_product`` equal to the
    name of *product_path*. Files written by older versions (``log10``
    values, no scale marker) or from another version of the same
    overpass (e.g. a near-real-time product that has since been
    replaced) are therefore reprocessed and overwritten, and the reason
    is logged. PNG-only outputs carry no readable record and are
    checked for existence only.

    Parameters
    ----------
    product_path : pathlib.Path
        Path to the extracted ``.SEN3`` directory.
    config : PipelineConfig
        Validated pipeline configuration.

    Returns
    -------
    PassResult
        The resampled arrays (see :attr:`PassResult.datasets` for how
        they are held) plus the paths written. For a pass that was
        skipped because its outputs are current, ``skipped`` is
        ``True``, nothing was written, and ``datasets`` holds the
        existing per-pass NetCDFs when ``netcdf`` is among the output
        formats and is empty otherwise.

    Raises
    ------
    ValueError
        If a resampled dataset still carries ``log10`` units (starting
        with ``"lg("``). Outputs are labelled as linear, so such data
        must never be written.
    """
    satellite = extract_satellite(product_path)
    sensing_time = extract_sensing_time(product_path)
    expected = {
        ds_name: _output_paths(config, ds_name, sensing_time, satellite)
        for ds_name in config.datasets
    }

    if not config.output.overwrite:
        reason = _stale_reason(expected, product_path.name)
        if reason is None:
            logger.info(
                "Skipping %s (outputs exist and are current)", product_path.name
            )
            return PassResult(
                product_path=product_path,
                satellite=satellite,
                sensing_time=sensing_time,
                datasets=_open_pass_datasets(expected),
                provenance={},
                output_files=[],
                skipped=True,
            )
        if any(path.exists() for paths in expected.values() for path in paths.values()):
            logger.info("Reprocessing %s: %s", product_path.name, reason)

    logger.info(
        "Processing %s (%s, %s)",
        product_path.name,
        satellite,
        sensing_time.strftime("%Y-%m-%d %H:%M"),
    )

    resampled = _load_masked_resampled(product_path, config)

    datasets: dict[str, xr.DataArray] = {}
    provenance_records: dict[str, Provenance] = {}
    output_files: list[Path] = []

    for ds_name, lazy_array in resampled.items():
        prov = create_pass_provenance(
            source_product=product_path.name,
            satellite=satellite,
            sensing_time=sensing_time,
            dataset=ds_name,
            masking_preset=config.masking.preset,
            masking_flags=config.masking.flags_for_product(ds_name),
            projection=config.output.projection,
            resolution_m=config.output.resolution_m,
        )
        # Resampling only returns datasets that were requested.
        out_paths = expected[ds_name]

        datasets[ds_name] = _export_computed(lazy_array, out_paths, prov, ds_name)
        provenance_records[ds_name] = prov
        output_files.extend(out_paths.values())

    return PassResult(
        product_path=product_path,
        satellite=satellite,
        sensing_time=sensing_time,
        datasets=datasets,
        provenance=provenance_records,
        output_files=output_files,
    )


def _output_paths(
    config: PipelineConfig,
    dataset: str,
    sensing_time: datetime,
    satellite: str,
) -> dict[str, Path]:
    """Per-pass output path of *dataset* for every requested format."""
    return {
        fmt: pass_output_path(
            base_dir=config.output.base_dir,
            fmt=fmt,
            dataset=dataset,
            sensing_time=sensing_time,
            satellite=satellite,
        )
        for fmt in config.output.formats
    }


def _stale_reason(
    expected: dict[str, dict[str, Path]],
    product_name: str,
) -> str | None:
    """Say why the existing outputs of a pass cannot be kept.

    Parameters
    ----------
    expected : dict[str, dict[str, pathlib.Path]]
        ``{dataset: {format: path}}`` of every file the pass produces.
    product_name : str
        Name of the ``.SEN3`` directory about to be processed.

    Returns
    -------
    str or None
        ``None`` when the outputs are current (the rules are given in
        :func:`process_single_pass`), otherwise a short reason for the
        log. A pass without any expected file is never current.
    """
    paths = [path for by_format in expected.values() for path in by_format.values()]
    missing = [path for path in paths if not path.exists()]
    if missing or not paths:
        return f"{len(missing)} of {len(paths)} output file(s) missing"

    for by_format in expected.values():
        record_path = checkable_output(by_format)
        if record_path is None:
            continue
        attrs = read_output_attrs(record_path)
        scale = attrs.get("s3bloom_value_scale")
        if scale != VALUE_SCALE:
            return (
                f"{record_path.name} has s3bloom_value_scale={scale!r}, "
                f"expected {VALUE_SCALE!r}"
            )
        source = attrs.get("s3bloom_source_product")
        if source != product_name:
            return (
                f"{record_path.name} has s3bloom_source_product={source!r}, "
                "which is not the product being processed"
            )

    return None


def _load_masked_resampled(
    product_path: Path,
    config: PipelineConfig,
) -> dict[str, xr.DataArray]:
    """Load, mask and resample one product; nothing is computed yet.

    Returns the lazy ``{dataset: DataArray}`` dict of
    :func:`s3bloom.processing.resampling.resample_scene`.

    Raises
    ------
    ValueError
        If a resampled dataset still carries ``log10`` units.
    """
    scene = load_scene(product_path, config.datasets)

    # Mask is built per-dataset because the flag list is product-aware
    # (BAC vs AAC, OC4ME_FAIL vs OCNN_FAIL, etc.). Building it once per
    # product keeps memory usage flat.
    for ds_name in config.datasets:
        if ds_name in scene:
            mask = build_quality_mask(scene, config.masking, product=ds_name)
            scene[ds_name] = apply_mask(scene[ds_name], mask)

    target_area = create_target_area(config.bbox, config.output)
    resampled = resample_scene(scene, target_area, config.datasets)

    # Every output is declared linear (units, ``s3bloom_value_scale``).
    # The reader converts log10-encoded products on load; refuse to
    # write anything if a log10 array got this far regardless.
    for ds_name, data_array in resampled.items():
        units = str(data_array.attrs.get("units", ""))
        if units.startswith("lg("):
            raise ValueError(
                f"Dataset {ds_name!r} from {product_path.name} is still "
                f"log10-encoded (units {units!r}); refusing to export it "
                "because outputs must hold linear values."
            )

    return resampled


def _export_computed(
    lazy_array: xr.DataArray,
    out_paths: dict[str, Path],
    provenance: Provenance,
    dataset_name: str,
) -> xr.DataArray:
    """Compute one resampled dataset once and write it in every format.

    Parameters
    ----------
    lazy_array : xarray.DataArray
        Resampled dataset; usually a dask graph that still reads from
        the raw product.
    out_paths : dict[str, pathlib.Path]
        ``{format: path}`` of the files to write.
    provenance : Provenance
        Lineage record embedded in every file.
    dataset_name : str
        Logical dataset name.

    Returns
    -------
    xarray.DataArray
        What the compositing stage should hold for this dataset: the
        NetCDF just written, opened lazily, so that the result costs no
        memory; or, when no NetCDF was written, the computed array.
    """
    # The one and only evaluation of the load → mask → resample graph
    # for this dataset; every format is written from memory.
    data_array = lazy_array.compute()

    for fmt, out_path in out_paths.items():
        export_dataset(data_array, out_path, fmt, provenance, dataset_name)
        logger.info("Exported: %s", out_path)

    if _REOPEN_FORMAT in out_paths:
        return _open_pass_dataset(out_paths[_REOPEN_FORMAT], dataset_name)
    return data_array


def _open_pass_datasets(
    expected: dict[str, dict[str, Path]],
) -> dict[str, xr.DataArray]:
    """Open the existing per-pass NetCDF of every dataset, lazily.

    Datasets without a NetCDF among their expected outputs are left
    out, so the result is empty when ``netcdf`` is not an output format.
    """
    return {
        ds_name: _open_pass_dataset(by_format[_REOPEN_FORMAT], ds_name)
        for ds_name, by_format in expected.items()
        if _REOPEN_FORMAT in by_format
    }


def _open_pass_dataset(path: Path, dataset_name: str) -> xr.DataArray:
    """Open one per-pass NetCDF as a lazy array ready for re-export.

    Parameters
    ----------
    path : pathlib.Path
        NetCDF written by :func:`s3bloom.export.netcdf.export_netcdf`.
    dataset_name : str
        Name of the data variable inside the file.

    Returns
    -------
    xarray.DataArray
        Dask-backed ``(y, x)`` array with the ``x``/``y`` coordinates of
        the file. Values are read from *path* each time the array is
        computed; xarray reopens the file as needed, so no handle has to
        stay open. What belongs to the file rather than to the data is
        left behind: the ``crs`` grid-mapping variable, the
        ``grid_mapping`` attribute pointing at it, and the on-disk
        encoding. The exporters add their own when the array, or a
        composite built from it, is written again.
    """
    data = xr.open_dataset(path, engine="netcdf4", chunks={})[dataset_name]
    detached = data.drop_encoding()
    detached.attrs = {
        key: value for key, value in data.attrs.items() if key != "grid_mapping"
    }
    return detached
