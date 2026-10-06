"""Filesystem layout and filename conventions for pipeline outputs.

Filenames embed enough information to identify a file in isolation
(dataset and, for a pass, sensing time and satellite; for a composite,
window length and centre date) so that downstream tools can parse them
without needing the provenance metadata. A composite name deliberately
leaves out the contributing satellites: there is exactly one composite
file per dataset, window length, centre date and format, whichever
passes went into it. The satellites and source products are recorded in
the provenance metadata instead. The top-level layout is::

    {base_dir}/processed/{format}/    — single-pass exports
    {base_dir}/composites/{format}/   — multi-day composite exports

If you change a filename pattern, update the corresponding example in
``README.md`` and ``docs/workflow.md``.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from s3bloom.export import FORMAT_EXTENSIONS


def pass_filename(
    *,
    dataset: str,
    sensing_time: datetime,
    satellite: str,
    ext: str,
) -> str:
    """Single-pass output filename.

    Pattern
    -------
    ``s3bloom_{dataset}_pass_{YYYYMMDDTHHMMSS}_{satellite}.{ext}``

    Example: ``s3bloom_chl_nn_pass_20240315T091500_S3A.tif``.
    """
    ts = sensing_time.strftime("%Y%m%dT%H%M%S")
    return f"s3bloom_{dataset}_pass_{ts}_{satellite}.{ext}"


def composite_filename(
    *,
    dataset: str,
    center_date: datetime,
    window_days: int,
    ext: str,
) -> str:
    """Composite-output filename.

    Pattern
    -------
    ``s3bloom_{dataset}_composite{N}d_{YYYYMMDD}.{ext}``

    The date is the centre of the window. The name does not depend on
    which satellites contributed, so a composite that gains or loses a
    pass is rewritten in place instead of leaving a second file for the
    same date.

    Example: ``s3bloom_chl_nn_composite3d_20240315.tif``.
    """
    ds = center_date.strftime("%Y%m%d")
    return f"s3bloom_{dataset}_composite{window_days}d_{ds}.{ext}"


def pass_output_path(
    *,
    base_dir: Path,
    fmt: str,
    dataset: str,
    sensing_time: datetime,
    satellite: str,
) -> Path:
    """Compose the full output path for a single-pass file.

    The path is ``{base_dir}/processed/{fmt}/{filename}``.
    """
    ext = FORMAT_EXTENSIONS[fmt]
    name = pass_filename(
        dataset=dataset,
        sensing_time=sensing_time,
        satellite=satellite,
        ext=ext,
    )
    return base_dir / "processed" / fmt / name


def composite_output_path(
    *,
    base_dir: Path,
    fmt: str,
    dataset: str,
    center_date: datetime,
    window_days: int,
) -> Path:
    """Compose the full output path for a composite file.

    The path is ``{base_dir}/composites/{fmt}/{filename}``.
    """
    ext = FORMAT_EXTENSIONS[fmt]
    name = composite_filename(
        dataset=dataset,
        center_date=center_date,
        window_days=window_days,
        ext=ext,
    )
    return base_dir / "composites" / fmt / name
