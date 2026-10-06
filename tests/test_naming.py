"""Tests for export naming module."""

from datetime import datetime, timezone
from pathlib import Path

import pytest

from s3bloom.export.naming import (
    composite_filename,
    composite_output_path,
    pass_filename,
    pass_output_path,
)


class TestPassFilename:
    def test_basic(self):
        name = pass_filename(
            dataset="chl_nn",
            sensing_time=datetime(2024, 3, 15, 9, 15, 0, tzinfo=timezone.utc),
            satellite="S3A",
            ext="tif",
        )
        assert name == "s3bloom_chl_nn_pass_20240315T091500_S3A.tif"

    def test_netcdf_ext(self):
        name = pass_filename(
            dataset="tsm_nn",
            sensing_time=datetime(2024, 3, 15, 9, 15, 0, tzinfo=timezone.utc),
            satellite="S3B",
            ext="nc",
        )
        assert name == "s3bloom_tsm_nn_pass_20240315T091500_S3B.nc"


class TestCompositeFilename:
    def test_basic(self):
        name = composite_filename(
            dataset="chl_nn",
            center_date=datetime(2024, 3, 15, tzinfo=timezone.utc),
            window_days=3,
            ext="tif",
        )
        assert name == "s3bloom_chl_nn_composite3d_20240315.tif"

    def test_other_dataset_window_and_extension(self):
        name = composite_filename(
            dataset="tsm_nn",
            center_date=datetime(2024, 12, 1, tzinfo=timezone.utc),
            window_days=7,
            ext="png",
        )
        assert name == "s3bloom_tsm_nn_composite7d_20241201.png"

    def test_no_satellite_segment(self):
        name = composite_filename(
            dataset="chl_nn",
            center_date=datetime(2024, 3, 15, tzinfo=timezone.utc),
            window_days=3,
            ext="nc",
        )
        assert "S3" not in name
        assert name.endswith("_20240315.nc")

    def test_satellites_argument_is_gone(self):
        # One file per date: the contributing satellites must not be
        # able to change the name any more.
        with pytest.raises(TypeError):
            composite_filename(
                dataset="chl_nn",
                center_date=datetime(2024, 3, 15, tzinfo=timezone.utc),
                satellites=["S3A", "S3B"],
                window_days=3,
                ext="tif",
            )


class TestOutputPaths:
    def test_pass_output_path(self):
        path = pass_output_path(
            base_dir=Path("data"),
            fmt="geotiff",
            dataset="chl_nn",
            sensing_time=datetime(2024, 3, 15, 9, 15, 0, tzinfo=timezone.utc),
            satellite="S3A",
        )
        assert path == Path("data/processed/geotiff/s3bloom_chl_nn_pass_20240315T091500_S3A.tif")

    def test_composite_output_path(self):
        path = composite_output_path(
            base_dir=Path("data"),
            fmt="netcdf",
            dataset="chl_nn",
            center_date=datetime(2024, 3, 15, tzinfo=timezone.utc),
            window_days=3,
        )
        assert path == Path("data/composites/netcdf/s3bloom_chl_nn_composite3d_20240315.nc")

    def test_composite_output_path_takes_no_satellites(self):
        with pytest.raises(TypeError):
            composite_output_path(
                base_dir=Path("data"),
                fmt="netcdf",
                dataset="chl_nn",
                center_date=datetime(2024, 3, 15, tzinfo=timezone.utc),
                satellites=["S3A"],
                window_days=3,
            )
