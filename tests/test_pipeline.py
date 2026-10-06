"""Tests for processing.pipeline."""

from __future__ import annotations

import logging
from datetime import date, datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import dask
import numpy as np
import pytest
import rasterio
import xarray as xr
from rasterio.transform import from_origin

from s3bloom.config import BoundingBox, OutputConfig, PipelineConfig, TimePeriod
from s3bloom.export import export_dataset
from s3bloom.export.naming import pass_output_path
from s3bloom.metadata.provenance import read_output_attrs
from s3bloom.processing import pipeline
from s3bloom.processing.pipeline import PassResult, process_single_pass

SENSING = datetime(2024, 3, 15, 9, 15, 0, tzinfo=timezone.utc)

# Two versions of the same overpass: the non-time-critical product and
# the near-real-time one it replaces. Both map to one output filename.
NT_PRODUCT = (
    "S3A_OL_2_WFR____20240315T091500_20240315T091800_20240316T173014"
    "_0179_110_122_1980_MAR_O_NT_003.SEN3"
)
NR_PRODUCT = (
    "S3A_OL_2_WFR____20240315T091500_20240315T091800_20240315T110000"
    "_0179_110_122_1980_MAR_O_NR_003.SEN3"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _config(
    tmp_path: Path,
    formats: list[str] | None = None,
    *,
    overwrite: bool = False,
    datasets: list[str] | None = None,
) -> PipelineConfig:
    return PipelineConfig(
        bbox=BoundingBox(lon_min=10.0, lat_min=56.5, lon_max=13.0, lat_max=59.0),
        time_period=TimePeriod(start_date=date(2024, 3, 15), end_date=date(2024, 3, 15)),
        datasets=datasets or ["chl_nn"],
        output=OutputConfig(
            base_dir=tmp_path / "out",
            formats=formats or ["netcdf"],
            overwrite=overwrite,
        ),
    )


def _out_path(config: PipelineConfig, fmt: str, dataset: str = "chl_nn") -> Path:
    return pass_output_path(
        base_dir=config.output.base_dir,
        fmt=fmt,
        dataset=dataset,
        sensing_time=SENSING,
        satellite="S3A",
    )


def _linear(data: xr.DataArray, offset: float = 0.0) -> xr.DataArray:
    """Copy of *data* shifted by *offset* and labelled as linear."""
    return (data + offset).astype(np.float32).assign_attrs(units="mg.m-3")


def _write_plain_netcdf(path: Path, data: xr.DataArray, **global_attrs: str) -> None:
    """Write a NetCDF the way an older version could have left it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    xr.Dataset({"chl_nn": data}, attrs=global_attrs).to_netcdf(path)


def _write_plain_geotiff(path: Path, data: xr.DataArray, **tags: str) -> None:
    """Write a GeoTIFF the way an older version could have left it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    values = np.asarray(data.values, dtype=np.float32)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=values.shape[0],
        width=values.shape[1],
        count=1,
        dtype="float32",
        crs="EPSG:3035",
        transform=from_origin(4_000_000.0, 3_001_200.0, 300.0, 300.0),
    ) as dst:
        dst.write(values, 1)
        dst.update_tags(**tags)


_PLAIN_WRITERS = {"netcdf": _write_plain_netcdf, "geotiff": _write_plain_geotiff}


def _stored_values(path: Path) -> np.ndarray:
    """Read the raster back from a per-pass NetCDF or GeoTIFF."""
    if path.suffix == ".nc":
        with xr.open_dataset(path) as ds:
            return ds["chl_nn"].values
    with rasterio.open(path) as src:
        return src.read(1)


# ---------------------------------------------------------------------------
# PassResult
# ---------------------------------------------------------------------------

class TestPassResult:
    def test_construction(self, tmp_path, simple_dataarray, provenance):
        result = PassResult(
            product_path=tmp_path / "product.SEN3",
            satellite="S3A",
            sensing_time=datetime(2024, 3, 15, tzinfo=timezone.utc),
            datasets={"chl_nn": simple_dataarray},
            provenance={"chl_nn": provenance},
            output_files=[tmp_path / "out.nc"],
        )
        assert result.satellite == "S3A"
        assert "chl_nn" in result.datasets

    def test_not_skipped_by_default(self, tmp_path):
        result = PassResult(
            product_path=tmp_path / "product.SEN3",
            satellite="S3A",
            sensing_time=datetime(2024, 3, 15, tzinfo=timezone.utc),
            datasets={},
            provenance={},
            output_files=[],
        )
        assert result.skipped is False

    def test_can_be_marked_skipped(self, tmp_path):
        result = PassResult(
            product_path=tmp_path / "product.SEN3",
            satellite="S3A",
            sensing_time=datetime(2024, 3, 15, tzinfo=timezone.utc),
            datasets={},
            provenance={},
            output_files=[],
            skipped=True,
        )
        assert result.skipped is True

    def test_frozen(self, tmp_path, simple_dataarray, provenance):
        result = PassResult(
            product_path=tmp_path / "product.SEN3",
            satellite="S3A",
            sensing_time=datetime(2024, 3, 15, tzinfo=timezone.utc),
            datasets={"chl_nn": simple_dataarray},
            provenance={"chl_nn": provenance},
            output_files=[],
        )
        with pytest.raises(Exception):
            result.satellite = "S3B"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# process_single_pass
# ---------------------------------------------------------------------------

class TestProcessSinglePass:
    def test_returns_pass_result(self, tmp_path, simple_dataarray, fake_stages):
        config = _config(tmp_path)
        fake_stages.resampled[NT_PRODUCT] = {"chl_nn": _linear(simple_dataarray)}

        result = process_single_pass(tmp_path / NT_PRODUCT, config)

        assert isinstance(result, PassResult)
        assert result.skipped is False
        assert result.product_path == tmp_path / NT_PRODUCT
        assert result.satellite == "S3A"
        assert result.sensing_time == SENSING
        assert set(result.datasets) == {"chl_nn"}
        assert result.provenance["chl_nn"].source_product == NT_PRODUCT

    def test_output_files_collected(self, tmp_path, simple_dataarray, fake_stages):
        config = _config(tmp_path, ["netcdf", "geotiff"])
        fake_stages.resampled[NT_PRODUCT] = {"chl_nn": _linear(simple_dataarray)}

        result = process_single_pass(tmp_path / NT_PRODUCT, config)

        # two formats × one dataset = 2 files
        assert sorted(result.output_files) == sorted(
            [_out_path(config, "netcdf"), _out_path(config, "geotiff")]
        )
        assert all(path.exists() for path in result.output_files)

    def test_outputs_record_source_product_and_scale(
        self, tmp_path, simple_dataarray, fake_stages
    ):
        config = _config(tmp_path, ["netcdf", "geotiff"])
        fake_stages.resampled[NT_PRODUCT] = {"chl_nn": _linear(simple_dataarray)}

        result = process_single_pass(tmp_path / NT_PRODUCT, config)

        for path in result.output_files:
            attrs = read_output_attrs(path)
            assert attrs["s3bloom_source_product"] == NT_PRODUCT
            assert attrs["s3bloom_value_scale"] == "linear"

    def test_dataset_not_in_scene_skips_masking(self, tmp_path, fake_stages, monkeypatch):
        config = _config(tmp_path)
        # Neither the scene nor the resampling result holds the dataset.
        fake_stages.resampled[NT_PRODUCT] = {}
        mock_mask = MagicMock()
        monkeypatch.setattr(pipeline, "build_quality_mask", mock_mask)

        result = process_single_pass(tmp_path / NT_PRODUCT, config)

        mock_mask.assert_not_called()
        assert result.output_files == []
        assert result.datasets == {}
        assert result.skipped is False

    def test_datasets_in_scene_are_masked_per_product(
        self, tmp_path, simple_dataarray, fake_stages, monkeypatch
    ):
        config = _config(tmp_path, datasets=["chl_nn", "tsm_nn"])
        raw = _linear(simple_dataarray)
        masked = _linear(simple_dataarray, offset=1.0)
        scene = {"chl_nn": raw, "tsm_nn": raw}
        mock_mask = MagicMock(return_value="mask")
        mock_apply = MagicMock(return_value=masked)
        monkeypatch.setattr(pipeline, "load_scene", lambda path, datasets: scene)
        monkeypatch.setattr(pipeline, "build_quality_mask", mock_mask)
        monkeypatch.setattr(pipeline, "apply_mask", mock_apply)
        monkeypatch.setattr(
            pipeline, "resample_scene", lambda scn, area, datasets: dict(scn)
        )

        result = process_single_pass(tmp_path / NT_PRODUCT, config)

        assert [call.kwargs["product"] for call in mock_mask.call_args_list] == [
            "chl_nn",
            "tsm_nn",
        ]
        assert mock_apply.call_count == 2
        np.testing.assert_array_equal(
            result.datasets["tsm_nn"].values, masked.values
        )

    @pytest.mark.parametrize("units", ["lg(re mg.m-3)", "lg(re g.m-3)", "lg(re m-1)"])
    def test_log10_scaled_data_is_rejected(
        self, tmp_path, simple_dataarray, fake_stages, monkeypatch, units
    ):
        config = _config(tmp_path)
        fake_stages.resampled[NT_PRODUCT] = {
            "chl_nn": simple_dataarray.assign_attrs(units=units)
        }
        export_mock = MagicMock()
        monkeypatch.setattr(pipeline, "export_dataset", export_mock)

        with pytest.raises(ValueError, match=r"lg\(") as excinfo:
            process_single_pass(tmp_path / NT_PRODUCT, config)

        message = str(excinfo.value)
        assert "chl_nn" in message
        assert units in message
        assert "linear" in message
        # Nothing may reach disk when the scale is wrong.
        export_mock.assert_not_called()

    def test_any_log10_scaled_dataset_is_rejected(
        self, tmp_path, simple_dataarray, fake_stages, monkeypatch
    ):
        config = _config(tmp_path, datasets=["chl_nn", "tsm_nn"])
        fake_stages.resampled[NT_PRODUCT] = {
            "chl_nn": simple_dataarray.assign_attrs(units="mg.m-3"),
            "tsm_nn": simple_dataarray.assign_attrs(units="lg(re g.m-3)"),
        }
        export_mock = MagicMock()
        monkeypatch.setattr(pipeline, "export_dataset", export_mock)

        with pytest.raises(ValueError, match="tsm_nn"):
            process_single_pass(tmp_path / NT_PRODUCT, config)

        export_mock.assert_not_called()

    @pytest.mark.parametrize("attrs", [{"units": "mg.m-3"}, {"units": ""}, {}])
    def test_linear_data_is_exported(
        self, tmp_path, simple_dataarray, fake_stages, monkeypatch, attrs
    ):
        config = _config(tmp_path)
        fake_stages.resampled[NT_PRODUCT] = {
            "chl_nn": simple_dataarray.assign_attrs(**attrs)
        }
        export_mock = MagicMock(wraps=export_dataset)
        monkeypatch.setattr(pipeline, "export_dataset", export_mock)

        result = process_single_pass(tmp_path / NT_PRODUCT, config)

        assert isinstance(result, PassResult)
        export_mock.assert_called_once()
        assert _out_path(config, "netcdf").exists()


# ---------------------------------------------------------------------------
# Each resampled array is computed once and exported from memory
# ---------------------------------------------------------------------------

class _RecordingExport:
    """Reads the values like every exporter does; writes all but the PNG."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def __call__(self, data, path, fmt, provenance, dataset_name):
        self.calls.append(
            {
                "fmt": fmt,
                "lazy": dask.is_dask_collection(data),
                "values": np.array(data.values),
            }
        )
        if fmt == "png":
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
        else:
            export_dataset(data, path, fmt, provenance, dataset_name)


class TestComputedOnce:
    @pytest.fixture
    def run(self, tmp_path, simple_dataarray, fake_stages, lazy_source, monkeypatch):
        source = lazy_source(_linear(simple_dataarray))
        fake_stages.resampled[NT_PRODUCT] = {"chl_nn": source.dataarray()}
        export = _RecordingExport()
        monkeypatch.setattr(pipeline, "export_dataset", export)
        config = _config(tmp_path, ["geotiff", "netcdf", "png"])
        result = process_single_pass(tmp_path / NT_PRODUCT, config)
        return source, export, result

    def test_resample_graph_is_computed_once_for_three_formats(self, run):
        source, export, _ = run

        assert [call["fmt"] for call in export.calls] == ["geotiff", "netcdf", "png"]
        assert source.reads == 1

    def test_exporters_receive_in_memory_arrays(self, run):
        _, export, _ = run

        assert [call["lazy"] for call in export.calls] == [False, False, False]

    def test_every_format_receives_the_same_values(self, run, simple_dataarray):
        _, export, _ = run

        expected = _linear(simple_dataarray).values
        for call in export.calls:
            np.testing.assert_array_equal(call["values"], expected)

    def test_using_the_result_does_not_recompute(self, run, simple_dataarray):
        source, _, result = run

        values = result.datasets["chl_nn"].values

        np.testing.assert_array_equal(values, _linear(simple_dataarray).values)
        assert source.reads == 1


# ---------------------------------------------------------------------------
# PassResult.datasets never depends on the raw product
# ---------------------------------------------------------------------------

class TestPassDatasets:
    def _run(self, tmp_path, simple_dataarray, fake_stages, lazy_source, formats):
        source = lazy_source(_linear(simple_dataarray))
        fake_stages.resampled[NT_PRODUCT] = {"chl_nn": source.dataarray()}
        config = _config(tmp_path, formats)
        result = process_single_pass(tmp_path / NT_PRODUCT, config)
        return source, config, result

    @pytest.mark.parametrize(
        "formats", [["netcdf"], ["netcdf", "geotiff"], ["geotiff"]]
    )
    def test_usable_after_source_is_gone(
        self, tmp_path, simple_dataarray, fake_stages, lazy_source, formats
    ):
        source, _, result = self._run(
            tmp_path, simple_dataarray, fake_stages, lazy_source, formats
        )

        # The .SEN3 directory is deleted and no file handle is cached.
        source.available = False
        with xr.set_options(file_cache_maxsize=1):
            values = result.datasets["chl_nn"].values

        np.testing.assert_array_equal(values, _linear(simple_dataarray).values)

    def test_with_netcdf_dataset_is_lazy_view_of_the_written_file(
        self, tmp_path, simple_dataarray, fake_stages, lazy_source
    ):
        _, config, result = self._run(
            tmp_path, simple_dataarray, fake_stages, lazy_source, ["netcdf", "geotiff"]
        )
        data = result.datasets["chl_nn"]

        # Dask-backed, so holding 1500 passes costs no array memory.
        assert dask.is_dask_collection(data)
        assert data.dtype == np.float32
        np.testing.assert_array_equal(
            data.values, _stored_values(_out_path(config, "netcdf"))
        )

    def test_with_netcdf_dataset_keeps_grid_but_not_grid_mapping(
        self, tmp_path, simple_dataarray, fake_stages, lazy_source
    ):
        _, _, result = self._run(
            tmp_path, simple_dataarray, fake_stages, lazy_source, ["netcdf"]
        )
        data = result.datasets["chl_nn"]

        assert data.dims == ("y", "x")
        assert set(data.coords) == {"y", "x"}
        np.testing.assert_array_equal(data["x"].values, simple_dataarray["x"].values)
        np.testing.assert_array_equal(data["y"].values, simple_dataarray["y"].values)
        assert "grid_mapping" not in data.attrs
        assert "grid_mapping" not in data.encoding

    @pytest.mark.parametrize("fmt", ["netcdf", "geotiff"])
    def test_with_netcdf_dataset_can_be_exported_again(
        self, tmp_path, simple_dataarray, fake_stages, lazy_source, provenance, fmt
    ):
        _, _, result = self._run(
            tmp_path, simple_dataarray, fake_stages, lazy_source, ["netcdf"]
        )
        ext = {"netcdf": "nc", "geotiff": "tif"}[fmt]
        again = tmp_path / f"again.{ext}"

        export_dataset(result.datasets["chl_nn"], again, fmt, provenance, "chl_nn")

        np.testing.assert_array_equal(
            _stored_values(again), _linear(simple_dataarray).values
        )
        if fmt == "netcdf":
            with xr.open_dataset(again, decode_coords="all") as ds:
                assert ds["chl_nn"].encoding["grid_mapping"] == "crs"
                assert "crs" in ds.coords

    def test_without_netcdf_dataset_is_held_in_memory(
        self, tmp_path, simple_dataarray, fake_stages, lazy_source
    ):
        _, _, result = self._run(
            tmp_path, simple_dataarray, fake_stages, lazy_source, ["geotiff"]
        )
        data = result.datasets["chl_nn"]

        assert not dask.is_dask_collection(data)
        np.testing.assert_array_equal(data.values, _linear(simple_dataarray).values)


# ---------------------------------------------------------------------------
# Skipping passes whose outputs exist and are current
# ---------------------------------------------------------------------------

class TestSkipCheck:
    @pytest.mark.parametrize(
        "formats", [["netcdf"], ["geotiff"], ["netcdf", "geotiff"]]
    )
    def test_current_outputs_are_not_reprocessed(
        self, tmp_path, simple_dataarray, fake_stages, formats
    ):
        config = _config(tmp_path, formats)
        fake_stages.resampled[NT_PRODUCT] = {"chl_nn": _linear(simple_dataarray)}
        first = process_single_pass(tmp_path / NT_PRODUCT, config)
        mtimes = {path: path.stat().st_mtime_ns for path in first.output_files}

        second = process_single_pass(tmp_path / NT_PRODUCT, config)

        assert isinstance(second, PassResult)
        assert second.skipped is True
        assert fake_stages.loads == [NT_PRODUCT]
        assert second.output_files == []
        assert mtimes == {path: path.stat().st_mtime_ns for path in mtimes}
        assert second.satellite == "S3A"
        assert second.sensing_time == SENSING
        assert second.product_path == tmp_path / NT_PRODUCT

    def test_skipped_pass_carries_datasets_from_its_netcdf(
        self, tmp_path, simple_dataarray, fake_stages
    ):
        config = _config(tmp_path, ["netcdf", "geotiff"])
        fake_stages.resampled[NT_PRODUCT] = {"chl_nn": _linear(simple_dataarray)}
        first = process_single_pass(tmp_path / NT_PRODUCT, config)

        second = process_single_pass(tmp_path / NT_PRODUCT, config)

        data = second.datasets["chl_nn"]
        assert dask.is_dask_collection(data)
        assert "grid_mapping" not in data.attrs
        assert set(data.coords) == {"y", "x"}
        np.testing.assert_array_equal(data.values, first.datasets["chl_nn"].values)
        assert data.values.tobytes() == first.datasets["chl_nn"].values.tobytes()
        xr.testing.assert_identical(data, first.datasets["chl_nn"])

    def test_skipped_pass_without_netcdf_has_no_datasets(
        self, tmp_path, simple_dataarray, fake_stages
    ):
        config = _config(tmp_path, ["geotiff"])
        fake_stages.resampled[NT_PRODUCT] = {"chl_nn": _linear(simple_dataarray)}
        process_single_pass(tmp_path / NT_PRODUCT, config)

        second = process_single_pass(tmp_path / NT_PRODUCT, config)

        assert second.skipped is True
        assert second.datasets == {}

    def test_png_only_falls_back_to_existence(
        self, tmp_path, simple_dataarray, fake_stages
    ):
        config = _config(tmp_path, ["png"])
        fake_stages.resampled[NT_PRODUCT] = {"chl_nn": _linear(simple_dataarray)}
        png = _out_path(config, "png")
        png.parent.mkdir(parents=True)
        png.write_bytes(b"\x89PNG")

        result = process_single_pass(tmp_path / NT_PRODUCT, config)

        assert result.skipped is True
        assert result.datasets == {}
        assert fake_stages.loads == []

    def test_missing_format_triggers_reprocessing(
        self, tmp_path, simple_dataarray, fake_stages
    ):
        config = _config(tmp_path, ["netcdf", "geotiff"])
        fake_stages.resampled[NT_PRODUCT] = {"chl_nn": _linear(simple_dataarray)}
        process_single_pass(tmp_path / NT_PRODUCT, config)
        _out_path(config, "geotiff").unlink()

        second = process_single_pass(tmp_path / NT_PRODUCT, config)

        assert second.skipped is False
        assert fake_stages.loads == [NT_PRODUCT, NT_PRODUCT]
        assert _out_path(config, "geotiff").exists()

    @pytest.mark.parametrize("fmt", ["netcdf", "geotiff"])
    def test_output_without_value_scale_is_reprocessed(
        self, tmp_path, simple_dataarray, fake_stages, caplog, fmt
    ):
        # Written by a version that stored log10 values and no marker.
        config = _config(tmp_path, [fmt])
        path = _out_path(config, fmt)
        old_values = np.log10(_linear(simple_dataarray, offset=1.0))
        _PLAIN_WRITERS[fmt](path, old_values, s3bloom_source_product=NT_PRODUCT)
        fake_stages.resampled[NT_PRODUCT] = {"chl_nn": _linear(simple_dataarray)}

        with caplog.at_level(logging.INFO, logger="s3bloom.processing.pipeline"):
            result = process_single_pass(tmp_path / NT_PRODUCT, config)

        assert result.skipped is False
        assert fake_stages.loads == [NT_PRODUCT]
        assert read_output_attrs(path)["s3bloom_value_scale"] == "linear"
        np.testing.assert_array_equal(
            _stored_values(path), _linear(simple_dataarray).values
        )
        assert "s3bloom_value_scale" in caplog.text
        assert "Reprocessing" in caplog.text

    @pytest.mark.parametrize("fmt", ["netcdf", "geotiff"])
    def test_output_from_another_source_product_is_reprocessed(
        self, tmp_path, simple_dataarray, fake_stages, caplog, fmt
    ):
        # The near-real-time product was processed first; the
        # non-time-critical one that replaces it maps to the same file.
        config = _config(tmp_path, [fmt])
        path = _out_path(config, fmt)
        fake_stages.resampled[NR_PRODUCT] = {"chl_nn": _linear(simple_dataarray)}
        fake_stages.resampled[NT_PRODUCT] = {
            "chl_nn": _linear(simple_dataarray, offset=5.0)
        }
        process_single_pass(tmp_path / NR_PRODUCT, config)
        assert read_output_attrs(path)["s3bloom_source_product"] == NR_PRODUCT

        with caplog.at_level(logging.INFO, logger="s3bloom.processing.pipeline"):
            result = process_single_pass(tmp_path / NT_PRODUCT, config)

        assert result.skipped is False
        assert fake_stages.loads == [NR_PRODUCT, NT_PRODUCT]
        assert read_output_attrs(path)["s3bloom_source_product"] == NT_PRODUCT
        np.testing.assert_array_equal(
            _stored_values(path), _linear(simple_dataarray, offset=5.0).values
        )
        assert "s3bloom_source_product" in caplog.text
        assert NR_PRODUCT in caplog.text

    def test_netcdf_is_preferred_over_geotiff_for_the_check(
        self, tmp_path, simple_dataarray, fake_stages
    ):
        config = _config(tmp_path, ["geotiff", "netcdf"])
        fake_stages.resampled[NT_PRODUCT] = {"chl_nn": _linear(simple_dataarray)}
        process_single_pass(tmp_path / NT_PRODUCT, config)
        # A stale NetCDF next to a current GeoTIFF must not be kept.
        nc_path = _out_path(config, "netcdf")
        nc_path.unlink()
        _write_plain_netcdf(nc_path, _linear(simple_dataarray))

        second = process_single_pass(tmp_path / NT_PRODUCT, config)

        assert second.skipped is False
        assert read_output_attrs(nc_path)["s3bloom_value_scale"] == "linear"

    def test_unreadable_output_is_reprocessed(
        self, tmp_path, simple_dataarray, fake_stages
    ):
        config = _config(tmp_path, ["netcdf"])
        path = _out_path(config, "netcdf")
        path.parent.mkdir(parents=True)
        path.write_bytes(b"truncated by an interrupted run")
        fake_stages.resampled[NT_PRODUCT] = {"chl_nn": _linear(simple_dataarray)}

        result = process_single_pass(tmp_path / NT_PRODUCT, config)

        assert result.skipped is False
        np.testing.assert_array_equal(
            _stored_values(path), _linear(simple_dataarray).values
        )

    def test_one_stale_dataset_reprocesses_the_pass(
        self, tmp_path, simple_dataarray, fake_stages
    ):
        config = _config(tmp_path, ["netcdf"], datasets=["chl_nn", "tsm_nn"])
        fake_stages.resampled[NT_PRODUCT] = {
            "chl_nn": _linear(simple_dataarray),
            "tsm_nn": _linear(simple_dataarray, offset=2.0),
        }
        process_single_pass(tmp_path / NT_PRODUCT, config)
        tsm_path = _out_path(config, "netcdf", "tsm_nn")
        tsm_path.unlink()
        _write_plain_netcdf(tsm_path, _linear(simple_dataarray))

        second = process_single_pass(tmp_path / NT_PRODUCT, config)

        assert second.skipped is False
        assert read_output_attrs(tsm_path)["s3bloom_value_scale"] == "linear"

    def test_overwrite_forces_reprocessing_of_current_outputs(
        self, tmp_path, simple_dataarray, fake_stages
    ):
        fake_stages.resampled[NT_PRODUCT] = {"chl_nn": _linear(simple_dataarray)}
        process_single_pass(tmp_path / NT_PRODUCT, _config(tmp_path, ["netcdf"]))
        fake_stages.resampled[NT_PRODUCT] = {
            "chl_nn": _linear(simple_dataarray, offset=3.0)
        }
        config = _config(tmp_path, ["netcdf"], overwrite=True)

        second = process_single_pass(tmp_path / NT_PRODUCT, config)

        assert second.skipped is False
        assert fake_stages.loads == [NT_PRODUCT, NT_PRODUCT]
        np.testing.assert_array_equal(
            _stored_values(_out_path(config, "netcdf")),
            _linear(simple_dataarray, offset=3.0).values,
        )
