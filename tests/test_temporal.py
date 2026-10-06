"""Tests for compositing/temporal.py."""

from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path

import dask
import numpy as np
import pytest
import rasterio
import xarray as xr

from s3bloom.compositing import temporal
from s3bloom.compositing.temporal import (
    _compute_composite_dates,
    _group_by_dataset,
    _geometric_mean_composite,
    create_composites,
)
from s3bloom.config import BoundingBox, PipelineConfig, TimePeriod
from s3bloom.export.naming import composite_output_path
from s3bloom.metadata.provenance import read_output_attrs
from s3bloom.processing.pipeline import PassResult


def _make_array(values: list[list[float]], attrs: dict | None = None) -> xr.DataArray:
    """Build a small (y, x) DataArray for tests."""
    a = np.asarray(values, dtype=np.float64)
    return xr.DataArray(
        a,
        dims=("y", "x"),
        coords={"y": np.arange(a.shape[0], dtype=float),
                "x": np.arange(a.shape[1], dtype=float)},
        attrs=attrs or {},
    )


def _make_pass(
    *,
    sensing: datetime,
    satellite: str,
    dataset: str,
    values: list[list[float]],
    product_name: str | None = None,
    skipped: bool = False,
) -> PassResult:
    da = _make_array(values, attrs={"crs": "EPSG:3035"})
    name = product_name or f"{satellite}_OL_2_WFR____{sensing:%Y%m%dT%H%M%S}.SEN3"
    return PassResult(
        product_path=Path(name),
        satellite=satellite,
        sensing_time=sensing,
        datasets={dataset: da},
        provenance={},
        output_files=[],
        skipped=skipped,
    )


def _make_config(
    tmp_path: Path,
    *,
    formats: list[str],
    window: int = 3,
    overwrite: bool = False,
) -> PipelineConfig:
    return PipelineConfig(
        bbox=BoundingBox(lon_min=10.0, lat_min=56.0, lon_max=12.0, lat_max=58.0),
        time_period=TimePeriod(
            start_date=date(2024, 3, 15), end_date=date(2024, 3, 17)
        ),
        datasets=["chl_nn"],
        output={
            "base_dir": tmp_path,
            "formats": formats,
            "composite_window_days": window,
            "overwrite": overwrite,
        },
    )


# --------------------------------------------------------------------------- #
# _geometric_mean_composite                                                   #
# --------------------------------------------------------------------------- #


class TestGeometricMeanComposite:
    def test_single_array_returned_as_copy(self):
        arr = _make_array([[1.0, 2.0], [3.0, 4.0]], attrs={"crs": "EPSG:3035"})
        out = _geometric_mean_composite([arr])

        np.testing.assert_array_equal(out.values, arr.values)
        # Modifying the input must not affect the output → copy semantics.
        arr.values[0, 0] = 99.0
        assert out.values[0, 0] == 1.0

    def test_single_array_keeps_attributes(self):
        arr = _make_array([[1.0]], attrs={"crs": "EPSG:3035", "units": "mg.m-3"})

        out = _geometric_mean_composite([arr])

        assert out.attrs == {"crs": "EPSG:3035", "units": "mg.m-3"}

    def test_geometric_mean_of_two_arrays(self):
        a = _make_array([[1.0, 2.0], [3.0, 4.0]])
        b = _make_array([[3.0, 4.0], [5.0, 6.0]])

        out = _geometric_mean_composite([a, b])

        np.testing.assert_allclose(
            out.values, np.sqrt([[3.0, 8.0], [15.0, 24.0]]), rtol=1e-12
        )

    def test_is_not_the_arithmetic_mean(self):
        a = _make_array([[1.0]])
        b = _make_array([[100.0]])

        out = _geometric_mean_composite([a, b])

        # Arithmetic mean would be 50.5; a bloom pixel seen once must not
        # dominate a log-normally distributed quantity like that.
        np.testing.assert_allclose(out.values, [[10.0]], rtol=1e-12)

    def test_equals_ten_to_the_mean_of_log10(self):
        rng = np.random.default_rng(7)
        stack = rng.lognormal(mean=0.5, sigma=1.0, size=(4, 3, 5))
        arrays = [_make_array(layer.tolist()) for layer in stack]

        out = _geometric_mean_composite(arrays)

        expected = 10.0 ** np.mean(np.log10(stack), axis=0)
        np.testing.assert_allclose(out.values, expected, rtol=1e-12)

    def test_matches_composite_of_log10_encoded_passes(self):
        # Composites used to be the plain mean of log10-encoded passes.
        # The linear composite must equal 10 ** (that old value).
        log_a = np.array([[-1.0, 0.25], [0.5, 1.5]])
        log_b = np.array([[-0.5, 0.75], [1.0, np.nan]])
        old_composite = np.nanmean(np.stack([log_a, log_b]), axis=0)

        out = _geometric_mean_composite(
            [
                _make_array((10.0**log_a).tolist()),
                _make_array((10.0**log_b).tolist()),
            ]
        )

        np.testing.assert_allclose(out.values, 10.0**old_composite, rtol=1e-12)

    def test_nan_is_skipped_per_pixel(self):
        a = _make_array([[1.0, np.nan], [np.nan, 4.0]])
        b = _make_array([[4.0, 5.0], [7.0, np.nan]])
        c = _make_array([[np.nan, 20.0], [np.nan, 9.0]])

        out = _geometric_mean_composite([a, b, c])

        # (0,0): gmean(1, 4) = 2;  (0,1): gmean(5, 20) = 10
        # (1,0): only 7;           (1,1): gmean(4, 9) = 6
        np.testing.assert_allclose(
            out.values, [[2.0, 10.0], [7.0, 6.0]], rtol=1e-12
        )

    @pytest.mark.filterwarnings("error")
    def test_all_nan_pixel_stays_nan(self):
        a = _make_array([[np.nan, 1.0], [1.0, 1.0]])
        b = _make_array([[np.nan, 4.0], [4.0, 4.0]])

        out = _geometric_mean_composite([a, b])

        assert np.isnan(out.values[0, 0])
        np.testing.assert_allclose(out.values[0, 1], 2.0, rtol=1e-12)

    @pytest.mark.filterwarnings("error")
    @pytest.mark.parametrize("bad", [0.0, -3.0])
    def test_non_positive_value_is_ignored(self, bad):
        # log10 is undefined for values <= 0; such a value must drop out
        # like a NaN instead of poisoning the pixel (or warning).
        a = _make_array([[bad, 2.0]])
        b = _make_array([[4.0, 8.0]])
        c = _make_array([[9.0, 4.0]])

        out = _geometric_mean_composite([a, b, c])

        np.testing.assert_allclose(out.values, [[6.0, 4.0]], rtol=1e-12)

    @pytest.mark.filterwarnings("error")
    def test_pixel_with_only_non_positive_values_is_nan(self):
        a = _make_array([[0.0, 1.0]])
        b = _make_array([[-1.0, 1.0]])

        out = _geometric_mean_composite([a, b])

        assert np.isnan(out.values[0, 0])
        np.testing.assert_allclose(out.values[0, 1], 1.0, rtol=1e-12)

    def test_result_is_positive_wherever_defined(self):
        a = _make_array([[1e-3, 0.5], [np.nan, 80.0]])
        b = _make_array([[2e-3, np.nan], [np.nan, 99.0]])

        out = _geometric_mean_composite([a, b])

        finite = out.values[np.isfinite(out.values)]
        assert finite.size == 3
        assert (finite > 0).all()

    def test_inputs_are_not_mutated(self):
        a = _make_array([[0.0, 2.0]], attrs={"units": "mg.m-3"})
        b = _make_array([[4.0, np.nan]], attrs={"units": "mg.m-3"})

        out = _geometric_mean_composite([a, b])
        out.attrs["touched"] = True

        np.testing.assert_array_equal(a.values, [[0.0, 2.0]])
        np.testing.assert_array_equal(b.values, [[4.0, np.nan]])
        assert a.attrs == {"units": "mg.m-3"}

    def test_float32_input_stays_float32(self):
        a = _make_array([[1.0, 2.0]]).astype(np.float32)
        b = _make_array([[4.0, 8.0]]).astype(np.float32)

        out = _geometric_mean_composite([a, b])

        assert out.dtype == np.float32
        np.testing.assert_allclose(out.values, [[2.0, 4.0]], rtol=1e-6)

    def test_attributes_preserved_from_first_array(self):
        a = _make_array([[1.0, 2.0], [3.0, 4.0]], attrs={"crs": "EPSG:3035", "units": "mg m-3"})
        b = _make_array([[3.0, 4.0], [5.0, 6.0]], attrs={"crs": "EPSG:9999"})

        out = _geometric_mean_composite([a, b])

        assert out.attrs["crs"] == "EPSG:3035"
        assert out.attrs["units"] == "mg m-3"

    def test_pass_idx_dim_is_consumed(self):
        a = _make_array([[1.0]])
        b = _make_array([[3.0]])
        out = _geometric_mean_composite([a, b])
        assert "pass_idx" not in out.dims
        assert out.dims == ("y", "x")


# --------------------------------------------------------------------------- #
# _compute_composite_dates                                                    #
# --------------------------------------------------------------------------- #


class TestComputeCompositeDates:
    def test_empty_input_returns_empty(self):
        assert _compute_composite_dates([], window=3) == []

    def test_single_date(self):
        d = date(2024, 3, 15)
        assert _compute_composite_dates([d], window=3) == [d]

    def test_fills_gaps_between_first_and_last(self):
        # Only two observed dates with a gap in between — every calendar
        # day in the range still gets a composite date.
        observed = [date(2024, 3, 15), date(2024, 3, 18)]
        out = _compute_composite_dates(observed, window=3)
        assert out == [
            date(2024, 3, 15),
            date(2024, 3, 16),
            date(2024, 3, 17),
            date(2024, 3, 18),
        ]

    def test_returns_sorted_dates(self):
        observed = [date(2024, 3, 18), date(2024, 3, 15), date(2024, 3, 16)]
        out = _compute_composite_dates(observed, window=3)
        assert out == sorted(out)

    def test_duplicates_collapsed(self):
        observed = [date(2024, 3, 15), date(2024, 3, 15), date(2024, 3, 16)]
        out = _compute_composite_dates(observed, window=3)
        assert out == [date(2024, 3, 15), date(2024, 3, 16)]


# --------------------------------------------------------------------------- #
# _group_by_dataset                                                           #
# --------------------------------------------------------------------------- #


class TestGroupByDataset:
    def test_groups_by_dataset_then_date(self):
        p1 = _make_pass(
            sensing=datetime(2024, 3, 15, 9, 15, tzinfo=timezone.utc),
            satellite="S3A",
            dataset="chl_nn",
            values=[[1.0]],
        )
        p2 = _make_pass(
            sensing=datetime(2024, 3, 16, 9, 30, tzinfo=timezone.utc),
            satellite="S3B",
            dataset="chl_nn",
            values=[[2.0]],
        )

        grouped = _group_by_dataset([p1, p2])

        assert set(grouped.keys()) == {"chl_nn"}
        assert set(grouped["chl_nn"].keys()) == {date(2024, 3, 15), date(2024, 3, 16)}

    def test_same_day_passes_share_a_bucket(self):
        # S3A and S3B passes on the same day should land together so they
        # contribute equally to that day's composite.
        p1 = _make_pass(
            sensing=datetime(2024, 3, 15, 9, 0, tzinfo=timezone.utc),
            satellite="S3A",
            dataset="chl_nn",
            values=[[1.0]],
        )
        p2 = _make_pass(
            sensing=datetime(2024, 3, 15, 11, 0, tzinfo=timezone.utc),
            satellite="S3B",
            dataset="chl_nn",
            values=[[2.0]],
        )

        grouped = _group_by_dataset([p1, p2])

        bucket = grouped["chl_nn"][date(2024, 3, 15)]
        assert len(bucket) == 2
        sats = {entry[1] for entry in bucket}
        assert sats == {"S3A", "S3B"}

    def test_multi_dataset_pass_split_per_dataset(self):
        da_chl = _make_array([[1.0]])
        da_tsm = _make_array([[10.0]])
        pr = PassResult(
            product_path=Path("S3A.SEN3"),
            satellite="S3A",
            sensing_time=datetime(2024, 3, 15, tzinfo=timezone.utc),
            datasets={"chl_nn": da_chl, "tsm_nn": da_tsm},
            provenance={},
            output_files=[],
        )

        grouped = _group_by_dataset([pr])

        assert set(grouped.keys()) == {"chl_nn", "tsm_nn"}

    def test_empty_input(self):
        assert _group_by_dataset([]) == {}

    def test_entries_carry_data_satellite_product_name(self):
        p = _make_pass(
            sensing=datetime(2024, 3, 15, tzinfo=timezone.utc),
            satellite="S3A",
            dataset="chl_nn",
            values=[[42.0]],
            product_name="S3A_TEST.SEN3",
        )

        grouped = _group_by_dataset([p])
        entry = grouped["chl_nn"][date(2024, 3, 15)][0]

        data, satellite, product_name = entry
        assert isinstance(data, xr.DataArray)
        assert satellite == "S3A"
        assert product_name == "S3A_TEST.SEN3"


# --------------------------------------------------------------------------- #
# create_composites                                                           #
# --------------------------------------------------------------------------- #


class _ExportRecorder:
    """Captures export_dataset calls so create_composites can be tested
    without exercising the real GeoTIFF/NetCDF/PNG writers."""

    def __init__(self):
        self.calls: list[dict] = []

    def __call__(self, data, path, fmt, provenance, dataset_name):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        self.calls.append(
            {
                "path": path,
                "fmt": fmt,
                "dataset": dataset_name,
                "shape": data.shape,
                "lazy": dask.is_dask_collection(data),
                "values": np.asarray(data.values).copy(),
                "provenance": provenance,
            }
        )


@pytest.fixture
def recorder(monkeypatch):
    rec = _ExportRecorder()
    monkeypatch.setattr(temporal, "export_dataset", rec)
    return rec


class TestCreateComposites:
    def test_empty_pass_results_returns_empty(self, tmp_path, recorder):
        cfg = _make_config(tmp_path, formats=["geotiff"])
        out = create_composites([], cfg)
        assert out == []
        assert recorder.calls == []

    def test_writes_one_file_per_format_per_date(self, tmp_path, recorder):
        # Three consecutive days, one pass each. With a 3-day window
        # every day gets a composite, in two formats.
        passes = [
            _make_pass(
                sensing=datetime(2024, 3, 15, 9, tzinfo=timezone.utc),
                satellite="S3A",
                dataset="chl_nn",
                values=[[1.0]],
            ),
            _make_pass(
                sensing=datetime(2024, 3, 16, 9, tzinfo=timezone.utc),
                satellite="S3A",
                dataset="chl_nn",
                values=[[2.0]],
            ),
            _make_pass(
                sensing=datetime(2024, 3, 17, 9, tzinfo=timezone.utc),
                satellite="S3A",
                dataset="chl_nn",
                values=[[3.0]],
            ),
        ]
        cfg = _make_config(tmp_path, formats=["geotiff", "netcdf"], window=3)

        files = create_composites(passes, cfg)

        # 3 dates × 2 formats = 6 outputs
        assert len(files) == 6
        assert len(recorder.calls) == 6
        formats = {c["fmt"] for c in recorder.calls}
        assert formats == {"geotiff", "netcdf"}

    def test_window_geometric_mean_is_computed(self, tmp_path, recorder):
        # With window=3, the composite for 2024-03-16 combines the
        # passes from 15, 16, 17 → geometric mean(1, 2, 3) = 6^(1/3).
        passes = [
            _make_pass(
                sensing=datetime(2024, 3, 15, tzinfo=timezone.utc),
                satellite="S3A", dataset="chl_nn", values=[[1.0]],
            ),
            _make_pass(
                sensing=datetime(2024, 3, 16, tzinfo=timezone.utc),
                satellite="S3A", dataset="chl_nn", values=[[2.0]],
            ),
            _make_pass(
                sensing=datetime(2024, 3, 17, tzinfo=timezone.utc),
                satellite="S3A", dataset="chl_nn", values=[[3.0]],
            ),
        ]
        cfg = _make_config(tmp_path, formats=["geotiff"], window=3)

        create_composites(passes, cfg)

        by_date = {c["path"].name: c["values"] for c in recorder.calls}
        # The middle date sees all three passes.
        middle = next(v for n, v in by_date.items() if "20240316" in n)
        np.testing.assert_allclose(middle, [[6.0 ** (1.0 / 3.0)]], rtol=1e-12)

    def test_edge_dates_only_see_their_neighbours(self, tmp_path, recorder):
        passes = [
            _make_pass(
                sensing=datetime(2024, 3, day, tzinfo=timezone.utc),
                satellite="S3A", dataset="chl_nn", values=[[value]],
            )
            for day, value in ((15, 1.0), (16, 4.0), (17, 16.0))
        ]
        cfg = _make_config(tmp_path, formats=["geotiff"], window=3)

        create_composites(passes, cfg)

        by_date = {c["path"].name: c["values"] for c in recorder.calls}
        first = next(v for n, v in by_date.items() if "20240315" in n)
        last = next(v for n, v in by_date.items() if "20240317" in n)
        np.testing.assert_allclose(first, [[2.0]], rtol=1e-12)
        np.testing.assert_allclose(last, [[8.0]], rtol=1e-12)

    def test_composite_provenance_names_the_method(self, tmp_path, recorder):
        passes = [
            _make_pass(
                sensing=datetime(2024, 3, 15, 9, tzinfo=timezone.utc),
                satellite="S3A", dataset="chl_nn", values=[[1.0]],
            ),
            _make_pass(
                sensing=datetime(2024, 3, 15, 11, tzinfo=timezone.utc),
                satellite="S3B", dataset="chl_nn", values=[[4.0]],
            ),
        ]
        cfg = _make_config(tmp_path, formats=["geotiff"], window=3)

        create_composites(passes, cfg)

        prov = recorder.calls[0]["provenance"]
        assert prov.composite_method == "geometric_mean"
        assert prov.value_scale == "linear"
        np.testing.assert_allclose(recorder.calls[0]["values"], [[2.0]], rtol=1e-12)

    def test_combines_multiple_satellites_in_provenance(self, tmp_path, recorder):
        # Two satellites on the same day → satellite tag "S3A-S3B".
        passes = [
            _make_pass(
                sensing=datetime(2024, 3, 15, 9, tzinfo=timezone.utc),
                satellite="S3A", dataset="chl_nn", values=[[1.0]],
            ),
            _make_pass(
                sensing=datetime(2024, 3, 15, 11, tzinfo=timezone.utc),
                satellite="S3B", dataset="chl_nn", values=[[3.0]],
            ),
        ]
        cfg = _make_config(tmp_path, formats=["geotiff"], window=3)

        create_composites(passes, cfg)

        prov = recorder.calls[0]["provenance"]
        assert prov.satellite == "S3A-S3B"
        assert prov.pass_count == 2
        assert prov.composite_window_days == 3

    def test_composite_filename_encodes_window_and_date(self, tmp_path, recorder):
        passes = [
            _make_pass(
                sensing=datetime(2024, 3, 15, 9, tzinfo=timezone.utc),
                satellite="S3A", dataset="chl_nn", values=[[1.0]],
            ),
        ]
        cfg = _make_config(tmp_path, formats=["geotiff"], window=3)

        files = create_composites(passes, cfg)

        assert len(files) == 1
        name = files[0].name
        assert "composite3d" in name
        assert "20240315" in name
        assert name.startswith("s3bloom_chl_nn_")
        assert name.endswith(".tif")

    @pytest.mark.parametrize("satellites", [["S3A"], ["S3B"], ["S3A", "S3B"]])
    def test_composite_filename_has_no_satellite_segment(
        self, tmp_path, recorder, satellites
    ):
        # One file per date, whichever spacecraft contributed.
        passes = [
            _make_pass(
                sensing=datetime(2024, 3, 15, 9 + i, tzinfo=timezone.utc),
                satellite=satellite, dataset="chl_nn", values=[[1.0]],
            )
            for i, satellite in enumerate(satellites)
        ]
        cfg = _make_config(tmp_path, formats=["geotiff"], window=3)

        files = create_composites(passes, cfg)

        assert [f.name for f in files] == ["s3bloom_chl_nn_composite3d_20240315.tif"]

    def test_passes_without_datasets_are_ignored(self, tmp_path, recorder):
        # A pass skipped on a run without NetCDF output carries no data.
        empty = PassResult(
            product_path=Path("S3B_OL_2_WFR____20240318T090000.SEN3"),
            satellite="S3B",
            sensing_time=datetime(2024, 3, 18, 9, tzinfo=timezone.utc),
            datasets={},
            provenance={},
            output_files=[],
            skipped=True,
        )
        real = _make_pass(
            sensing=datetime(2024, 3, 15, 9, tzinfo=timezone.utc),
            satellite="S3A", dataset="chl_nn", values=[[1.0]],
        )
        cfg = _make_config(tmp_path, formats=["geotiff"], window=3)

        assert create_composites([empty], cfg) == []
        files = create_composites([empty, real], cfg)

        assert [f.name for f in files] == ["s3bloom_chl_nn_composite3d_20240315.tif"]
        assert recorder.calls[0]["provenance"].pass_count == 1

    def test_skipped_passes_with_datasets_are_composited(self, tmp_path, recorder):
        fresh = _make_pass(
            sensing=datetime(2024, 3, 15, 9, tzinfo=timezone.utc),
            satellite="S3A", dataset="chl_nn", values=[[1.0]],
        )
        resumed = _make_pass(
            sensing=datetime(2024, 3, 15, 11, tzinfo=timezone.utc),
            satellite="S3B", dataset="chl_nn", values=[[4.0]], skipped=True,
        )
        cfg = _make_config(tmp_path, formats=["geotiff"], window=3)

        create_composites([fresh, resumed], cfg)

        assert recorder.calls[0]["provenance"].pass_count == 2
        np.testing.assert_allclose(recorder.calls[0]["values"], [[2.0]], rtol=1e-12)

    def test_source_products_are_recorded_in_name_order(self, tmp_path, recorder):
        names = ["S3B_b.SEN3", "S3A_c.SEN3", "S3A_a.SEN3"]
        passes = [
            _make_pass(
                sensing=datetime(2024, 3, 15, 9 + i, tzinfo=timezone.utc),
                satellite=name[:3], dataset="chl_nn", values=[[1.0]],
                product_name=name,
            )
            for i, name in enumerate(names)
        ]
        cfg = _make_config(tmp_path, formats=["geotiff"], window=3)

        create_composites(passes, cfg)

        prov = recorder.calls[0]["provenance"]
        assert prov.source_products == ("S3A_a.SEN3", "S3A_c.SEN3", "S3B_b.SEN3")
        assert prov.satellite == "S3A-S3B"

    def test_composite_is_computed_once_for_all_formats(
        self, tmp_path, recorder, lazy_source
    ):
        sources = [
            lazy_source(_make_array([[1.0, 2.0], [3.0, 4.0]])),
            lazy_source(_make_array([[4.0, 8.0], [3.0, 1.0]])),
        ]
        passes = [
            PassResult(
                product_path=Path(f"S3A_{i}.SEN3"),
                satellite="S3A",
                sensing_time=datetime(2024, 3, 15, 9 + i, tzinfo=timezone.utc),
                datasets={"chl_nn": source.dataarray()},
                provenance={},
                output_files=[],
            )
            for i, source in enumerate(sources)
        ]
        cfg = _make_config(tmp_path, formats=["geotiff", "netcdf", "png"], window=3)

        create_composites(passes, cfg)

        assert len(recorder.calls) == 3
        assert [source.reads for source in sources] == [1, 1]
        assert [call["lazy"] for call in recorder.calls] == [False, False, False]
        for call in recorder.calls:
            np.testing.assert_allclose(
                call["values"], [[2.0, 4.0], [3.0, 2.0]], rtol=1e-12
            )

    def test_dates_with_no_pass_in_their_window_get_no_composite(
        self, tmp_path, recorder
    ):
        # Passes five days apart with a 3-day window: the 17th and 18th
        # are covered by neither.
        passes = [
            _make_pass(
                sensing=datetime(2024, 3, day, 9, tzinfo=timezone.utc),
                satellite="S3A", dataset="chl_nn", values=[[1.0]],
            )
            for day in (15, 20)
        ]
        cfg = _make_config(tmp_path, formats=["geotiff"], window=3)

        files = create_composites(passes, cfg)

        assert [f.name[-12:-4] for f in files] == [
            "20240315", "20240316", "20240319", "20240320",
        ]

    def test_per_dataset_composites(self, tmp_path, recorder):
        da_chl = _make_array([[1.0]])
        da_tsm = _make_array([[10.0]])
        pr = PassResult(
            product_path=Path("S3A.SEN3"),
            satellite="S3A",
            sensing_time=datetime(2024, 3, 15, tzinfo=timezone.utc),
            datasets={"chl_nn": da_chl, "tsm_nn": da_tsm},
            provenance={},
            output_files=[],
        )
        cfg = _make_config(tmp_path, formats=["geotiff"], window=3)

        create_composites([pr], cfg)

        datasets = {c["dataset"] for c in recorder.calls}
        assert datasets == {"chl_nn", "tsm_nn"}


# --------------------------------------------------------------------------- #
# create_composites: skipping composites that are up to date                  #
# --------------------------------------------------------------------------- #


def _day_pass(
    day: int, hour: int, satellite: str, values: list[list[float]], tag: str = "NT"
) -> PassResult:
    sensing = datetime(2024, 3, day, hour, tzinfo=timezone.utc)
    return _make_pass(
        sensing=sensing,
        satellite=satellite,
        dataset="chl_nn",
        values=values,
        product_name=f"{satellite}_OL_2_WFR____{sensing:%Y%m%dT%H%M%S}_{tag}_003.SEN3",
    )


def _composite_path(cfg: PipelineConfig, fmt: str, day: int) -> Path:
    return composite_output_path(
        base_dir=cfg.output.base_dir,
        fmt=fmt,
        dataset="chl_nn",
        center_date=datetime(2024, 3, day, tzinfo=timezone.utc),
        window_days=cfg.output.composite_window_days,
    )


def _composite_values(path: Path) -> np.ndarray:
    if path.suffix == ".nc":
        with xr.open_dataset(path) as ds:
            return ds["chl_nn"].values
    with rasterio.open(path) as src:
        return src.read(1)


A = [[1.0, 2.0], [3.0, np.nan]]
B = [[4.0, 8.0], [np.nan, np.nan]]
C = [[16.0, 0.5], [3.0, 7.0]]


class TestCompositeSkipCheck:
    """Real exporters: the check reads what is recorded in the files."""

    @pytest.mark.parametrize("fmt", ["netcdf", "geotiff"])
    def test_same_sources_are_skipped(self, tmp_path, caplog, fmt):
        cfg = _make_config(tmp_path, formats=[fmt])
        passes = [_day_pass(15, 9, "S3A", A), _day_pass(15, 11, "S3B", B)]
        first = create_composites(passes, cfg)
        mtimes = {path: path.stat().st_mtime_ns for path in first}

        with caplog.at_level("INFO", logger="s3bloom.compositing.temporal"):
            second = create_composites(passes, cfg)

        assert first == [_composite_path(cfg, fmt, 15)]
        assert second == []
        assert mtimes == {path: path.stat().st_mtime_ns for path in mtimes}
        assert "up to date" in caplog.text

    @pytest.mark.parametrize("fmt", ["netcdf", "geotiff"])
    def test_added_source_rebuilds_the_composite(self, tmp_path, fmt):
        cfg = _make_config(tmp_path, formats=[fmt])
        only_a = [_day_pass(15, 9, "S3A", A)]
        both = [*only_a, _day_pass(15, 11, "S3B", B)]
        create_composites(only_a, cfg)

        second = create_composites(both, cfg)

        path = _composite_path(cfg, fmt, 15)
        assert second == [path]
        attrs = read_output_attrs(path)
        assert attrs["s3bloom_pass_count"] == "2"
        assert attrs["s3bloom_satellite"] == "S3A-S3B"
        assert attrs["s3bloom_source_products"].split(",") == [
            p.product_path.name for p in both
        ]
        np.testing.assert_allclose(
            _composite_values(path), [[2.0, 4.0], [3.0, np.nan]], rtol=1e-6
        )
        # Still one file for the date, although the satellite set changed.
        assert [f.name for f in path.parent.iterdir()] == [path.name]

    @pytest.mark.parametrize("fmt", ["netcdf", "geotiff"])
    def test_removed_source_rebuilds_the_composite(self, tmp_path, fmt):
        cfg = _make_config(tmp_path, formats=[fmt])
        both = [_day_pass(15, 9, "S3A", A), _day_pass(15, 11, "S3B", B)]
        create_composites(both, cfg)

        second = create_composites(both[:1], cfg)

        path = _composite_path(cfg, fmt, 15)
        assert second == [path]
        assert read_output_attrs(path)["s3bloom_pass_count"] == "1"

    def test_composite_is_kept_when_a_window_pass_has_no_data(
        self, tmp_path, caplog
    ):
        # Without per-pass NetCDFs a skipped pass cannot be read back. The
        # full composite from the earlier run must not be replaced by one
        # built from the remaining passes only.
        cfg = _make_config(tmp_path, formats=["geotiff"])
        pass_a, pass_b = _day_pass(15, 9, "S3A", A), _day_pass(15, 11, "S3B", B)
        create_composites([pass_a, pass_b], cfg)
        path = _composite_path(cfg, "geotiff", 15)
        before = _composite_values(path)
        dataless_a = PassResult(
            product_path=pass_a.product_path,
            satellite=pass_a.satellite,
            sensing_time=pass_a.sensing_time,
            datasets={},
            provenance={},
            output_files=[],
            skipped=True,
        )

        with caplog.at_level("WARNING", logger="s3bloom.compositing.temporal"):
            second = create_composites([dataless_a, pass_b], cfg)

        assert second == []
        np.testing.assert_array_equal(_composite_values(path), before)
        assert read_output_attrs(path)["s3bloom_pass_count"] == "2"
        assert "without data" in caplog.text

    def test_missing_composite_is_built_from_the_passes_with_data(self, tmp_path):
        # Same situation, but nothing on disk to protect: build what we can.
        cfg = _make_config(tmp_path, formats=["geotiff"])
        pass_a, pass_b = _day_pass(15, 9, "S3A", A), _day_pass(15, 11, "S3B", B)
        dataless_a = PassResult(
            product_path=pass_a.product_path,
            satellite=pass_a.satellite,
            sensing_time=pass_a.sensing_time,
            datasets={},
            provenance={},
            output_files=[],
            skipped=True,
        )

        written = create_composites([dataless_a, pass_b], cfg)

        path = _composite_path(cfg, "geotiff", 15)
        assert written == [path]
        assert read_output_attrs(path)["s3bloom_pass_count"] == "1"

    def test_replaced_source_product_rebuilds_the_composite(self, tmp_path):
        # Same overpass, but the NR product was replaced by the NT one.
        cfg = _make_config(tmp_path, formats=["netcdf"])
        create_composites([_day_pass(15, 9, "S3A", A, tag="NR")], cfg)

        second = create_composites([_day_pass(15, 9, "S3A", C, tag="NT")], cfg)

        path = _composite_path(cfg, "netcdf", 15)
        assert second == [path]
        assert "_NT_" in read_output_attrs(path)["s3bloom_source_products"]
        np.testing.assert_allclose(_composite_values(path), C, rtol=1e-6)

    def test_only_affected_dates_are_rebuilt(self, tmp_path):
        cfg = _make_config(tmp_path, formats=["netcdf"], window=3)
        first_run = [_day_pass(15, 9, "S3A", A), _day_pass(16, 9, "S3A", B)]
        create_composites(first_run, cfg)

        # A pass on the 18th: the windows of the 15th and 16th are unchanged.
        second = create_composites([*first_run, _day_pass(18, 9, "S3A", C)], cfg)

        assert [f.name for f in second] == [
            "s3bloom_chl_nn_composite3d_20240317.nc",
            "s3bloom_chl_nn_composite3d_20240318.nc",
        ]

    def test_overwrite_forces_rebuild(self, tmp_path):
        passes = [_day_pass(15, 9, "S3A", A)]
        create_composites(passes, _make_config(tmp_path, formats=["netcdf"]))
        cfg = _make_config(tmp_path, formats=["netcdf"], overwrite=True)

        second = create_composites(passes, cfg)

        assert second == [_composite_path(cfg, "netcdf", 15)]

    def test_missing_format_rebuilds_the_composite(self, tmp_path):
        cfg = _make_config(tmp_path, formats=["netcdf", "geotiff"])
        passes = [_day_pass(15, 9, "S3A", A)]
        create_composites(passes, cfg)
        _composite_path(cfg, "geotiff", 15).unlink()

        second = create_composites(passes, cfg)

        assert len(second) == 2
        assert _composite_path(cfg, "geotiff", 15).exists()

    def test_stale_netcdf_next_to_current_geotiff_is_rebuilt(self, tmp_path):
        cfg = _make_config(tmp_path, formats=["geotiff", "netcdf"])
        create_composites([_day_pass(15, 9, "S3A", A)], cfg)
        # Only the GeoTIFF is brought up to date with the second source.
        both = [_day_pass(15, 9, "S3A", A), _day_pass(15, 11, "S3B", B)]
        create_composites(both, _make_config(tmp_path, formats=["geotiff"]))

        files = create_composites(both, cfg)

        assert len(files) == 2
        nc_attrs = read_output_attrs(_composite_path(cfg, "netcdf", 15))
        assert nc_attrs["s3bloom_pass_count"] == "2"

    def test_unreadable_composite_is_rebuilt(self, tmp_path):
        cfg = _make_config(tmp_path, formats=["netcdf"])
        path = _composite_path(cfg, "netcdf", 15)
        path.parent.mkdir(parents=True)
        path.write_bytes(b"truncated by an interrupted run")

        files = create_composites([_day_pass(15, 9, "S3A", A)], cfg)

        assert files == [path]
        np.testing.assert_allclose(_composite_values(path), A, rtol=1e-6)

    def test_png_only_falls_back_to_existence(self, tmp_path, recorder):
        cfg = _make_config(tmp_path, formats=["png"])
        first = create_composites([_day_pass(15, 9, "S3A", A)], cfg)

        second = create_composites(
            [_day_pass(15, 9, "S3A", A), _day_pass(15, 11, "S3B", B)], cfg
        )

        assert len(first) == 1
        assert second == []

    @pytest.mark.parametrize("fmt", ["netcdf", "geotiff"])
    def test_pass_order_does_not_change_the_composite(self, tmp_path, fmt):
        # Resumed runs hand the passes over in a different order
        # (already-downloaded products come first).
        rng = np.random.default_rng(3)
        stack = rng.lognormal(mean=0.5, sigma=1.5, size=(5, 6, 7)).astype(np.float32)
        passes = [
            _day_pass(15, 8 + i, "S3A" if i % 2 else "S3B", layer.tolist())
            for i, layer in enumerate(stack)
        ]
        forward = _make_config(tmp_path / "forward", formats=[fmt])
        backward = _make_config(tmp_path / "backward", formats=[fmt])

        create_composites(passes, forward)
        create_composites(passes[::-1], backward)

        a = _composite_values(_composite_path(forward, fmt, 15))
        b = _composite_values(_composite_path(backward, fmt, 15))
        assert a.tobytes() == b.tobytes()
        assert (
            read_output_attrs(_composite_path(forward, fmt, 15))["s3bloom_source_products"]
            == read_output_attrs(_composite_path(backward, fmt, 15))["s3bloom_source_products"]
        )
        # ... and the reordered window counts as the same composite.
        assert create_composites(passes[::-1], forward) == []
