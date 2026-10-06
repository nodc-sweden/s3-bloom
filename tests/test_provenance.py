"""Tests for provenance module."""

import pytest
from datetime import datetime, timezone

from s3bloom.metadata.provenance import (
    Provenance,
    create_composite_provenance,
    create_pass_provenance,
)


class TestCreatePassProvenance:
    def test_basic(self):
        prov = create_pass_provenance(
            source_product="S3A_OL_2_WFR____20240315T091500.SEN3",
            satellite="S3A",
            sensing_time=datetime(2024, 3, 15, 9, 15, 0, tzinfo=timezone.utc),
            dataset="chl_nn",
            masking_preset="strict",
            masking_flags=["CLOUD", "INVALID"],
            projection="EPSG:3035",
            resolution_m=300,
        )
        assert prov.satellite == "S3A"
        assert prov.dataset == "chl_nn"
        assert prov.masking_flags == ("CLOUD", "INVALID")
        assert prov.composite_window_days is None

    def test_to_dict(self):
        prov = create_pass_provenance(
            source_product="test.SEN3",
            satellite="S3B",
            sensing_time=datetime(2024, 3, 15, tzinfo=timezone.utc),
            dataset="chl_nn",
            masking_preset="moderate",
            masking_flags=["CLOUD"],
            projection="EPSG:3035",
            resolution_m=300,
        )
        d = prov.to_dict()
        assert d["satellite"] == "S3B"
        assert d["masking_flags"] == "CLOUD"


class TestCreateCompositeProvenance:
    def test_basic(self):
        prov = create_composite_provenance(
            source_products=["prod1.SEN3", "prod2.SEN3"],
            satellites=["S3A", "S3B"],
            center_date=datetime(2024, 3, 15, tzinfo=timezone.utc),
            dataset="chl_nn",
            masking_preset="strict",
            masking_flags=["CLOUD"],
            projection="EPSG:3035",
            resolution_m=300,
            composite_window_days=3,
        )
        assert prov.satellite == "S3A-S3B"
        assert prov.pass_count == 2
        assert prov.composite_window_days == 3
        assert prov.source_products == ("prod1.SEN3", "prod2.SEN3")

    def test_deduplicates_satellites(self):
        prov = create_composite_provenance(
            source_products=["p1.SEN3", "p2.SEN3", "p3.SEN3"],
            satellites=["S3A", "S3A", "S3B"],
            center_date=datetime(2024, 3, 15, tzinfo=timezone.utc),
            dataset="chl_nn",
            masking_preset="strict",
            masking_flags=["CLOUD"],
            projection="EPSG:3035",
            resolution_m=300,
            composite_window_days=7,
        )
        assert prov.satellite == "S3A-S3B"
        assert prov.pass_count == 3


class TestToNetcdfAttrs:
    def test_prefixes_all_keys(self):
        prov = create_pass_provenance(
            source_product="test.SEN3",
            satellite="S3A",
            sensing_time=datetime(2024, 3, 15, tzinfo=timezone.utc),
            dataset="chl_nn",
            masking_preset="strict",
            masking_flags=["CLOUD"],
            projection="EPSG:3035",
            resolution_m=300,
        )
        attrs = prov.to_netcdf_attrs()
        for key in attrs:
            assert key.startswith("s3bloom_")

    def test_skips_empty_fields(self):
        prov = create_pass_provenance(
            source_product="test.SEN3",
            satellite="S3A",
            sensing_time=datetime(2024, 3, 15, tzinfo=timezone.utc),
            dataset="chl_nn",
            masking_preset="strict",
            masking_flags=["CLOUD"],
            projection="EPSG:3035",
            resolution_m=300,
        )
        attrs = prov.to_netcdf_attrs()
        # composite_window_days and pass_count are None for a pass → should be absent
        assert "s3bloom_composite_window_days" not in attrs
        assert "s3bloom_pass_count" not in attrs

    def test_all_values_are_strings(self):
        prov = create_pass_provenance(
            source_product="test.SEN3",
            satellite="S3B",
            sensing_time=datetime(2024, 3, 15, tzinfo=timezone.utc),
            dataset="tsm_nn",
            masking_preset="moderate",
            masking_flags=["CLOUD", "INVALID"],
            projection="EPSG:3035",
            resolution_m=300,
        )
        attrs = prov.to_netcdf_attrs()
        for v in attrs.values():
            assert isinstance(v, str)

    def test_composite_includes_window_and_count(self):
        prov = create_composite_provenance(
            source_products=["p1.SEN3", "p2.SEN3"],
            satellites=["S3A", "S3B"],
            center_date=datetime(2024, 3, 15, tzinfo=timezone.utc),
            dataset="chl_nn",
            masking_preset="strict",
            masking_flags=["CLOUD"],
            projection="EPSG:3035",
            resolution_m=300,
            composite_window_days=3,
        )
        attrs = prov.to_netcdf_attrs()
        assert "s3bloom_composite_window_days" in attrs
        assert attrs["s3bloom_composite_window_days"] == "3"
        assert "s3bloom_pass_count" in attrs


def _pass_provenance() -> Provenance:
    return create_pass_provenance(
        source_product="test.SEN3",
        satellite="S3A",
        sensing_time=datetime(2024, 3, 15, tzinfo=timezone.utc),
        dataset="chl_nn",
        masking_preset="strict",
        masking_flags=["CLOUD"],
        projection="EPSG:3035",
        resolution_m=300,
    )


def _composite_provenance() -> Provenance:
    return create_composite_provenance(
        source_products=["p1.SEN3", "p2.SEN3"],
        satellites=["S3A", "S3B"],
        center_date=datetime(2024, 3, 15, tzinfo=timezone.utc),
        dataset="chl_nn",
        masking_preset="strict",
        masking_flags=["CLOUD"],
        projection="EPSG:3035",
        resolution_m=300,
        composite_window_days=3,
    )


class TestValueScale:
    def test_defaults_on_direct_construction(self):
        prov = Provenance(
            source_product="test.SEN3",
            satellite="S3A",
            sensing_time=datetime(2024, 3, 15, tzinfo=timezone.utc),
            dataset="chl_nn",
            masking_preset="strict",
            masking_flags=("CLOUD",),
            projection="EPSG:3035",
            resolution_m=300,
            pipeline_version="0.2.0",
        )
        assert prov.value_scale == "linear"
        assert prov.composite_method == ""

    def test_pass_is_linear(self):
        assert _pass_provenance().value_scale == "linear"

    def test_composite_is_linear(self):
        assert _composite_provenance().value_scale == "linear"

    def test_in_to_dict(self):
        assert _pass_provenance().to_dict()["value_scale"] == "linear"
        assert _composite_provenance().to_dict()["value_scale"] == "linear"

    def test_netcdf_attribute_has_exact_name_and_value(self):
        # External R scripts key on this exact attribute name and value.
        assert _pass_provenance().to_netcdf_attrs()["s3bloom_value_scale"] == "linear"
        assert (
            _composite_provenance().to_netcdf_attrs()["s3bloom_value_scale"]
            == "linear"
        )


class TestCompositeMethod:
    def test_empty_for_pass(self):
        assert _pass_provenance().composite_method == ""

    def test_geometric_mean_for_composite(self):
        assert _composite_provenance().composite_method == "geometric_mean"

    def test_in_to_dict(self):
        assert _pass_provenance().to_dict()["composite_method"] == ""
        assert (
            _composite_provenance().to_dict()["composite_method"] == "geometric_mean"
        )

    def test_absent_from_pass_netcdf_attrs(self):
        assert "s3bloom_composite_method" not in _pass_provenance().to_netcdf_attrs()

    def test_present_in_composite_netcdf_attrs(self):
        attrs = _composite_provenance().to_netcdf_attrs()
        assert attrs["s3bloom_composite_method"] == "geometric_mean"


class TestScaleMarkersInGeotiffTags:
    """The GeoTIFF exporter writes every truthy ``to_dict`` entry as a tag."""

    @staticmethod
    def _write(tmp_path, prov: Provenance) -> dict[str, str]:
        import numpy as np
        import rasterio
        import xarray as xr

        from s3bloom.export.geotiff import export_geotiff

        data = xr.DataArray(
            np.linspace(0.1, 20.0, 12, dtype=np.float32).reshape(3, 4),
            dims=("y", "x"),
            coords={
                "y": 3_500_000.0 - 300.0 * np.arange(3),
                "x": 4_300_000.0 + 300.0 * np.arange(4),
            },
        )
        out = export_geotiff(data, tmp_path / "out.tif", prov)
        with rasterio.open(out) as src:
            return dict(src.tags())

    def test_pass_geotiff_declares_linear_scale(self, tmp_path):
        tags = self._write(tmp_path, _pass_provenance())
        assert tags["s3bloom_value_scale"] == "linear"
        assert "s3bloom_composite_method" not in tags

    def test_composite_geotiff_declares_method(self, tmp_path):
        tags = self._write(tmp_path, _composite_provenance())
        assert tags["s3bloom_value_scale"] == "linear"
        assert tags["s3bloom_composite_method"] == "geometric_mean"


class TestProvenanceFrozen:
    def test_immutable(self):
        prov = create_pass_provenance(
            source_product="test.SEN3",
            satellite="S3A",
            sensing_time=datetime(2024, 3, 15, tzinfo=timezone.utc),
            dataset="chl_nn",
            masking_preset="strict",
            masking_flags=["CLOUD"],
            projection="EPSG:3035",
            resolution_m=300,
        )
        with pytest.raises(Exception):
            prov.satellite = "S3B"  # type: ignore[misc]


class TestReadOutputAttrs:
    """Reading the ``s3bloom_*`` record back from a written output."""

    @staticmethod
    def _data():
        import numpy as np
        import xarray as xr

        return xr.DataArray(
            np.linspace(0.1, 20.0, 12, dtype=np.float32).reshape(3, 4),
            dims=("y", "x"),
            coords={
                "y": 3_500_000.0 - 300.0 * np.arange(3),
                "x": 4_300_000.0 + 300.0 * np.arange(4),
            },
        )

    def _export(self, tmp_path, fmt: str, prov: Provenance):
        from s3bloom.export import FORMAT_EXTENSIONS, export_dataset

        path = tmp_path / f"out.{FORMAT_EXTENSIONS[fmt]}"
        export_dataset(self._data(), path, fmt, prov, "chl_nn")
        return path

    @pytest.mark.parametrize("fmt", ["netcdf", "geotiff"])
    def test_pass_record_round_trips(self, tmp_path, fmt):
        from s3bloom.metadata.provenance import read_output_attrs

        prov = _pass_provenance()

        attrs = read_output_attrs(self._export(tmp_path, fmt, prov))

        assert attrs["s3bloom_value_scale"] == "linear"
        assert attrs["s3bloom_source_product"] == prov.source_product
        assert attrs == prov.to_netcdf_attrs()

    @pytest.mark.parametrize("fmt", ["netcdf", "geotiff"])
    def test_composite_record_round_trips(self, tmp_path, fmt):
        from s3bloom.metadata.provenance import read_output_attrs

        prov = _composite_provenance()

        attrs = read_output_attrs(self._export(tmp_path, fmt, prov))

        assert attrs["s3bloom_source_products"] == ",".join(prov.source_products)
        assert attrs == prov.to_netcdf_attrs()

    @pytest.mark.parametrize("fmt", ["netcdf", "geotiff"])
    def test_only_s3bloom_attributes_are_returned(self, tmp_path, fmt):
        from s3bloom.metadata.provenance import read_output_attrs

        attrs = read_output_attrs(self._export(tmp_path, fmt, _pass_provenance()))

        assert attrs
        assert all(key.startswith("s3bloom_") for key in attrs)
        assert all(isinstance(value, str) for value in attrs.values())

    @pytest.mark.parametrize("name", ["missing.nc", "missing.tif"])
    def test_missing_file_gives_empty_record(self, tmp_path, name):
        from s3bloom.metadata.provenance import read_output_attrs

        assert read_output_attrs(tmp_path / name) == {}

    @pytest.mark.parametrize("name", ["broken.nc", "broken.tif"])
    def test_unreadable_file_gives_empty_record(self, tmp_path, name):
        # e.g. a file truncated by an interrupted run: it must count as
        # "no record" (and so be rebuilt), not abort the whole run.
        from s3bloom.metadata.provenance import read_output_attrs

        path = tmp_path / name
        path.write_bytes(b"not a real file")

        assert read_output_attrs(path) == {}

    def test_png_carries_no_record(self, tmp_path):
        from s3bloom.metadata.provenance import read_output_attrs

        path = tmp_path / "map.png"
        path.write_bytes(b"\x89PNG")

        assert read_output_attrs(path) == {}

    def test_file_is_closed_again(self, tmp_path):
        # The caller may overwrite the file right after reading it.
        from s3bloom.metadata.provenance import read_output_attrs

        path = self._export(tmp_path, "netcdf", _pass_provenance())
        read_output_attrs(path)

        self._export(tmp_path, "netcdf", _composite_provenance())

        assert "s3bloom_source_products" in read_output_attrs(path)


class TestCheckableOutput:
    def test_netcdf_is_preferred(self, tmp_path):
        from s3bloom.metadata.provenance import checkable_output

        outputs = {
            "png": tmp_path / "a.png",
            "geotiff": tmp_path / "a.tif",
            "netcdf": tmp_path / "a.nc",
        }

        assert checkable_output(outputs) == tmp_path / "a.nc"

    def test_geotiff_when_there_is_no_netcdf(self, tmp_path):
        from s3bloom.metadata.provenance import checkable_output

        outputs = {"png": tmp_path / "a.png", "geotiff": tmp_path / "a.tif"}

        assert checkable_output(outputs) == tmp_path / "a.tif"

    @pytest.mark.parametrize("formats", [["png"], []])
    def test_nothing_to_check_without_netcdf_or_geotiff(self, tmp_path, formats):
        from s3bloom.metadata.provenance import checkable_output

        outputs = {fmt: tmp_path / f"a.{fmt}" for fmt in formats}

        assert checkable_output(outputs) is None
