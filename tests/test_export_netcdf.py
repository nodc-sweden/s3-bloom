"""Tests for export.netcdf."""
from dataclasses import replace

import numpy as np
import pytest
import rioxarray  # noqa: F401  (registers the ``.rio`` accessor)
import xarray as xr
from pyproj import CRS

from s3bloom.export.netcdf import (
    _build_dataset,
    _long_name,
    _standard_name,
    export_netcdf,
)


class TestLongName:
    def test_chl_nn(self):
        assert "chlorophyll" in _long_name("chl_nn").lower()

    def test_chl_oc4me(self):
        assert "chlorophyll" in _long_name("chl_oc4me").lower()

    def test_tsm_nn(self):
        assert "suspended" in _long_name("tsm_nn").lower()

    def test_unknown_returns_name_itself(self):
        assert _long_name("unknown_var") == "unknown_var"


class TestStandardName:
    def test_chl_nn(self):
        assert _standard_name("chl_nn") == "mass_concentration_of_chlorophyll_a_in_sea_water"

    def test_chl_oc4me(self):
        assert _standard_name("chl_oc4me") == "mass_concentration_of_chlorophyll_a_in_sea_water"

    def test_tsm_nn(self):
        assert "suspended_matter" in _standard_name("tsm_nn")

    def test_unknown_returns_empty_string(self):
        assert _standard_name("unknown") == ""


class TestBuildDataset:
    def test_returns_dataset(self, simple_dataarray, provenance):
        ds = _build_dataset(simple_dataarray, "chl_nn", provenance)
        assert isinstance(ds, xr.Dataset)

    def test_variable_present(self, simple_dataarray, provenance):
        ds = _build_dataset(simple_dataarray, "chl_nn", provenance)
        assert "chl_nn" in ds.data_vars

    def test_cf_conventions_attribute(self, simple_dataarray, provenance):
        ds = _build_dataset(simple_dataarray, "chl_nn", provenance)
        assert ds.attrs["Conventions"] == "CF-1.8"

    def test_provenance_attrs_prefixed(self, simple_dataarray, provenance):
        ds = _build_dataset(simple_dataarray, "chl_nn", provenance)
        assert "s3bloom_source_product" in ds.attrs
        assert "s3bloom_satellite" in ds.attrs

    def test_crs_stored_as_cf_grid_mapping_variable(self, simple_dataarray, provenance):
        ds = _build_dataset(simple_dataarray, "chl_nn", provenance)
        assert "crs" in ds.data_vars
        assert ds["crs"].ndim == 0
        assert np.issubdtype(ds["crs"].dtype, np.integer)
        assert CRS.from_cf(ds["crs"].attrs).to_epsg() == 3035

    def test_crs_variable_carries_wkt_for_gdal(self, simple_dataarray, provenance):
        ds = _build_dataset(simple_dataarray, "chl_nn", provenance)
        attrs = ds["crs"].attrs
        assert CRS.from_wkt(attrs["spatial_ref"]).to_epsg() == 3035
        assert CRS.from_wkt(attrs["crs_wkt"]).to_epsg() == 3035

    def test_data_variable_references_grid_mapping(self, simple_dataarray, provenance):
        ds = _build_dataset(simple_dataarray, "chl_nn", provenance)
        assert ds["chl_nn"].attrs["grid_mapping"] == "crs"

    def test_no_bare_authority_code_global_attribute(self, simple_dataarray, provenance):
        ds = _build_dataset(simple_dataarray, "chl_nn", provenance)
        assert "crs_wkt" not in ds.attrs

    def test_crs_follows_provenance_projection(self, simple_dataarray, provenance):
        utm = replace(provenance, projection="EPSG:32633")
        ds = _build_dataset(simple_dataarray, "chl_nn", utm)
        assert CRS.from_cf(ds["crs"].attrs).to_epsg() == 32633
        assert CRS.from_wkt(ds["crs"].attrs["spatial_ref"]).to_epsg() == 32633

    def test_variable_has_long_name(self, simple_dataarray, provenance):
        ds = _build_dataset(simple_dataarray, "chl_nn", provenance)
        assert "long_name" in ds["chl_nn"].attrs
        assert "units" in ds["chl_nn"].attrs

    @pytest.mark.parametrize(
        ("dataset_name", "units"),
        [("chl_nn", "mg m-3"), ("chl_oc4me", "mg m-3"), ("tsm_nn", "g m-3")],
    )
    def test_units_are_linear_ascii_cf(
        self, simple_dataarray, provenance, dataset_name, units
    ):
        ds = _build_dataset(simple_dataarray, dataset_name, provenance)
        assert ds[dataset_name].attrs["units"] == units

    def test_chl_valid_range_is_linear_physical_range(self, simple_dataarray, provenance):
        ds = _build_dataset(simple_dataarray, "chl_nn", provenance)
        assert ds["chl_nn"].attrs["valid_min"] == 0.0
        assert ds["chl_nn"].attrs["valid_max"] == 100.0

    def test_valid_range_is_not_the_colour_bar_range(self, simple_dataarray, provenance):
        ds = _build_dataset(simple_dataarray, "tsm_nn", provenance)
        assert ds["tsm_nn"].attrs["valid_min"] == 0.0
        assert ds["tsm_nn"].attrs["valid_max"] == 1000.0

    def test_unknown_dataset_has_no_valid_range(self, simple_dataarray, provenance):
        ds = _build_dataset(simple_dataarray, "unknown_var", provenance)
        assert "valid_min" not in ds["unknown_var"].attrs
        assert "valid_max" not in ds["unknown_var"].attrs

    def test_value_scale_global_attribute(self, simple_dataarray, provenance):
        ds = _build_dataset(simple_dataarray, "chl_nn", provenance)
        assert ds.attrs["s3bloom_value_scale"] == "linear"

    def test_values_are_stored_as_given(self, simple_dataarray, provenance):
        ds = _build_dataset(simple_dataarray, "chl_nn", provenance)
        np.testing.assert_array_equal(ds["chl_nn"].values, simple_dataarray.values)

    def test_x_y_coord_attrs(self, simple_dataarray, provenance):
        ds = _build_dataset(simple_dataarray, "chl_nn", provenance)
        assert ds.x.attrs.get("axis") == "X"
        assert ds.y.attrs.get("axis") == "Y"
        assert ds.x.attrs.get("standard_name") == "projection_x_coordinate"

    def test_drops_non_spatial_extra_coords(self, simple_dataarray, provenance):
        da = simple_dataarray.assign_coords({"spatial_ref": 0})
        ds = _build_dataset(da, "chl_nn", provenance)
        assert "spatial_ref" not in ds.coords

    def test_unknown_dataset_still_builds(self, simple_dataarray, provenance):
        ds = _build_dataset(simple_dataarray, "unknown_var", provenance)
        assert "unknown_var" in ds.data_vars


class TestExportNetcdf:
    def test_writes_file(self, tmp_path, simple_dataarray, provenance):
        out = tmp_path / "out.nc"
        result = export_netcdf(simple_dataarray, out, provenance, "chl_nn")
        assert result == out
        assert out.exists()
        assert out.stat().st_size > 0

    def test_creates_parent_directories(self, tmp_path, simple_dataarray, provenance):
        out = tmp_path / "nested" / "out.nc"
        export_netcdf(simple_dataarray, out, provenance, "chl_nn")
        assert out.exists()

    def test_returns_path(self, tmp_path, simple_dataarray, provenance):
        out = tmp_path / "out.nc"
        result = export_netcdf(simple_dataarray, out, provenance, "chl_nn")
        assert result == out

    def test_output_is_readable_netcdf(self, tmp_path, simple_dataarray, provenance):
        out = tmp_path / "out.nc"
        export_netcdf(simple_dataarray, out, provenance, "chl_nn")
        ds = xr.open_dataset(out)
        assert "chl_nn" in ds.data_vars
        ds.close()

    def test_netcdf_has_cf_conventions(self, tmp_path, simple_dataarray, provenance):
        out = tmp_path / "out.nc"
        export_netcdf(simple_dataarray, out, provenance, "chl_nn")
        ds = xr.open_dataset(out)
        assert ds.attrs.get("Conventions") == "CF-1.8"
        ds.close()

    def test_written_file_declares_linear_scale(self, tmp_path, simple_dataarray, provenance):
        out = tmp_path / "out.nc"
        export_netcdf(simple_dataarray, out, provenance, "chl_nn")
        with xr.open_dataset(out) as ds:
            assert ds.attrs["s3bloom_value_scale"] == "linear"
            assert ds["chl_nn"].attrs["units"] == "mg m-3"
            assert ds["chl_nn"].attrs["valid_min"] == 0.0
            assert ds["chl_nn"].attrs["valid_max"] == 100.0

    def test_written_values_round_trip_unchanged(self, tmp_path, simple_dataarray, provenance):
        out = tmp_path / "out.nc"
        export_netcdf(simple_dataarray, out, provenance, "chl_nn")
        with xr.open_dataset(out) as ds:
            np.testing.assert_array_equal(ds["chl_nn"].values, simple_dataarray.values)

    def test_written_unknown_dataset_has_no_valid_range(
        self, tmp_path, simple_dataarray, provenance
    ):
        out = tmp_path / "out.nc"
        export_netcdf(simple_dataarray, out, provenance, "unknown_var")
        with xr.open_dataset(out) as ds:
            assert "valid_min" not in ds["unknown_var"].attrs
            assert "valid_max" not in ds["unknown_var"].attrs

    def test_written_file_has_grid_mapping_variable(self, tmp_path, simple_dataarray, provenance):
        out = tmp_path / "out.nc"
        export_netcdf(simple_dataarray, out, provenance, "chl_nn")
        with xr.open_dataset(out) as ds:
            assert ds["chl_nn"].attrs["grid_mapping"] == "crs"
            assert CRS.from_cf(ds["crs"].attrs).to_epsg() == 3035

    def test_written_file_crs_decoded_by_rioxarray(self, tmp_path, simple_dataarray, provenance):
        out = tmp_path / "out.nc"
        export_netcdf(simple_dataarray, out, provenance, "chl_nn")
        with xr.open_dataset(out, decode_coords="all") as ds:
            assert ds["chl_nn"].rio.crs.to_epsg() == 3035

    def test_written_file_crs_follows_provenance_projection(
        self, tmp_path, simple_dataarray, provenance
    ):
        out = tmp_path / "out.nc"
        utm = replace(provenance, projection="EPSG:32633")
        export_netcdf(simple_dataarray, out, utm, "chl_nn")
        with xr.open_dataset(out, decode_coords="all") as ds:
            assert ds["chl_nn"].rio.crs.to_epsg() == 32633
