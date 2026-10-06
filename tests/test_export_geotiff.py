"""Tests for export.geotiff."""
import numpy as np
import pytest
import rasterio
import xarray as xr
from pyresample import create_area_def

from s3bloom.export.geotiff import _ensure_2d, export_geotiff
from s3bloom.processing.resampling import _add_spatial_coords


class TestEnsure2d:
    def test_2d_passthrough(self):
        da = xr.DataArray(np.ones((4, 6)), dims=["y", "x"])
        result = _ensure_2d(da)
        assert result.dims == ("y", "x")
        assert result is da

    def test_3d_singleton_leading_dim_squeezed(self):
        da = xr.DataArray(np.ones((1, 4, 6)), dims=["time", "y", "x"])
        result = _ensure_2d(da)
        assert result.dims == ("y", "x")
        assert result.shape == (4, 6)

    def test_3d_non_singleton_raises(self):
        da = xr.DataArray(np.ones((2, 4, 6)), dims=["time", "y", "x"])
        with pytest.raises(ValueError, match="Expected 2D"):
            _ensure_2d(da)

    def test_4d_raises(self):
        da = xr.DataArray(np.ones((1, 1, 4, 6)), dims=["a", "b", "y", "x"])
        with pytest.raises(ValueError, match="Expected 2D"):
            _ensure_2d(da)


class TestExportGeotiff:
    def test_writes_file(self, tmp_path, simple_dataarray, provenance):
        out = tmp_path / "out.tif"
        result = export_geotiff(simple_dataarray, out, provenance)
        assert result == out
        assert out.exists()
        assert out.stat().st_size > 0

    def test_creates_parent_directories(self, tmp_path, simple_dataarray, provenance):
        out = tmp_path / "nested" / "deeper" / "out.tif"
        export_geotiff(simple_dataarray, out, provenance)
        assert out.exists()

    def test_returns_path(self, tmp_path, simple_dataarray, provenance):
        out = tmp_path / "out.tif"
        result = export_geotiff(simple_dataarray, out, provenance)
        assert result == out

    def test_accepts_3d_singleton(self, tmp_path, provenance):
        data_3d = xr.DataArray(
            np.ones((1, 4, 6), dtype=np.float32),
            dims=["time", "y", "x"],
            coords={
                "y": 3_001_050.0 - 300.0 * np.arange(4),
                "x": 4_000_150.0 + 300.0 * np.arange(6),
            },
        )
        out = tmp_path / "out3d.tif"
        result = export_geotiff(data_3d, out, provenance)
        assert result.exists()


class TestGeotiffGeoreferencing:
    def test_pixel_size_is_exactly_300m(self, tmp_path, simple_dataarray, provenance):
        out = export_geotiff(simple_dataarray, tmp_path / "out.tif", provenance)
        with rasterio.open(out) as src:
            assert src.transform.a == pytest.approx(300.0, abs=1e-9)
            assert src.transform.e == pytest.approx(-300.0, abs=1e-9)
            assert src.transform.b == 0
            assert src.transform.d == 0

    def test_origin_is_half_pixel_outside_first_centre(
        self, tmp_path, simple_dataarray, provenance
    ):
        out = export_geotiff(simple_dataarray, tmp_path / "out.tif", provenance)
        x0 = float(simple_dataarray.x.values[0])
        y0 = float(simple_dataarray.y.values[0])
        with rasterio.open(out) as src:
            assert src.transform.c == pytest.approx(x0 - 150.0, abs=1e-6)
            assert src.transform.f == pytest.approx(y0 + 150.0, abs=1e-6)

    def test_rows_written_north_up(self, tmp_path, simple_dataarray, provenance):
        out = export_geotiff(simple_dataarray, tmp_path / "out.tif", provenance)
        with rasterio.open(out) as src:
            written = src.read(1)
            assert src.bounds.top > src.bounds.bottom
        np.testing.assert_array_equal(written, simple_dataarray.values)

    def test_crs_written(self, tmp_path, simple_dataarray, provenance):
        out = export_geotiff(simple_dataarray, tmp_path / "out.tif", provenance)
        with rasterio.open(out) as src:
            assert src.crs.to_epsg() == 3035

    def test_resampled_grid_round_trips_to_area_extent(self, tmp_path, provenance):
        """Coords from the resampler must georeference onto the target area."""
        area = create_area_def(
            area_id="exact",
            projection="EPSG:3035",
            area_extent=(4_000_000.0, 3_000_000.0, 4_001_800.0, 3_001_200.0),
            resolution=300,
        )
        raw = xr.DataArray(np.ones((area.height, area.width), dtype=np.float32), dims=["y", "x"])
        gridded = _add_spatial_coords(raw, area)

        out = export_geotiff(gridded, tmp_path / "grid.tif", provenance)

        with rasterio.open(out) as src:
            assert src.transform.a == pytest.approx(300.0, abs=1e-6)
            assert src.transform.e == pytest.approx(-300.0, abs=1e-6)
            assert src.transform.c == pytest.approx(area.area_extent[0], abs=1e-6)
            assert src.transform.f == pytest.approx(area.area_extent[3], abs=1e-6)
            assert tuple(src.bounds) == pytest.approx(area.area_extent, abs=1e-6)
