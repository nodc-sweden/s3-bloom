"""Tests for processing.resampling.

The swath tests build a real :class:`satpy.Scene` on a synthetic
~300 m lattice and run pyresample's nearest-neighbour resampler for
real, because the behaviour under test (whether masked source pixels
survive resampling) lives inside the resampler, not in our wrapper.
"""
from dataclasses import dataclass
from unittest.mock import MagicMock

import dask.array as da
import numpy as np
import pytest
import xarray as xr
from pyproj import CRS, Transformer
from pyresample import create_area_def
from pyresample.geometry import AreaDefinition, SwathDefinition
from satpy import Scene

from s3bloom.config import BoundingBox, OutputConfig
from s3bloom.processing.resampling import (
    _add_spatial_coords,
    _bbox_to_extent,
    create_target_area,
    resample_scene,
)

PROJECTION = "EPSG:3035"
SOURCE_PITCH_M = 300.0
# Source lattice extends this far beyond the target so every target pixel
# has a source pixel well within the radius of influence.
SOURCE_MARGIN_M = 3000.0
# Sub-pixel shift so source and target pixel centres never coincide and
# nearest-neighbour ties cannot occur.
SOURCE_OFFSET_X_M = 37.0
SOURCE_OFFSET_Y_M = -53.0
HOLE_HALF_WIDTH = 7  # hole is (2*7+1) = 15 source pixels square


@dataclass(frozen=True)
class SourceLattice:
    """Synthetic swath: projected pixel centres plus their lon/lat."""

    xs: np.ndarray
    ys: np.ndarray
    lons: np.ndarray
    lats: np.ndarray

    @property
    def shape(self) -> tuple[int, int]:
        return (self.ys.size, self.xs.size)

    @property
    def centre(self) -> tuple[int, int]:
        return (self.ys.size // 2, self.xs.size // 2)


@pytest.fixture(scope="module")
def small_bbox():
    return BoundingBox(lon_min=11.0, lat_min=58.0, lon_max=11.2, lat_max=58.1)


@pytest.fixture(scope="module")
def target_area(small_bbox):
    return create_target_area(small_bbox, OutputConfig())


@pytest.fixture(scope="module")
def lattice(target_area):
    """~300 m source lattice covering *target_area* with a margin."""
    x_min, y_min, x_max, y_max = target_area.area_extent
    xs = (
        np.arange(x_min - SOURCE_MARGIN_M, x_max + SOURCE_MARGIN_M, SOURCE_PITCH_M)
        + SOURCE_OFFSET_X_M
    )
    ys = (
        np.arange(y_max + SOURCE_MARGIN_M, y_min - SOURCE_MARGIN_M, -SOURCE_PITCH_M)
        + SOURCE_OFFSET_Y_M
    )
    grid_x, grid_y = np.meshgrid(xs, ys)
    to_lonlat = Transformer.from_crs(PROJECTION, "EPSG:4326", always_xy=True)
    lons, lats = to_lonlat.transform(grid_x, grid_y)
    return SourceLattice(xs=xs, ys=ys, lons=lons, lats=lats)


def _make_scene(lattice: SourceLattice, values: np.ndarray, name: str = "chl_nn") -> Scene:
    """Wrap *values* as a swath dataset in a real, reader-less Scene."""
    lons = xr.DataArray(da.from_array(lattice.lons, chunks=-1), dims=("y", "x"))
    lats = xr.DataArray(da.from_array(lattice.lats, chunks=-1), dims=("y", "x"))
    swath = SwathDefinition(lons=lons, lats=lats)
    scene = Scene()
    scene[name] = xr.DataArray(
        da.from_array(values, chunks=-1),
        dims=("y", "x"),
        attrs={"area": swath, "name": name},
    )
    return scene


def _unique_values(lattice: SourceLattice) -> np.ndarray:
    """Every source pixel gets a distinct finite value (1, 2, 3, ...)."""
    size = lattice.shape[0] * lattice.shape[1]
    return (np.arange(size, dtype=np.float32) + 1).reshape(lattice.shape)


def _values_with_hole(lattice: SourceLattice) -> np.ndarray:
    """Distinct values with a 15x15 NaN block in the middle."""
    values = _unique_values(lattice)
    cy, cx = lattice.centre
    values[
        cy - HOLE_HALF_WIDTH : cy + HOLE_HALF_WIDTH + 1,
        cx - HOLE_HALF_WIDTH : cx + HOLE_HALF_WIDTH + 1,
    ] = np.nan
    return values


class TestMaskedPixelsSurviveResampling:
    def test_hole_interior_stays_nan(self, lattice, target_area):
        values = _values_with_hole(lattice)
        scene = _make_scene(lattice, values)

        result = resample_scene(scene, target_area, ["chl_nn"])["chl_nn"].values

        # Target pixels whose centre lies inside the hole, shrunk by one
        # source pixel on each side so projection distortion at the rim
        # cannot move a pixel across the boundary. Located from the area
        # itself, not from the output coords, to keep this independent of
        # how x/y are labelled.
        cy, cx = lattice.centre
        inner = HOLE_HALF_WIDTH - 1
        target_x = target_area.projection_x_coords
        target_y = target_area.projection_y_coords
        in_x = (target_x >= lattice.xs[cx - inner]) & (target_x <= lattice.xs[cx + inner])
        in_y = (target_y <= lattice.ys[cy - inner]) & (target_y >= lattice.ys[cy + inner])
        inside = np.outer(in_y, in_x)

        assert inside.sum() >= 81, "hole must cover at least 9x9 target pixels"
        filled = int(np.isfinite(result[inside]).sum())
        assert filled == 0, (
            f"{filled} of {int(inside.sum())} target pixels inside the masked "
            "block were filled from neighbouring valid pixels"
        )

    def test_valid_count_bounded_by_valid_source_fraction(self, lattice, target_area):
        values = _values_with_hole(lattice)
        scene = _make_scene(lattice, values)

        result = resample_scene(scene, target_area, ["chl_nn"])["chl_nn"].values

        # Valid fraction of the source pixels that fall on the target.
        x_min, y_min, x_max, y_max = target_area.area_extent
        on_target = np.outer(
            (lattice.ys >= y_min) & (lattice.ys <= y_max),
            (lattice.xs >= x_min) & (lattice.xs <= x_max),
        )
        valid_fraction = np.isfinite(values[on_target]).mean()
        # One target row + column of slack for pixels straddling the edges.
        slack = target_area.width + target_area.height
        allowed = int(np.ceil(valid_fraction * result.size)) + slack

        n_valid = int(np.isfinite(result).sum())
        assert n_valid <= allowed, (
            f"{n_valid} valid target pixels, but only {valid_fraction:.1%} of "
            f"the source is valid (max {allowed} of {result.size})"
        )

    def test_valid_pixels_keep_source_values(self, lattice, target_area):
        values = _values_with_hole(lattice)
        scene = _make_scene(lattice, values)

        result = resample_scene(scene, target_area, ["chl_nn"])["chl_nn"].values

        finite = result[np.isfinite(result)]
        assert finite.size > 0
        assert np.isin(finite, values[np.isfinite(values)]).all()

    def test_fully_valid_source_fills_target(self, lattice, target_area):
        scene = _make_scene(lattice, _unique_values(lattice))

        result = resample_scene(scene, target_area, ["chl_nn"])["chl_nn"].values

        assert np.isfinite(result).all()

    def test_isolated_valid_pixel_does_not_grow(self, lattice, target_area):
        values = np.full(lattice.shape, np.nan, dtype=np.float32)
        cy, cx = lattice.centre
        values[cy, cx] = 5.0
        scene = _make_scene(lattice, values)

        result = resample_scene(scene, target_area, ["chl_nn"])["chl_nn"].values

        # Target pitch is marginally finer than the source pitch, so one
        # source pixel may own up to 2x2 target centres, never more.
        n_valid = int(np.isfinite(result).sum())
        assert 1 <= n_valid <= 4, f"single source pixel became {n_valid} target pixels"
        assert np.all(result[np.isfinite(result)] == 5.0)

    def test_all_nan_source_stays_all_nan(self, lattice, target_area):
        values = np.full(lattice.shape, np.nan, dtype=np.float32)
        scene = _make_scene(lattice, values)

        result = resample_scene(scene, target_area, ["chl_nn"])["chl_nn"].values

        assert np.isnan(result).all()

    def test_target_outside_swath_is_nan(self, lattice):
        far_bbox = BoundingBox(lon_min=15.0, lat_min=60.0, lon_max=15.1, lat_max=60.05)
        far_area = create_target_area(far_bbox, OutputConfig())
        scene = _make_scene(lattice, _unique_values(lattice))

        result = resample_scene(scene, far_area, ["chl_nn"])["chl_nn"].values

        assert np.isnan(result).all()


class TestResampleSceneOutput:
    def test_output_shape_matches_target(self, lattice, target_area):
        scene = _make_scene(lattice, _unique_values(lattice))

        result = resample_scene(scene, target_area, ["chl_nn"])

        assert list(result) == ["chl_nn"]
        assert result["chl_nn"].dims == ("y", "x")
        assert result["chl_nn"].shape == (target_area.height, target_area.width)

    def test_x_y_are_pixel_centres(self, lattice, target_area):
        scene = _make_scene(lattice, _unique_values(lattice))

        out = resample_scene(scene, target_area, ["chl_nn"])["chl_nn"]

        np.testing.assert_allclose(out.x.values, target_area.projection_x_coords, rtol=0, atol=1e-6)
        np.testing.assert_allclose(out.y.values, target_area.projection_y_coords, rtol=0, atol=1e-6)

    def test_only_requested_datasets_returned(self, lattice, target_area):
        scene = _make_scene(lattice, _unique_values(lattice))
        scene["wqsf"] = scene["chl_nn"].copy()

        result = resample_scene(scene, target_area, ["chl_nn"])

        assert list(result) == ["chl_nn"]

    def test_missing_dataset_warned_and_skipped(self, lattice, target_area, caplog):
        scene = _make_scene(lattice, _unique_values(lattice))

        with caplog.at_level("WARNING", logger="s3bloom.processing.resampling"):
            result = resample_scene(scene, target_area, ["chl_nn", "tsm_nn"])

        assert list(result) == ["chl_nn"]
        assert "tsm_nn" in caplog.text

    def test_no_datasets_present_returns_empty(self, lattice, target_area):
        scene = _make_scene(lattice, _unique_values(lattice))

        assert resample_scene(scene, target_area, ["tsm_nn"]) == {}

    def test_source_scene_not_modified(self, lattice, target_area):
        values = _values_with_hole(lattice)
        scene = _make_scene(lattice, values)

        resample_scene(scene, target_area, ["chl_nn"])

        assert scene["chl_nn"].shape == lattice.shape
        np.testing.assert_array_equal(scene["chl_nn"].values, values)


class TestResampleSceneCallArguments:
    @staticmethod
    def _mock_scene(present: set[str]) -> MagicMock:
        scene = MagicMock()
        scene.__contains__.side_effect = lambda name: name in present
        scene.resample.return_value = {}
        return scene

    def test_masked_source_pixels_kept_in_search(self):
        scene = self._mock_scene({"chl_nn"})

        resample_scene(scene, MagicMock(), ["chl_nn"])

        scene.resample.assert_called_once()
        assert scene.resample.call_args.kwargs["mask_area"] is False

    def test_only_requested_datasets_resampled(self):
        scene = self._mock_scene({"chl_nn", "tsm_nn", "wqsf"})
        target = MagicMock()

        resample_scene(scene, target, ["chl_nn", "tsm_nn"])

        assert scene.resample.call_args.args == (target,)
        assert scene.resample.call_args.kwargs["datasets"] == ["chl_nn", "tsm_nn"]

    def test_nearest_neighbour_with_nan_fill(self):
        scene = self._mock_scene({"chl_nn"})

        resample_scene(scene, MagicMock(), ["chl_nn"])

        kwargs = scene.resample.call_args.kwargs
        assert kwargs["resampler"] == "nearest"
        assert kwargs["radius_of_influence"] == 1500
        assert np.isnan(kwargs["fill_value"])

    def test_composite_regeneration_disabled(self):
        # With datasets= limiting what is resampled, generate=True makes
        # satpy warn that the left-out wishlist entries were "not created".
        scene = self._mock_scene({"chl_nn", "wqsf"})

        resample_scene(scene, MagicMock(), ["chl_nn"])

        assert scene.resample.call_args.kwargs["generate"] is False

    def test_datasets_absent_from_scene_not_requested(self):
        scene = self._mock_scene({"chl_nn"})

        result = resample_scene(scene, MagicMock(), ["chl_nn", "tsm_nn"])

        assert scene.resample.call_args.kwargs["datasets"] == ["chl_nn"]
        assert result == {}

    def test_caller_dataset_list_not_mutated(self):
        scene = self._mock_scene({"chl_nn"})
        requested = ["chl_nn", "tsm_nn"]

        resample_scene(scene, MagicMock(), requested)

        assert requested == ["chl_nn", "tsm_nn"]


class TestAddSpatialCoords:
    @pytest.fixture
    def exact_area(self):
        """6x4 area on an exact 300 m grid."""
        return create_area_def(
            area_id="exact",
            projection=PROJECTION,
            area_extent=(4_000_000.0, 3_000_000.0, 4_001_800.0, 3_001_200.0),
            resolution=300,
        )

    @pytest.fixture
    def raw(self, exact_area):
        return xr.DataArray(
            np.zeros((exact_area.height, exact_area.width), dtype=np.float32),
            dims=("y", "x"),
            attrs={"units": "mg m-3"},
        )

    def test_x_matches_projection_x_coords(self, raw, exact_area):
        out = _add_spatial_coords(raw, exact_area)
        np.testing.assert_array_equal(out.x.values, exact_area.projection_x_coords)

    def test_y_matches_projection_y_coords(self, raw, exact_area):
        out = _add_spatial_coords(raw, exact_area)
        np.testing.assert_array_equal(out.y.values, exact_area.projection_y_coords)

    def test_first_coord_is_half_pixel_inside_extent(self, raw, exact_area):
        out = _add_spatial_coords(raw, exact_area)
        x_min, y_min, x_max, y_max = exact_area.area_extent
        assert out.x.values[0] == pytest.approx(x_min + exact_area.pixel_size_x / 2)
        assert out.x.values[-1] == pytest.approx(x_max - exact_area.pixel_size_x / 2)
        assert out.y.values[0] == pytest.approx(y_max - exact_area.pixel_size_y / 2)
        assert out.y.values[-1] == pytest.approx(y_min + exact_area.pixel_size_y / 2)

    def test_exact_grid_values(self, raw, exact_area):
        out = _add_spatial_coords(raw, exact_area)
        np.testing.assert_allclose(out.x.values, 4_000_150.0 + 300.0 * np.arange(6))
        np.testing.assert_allclose(out.y.values, 3_001_050.0 - 300.0 * np.arange(4))

    def test_spacing_equals_pixel_size(self, raw, exact_area):
        out = _add_spatial_coords(raw, exact_area)
        np.testing.assert_allclose(np.diff(out.x.values), exact_area.pixel_size_x)
        np.testing.assert_allclose(np.diff(out.y.values), -exact_area.pixel_size_y)

    def test_spacing_equals_pixel_size_on_unsnapped_area(self, target_area):
        # create_target_area does not snap the extent, so the pixel size is
        # close to but not exactly the requested resolution.
        raw = xr.DataArray(
            np.zeros((target_area.height, target_area.width), dtype=np.float32),
            dims=("y", "x"),
        )
        out = _add_spatial_coords(raw, target_area)
        np.testing.assert_allclose(np.diff(out.x.values), target_area.pixel_size_x)
        np.testing.assert_allclose(np.diff(out.y.values), -target_area.pixel_size_y)
        assert out.x.values[0] == pytest.approx(
            target_area.area_extent[0] + target_area.pixel_size_x / 2
        )
        assert out.y.values[0] == pytest.approx(
            target_area.area_extent[3] - target_area.pixel_size_y / 2
        )

    def test_y_descending_x_ascending(self, raw, exact_area):
        out = _add_spatial_coords(raw, exact_area)
        assert np.all(np.diff(out.y.values) < 0)
        assert np.all(np.diff(out.x.values) > 0)

    def test_crs_and_area_id_attrs(self, raw, exact_area):
        out = _add_spatial_coords(raw, exact_area)
        assert CRS.from_wkt(out.attrs["crs"]) == CRS.from_user_input(PROJECTION)
        assert out.attrs["area_id"] == "exact"
        assert out.attrs["units"] == "mg m-3"

    def test_renames_non_xy_dims(self, exact_area):
        raw = xr.DataArray(
            np.zeros((exact_area.height, exact_area.width)), dims=("rows", "columns")
        )
        out = _add_spatial_coords(raw, exact_area)
        assert out.dims == ("y", "x")

    def test_input_not_mutated(self, raw, exact_area):
        _add_spatial_coords(raw, exact_area)
        assert raw.attrs == {"units": "mg m-3"}
        assert "x" not in raw.coords
        assert "y" not in raw.coords


class TestBboxToExtent:
    def test_ordering_is_xmin_ymin_xmax_ymax(self, small_bbox):
        x_min, y_min, x_max, y_max = _bbox_to_extent(small_bbox, PROJECTION)
        assert x_min < x_max
        assert y_min < y_max

    def test_wgs84_is_identity(self, small_bbox):
        extent = _bbox_to_extent(small_bbox, "EPSG:4326")
        assert extent == pytest.approx((11.0, 58.0, 11.2, 58.1))

    def test_matches_transformed_corners(self, small_bbox):
        transformer = Transformer.from_crs("EPSG:4326", PROJECTION, always_xy=True)
        x_ll, y_ll = transformer.transform(small_bbox.lon_min, small_bbox.lat_min)
        x_ur, y_ur = transformer.transform(small_bbox.lon_max, small_bbox.lat_max)

        extent = _bbox_to_extent(small_bbox, PROJECTION)

        assert extent == pytest.approx(
            (min(x_ll, x_ur), min(y_ll, y_ur), max(x_ll, x_ur), max(y_ll, y_ur))
        )

    def test_extent_in_projected_metres(self, small_bbox):
        x_min, y_min, x_max, y_max = _bbox_to_extent(small_bbox, PROJECTION)
        # 0.2 deg lon x 0.1 deg lat at 58 N is roughly 12 km x 11 km.
        assert 10_000 < x_max - x_min < 14_000
        assert 9_000 < y_max - y_min < 13_000

    def test_southern_hemisphere_utm_stays_ordered(self):
        bbox = BoundingBox(lon_min=18.0, lat_min=-34.5, lon_max=19.0, lat_max=-33.5)
        x_min, y_min, x_max, y_max = _bbox_to_extent(bbox, "EPSG:32734")
        assert x_min < x_max
        assert y_min < y_max

    def test_returns_four_floats(self, small_bbox):
        extent = _bbox_to_extent(small_bbox, PROJECTION)
        assert len(extent) == 4
        assert all(isinstance(v, float) for v in extent)


class TestCreateTargetArea:
    def test_returns_area_definition(self, target_area):
        assert isinstance(target_area, AreaDefinition)
        assert target_area.area_id == "s3bloom_target"

    def test_crs_matches_output_config(self, target_area):
        assert target_area.crs == CRS.from_user_input(PROJECTION)

    def test_custom_projection(self, small_bbox):
        area = create_target_area(small_bbox, OutputConfig(projection="EPSG:32632"))
        assert area.crs == CRS.from_user_input("EPSG:32632")

    def test_extent_matches_bbox_extent(self, small_bbox, target_area):
        assert target_area.area_extent == pytest.approx(
            _bbox_to_extent(small_bbox, PROJECTION)
        )

    def test_extent_ordering(self, target_area):
        x_min, y_min, x_max, y_max = target_area.area_extent
        assert x_min < x_max
        assert y_min < y_max

    def test_shape_derived_from_extent_and_resolution(self, target_area):
        x_min, y_min, x_max, y_max = target_area.area_extent
        assert target_area.width == round((x_max - x_min) / 300)
        assert target_area.height == round((y_max - y_min) / 300)

    def test_pixel_size_close_to_resolution(self, target_area):
        assert target_area.pixel_size_x == pytest.approx(300, rel=0.02)
        assert target_area.pixel_size_y == pytest.approx(300, rel=0.02)

    def test_coarser_resolution_gives_fewer_pixels(self, small_bbox, target_area):
        coarse = create_target_area(small_bbox, OutputConfig(resolution_m=1000))
        assert coarse.width < target_area.width
        assert coarse.height < target_area.height
        assert coarse.pixel_size_x == pytest.approx(1000, rel=0.1)

    def test_default_aoi_grid_size(self):
        area = create_target_area(
            BoundingBox.from_string("swedish_west_coast"), OutputConfig()
        )
        assert (area.width, area.height) == (576, 939)
