"""Tests for export.png.

The Natural Earth land/coastline features are replaced by empty
features so nothing is downloaded and rendering stays fast.
"""

from __future__ import annotations

from dataclasses import replace

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import dask.array as da
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr
from cartopy.mpl.geoaxes import GeoAxes

from s3bloom.defaults import COLORMAP_SETTINGS
from s3bloom.export import png
from s3bloom.export.png import (
    _build_title,
    _create_norm,
    _projection_from_epsg,
    export_png,
)


def _linear_array(*, lazy: bool = False) -> xr.DataArray:
    """4x6 array of linear concentrations on a 300 m EPSG:3035 grid."""
    values = np.linspace(0.05, 45.0, 24, dtype=np.float32).reshape(4, 6)
    values[0, 0] = np.nan
    data = da.from_array(values, chunks=2) if lazy else values
    return xr.DataArray(
        data,
        dims=("y", "x"),
        coords={
            "y": 3_500_000.0 - 300.0 * np.arange(4),
            "x": 4_300_000.0 + 300.0 * np.arange(6),
        },
        name="chl_nn",
    )


def _expected(data: xr.DataArray) -> np.ndarray:
    return np.asarray(data.values, dtype=np.float64)


@pytest.fixture(autouse=True)
def offline_features(monkeypatch):
    """Swap the Natural Earth features for empty ones (no downloads)."""
    empty = cfeature.ShapelyFeature([], ccrs.PlateCarree())
    monkeypatch.setattr(png, "_LAND_FEATURE", empty)
    monkeypatch.setattr(png, "_COASTLINE_FEATURE", empty)


@pytest.fixture
def plotted(monkeypatch):
    """Record every ``pcolormesh`` call while still drawing for real."""
    calls: list[dict] = []
    original = GeoAxes.pcolormesh

    def _spy(self, *args, **kwargs):
        calls.append({"args": args, "kwargs": kwargs})
        return original(self, *args, **kwargs)

    monkeypatch.setattr(GeoAxes, "pcolormesh", _spy)
    return calls


# --------------------------------------------------------------------------- #
# export_png                                                                  #
# --------------------------------------------------------------------------- #


class TestExportPngValues:
    @pytest.mark.parametrize("dataset_name", ["chl_nn", "chl_oc4me", "tsm_nn"])
    def test_values_are_plotted_unchanged(
        self, tmp_path, provenance, plotted, dataset_name
    ):
        data = _linear_array()

        export_png(data, tmp_path / "out.png", provenance, dataset_name)

        assert len(plotted) == 1
        np.testing.assert_array_equal(plotted[0]["args"][2], _expected(data))

    def test_stale_log10_encoded_flag_is_ignored(
        self, tmp_path, provenance, plotted, monkeypatch
    ):
        # Outputs are linear end to end; a leftover flag must not make
        # the exporter exponentiate the values again.
        stale = {**COLORMAP_SETTINGS["chl_nn"], "log10_encoded": True}
        monkeypatch.setitem(png.COLORMAP_SETTINGS, "chl_nn", stale)
        data = _linear_array()

        export_png(data, tmp_path / "out.png", provenance, "chl_nn")

        np.testing.assert_array_equal(plotted[0]["args"][2], _expected(data))

    def test_dask_backed_values_are_plotted_unchanged(
        self, tmp_path, provenance, plotted
    ):
        data = _linear_array(lazy=True)

        export_png(data, tmp_path / "out.png", provenance, "chl_nn")

        np.testing.assert_array_equal(plotted[0]["args"][2], _expected(data))

    def test_input_is_not_mutated(self, tmp_path, provenance):
        data = _linear_array()
        before = data.values.copy()

        export_png(data, tmp_path / "out.png", provenance, "chl_nn")

        np.testing.assert_array_equal(data.values, before)

    def test_coordinates_are_passed_through(self, tmp_path, provenance, plotted):
        data = _linear_array()

        export_png(data, tmp_path / "out.png", provenance, "chl_nn")

        np.testing.assert_array_equal(plotted[0]["args"][0], data["x"].values)
        np.testing.assert_array_equal(plotted[0]["args"][1], data["y"].values)

    def test_colour_range_comes_from_settings(self, tmp_path, provenance, plotted):
        export_png(_linear_array(), tmp_path / "out.png", provenance, "tsm_nn")

        norm = plotted[0]["kwargs"]["norm"]
        assert norm.vmin == COLORMAP_SETTINGS["tsm_nn"]["vmin"]
        assert norm.vmax == COLORMAP_SETTINGS["tsm_nn"]["vmax"]

    def test_unknown_dataset_uses_fallback_settings(
        self, tmp_path, provenance, plotted
    ):
        data = _linear_array()

        export_png(data, tmp_path / "out.png", provenance, "mystery_var")

        assert plotted[0]["kwargs"]["cmap"].name == "viridis"
        np.testing.assert_array_equal(plotted[0]["args"][2], _expected(data))


class TestExportPngFile:
    def test_writes_png_and_returns_path(self, tmp_path, provenance):
        out = tmp_path / "out.png"

        result = export_png(_linear_array(), out, provenance, "chl_nn")

        assert result == out
        assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"

    def test_creates_parent_directories(self, tmp_path, provenance):
        out = tmp_path / "nested" / "deeper" / "out.png"

        export_png(_linear_array(), out, provenance, "chl_nn")

        assert out.exists()

    def test_figure_is_closed(self, tmp_path, provenance):
        open_before = set(plt.get_fignums())

        export_png(_linear_array(), tmp_path / "out.png", provenance, "chl_nn")

        assert set(plt.get_fignums()) == open_before

    def test_composite_provenance_renders(self, tmp_path, composite_provenance):
        out = tmp_path / "composite.png"

        export_png(_linear_array(), out, composite_provenance, "chl_nn")

        assert out.stat().st_size > 0


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


class TestCreateNorm:
    def test_linear_norm_by_default(self):
        norm = _create_norm({"vmin": 0.0, "vmax": 30.0, "log_scale": False})

        assert type(norm) is mcolors.Normalize
        assert (norm.vmin, norm.vmax) == (0.0, 30.0)

    def test_missing_log_scale_means_linear(self):
        norm = _create_norm({"vmin": 1.0, "vmax": 2.0})

        assert type(norm) is mcolors.Normalize

    def test_log_norm_when_requested(self):
        norm = _create_norm({"vmin": 0.1, "vmax": 30.0, "log_scale": True})

        assert isinstance(norm, mcolors.LogNorm)
        assert (norm.vmin, norm.vmax) == (0.1, 30.0)

    @pytest.mark.parametrize("dataset_name", sorted(COLORMAP_SETTINGS))
    def test_every_default_setting_builds_a_norm(self, dataset_name):
        cfg = COLORMAP_SETTINGS[dataset_name]

        norm = _create_norm(cfg)

        assert norm.vmin == cfg["vmin"]
        assert norm.vmax == cfg["vmax"]


class TestBuildTitle:
    def test_pass_title(self, provenance):
        title = _build_title(provenance, "chl_nn")

        assert title == "chl_nn — S3A\n2024-03-15 09:15 UTC | Mask: strict"

    def test_composite_title(self, composite_provenance):
        title = _build_title(composite_provenance, "chl_nn")

        first, second = title.split("\n")
        assert first == "chl_nn — 3-day composite"
        assert second == "2024-03-15 09:15 UTC | S3A-S3B | Mask: strict"

    def test_title_follows_masking_preset(self, provenance):
        relaxed = replace(provenance, masking_preset="relaxed")

        assert _build_title(relaxed, "tsm_nn").endswith("Mask: relaxed")


class TestProjectionFromEpsg:
    def test_laea_gets_parameterised_projection(self):
        proj = _projection_from_epsg("EPSG:3035")

        assert isinstance(proj, ccrs.LambertAzimuthalEqualArea)
        params = proj.proj4_params
        assert params["lon_0"] == 10.0
        assert params["lat_0"] == 52.0
        assert params["x_0"] == 4_321_000.0
        assert params["y_0"] == 3_210_000.0

    def test_other_projections_use_cartopy_epsg(self, monkeypatch):
        sentinel = object()
        requested: list[int] = []

        def _fake_epsg(code: int) -> object:
            requested.append(code)
            return sentinel

        monkeypatch.setattr(png.ccrs, "epsg", _fake_epsg)

        assert _projection_from_epsg("EPSG:32633") is sentinel
        assert requested == [32633]
