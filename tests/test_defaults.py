"""Tests for the display and valid-range tables in defaults."""

from __future__ import annotations

import pytest

from s3bloom import defaults
from s3bloom.defaults import COLORMAP_SETTINGS

SCIENCE_DATASETS = ("chl_nn", "chl_oc4me", "tsm_nn")


class TestColormapSettings:
    def test_covers_the_science_datasets(self):
        assert set(COLORMAP_SETTINGS) >= set(SCIENCE_DATASETS)

    @pytest.mark.parametrize("dataset_name", sorted(COLORMAP_SETTINGS))
    def test_no_log10_encoded_key(self, dataset_name):
        # Every stage of the pipeline works on linear values, so there
        # is nothing for an exporter to decode.
        assert "log10_encoded" not in COLORMAP_SETTINGS[dataset_name]

    @pytest.mark.parametrize("dataset_name", sorted(COLORMAP_SETTINGS))
    def test_display_keys_present(self, dataset_name):
        assert set(COLORMAP_SETTINGS[dataset_name]) == {
            "cmap",
            "vmin",
            "vmax",
            "log_scale",
            "label",
            "units",
        }

    @pytest.mark.parametrize(
        ("dataset_name", "units"),
        [("chl_nn", "mg m-3"), ("chl_oc4me", "mg m-3"), ("tsm_nn", "g m-3")],
    )
    def test_units_are_plain_ascii_cf(self, dataset_name, units):
        assert COLORMAP_SETTINGS[dataset_name]["units"] == units
        assert COLORMAP_SETTINGS[dataset_name]["units"].isascii()

    @pytest.mark.parametrize("dataset_name", sorted(COLORMAP_SETTINGS))
    def test_colour_bar_range_is_ordered(self, dataset_name):
        cfg = COLORMAP_SETTINGS[dataset_name]
        assert cfg["vmin"] < cfg["vmax"]


class TestValidRange:
    def test_covers_the_science_datasets(self):
        assert set(defaults.VALID_RANGE) == set(SCIENCE_DATASETS)

    @pytest.mark.parametrize(
        ("dataset_name", "expected"),
        [
            ("chl_nn", (0.0, 100.0)),
            ("chl_oc4me", (0.0, 100.0)),
            ("tsm_nn", (0.0, 1000.0)),
        ],
    )
    def test_linear_ranges(self, dataset_name, expected):
        assert defaults.VALID_RANGE[dataset_name] == expected

    @pytest.mark.parametrize("dataset_name", SCIENCE_DATASETS)
    def test_values_are_floats(self, dataset_name):
        low, high = defaults.VALID_RANGE[dataset_name]
        assert isinstance(low, float)
        assert isinstance(high, float)

    @pytest.mark.parametrize("dataset_name", SCIENCE_DATASETS)
    def test_valid_range_is_not_the_colour_bar_range(self, dataset_name):
        # The colour bar is a display choice and may clip; the valid
        # range has to contain it.
        low, high = defaults.VALID_RANGE[dataset_name]
        cfg = COLORMAP_SETTINGS[dataset_name]
        assert low <= cfg["vmin"]
        assert high > cfg["vmax"]
