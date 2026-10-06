"""Tests for processing.reader."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import dask.array as da
import numpy as np
import pytest
import xarray as xr

from s3bloom.processing import reader
from s3bloom.processing.reader import (
    _collect_filenames,
    extract_satellite,
    extract_sensing_time,
    load_scene,
)

REAL_PRODUCT = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "raw"
    / (
        "S3A_OL_2_WFR____20240310T085903_20240310T090203_20240311T152230"
        "_0180_110_050_1980_MAR_O_NT_003.SEN3"
    )
)

LOG_VALUES = [[-1.0, 0.0], [1.0, np.nan]]
LINEAR_VALUES = [[0.1, 1.0], [10.0, np.nan]]


def _array(
    values: list[list[float]] | None = None,
    *,
    units: str | None = "lg(re mg.m-3)",
    lazy: bool = True,
    **extra_attrs,
) -> xr.DataArray:
    """Build a small float32 swath-like DataArray with satpy-style attrs."""
    raw = np.asarray(LOG_VALUES if values is None else values, dtype=np.float32)
    data = da.from_array(raw, chunks=1) if lazy else raw
    attrs = dict(extra_attrs)
    if units is not None:
        attrs["units"] = units
    return xr.DataArray(data, dims=("y", "x"), attrs=attrs)


# --------------------------------------------------------------------------- #
# to_linear                                                                   #
# --------------------------------------------------------------------------- #


class TestToLinear:
    def test_log10_values_become_linear(self):
        out = reader.to_linear(_array())

        np.testing.assert_allclose(out.values, LINEAR_VALUES, rtol=1e-6)

    def test_equals_ten_to_the_power_of_input(self):
        arr = _array([[-1.8, -0.35], [0.72, 2.0]])

        out = reader.to_linear(arr)

        np.testing.assert_allclose(out.values, 10.0 ** arr.values, rtol=1e-6)

    @pytest.mark.parametrize(
        ("log_units", "linear_units"),
        [
            ("lg(re mg.m-3)", "mg.m-3"),
            ("lg(re g.m-3)", "g.m-3"),
            ("lg(re m-1)", "m-1"),
            # Tolerate spellings without the "re " reference or the
            # closing parenthesis rather than producing a mangled unit.
            ("lg(mg.m-3)", "mg.m-3"),
            ("lg(re mg.m-3", "mg.m-3"),
        ],
    )
    def test_units_are_the_inner_part(self, log_units, linear_units):
        out = reader.to_linear(_array(units=log_units))

        assert out.attrs["units"] == linear_units

    def test_all_other_attrs_preserved(self):
        area = object()
        arr = _array(
            area=area,
            platform_name="Sentinel-3A",
            sensor="olci",
            standard_name="mass_concentration_of_chlorophyll_a_in_sea_water",
            _satpy_id="chl_nn-id",
        )

        out = reader.to_linear(arr)

        assert out.attrs["area"] is area
        assert set(out.attrs) == set(arr.attrs)
        for key in ("platform_name", "sensor", "standard_name", "_satpy_id"):
            assert out.attrs[key] == arr.attrs[key]

    def test_log10_prefix_is_dropped_from_long_name(self):
        # satpy's description of the stored values; it ends up as the
        # GeoTIFF band description and must not call linear data log10.
        arr = _array(
            long_name="log10 scaled (Neural Net) Algal pigment concentration"
        )

        out = reader.to_linear(arr)

        assert out.attrs["long_name"] == "(Neural Net) Algal pigment concentration"

    @pytest.mark.parametrize(
        "long_name",
        ["Log10 Scaled Total suspended matter", "LOG10 SCALED Total suspended matter"],
    )
    def test_long_name_prefix_match_ignores_case(self, long_name):
        out = reader.to_linear(_array(long_name=long_name))

        assert out.attrs["long_name"] == "Total suspended matter"

    @pytest.mark.parametrize(
        "long_name",
        [
            "Algal pigment concentration",
            "Concentration (log10 scaled elsewhere)",
            "log10 scaledness",
            "",
        ],
    )
    def test_other_long_names_are_left_alone(self, long_name):
        out = reader.to_linear(_array(long_name=long_name))

        assert out.attrs["long_name"] == long_name

    def test_long_name_is_not_invented(self):
        out = reader.to_linear(_array())

        assert "long_name" not in out.attrs

    def test_non_string_long_name_is_left_alone(self):
        out = reader.to_linear(_array(long_name=7))

        assert out.attrs["long_name"] == 7

    def test_long_name_of_input_is_not_mutated(self):
        arr = _array(long_name="log10 scaled Algal pigment concentration")

        reader.to_linear(arr)

        assert arr.attrs["long_name"] == "log10 scaled Algal pigment concentration"

    def test_long_name_of_linear_input_is_not_touched(self):
        arr = _array(units="mg.m-3", long_name="log10 scaled something")

        out = reader.to_linear(arr)

        assert out is arr
        assert out.attrs["long_name"] == "log10 scaled something"

    def test_dims_and_coords_preserved(self):
        arr = _array().assign_coords(y=[1.0, 0.0], x=[5.0, 6.0])

        out = reader.to_linear(arr)

        assert out.dims == arr.dims
        np.testing.assert_array_equal(out["y"].values, [1.0, 0.0])
        np.testing.assert_array_equal(out["x"].values, [5.0, 6.0])

    def test_nan_stays_nan(self):
        out = reader.to_linear(_array())

        assert np.isnan(out.values[1, 1])

    def test_input_is_not_mutated(self):
        arr = _array(area=object())
        attrs_before = dict(arr.attrs)

        out = reader.to_linear(arr)
        out.attrs["added_later"] = True

        assert arr.attrs == attrs_before
        assert arr.attrs["units"] == "lg(re mg.m-3)"
        np.testing.assert_array_equal(
            arr.values, np.asarray(LOG_VALUES, dtype=np.float32)
        )

    def test_source_packing_encoding_is_not_carried_over(self):
        # The product packs log10 values as uint8 with a scale/offset.
        # Those parameters describe log10 data and would corrupt linear
        # values if an encoding-aware writer ever picked them up.
        packing = {
            "dtype": np.dtype("uint8"),
            "scale_factor": np.float32(0.015748031),
            "add_offset": np.float32(-2.0),
            "_FillValue": np.uint8(255),
        }
        arr = _array()
        arr.encoding = dict(packing)

        out = reader.to_linear(arr)

        assert out.encoding == {}
        assert arr.encoding == packing

    def test_returns_new_object_for_log_data(self):
        arr = _array()

        assert reader.to_linear(arr) is not arr

    def test_stays_float32(self):
        out = reader.to_linear(_array())

        assert out.dtype == np.float32
        assert out.values.dtype == np.float32

    def test_stays_lazy(self):
        computed: list[int] = []

        def _block(block: np.ndarray) -> np.ndarray:
            computed.append(1)
            return block

        raw = da.from_array(np.asarray(LOG_VALUES, dtype=np.float32), chunks=1)
        lazy = raw.map_blocks(_block, dtype=np.float32, meta=np.array((), np.float32))
        arr = xr.DataArray(lazy, dims=("y", "x"), attrs={"units": "lg(re mg.m-3)"})

        out = reader.to_linear(arr)

        assert isinstance(out.data, da.Array)
        assert computed == []
        np.testing.assert_allclose(out.values, LINEAR_VALUES, rtol=1e-6)
        assert computed

    def test_numpy_backed_input_also_converted(self):
        out = reader.to_linear(_array(lazy=False))

        np.testing.assert_allclose(out.values, LINEAR_VALUES, rtol=1e-6)
        assert out.dtype == np.float32

    @pytest.mark.parametrize("units", ["mg.m-3", "kg.m-2", "1", "", "%"])
    def test_non_log_units_left_untouched(self, units):
        arr = _array(LINEAR_VALUES, units=units, area=object())

        out = reader.to_linear(arr)

        assert out.attrs["units"] == units
        assert out.attrs["area"] is arr.attrs["area"]
        np.testing.assert_array_equal(out.values, arr.values)

    def test_missing_units_attr_left_untouched(self):
        # The WQSF flag layer carries no ``units`` at all; this is what
        # breaks satpy's own ``unlog`` option.
        arr = _array([[1.0, 2.0], [4.0, 8.0]], units=None, flag_masks=[1, 2])

        out = reader.to_linear(arr)

        assert "units" not in out.attrs
        np.testing.assert_array_equal(out.values, arr.values)

    def test_non_string_units_left_untouched(self):
        arr = _array(LINEAR_VALUES, units=None).assign_attrs(units=1)

        out = reader.to_linear(arr)

        assert out.attrs["units"] == 1
        np.testing.assert_array_equal(out.values, arr.values)


# --------------------------------------------------------------------------- #
# load_scene                                                                  #
# --------------------------------------------------------------------------- #


class _FakeScene:
    """Dict-backed stand-in for :class:`satpy.Scene`."""

    def __init__(
        self,
        contents: dict[str, xr.DataArray],
        available: list[str] | None = None,
    ) -> None:
        self._contents = dict(contents)
        self._available = list(contents) if available is None else list(available)
        self._loaded: dict[str, xr.DataArray] = {}
        self.load_calls: list[list[str]] = []

    def available_dataset_names(self) -> list[str]:
        return list(self._available)

    def load(self, names: list[str]) -> None:
        self.load_calls.append(list(names))
        for name in names:
            if name in self._contents:
                self._loaded[name] = self._contents[name]

    def __getitem__(self, name: str) -> xr.DataArray:
        return self._loaded[name]

    def __setitem__(self, name: str, value: xr.DataArray) -> None:
        self._loaded[name] = value

    def __contains__(self, name: object) -> bool:
        return name in self._loaded

    def keys(self) -> list[str]:
        return list(self._loaded)


def _wqsf() -> xr.DataArray:
    """Flag layer without a ``units`` attribute, as satpy delivers it."""
    return xr.DataArray(
        da.from_array(np.array([[0, 1], [2, 4]], dtype=np.uint64), chunks=1),
        dims=("y", "x"),
        attrs={"flag_masks": [1, 2, 4], "flag_meanings": "INVALID WATER CLOUD"},
    )


@pytest.fixture
def scene_factory(monkeypatch):
    """Patch ``Scene`` and file discovery; return an installer.

    The installer takes the fake scene to hand out and returns the list
    that records every ``Scene(...)`` keyword-argument dict.
    """

    def _install(scene: _FakeScene) -> list[dict]:
        calls: list[dict] = []

        def _factory(*args, **kwargs):
            calls.append({"args": args, **kwargs})
            return scene

        monkeypatch.setattr(reader, "Scene", _factory)
        monkeypatch.setattr(
            reader, "_collect_filenames", lambda _path: ["/fake/chl_nn.nc"]
        )
        return calls

    return _install


PRODUCT = Path("S3A_OL_2_WFR____20240315T091500_TEST.SEN3")


class TestLoadScene:
    def test_returns_the_scene(self, scene_factory):
        scene = _FakeScene({"chl_nn": _array(), "wqsf": _wqsf()})
        scene_factory(scene)

        assert load_scene(PRODUCT, ["chl_nn"]) is scene

    def test_scene_built_with_olci_l2_reader_and_filenames(self, scene_factory):
        scene = _FakeScene({"chl_nn": _array(), "wqsf": _wqsf()})
        calls = scene_factory(scene)

        load_scene(PRODUCT, ["chl_nn"])

        assert len(calls) == 1
        assert calls[0]["reader"] == "olci_l2"
        assert calls[0]["filenames"] == ["/fake/chl_nn.nc"]

    def test_does_not_ask_satpy_to_unlog(self, scene_factory):
        # satpy's ``unlog`` reader option breaks loading of ``wqsf``
        # (KeyError: 'units'), so the conversion must stay in our code.
        scene = _FakeScene({"chl_nn": _array(), "wqsf": _wqsf()})
        calls = scene_factory(scene)

        load_scene(PRODUCT, ["chl_nn"])

        reader_kwargs = calls[0].get("reader_kwargs") or {}
        assert "unlog" not in reader_kwargs

    def test_always_loads_wqsf(self, scene_factory):
        scene = _FakeScene({"chl_nn": _array(), "wqsf": _wqsf()})
        scene_factory(scene)

        out = load_scene(PRODUCT, ["chl_nn"])

        assert scene.load_calls == [["chl_nn", "wqsf"]]
        assert "wqsf" in out

    def test_log10_dataset_is_converted_to_linear(self, scene_factory):
        scene = _FakeScene({"chl_nn": _array(), "wqsf": _wqsf()})
        scene_factory(scene)

        out = load_scene(PRODUCT, ["chl_nn"])

        assert out["chl_nn"].attrs["units"] == "mg.m-3"
        np.testing.assert_allclose(out["chl_nn"].values, LINEAR_VALUES, rtol=1e-6)

    def test_every_requested_log10_dataset_is_converted(self, scene_factory):
        scene = _FakeScene(
            {
                "chl_nn": _array(),
                "chl_oc4me": _array(),
                "tsm_nn": _array(units="lg(re g.m-3)"),
                "wqsf": _wqsf(),
            }
        )
        scene_factory(scene)

        out = load_scene(PRODUCT, ["chl_nn", "chl_oc4me", "tsm_nn"])

        assert out["chl_nn"].attrs["units"] == "mg.m-3"
        assert out["chl_oc4me"].attrs["units"] == "mg.m-3"
        assert out["tsm_nn"].attrs["units"] == "g.m-3"
        for name in ("chl_nn", "chl_oc4me", "tsm_nn"):
            np.testing.assert_allclose(out[name].values, LINEAR_VALUES, rtol=1e-6)

    def test_converted_dataset_keeps_area_and_stays_lazy(self, scene_factory):
        area = object()
        scene = _FakeScene({"chl_nn": _array(area=area), "wqsf": _wqsf()})
        scene_factory(scene)

        out = load_scene(PRODUCT, ["chl_nn"])

        assert out["chl_nn"].attrs["area"] is area
        assert isinstance(out["chl_nn"].data, da.Array)
        assert out["chl_nn"].dtype == np.float32

    def test_wqsf_is_not_touched(self, scene_factory):
        wqsf = _wqsf()
        scene = _FakeScene({"chl_nn": _array(), "wqsf": wqsf})
        scene_factory(scene)

        out = load_scene(PRODUCT, ["chl_nn"])

        assert out["wqsf"] is wqsf
        assert out["wqsf"].dtype == np.uint64
        assert out["wqsf"].attrs["flag_masks"] == [1, 2, 4]

    def test_non_log_dataset_is_not_touched(self, scene_factory):
        iwv = _array(LINEAR_VALUES, units="kg.m-2")
        scene = _FakeScene({"iwv": iwv, "wqsf": _wqsf()})
        scene_factory(scene)

        out = load_scene(PRODUCT, ["iwv"])

        assert out["iwv"] is iwv

    def test_source_array_is_not_mutated(self, scene_factory):
        source = _array()
        scene = _FakeScene({"chl_nn": source, "wqsf": _wqsf()})
        scene_factory(scene)

        load_scene(PRODUCT, ["chl_nn"])

        assert source.attrs["units"] == "lg(re mg.m-3)"

    def test_unavailable_dataset_is_warned_about_and_skipped(
        self, scene_factory, caplog
    ):
        scene = _FakeScene({"chl_nn": _array(), "wqsf": _wqsf()})
        scene_factory(scene)

        with caplog.at_level(logging.WARNING, logger=reader.__name__):
            out = load_scene(PRODUCT, ["chl_nn", "does_not_exist"])

        assert scene.load_calls == [["chl_nn", "wqsf"]]
        assert "does_not_exist" not in out
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert any("does_not_exist" in r.getMessage() for r in warnings)

    def test_raises_when_no_requested_dataset_is_available(self, scene_factory):
        scene = _FakeScene({"chl_nn": _array(), "wqsf": _wqsf()})
        scene_factory(scene)

        with pytest.raises(ValueError, match="None of the requested datasets"):
            load_scene(PRODUCT, ["nope", "also_nope"])

        assert scene.load_calls == []

    def test_dataset_that_failed_to_load_is_skipped(self, scene_factory):
        # satpy lists the name as available but could not produce it.
        scene = _FakeScene(
            {"chl_nn": _array(), "wqsf": _wqsf()},
            available=["chl_nn", "tsm_nn", "wqsf"],
        )
        scene_factory(scene)

        out = load_scene(PRODUCT, ["chl_nn", "tsm_nn"])

        assert "tsm_nn" not in out
        assert out["chl_nn"].attrs["units"] == "mg.m-3"

    def test_empty_product_directory_raises(self, tmp_path, monkeypatch):
        product = tmp_path / "S3A_EMPTY.SEN3"
        product.mkdir()
        built: list[dict] = []
        monkeypatch.setattr(reader, "Scene", lambda **kw: built.append(kw))

        with pytest.raises(FileNotFoundError):
            load_scene(product, ["chl_nn"])

        assert built == []


# --------------------------------------------------------------------------- #
# extract_sensing_time / extract_satellite                                    #
# --------------------------------------------------------------------------- #


class TestExtractSensingTime:
    def test_parses_sensing_start(self):
        out = extract_sensing_time(REAL_PRODUCT)

        assert out == datetime(2024, 3, 10, 8, 59, 3, tzinfo=timezone.utc)

    def test_result_is_utc_aware(self):
        out = extract_sensing_time(Path("S3B_OL_2_WFR____20240315T091500_X.SEN3"))

        assert out.tzinfo == timezone.utc

    def test_first_timestamp_wins(self):
        name = "S3A_OL_2_WFR____20240315T091500_20240315T091800_20240316T120000.SEN3"

        assert extract_sensing_time(Path(name)).hour == 9
        assert extract_sensing_time(Path(name)).minute == 15

    def test_only_directory_name_is_used(self):
        path = Path("/data/20990101T000000/S3A_OL_2_WFR____20240315T091500_X.SEN3")

        assert extract_sensing_time(path).year == 2024

    def test_malformed_token_is_skipped(self):
        # Right length, starts with "2" and contains "T", but not a date.
        name = "S3A_OL_2_WFR____2024ABCDTXXXXXX_20240315T091500_X.SEN3"

        assert extract_sensing_time(Path(name)) == datetime(
            2024, 3, 15, 9, 15, tzinfo=timezone.utc
        )

    @pytest.mark.parametrize(
        "name", ["", "product.SEN3", "S3A_OL_2_WFR____2024ABCDTXXXXXX.SEN3"]
    )
    def test_raises_when_no_timestamp(self, name):
        with pytest.raises(ValueError, match="Could not parse sensing time"):
            extract_sensing_time(Path(name))


class TestExtractSatellite:
    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("S3A_OL_2_WFR____20240315T091500.SEN3", "S3A"),
            ("S3B_OL_2_WFR____20240315T091500.SEN3", "S3B"),
            ("S3C_OL_2_WFR____20240315T091500.SEN3", "S3X"),
            ("something_else", "S3X"),
        ],
    )
    def test_satellite_tag(self, name, expected):
        assert extract_satellite(Path(name)) == expected

    def test_only_directory_name_is_used(self):
        assert extract_satellite(Path("/S3B/raw/S3A_OL_2_WFR.SEN3")) == "S3A"


# --------------------------------------------------------------------------- #
# _collect_filenames                                                          #
# --------------------------------------------------------------------------- #


class TestCollectFilenames:
    def test_returns_sorted_nc_files_as_strings(self, tmp_path):
        for name in ("wqsf.nc", "chl_nn.nc", "geo_coordinates.nc"):
            (tmp_path / name).touch()

        out = _collect_filenames(tmp_path)

        assert all(isinstance(f, str) for f in out)
        assert [Path(f).name for f in out] == [
            "chl_nn.nc",
            "geo_coordinates.nc",
            "wqsf.nc",
        ]

    def test_ignores_non_nc_files(self, tmp_path):
        (tmp_path / "chl_nn.nc").touch()
        (tmp_path / "xfdumanifest.xml").touch()
        (tmp_path / "notes.nc.txt").touch()

        out = _collect_filenames(tmp_path)

        assert [Path(f).name for f in out] == ["chl_nn.nc"]

    def test_paths_point_into_the_product_directory(self, tmp_path):
        (tmp_path / "chl_nn.nc").touch()

        out = _collect_filenames(tmp_path)

        assert Path(out[0]).parent == tmp_path.resolve()

    def test_empty_directory_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            _collect_filenames(tmp_path)

    def test_directory_without_nc_files_raises(self, tmp_path):
        (tmp_path / "xfdumanifest.xml").touch()

        with pytest.raises(FileNotFoundError):
            _collect_filenames(tmp_path)


# --------------------------------------------------------------------------- #
# Real product (slow, needs data/raw)                                         #
# --------------------------------------------------------------------------- #


@pytest.mark.integration
@pytest.mark.skipif(
    not REAL_PRODUCT.is_dir(), reason=f"real product not available: {REAL_PRODUCT}"
)
class TestLoadSceneRealProduct:
    def test_chl_nn_is_linear_and_wqsf_survives(self):
        scene = load_scene(REAL_PRODUCT, ["chl_nn"])

        chl = scene["chl_nn"]
        assert chl.attrs["units"] == "mg.m-3"
        assert "area" in chl.attrs
        assert chl.dtype == np.float32

        values = np.asarray(chl.values)
        finite = values[np.isfinite(values)]
        assert finite.size > 0
        assert finite.min() > 0.0
        assert finite.max() <= 100.0

        assert "wqsf" in scene
        assert "flag_masks" in scene["wqsf"].attrs
