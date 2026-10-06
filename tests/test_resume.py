"""Per-pass processing and compositing together.

Covers the hand-off between :func:`process_single_pass` and
:func:`create_composites`: results must not depend on the raw product,
and a resumed run must composite exactly what a fresh run composites.
The load/mask/resample stages are replaced by synthetic arrays; the
exporters and the files they write are real.
"""

from __future__ import annotations

import shutil
import warnings
from datetime import date
from pathlib import Path

import numpy as np
import pytest
import rasterio
import xarray as xr

from s3bloom.compositing.temporal import create_composites
from s3bloom.config import BoundingBox, PipelineConfig, TimePeriod
from s3bloom.metadata.provenance import read_output_attrs
from s3bloom.processing.pipeline import PassResult, process_single_pass

PRODUCTS = [
    "S3A_OL_2_WFR____20240315T091500_20240315T091800_20240316T173014"
    "_0179_110_122_1980_MAR_O_NT_003.SEN3",
    "S3B_OL_2_WFR____20240315T101000_20240315T101300_20240316T180000"
    "_0179_091_122_1980_MAR_O_NT_003.SEN3",
    "S3A_OL_2_WFR____20240316T094400_20240316T094700_20240317T173226"
    "_0179_110_136_1980_MAR_O_NT_003.SEN3",
    "S3B_OL_2_WFR____20240317T100000_20240317T100300_20240318T160039"
    "_0180_091_150_1980_MAR_O_NT_003.SEN3",
]
SHAPE = (20, 30)


def _config(base_dir: Path, formats: list[str]) -> PipelineConfig:
    return PipelineConfig(
        bbox=BoundingBox(lon_min=10.0, lat_min=56.5, lon_max=13.0, lat_max=59.0),
        time_period=TimePeriod(start_date=date(2024, 3, 15), end_date=date(2024, 3, 17)),
        datasets=["chl_nn"],
        output={"base_dir": base_dir, "formats": formats, "composite_window_days": 3},
    )


def _scene_array(seed: int) -> xr.DataArray:
    """Log-normal "chlorophyll" on a 300 m grid with a cloudy quarter."""
    rng = np.random.default_rng(seed)
    values = rng.lognormal(mean=0.3, sigma=1.2, size=SHAPE).astype(np.float32)
    values[rng.random(SHAPE) < 0.25] = np.nan
    return xr.DataArray(
        values,
        dims=("y", "x"),
        coords={
            "y": 3_600_000.0 - 300.0 * np.arange(SHAPE[0]),
            "x": 4_300_000.0 + 300.0 * np.arange(SHAPE[1]),
        },
        attrs={"units": "mg.m-3"},
    )


@pytest.fixture
def sources(fake_stages, lazy_source):
    """One lazily-read synthetic scene per product in :data:`PRODUCTS`."""
    by_product = {
        name: lazy_source(_scene_array(seed)) for seed, name in enumerate(PRODUCTS)
    }
    for name, source in by_product.items():
        fake_stages.resampled[name] = {"chl_nn": source.dataarray()}
    return by_product


def _process(
    products: list[str], config: PipelineConfig, sources: dict, raw_dir: Path
) -> list[PassResult]:
    """Process *products* like the CLI does with ``--delete-raw``."""
    results = []
    for name in products:
        result = process_single_pass(raw_dir / name, config)
        results.append(result)
        if not result.skipped:
            # The .SEN3 directory is removed as soon as the pass is done.
            sources[name].available = False
    return results


def _composite_bytes(base_dir: Path) -> dict[str, bytes]:
    """Raw array bytes of every composite raster below *base_dir*."""
    found: dict[str, bytes] = {}
    for path in sorted((base_dir / "composites").rglob("*.nc")):
        with xr.open_dataset(path) as ds:
            found[path.name] = ds["chl_nn"].values.tobytes()
    for path in sorted((base_dir / "composites").rglob("*.tif")):
        with rasterio.open(path) as src:
            found[path.name] = src.read(1).tobytes()
    return found


def _source_products(base_dir: Path) -> dict[str, str]:
    return {
        path.name: read_output_attrs(path)["s3bloom_source_products"]
        for path in sorted((base_dir / "composites").rglob("*.nc"))
    }


class TestCompositesWithoutRawProducts:
    @pytest.mark.parametrize(
        "formats", [["netcdf", "geotiff"], ["netcdf"], ["geotiff"]]
    )
    def test_compositing_works_after_raw_products_are_gone(
        self, tmp_path, sources, formats
    ):
        config = _config(tmp_path / "out", formats)

        with xr.set_options(file_cache_maxsize=1):
            results = _process(PRODUCTS, config, sources, tmp_path / "raw")
            assert not any(source.available for source in sources.values())
            files = create_composites(results, config)

        # 3 dates × requested formats
        assert len(files) == 3 * len(formats)
        assert all(path.exists() for path in files)

    def test_composite_values_are_the_geometric_mean_of_the_passes(
        self, tmp_path, sources
    ):
        config = _config(tmp_path / "out", ["netcdf"])

        with xr.set_options(file_cache_maxsize=1):
            results = _process(PRODUCTS, config, sources, tmp_path / "raw")
            create_composites(results, config)

        # 2024-03-16 ± 1 day sees all four passes.
        stack = np.stack([_scene_array(seed).values for seed in range(4)])
        with warnings.catch_warnings():
            # Pixels that are cloudy in all four passes: mean of nothing.
            warnings.simplefilter("ignore", RuntimeWarning)
            expected = 10.0 ** np.nanmean(np.log10(stack.astype(np.float64)), axis=0)
        path = (
            config.output.base_dir
            / "composites/netcdf/s3bloom_chl_nn_composite3d_20240316.nc"
        )
        with xr.open_dataset(path) as ds:
            np.testing.assert_allclose(ds["chl_nn"].values, expected, rtol=1e-5)

    def test_each_raw_product_is_read_exactly_once(self, tmp_path, sources):
        config = _config(tmp_path / "out", ["netcdf", "geotiff"])

        results = _process(PRODUCTS, config, sources, tmp_path / "raw")
        create_composites(results, config)

        assert [source.reads for source in sources.values()] == [1, 1, 1, 1]


class TestResumedRun:
    @pytest.mark.parametrize("formats", [["netcdf"], ["netcdf", "geotiff"]])
    def test_resumed_run_builds_byte_identical_composites(
        self, tmp_path, sources, fake_stages, formats
    ):
        config = _config(tmp_path / "out", formats)
        fresh_results = _process(PRODUCTS, config, sources, tmp_path / "raw")
        fresh_files = create_composites(fresh_results, config)
        fresh = _composite_bytes(config.output.base_dir)
        fresh_sources = _source_products(config.output.base_dir)
        del fresh_results
        shutil.rmtree(config.output.base_dir / "composites")

        # Second invocation: every per-pass output already exists.
        resumed_results = _process(PRODUCTS, config, sources, tmp_path / "raw")
        resumed_files = create_composites(resumed_results, config)

        assert all(result.skipped for result in resumed_results)
        assert fake_stages.loads == PRODUCTS
        assert len(fresh) == 3 * len(formats)
        assert sorted(resumed_files) == sorted(fresh_files)
        assert _composite_bytes(config.output.base_dir) == fresh
        assert _source_products(config.output.base_dir) == fresh_sources

    def test_resumed_run_leaves_current_composites_alone(self, tmp_path, sources):
        config = _config(tmp_path / "out", ["netcdf", "geotiff"])
        files = create_composites(
            _process(PRODUCTS, config, sources, tmp_path / "raw"), config
        )
        mtimes = {path: path.stat().st_mtime_ns for path in files}

        resumed = create_composites(
            _process(PRODUCTS, config, sources, tmp_path / "raw"), config
        )

        assert resumed == []
        assert mtimes == {path: path.stat().st_mtime_ns for path in mtimes}

    @pytest.mark.parametrize("formats", [["netcdf"], ["netcdf", "geotiff"]])
    def test_interrupted_then_resumed_run_equals_a_fresh_run(
        self, tmp_path, sources, fake_stages, lazy_source, formats
    ):
        # Run 1 got through two passes only; run 2 sees all four, with
        # the already-processed ones handed over last.
        resumed_cfg = _config(tmp_path / "resumed", formats)
        create_composites(
            _process(PRODUCTS[:2], resumed_cfg, sources, tmp_path / "raw"), resumed_cfg
        )
        results = _process(
            PRODUCTS[2:] + PRODUCTS[:2], resumed_cfg, sources, tmp_path / "raw"
        )
        create_composites(results, resumed_cfg)

        # Reference: everything in one go into an empty directory.
        for seed, name in enumerate(PRODUCTS):
            fake_stages.resampled[name] = {
                "chl_nn": lazy_source(_scene_array(seed)).dataarray()
            }
        fresh_cfg = _config(tmp_path / "fresh", formats)
        create_composites(
            [process_single_pass(tmp_path / "raw" / name, fresh_cfg) for name in PRODUCTS],
            fresh_cfg,
        )

        assert [result.skipped for result in results] == [False, False, True, True]
        resumed = _composite_bytes(resumed_cfg.output.base_dir)
        assert len(resumed) == 3 * len(formats)
        assert resumed == _composite_bytes(fresh_cfg.output.base_dir)
        assert _source_products(resumed_cfg.output.base_dir) == _source_products(
            fresh_cfg.output.base_dir
        )

    def test_resumed_run_without_netcdf_cannot_composite_skipped_passes(
        self, tmp_path, sources
    ):
        config = _config(tmp_path / "out", ["geotiff"])
        files = create_composites(
            _process(PRODUCTS, config, sources, tmp_path / "raw"), config
        )
        mtimes = {path: path.stat().st_mtime_ns for path in files}

        results = _process(PRODUCTS, config, sources, tmp_path / "raw")
        resumed = create_composites(results, config)

        assert all(result.skipped and not result.datasets for result in results)
        assert resumed == []
        assert mtimes == {path: path.stat().st_mtime_ns for path in mtimes}
