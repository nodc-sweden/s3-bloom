"""Shared pytest fixtures."""
from datetime import datetime, timezone

import dask
import dask.array as da
import numpy as np
import pytest
import xarray as xr

from s3bloom.metadata.provenance import Provenance
from s3bloom.processing import pipeline


class _FakeScene(dict):
    """Empty scene that remembers which product it was loaded from."""

    def __init__(self, product_name: str) -> None:
        super().__init__()
        self.product_name = product_name


class FakeStages:
    """Stand-in for the load → mask → resample stages of a pass.

    ``resampled`` maps a product directory name to the
    ``{dataset: DataArray}`` dict that resampling "produces" for it;
    ``loads`` lists the products whose scene was actually loaded, i.e.
    the passes that were processed rather than skipped.
    """

    def __init__(self) -> None:
        self.resampled: dict[str, dict[str, xr.DataArray]] = {}
        self.loads: list[str] = []

    def load_scene(self, product_path, datasets):
        self.loads.append(product_path.name)
        return _FakeScene(product_path.name)

    def resample_scene(self, scene, target_area, datasets):
        return dict(self.resampled[scene.product_name])


class LazySource:
    """Dask-backed stand-in for an array that is read from a raw product.

    Every computation of :meth:`dataarray`'s graph calls :meth:`read`,
    which counts the call, or raises once the source was made
    unavailable (the ``.SEN3`` directory was deleted).
    """

    def __init__(self, template: xr.DataArray) -> None:
        self._template = template
        self._values = np.array(template.values)
        self.reads = 0
        self.available = True

    def read(self) -> np.ndarray:
        if not self.available:
            raise FileNotFoundError("raw product is gone")
        self.reads += 1
        return self._values

    def dataarray(self) -> xr.DataArray:
        lazy = da.from_delayed(
            dask.delayed(self.read, pure=False)(),
            shape=self._values.shape,
            dtype=self._values.dtype,
        )
        return self._template.copy(data=lazy)


@pytest.fixture
def lazy_source():
    """Factory: wrap a DataArray's values in a :class:`LazySource`."""
    return LazySource


@pytest.fixture
def fake_stages(monkeypatch):
    """Run ``process_single_pass`` on synthetic arrays, real exporters."""
    stages = FakeStages()
    monkeypatch.setattr(pipeline, "load_scene", stages.load_scene)
    monkeypatch.setattr(pipeline, "create_target_area", lambda bbox, output: None)
    monkeypatch.setattr(pipeline, "resample_scene", stages.resample_scene)
    return stages


@pytest.fixture
def sensing_time():
    return datetime(2024, 3, 15, 9, 15, 0, tzinfo=timezone.utc)


@pytest.fixture
def provenance(sensing_time):
    return Provenance(
        source_product="S3A_OL_2_WFR____20240315T091500.SEN3",
        satellite="S3A",
        sensing_time=sensing_time,
        dataset="chl_nn",
        masking_preset="strict",
        masking_flags=("CLOUD", "INVALID"),
        projection="EPSG:3035",
        resolution_m=300,
        pipeline_version="0.1.0",
    )


@pytest.fixture
def composite_provenance(sensing_time):
    return Provenance(
        source_product="composite_20240315",
        satellite="S3A-S3B",
        sensing_time=sensing_time,
        dataset="chl_nn",
        masking_preset="strict",
        masking_flags=("CLOUD", "INVALID"),
        projection="EPSG:3035",
        resolution_m=300,
        pipeline_version="0.1.0",
        source_products=("prod1.SEN3", "prod2.SEN3"),
        composite_window_days=3,
        pass_count=2,
    )


@pytest.fixture
def simple_dataarray():
    """Small 4×6 2D DataArray with LAEA-projected x/y coords.

    Coordinates are pixel centres on an exact 300 m grid whose outer
    edge starts at (4 000 000, 3 001 200); y runs north to south.
    """
    rng = np.random.default_rng(42)
    data = rng.random((4, 6)).astype(np.float32)
    data[0, 0] = np.nan
    x = 4_000_150.0 + 300.0 * np.arange(6)
    y = 3_001_050.0 - 300.0 * np.arange(4)
    return xr.DataArray(
        data,
        dims=["y", "x"],
        coords={"y": y, "x": x},
        name="chl_nn",
    )
