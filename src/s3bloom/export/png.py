"""Quick-look PNG maps with cartopy.

Each PNG is a publication-quality map: data layer on top, Natural Earth
10 m land polygon below it, thin coastline trace, optional graticule,
and a colorbar matching the dataset's recommended palette
(``cmocean.algae`` for chlorophyll, ``cmocean.turbid`` for TSM).

The values are plotted exactly as they are passed in: linear
concentrations, the same numbers the GeoTIFF and NetCDF outputs hold.

The ``Agg`` matplotlib backend is forced at import time to keep this
module headless / display-free (CI, server, etc.).
"""

from __future__ import annotations

import logging
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pyproj
import xarray as xr

matplotlib.use("Agg")

import cmocean  # noqa: E402, F401 -- registers colormaps

from s3bloom.defaults import COLORMAP_SETTINGS
from s3bloom.metadata.provenance import Provenance

logger = logging.getLogger(__name__)

_LAND_FEATURE = cfeature.NaturalEarthFeature(
    "physical", "land", "10m", facecolor="#f0e6d2", edgecolor="none"
)
_COASTLINE_FEATURE = cfeature.NaturalEarthFeature(
    "physical", "coastline", "10m", facecolor="none", edgecolor="#444444", linewidth=0.6
)


def _projection_from_epsg(epsg_str: str) -> ccrs.Projection:
    """Convert an ``"EPSG:nnnn"`` string to a cartopy projection.

    Lambert Azimuthal Equal Area (used by EPSG:3035, the default) gets
    a fully-parameterised :class:`cartopy.crs.LambertAzimuthalEqualArea`
    so that gridlines render correctly. All other CRSs go through
    :func:`cartopy.crs.epsg`, which works for most projected CRSs but
    can fail on niche ones.
    """
    code = int(epsg_str.split(":")[-1])
    proj = pyproj.CRS.from_epsg(code)
    cf = proj.to_cf()

    if "lambert_azimuthal_equal_area" in cf.get("grid_mapping_name", ""):
        return ccrs.LambertAzimuthalEqualArea(
            central_longitude=cf["longitude_of_projection_origin"],
            central_latitude=cf["latitude_of_projection_origin"],
            false_easting=cf.get("false_easting", 0),
            false_northing=cf.get("false_northing", 0),
        )

    # Fallback for other projections
    return ccrs.epsg(code)


def export_png(
    data: xr.DataArray,
    path: Path,
    provenance: Provenance,
    dataset_name: str,
) -> Path:
    """Render *data* as a quick-look PNG map.

    Parameters
    ----------
    data : xarray.DataArray
        Resampled array on the target grid, in linear units; must
        have ``x``/``y`` projected coordinates. Plotted as is.
    path : pathlib.Path
        Output path.
    provenance : Provenance
        Used for the title and to derive the cartopy CRS.
    dataset_name : str
        Used to look up the colormap, colour-bar limits and label in
        :data:`s3bloom.defaults.COLORMAP_SETTINGS`.

    Returns
    -------
    pathlib.Path
        The path that was written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    color_cfg = COLORMAP_SETTINGS.get(
        dataset_name,
        {"cmap": "viridis", "vmin": 0, "vmax": 100, "log_scale": False, "label": dataset_name},
    )

    values = data.values if hasattr(data, "values") else np.array(data)
    if hasattr(values, "compute"):
        values = values.compute()
    values = np.asarray(values, dtype=np.float64)

    data_crs = _projection_from_epsg(provenance.projection)

    fig, ax = plt.subplots(
        1, 1, figsize=(10, 8),
        subplot_kw={"projection": data_crs},
    )

    norm = _create_norm(color_cfg)
    cmap = plt.get_cmap(color_cfg["cmap"]).copy()
    # Render NaN (= masked or land) pixels in neutral gray instead of
    # making them transparent — preserves the dataset's footprint.
    cmap.set_bad(color="#d9d9d9")

    y_vals = np.asarray(data.coords.get("y", np.arange(values.shape[0])))
    x_vals = np.asarray(data.coords.get("x", np.arange(values.shape[1])))

    im = ax.pcolormesh(
        x_vals,
        y_vals,
        values,
        cmap=cmap,
        norm=norm,
        shading="auto",
        transform=data_crs,
    )

    ax.add_feature(_LAND_FEATURE, zorder=2)
    ax.add_feature(_COASTLINE_FEATURE, zorder=3)

    gl = ax.gridlines(
        draw_labels=True,
        linewidth=0.4,
        color="gray",
        alpha=0.5,
        linestyle="--",
    )
    gl.top_labels = False
    gl.right_labels = False

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label(color_cfg.get("label", dataset_name), fontsize=11)

    ax.set_title(_build_title(provenance, dataset_name), fontsize=12, pad=10)

    fig.tight_layout()
    fig.savefig(str(path), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    logger.info("PNG exported: %s", path)
    return path


def _create_norm(color_cfg: dict) -> mcolors.Normalize:
    """Build a matplotlib norm — log when ``log_scale`` is set, else linear."""
    if color_cfg.get("log_scale"):
        return mcolors.LogNorm(
            vmin=color_cfg["vmin"],
            vmax=color_cfg["vmax"],
        )
    return mcolors.Normalize(
        vmin=color_cfg["vmin"],
        vmax=color_cfg["vmax"],
    )


def _build_title(provenance: Provenance, dataset_name: str) -> str:
    """Compose a two-line title summarising the data shown."""
    time_str = provenance.sensing_time.strftime("%Y-%m-%d %H:%M UTC")

    if provenance.composite_window_days:
        return (
            f"{dataset_name} — {provenance.composite_window_days}-day composite\n"
            f"{time_str} | {provenance.satellite} | "
            f"Mask: {provenance.masking_preset}"
        )

    return (
        f"{dataset_name} — {provenance.satellite}\n"
        f"{time_str} | Mask: {provenance.masking_preset}"
    )
