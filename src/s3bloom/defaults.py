"""Pipeline-wide defaults: bounding-box presets, WQSF masking flag tables,
default projection/resolution, colormap settings and valid data ranges.

This module is the single source of truth for tunable constants. Anything
that a user might reasonably want to change without editing application
logic lives here.

Notes
-----
Masking flags are split per the EUMETSAT Matchup Protocols v8B (Appendix A,
Table 1) into three categories:

1. *Common flags* — apply to every Ocean Colour product. The strictness
   preset (``strict`` / ``moderate`` / ``relaxed``) selects which of these
   are used.
2. *Processing-chain flags* — only meaningful for products processed with
   the **Baseline Atmospheric Correction** (BAC, Open Water case), e.g.
   ``chl_oc4me``. They are *not* applied to AAC products such as
   ``chl_nn`` / ``tsm_nn``.
3. *Product flags* — algorithm-failure flags specific to each retrieval
   (e.g. ``OC4ME_FAIL``, ``OCNN_FAIL``). These are always applied.

See :func:`get_masking_flags` for the combination logic.
"""

from __future__ import annotations

# Geographic regions that the pipeline's CLI accepts as ``--bbox <name>``.
# Tuples are ``(lon_min, lat_min, lon_max, lat_max)`` in WGS84 (EPSG:4326).
# Add new entries here to expose them to the CLI automatically.
BBOX_PRESETS: dict[str, tuple[float, float, float, float]] = {
    "swedish_west_coast": (10.0, 56.5, 13.0, 59.0),
    "kattegat": (10.0, 55.5, 13.0, 58.0),
    "skagerrak": (7.0, 57.0, 12.0, 59.5),
    "kattegat_skagerrak": (7.0, 55.5, 13.5, 59.5),
    "kattegat_skagerrak_extended": (5.0, 55.5, 13.5, 60.0),
    # Egentliga Östersjön (HELCOM Baltic Proper sub-basins: Arkona, Bornholm,
    # Gdansk, Eastern/Western Gotland, Northern Baltic Proper). Excludes
    # Bothnian Sea/Bay, Gulf of Finland, Gulf of Riga.
    "baltic_proper": (13.0, 54.0, 23.0, 60.0),
    # Whole Baltic Sea incl. Bothnian Sea/Bay, Gulfs of Finland and Riga.
    "baltic_all": (13.0, 54.0, 30.0, 66.0),
}

# Common flags by strictness — applied to every product. See module docstring.
# Keep ordered roughly by impact on coverage so the diff between presets is
# easy to read.
_COMMON_FLAGS: dict[str, list[str]] = {
    "strict": [
        "CLOUD",
        "CLOUD_AMBIGUOUS",
        "CLOUD_MARGIN",
        "INVALID",
        "COSMETIC",
        "SATURATED",
        "SUSPECT",
        "HISOLZEN",
        "HIGHGLINT",
        "SNOW_ICE",
    ],
    "moderate": [
        "CLOUD",
        "CLOUD_AMBIGUOUS",
        "INVALID",
        "COSMETIC",
        "SATURATED",
        "SUSPECT",
        "SNOW_ICE",
    ],
    "relaxed": [
        "CLOUD",
        "INVALID",
        "SATURATED",
        "SNOW_ICE",
    ],
}

# BAC (Baseline Atmospheric Correction) processing-chain flags. Only attached
# to products in :data:`BAC_PRODUCTS`. Includes the per-band negative
# water-leaving reflectance flags (``RWNEG_O2``..``RWNEG_O8``) for the bands
# the OC4Me ratio uses.
_BAC_FLAGS: dict[str, list[str]] = {
    "strict": [
        "AC_FAIL",
        "WHITECAPS",
        "ADJAC",
        "RWNEG_O2",
        "RWNEG_O3",
        "RWNEG_O4",
        "RWNEG_O5",
        "RWNEG_O6",
        "RWNEG_O7",
        "RWNEG_O8",
    ],
    "moderate": [
        "AC_FAIL",
        "WHITECAPS",
    ],
    "relaxed": [],
}

# Products processed with the Baseline Atmospheric Correction (Open Water)
BAC_PRODUCTS: frozenset[str] = frozenset({"chl_oc4me"})

# Per-product algorithm failure flags (applied regardless of strictness)
PRODUCT_FLAGS: dict[str, list[str]] = {
    "chl_oc4me": ["OC4ME_FAIL"],
    "chl_nn": ["OCNN_FAIL"],
    "tsm_nn": ["OCNN_FAIL"],
    "iop_nn": ["OCNN_FAIL"],
}


def get_masking_flags(preset: str, product: str) -> list[str]:
    """Return the WQSF flag list for a product at a given strictness.

    The returned list is the union of common flags (strictness-dependent),
    BAC processing-chain flags (only for Open Water products such as
    ``chl_oc4me``), and the product's algorithm-failure flag.

    Parameters
    ----------
    preset : {"strict", "moderate", "relaxed"}
        Strictness preset name. Controls which *common* flags are applied.
    product : str
        Dataset name, e.g. ``"chl_nn"`` or ``"chl_oc4me"``. Selects which
        processing-chain and per-product flags are attached.

    Returns
    -------
    list of str
        Flag names suitable for matching against the product's
        ``flag_meanings`` attribute. The list may contain entries that are
        not present in a specific product's metadata; the masking layer
        will warn and skip those.

    Raises
    ------
    ValueError
        If ``preset`` is not a known strictness preset.

    Notes
    -----
    Reference: EUMETSAT Matchup Protocols v8B, Appendix A, Table 1.
    """
    if preset not in _COMMON_FLAGS:
        raise ValueError(
            f"Unknown masking preset: {preset!r}. "
            f"Choose from: {', '.join(_COMMON_FLAGS)}"
        )

    flags = list(_COMMON_FLAGS[preset])

    if product in BAC_PRODUCTS:
        flags.extend(_BAC_FLAGS[preset])

    flags.extend(PRODUCT_FLAGS.get(product, []))

    return flags


# Public copy of the common-flag table. Used by CLI validation and the
# ``list-presets`` command. ``_COMMON_FLAGS`` stays private so that callers
# go through :func:`get_masking_flags` and pick up product-aware logic.
MASKING_PRESETS: dict[str, list[str]] = {
    name: list(flags) for name, flags in _COMMON_FLAGS.items()
}

# Strict is the default because for bloom monitoring a false negative
# (keeping a bad pixel) corrupts composites silently, while a false positive
# (masking a good pixel) is recoverable through compositing.
DEFAULT_MASKING = "strict"

DEFAULT_DATASETS: list[str] = ["chl_nn"]

# EPSG:3035 (ETRS89 / LAEA Europe): equal-area, low distortion at Nordic
# latitudes, INSPIRE-standard for EU environmental reporting.
DEFAULT_PROJECTION = "EPSG:3035"

# Native OLCI Full Resolution pixel is ~300 m; resampling to 300 m avoids
# both up- and down-sampling artefacts.
DEFAULT_RESOLUTION = 300  # metres

COMPOSITE_WINDOW_DAYS = 3

# Per-dataset display and unit defaults.
#
# Every array the pipeline handles after loading holds LINEAR values
# (mg m-3, g m-3): the reader undoes the log10 encoding of the OLCI L2
# products, and the GeoTIFF, NetCDF and PNG outputs all carry the same
# linear values. Nothing in this table changes the data.
#   cmap        : matplotlib/cmocean colormap name
#   vmin / vmax : colour-bar limits of the PNG maps, in linear units.
#                 Display only — values outside are clipped on the map
#                 but are still valid data; see VALID_RANGE for the
#                 physical range.
#   log_scale   : apply LogNorm to the colour bar (keeps low values visible)
#   label       : colour-bar label (display text)
#   units       : CF ``units`` attribute written to NetCDF, plain ASCII
COLORMAP_SETTINGS: dict[str, dict] = {
    "chl_nn": {
        "cmap": "cmo.algae",
        "vmin": 0.0,
        "vmax": 30.0,
        "log_scale": False,
        "label": "Chl-a (mg m⁻³)",
        "units": "mg m-3",
    },
    "chl_oc4me": {
        "cmap": "cmo.algae",
        "vmin": 0.0,
        "vmax": 30.0,
        "log_scale": False,
        "label": "Chl-a OC4Me (mg m⁻³)",
        "units": "mg m-3",
    },
    "tsm_nn": {
        "cmap": "cmo.turbid",
        "vmin": 0.5,
        "vmax": 50.0,
        "log_scale": False,
        "label": "TSM (g m⁻³)",
        "units": "g m-3",
    },
}

# Physically valid ``(min, max)`` per dataset, in the linear units above.
# Written as CF ``valid_min`` / ``valid_max`` by the NetCDF exporter and
# deliberately separate from the colour-bar limits. 100 mg m-3 is the
# upper bound of the OLCI L2 chlorophyll encoding. Datasets without an
# entry get no valid-range attributes.
VALID_RANGE: dict[str, tuple[float, float]] = {
    "chl_nn": (0.0, 100.0),
    "chl_oc4me": (0.0, 100.0),
    "tsm_nn": (0.0, 1000.0),
}

# CDSE rate-limits aggressive downloaders. Two parallel streams is the
# largest value that has been observed to be reliably accepted; do not
# raise this without confirming current CDSE policy.
MAX_PARALLEL_DOWNLOADS = 2
