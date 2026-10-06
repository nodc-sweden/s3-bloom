"""Discover Sentinel-3 OLCI L2 products via the CDSE OData catalogue.

Search is read-only and does *not* require authentication, so this module
can be exercised offline with a network mock. Authentication is required
only for the download step (see :mod:`s3bloom.discovery.download`).

The OData query combines four filters:

* ``contains(Name, 'OL_2_WFR___')`` — Ocean Colour Level-2 Water Full
  Resolution products only.
* ``ContentDate/Start ge/le ...`` — sensing-time range.
* ``OData.CSC.Intersects(area=...)`` — geographic intersection with the
  bounding-box polygon (WGS84).

Results are paginated server-side; this module follows the
``@odata.nextLink`` cursor until exhausted.

The catalogue can list the same overpass more than once: a near-real-time
(``NR``) product is published within hours and is later joined by a
short-time-critical (``ST``) and/or non-time-critical (``NT``) version,
and reprocessing adds further copies with a newer creation time. All of
them share one sensing start, which is what downstream output filenames
are built from, so :func:`search_products` keeps only the best version of
each overpass (see :func:`dedupe_products`).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import requests

from s3bloom.config import BoundingBox, TimePeriod

logger = logging.getLogger(__name__)

ODATA_BASE = "https://catalogue.dataspace.copernicus.eu/odata/v1"
PRODUCT_TYPE = "OL_2_WFR___"
PAGE_SIZE = 100

# Timeliness tokens in product names, mapped to their preference rank
# (higher wins): non-time-critical > short-time-critical > near-real-time.
TIMELINESS_RANK = {"NR": 0, "ST": 1, "NT": 2}

_TIMESTAMP_FORMAT = "%Y%m%dT%H%M%S"
_TIMESTAMP_LENGTH = 15
# Position of the creation time among a title's timestamp tokens
# (sensing start, sensing stop, creation).
_CREATION_TOKEN_INDEX = 2


@dataclass(frozen=True)
class ProductInfo:
    """Catalogue metadata for one Sentinel-3 OLCI L2 product.

    Attributes
    ----------
    product_id : str
        Stable CDSE UUID. Use this to construct download URLs; titles can
        change when products are reprocessed.
    title : str
        Product file-system name (without the trailing ``.zip``), e.g.
        ``S3A_OL_2_WFR____20240315T091500_..._...``.
    sensing_start : datetime
        UTC start of the sensing window.
    satellite : {"S3A", "S3B", "S3X"}
        Spacecraft identifier extracted from ``title``. ``S3X`` is a
        sentinel for unknown / future spacecraft.
    footprint : str or None
        Stringified GeoJSON footprint as returned by CDSE, or ``None`` if
        the catalogue entry omitted it.
    size_mb : float or None
        Reported product size in megabytes, or ``None`` if unknown.
    online : bool
        ``True`` if the product is available for immediate download.
        Offline products require a CDSE retrieval-from-archive request.
    timeliness : {"NR", "ST", "NT", ""}
        Timeliness class parsed from ``title``: near-real-time,
        short-time-critical or non-time-critical. Empty string when the
        title carries no recognisable timeliness token.
    """

    product_id: str
    title: str
    sensing_start: datetime
    satellite: str
    footprint: str | None
    size_mb: float | None
    online: bool
    timeliness: str = ""

    @classmethod
    def from_odata(cls, entry: dict[str, Any]) -> ProductInfo:
        """Build a :class:`ProductInfo` from one OData catalogue entry.

        ``satellite`` and ``timeliness`` are parsed from the entry's
        ``Name``; ``sensing_start`` comes from ``ContentDate/Start``.
        """
        title = entry.get("Name", "")
        satellite = _extract_satellite(title)

        content_date = entry.get("ContentDate", {})
        sensing_start = _parse_datetime(content_date.get("Start", ""))

        size_bytes = entry.get("ContentLength", 0)
        size_mb = size_bytes / (1024 * 1024) if size_bytes else None

        geo = entry.get("GeoFootprint", {})
        footprint = str(geo) if geo else None

        return cls(
            product_id=entry.get("Id", ""),
            title=title,
            sensing_start=sensing_start,
            satellite=satellite,
            footprint=footprint,
            size_mb=size_mb,
            online=entry.get("Online", True),
            timeliness=_extract_timeliness(title),
        )


def search_products(
    bbox: BoundingBox,
    time_period: TimePeriod,
) -> list[ProductInfo]:
    """Query CDSE for OLCI L2 WFR products in *bbox* and *time_period*.

    Parameters
    ----------
    bbox : BoundingBox
        Geographic AOI in WGS84 lon/lat. The query uses an
        ``Intersects`` filter, so partial overlaps are returned.
    time_period : TimePeriod
        Sensing-time range. Inclusive on both ends, treated as UTC.

    Returns
    -------
    list of ProductInfo
        One entry per overpass, sorted by sensing start time, ascending.
        Empty list if nothing matches (no exception is raised in that
        case).

    Notes
    -----
    Pagination is handled transparently via the ``@odata.nextLink``
    cursor. The session reuses the retry-aware :func:`_create_session`
    helper from the download module so transient 5xx responses are
    retried automatically.

    The catalogue may hold several versions of one overpass (``NR``,
    ``ST``, ``NT``, reprocessed copies). They are reduced to the best
    available version with :func:`dedupe_products` before returning.
    """
    logger.info(
        "Searching CDSE for %s products: %s to %s, bbox=%s",
        PRODUCT_TYPE,
        time_period.start_date,
        time_period.end_date,
        bbox.as_tuple(),
    )

    filter_parts = [
        f"contains(Name,'{PRODUCT_TYPE}')",
        f"ContentDate/Start ge {time_period.start_date.isoformat()}T00:00:00.000Z",
        f"ContentDate/Start le {time_period.end_date.isoformat()}T23:59:59.999Z",
        _bbox_filter(bbox),
    ]
    filter_str = " and ".join(filter_parts)

    products: list[ProductInfo] = []
    url: str | None = (
        f"{ODATA_BASE}/Products"
        f"?$filter={filter_str}"
        f"&$orderby=ContentDate/Start asc"
        f"&$top={PAGE_SIZE}"
    )

    # Imported lazily to avoid a circular import — download.py imports
    # ProductInfo from this module.
    from s3bloom.discovery.download import _create_session

    session = _create_session()

    while url:
        logger.debug("Fetching: %s", url)
        resp = session.get(url, timeout=60)
        resp.raise_for_status()
        data = resp.json()

        for entry in data.get("value", []):
            products.append(ProductInfo.from_odata(entry))

        url = data.get("@odata.nextLink")

    logger.info("Found %d catalogue entries", len(products))

    unique = dedupe_products(products)
    dropped = len(products) - len(unique)
    if dropped:
        logger.info(
            "Dropped %d duplicate products (superseded versions of the same "
            "overpass); %d remain",
            dropped,
            len(unique),
        )
    return unique


def dedupe_products(products: list[ProductInfo]) -> list[ProductInfo]:
    """Keep one product per overpass, preferring the best version.

    Products are grouped by satellite and the ``YYYYMMDDTHHMMSS``
    sensing-start token in their title. That token is what
    :func:`s3bloom.processing.reader.extract_sensing_time` reads when the
    output filename is built, so two products sharing it would overwrite
    each other downstream. ``ContentDate`` is deliberately not used: it
    has sub-second precision and can differ between versions.

    Within a group the winner is the highest timeliness class (``NT``
    over ``ST`` over ``NR``); ties are broken by the latest creation
    timestamp in the title, which selects reprocessed versions. If that
    is also equal, the product listed first is kept.

    Parameters
    ----------
    products : list of ProductInfo
        Catalogue entries, typically in ascending sensing-start order.
        Not modified.

    Returns
    -------
    list of ProductInfo
        New list holding the surviving products in their original
        relative order, so an ascending input stays ascending.

    Notes
    -----
    A product whose title has no recognisable sensing-start token or no
    timeliness token is never grouped with anything and is always passed
    through.
    """
    keys = [_dedupe_key(product) for product in products]

    # Index of the preferred product for each overpass key.
    best: dict[tuple[str, str], int] = {}
    for index, key in enumerate(keys):
        if key is None:
            continue
        current = best.get(key)
        if current is None or (
            _preference(products[index]) > _preference(products[current])
        ):
            best[key] = index

    keep = set(best.values())
    return [
        product
        for index, (product, key) in enumerate(zip(products, keys))
        if key is None or index in keep
    ]


def _dedupe_key(product: ProductInfo) -> tuple[str, str] | None:
    """Return the ``(satellite, sensing-start token)`` overpass key.

    ``None`` means the title could not be parsed well enough to identify
    the overpass, so the product must not be merged with any other. Both
    tokens are read from the title rather than from derived fields, so
    the result does not depend on how the :class:`ProductInfo` was built.
    """
    tokens = _timestamp_tokens(product.title)
    if not tokens or not _extract_timeliness(product.title):
        return None
    return (product.satellite, tokens[0])


def _preference(product: ProductInfo) -> tuple[int, str]:
    """Sort key ranking versions of one overpass; the largest wins.

    Creation tokens are fixed-width ``YYYYMMDDTHHMMSS`` strings, so they
    compare chronologically without being parsed. A missing creation
    token sorts before any real one.
    """
    tokens = _timestamp_tokens(product.title)
    creation = (
        tokens[_CREATION_TOKEN_INDEX] if len(tokens) > _CREATION_TOKEN_INDEX else ""
    )
    return (TIMELINESS_RANK[_extract_timeliness(product.title)], creation)


def _timestamp_tokens(title: str) -> list[str]:
    """Return the ``YYYYMMDDTHHMMSS`` tokens of *title*, in order.

    For a standard product name these are sensing start, sensing stop
    and creation time. The token test mirrors
    :func:`s3bloom.processing.reader.extract_sensing_time` so the first
    token is exactly the one used for output filenames.
    """
    return [part for part in title.split("_") if _is_timestamp_token(part)]


def _is_timestamp_token(part: str) -> bool:
    """Return ``True`` if *part* is a valid ``YYYYMMDDTHHMMSS`` token."""
    if len(part) != _TIMESTAMP_LENGTH or part[0] != "2" or "T" not in part:
        return False
    try:
        datetime.strptime(part, _TIMESTAMP_FORMAT)
    except ValueError:
        return False
    return True


def _extract_timeliness(title: str) -> str:
    """Return ``"NR"``, ``"ST"`` or ``"NT"`` from a product title.

    The timeliness class is the second-to-last underscore-separated
    field of a standard name, so the title is scanned from the end.
    Falls back to ``""`` when no such token is present.
    """
    for part in reversed(title.split("_")):
        if part in TIMELINESS_RANK:
            return part
    return ""


def _bbox_filter(bbox: BoundingBox) -> str:
    """Build the OData ``Intersects`` clause for *bbox*.

    Returns the filter substring (without the leading ``and``). The
    polygon is closed by repeating the first vertex, as required by WKT.
    """
    wkt = (
        f"POLYGON(("
        f"{bbox.lon_min} {bbox.lat_min},"
        f"{bbox.lon_max} {bbox.lat_min},"
        f"{bbox.lon_max} {bbox.lat_max},"
        f"{bbox.lon_min} {bbox.lat_max},"
        f"{bbox.lon_min} {bbox.lat_min}"
        f"))"
    )
    return f"OData.CSC.Intersects(area=geography'SRID=4326;{wkt}')"


def _extract_satellite(title: str) -> str:
    """Return ``"S3A"`` or ``"S3B"`` from a product title.

    Falls back to ``"S3X"`` for unknown prefixes (e.g. future S3C/S3D
    missions) so the caller never has to handle ``None``.
    """
    if title.startswith("S3A"):
        return "S3A"
    if title.startswith("S3B"):
        return "S3B"
    return "S3X"


def _parse_datetime(dt_str: str) -> datetime:
    """Parse an ISO-8601 datetime string from CDSE into a tz-aware datetime."""
    if not dt_str:
        raise ValueError("Empty datetime string in CDSE product metadata")
    dt_str = dt_str.replace("Z", "+00:00")
    return datetime.fromisoformat(dt_str)
