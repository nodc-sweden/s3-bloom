"""Tests for discovery.search."""
import logging
from datetime import date, datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from s3bloom.config import BoundingBox, TimePeriod
from s3bloom.discovery.search import (
    ProductInfo,
    _bbox_filter,
    _extract_satellite,
    _parse_datetime,
    dedupe_products,
    search_products,
)

REAL_NR = (
    "S3A_OL_2_WFR____20260315T094407_20260315T094707_20260315T114512"
    "_0179_137_136_1980_MAR_O_NR_004.SEN3"
)
REAL_NT = (
    "S3A_OL_2_WFR____20260315T094407_20260315T094707_20260316T172233"
    "_0179_137_136_1980_MAR_O_NT_004.SEN3"
)


def _title(
    start="20240315T091500",
    timeliness="NT",
    created="20240315T111500",
    satellite="S3A",
):
    """Build a product name; ``timeliness=None`` gives the short legacy form."""
    if timeliness is None:
        return f"{satellite}_OL_2_WFR____{start}"
    stop = f"{start[:9]}{int(start[9:]) + 300:06d}"
    return (
        f"{satellite}_OL_2_WFR____{start}_{stop}_{created}"
        f"_0179_137_136_1980_MAR_O_{timeliness}_004.SEN3"
    )


def _iso(start):
    """Convert a ``YYYYMMDDTHHMMSS`` token to a CDSE ISO-8601 string."""
    parsed = datetime.strptime(start, "%Y%m%dT%H%M%S")
    return parsed.strftime("%Y-%m-%dT%H:%M:%S.000Z")


def _product(product_id="abc-123", **title_kwargs):
    """Build a :class:`ProductInfo` the way the catalogue parser would."""
    start = title_kwargs.get("start", "20240315T091500")
    return ProductInfo.from_odata(
        {
            "Id": product_id,
            "Name": _title(**title_kwargs),
            "ContentDate": {"Start": _iso(start)},
            "ContentLength": 0,
            "GeoFootprint": {},
            "Online": True,
        }
    )


class TestExtractSatellite:
    def test_s3a(self):
        assert _extract_satellite("S3A_OL_2_WFR____20240315T091500") == "S3A"

    def test_s3b(self):
        assert _extract_satellite("S3B_OL_2_WFR____20240315T091500") == "S3B"

    def test_unknown_prefix_returns_s3x(self):
        assert _extract_satellite("S3C_OL_2_WFR____20240315T091500") == "S3X"

    def test_empty_string_returns_s3x(self):
        assert _extract_satellite("") == "S3X"


class TestParseDatetime:
    def test_z_suffix(self):
        dt = _parse_datetime("2024-03-15T09:15:00.000Z")
        assert dt == datetime(2024, 3, 15, 9, 15, 0, tzinfo=timezone.utc)

    def test_utc_offset(self):
        dt = _parse_datetime("2024-03-15T09:15:00+00:00")
        assert dt.year == 2024
        assert dt.tzinfo is not None

    def test_empty_string_raises(self):
        with pytest.raises(ValueError, match="Empty datetime string"):
            _parse_datetime("")


class TestBboxFilter:
    def test_contains_polygon_keyword(self):
        bbox = BoundingBox(lon_min=10.0, lat_min=56.5, lon_max=13.0, lat_max=59.0)
        result = _bbox_filter(bbox)
        assert "POLYGON" in result

    def test_contains_odata_intersects(self):
        bbox = BoundingBox(lon_min=10.0, lat_min=56.5, lon_max=13.0, lat_max=59.0)
        result = _bbox_filter(bbox)
        assert "OData.CSC.Intersects" in result

    def test_contains_srid(self):
        bbox = BoundingBox(lon_min=10.0, lat_min=56.5, lon_max=13.0, lat_max=59.0)
        result = _bbox_filter(bbox)
        assert "SRID=4326" in result

    def test_polygon_is_closed(self):
        bbox = BoundingBox(lon_min=10.0, lat_min=56.5, lon_max=13.0, lat_max=59.0)
        result = _bbox_filter(bbox)
        inner = result.split("POLYGON((")[1].split("))")[0]
        vertices = [v.strip() for v in inner.split(",")]
        assert vertices[0] == vertices[-1]

    def test_bbox_coords_present(self):
        bbox = BoundingBox(lon_min=7.5, lat_min=55.0, lon_max=12.5, lat_max=60.0)
        result = _bbox_filter(bbox)
        assert "7.5" in result
        assert "55.0" in result


class TestProductInfoFromOdata:
    def _entry(self, **overrides):
        base = {
            "Id": "abc-123",
            "Name": "S3A_OL_2_WFR____20240315T091500",
            "ContentDate": {"Start": "2024-03-15T09:15:00.000Z"},
            "ContentLength": 524_288_000,  # 500 MB
            "GeoFootprint": {"type": "Polygon", "coordinates": []},
            "Online": True,
        }
        base.update(overrides)
        return base

    def test_product_id(self):
        pi = ProductInfo.from_odata(self._entry())
        assert pi.product_id == "abc-123"

    def test_satellite_extraction(self):
        pi = ProductInfo.from_odata(self._entry())
        assert pi.satellite == "S3A"

    def test_s3b_satellite(self):
        pi = ProductInfo.from_odata(self._entry(Name="S3B_OL_2_WFR____20240315T091500"))
        assert pi.satellite == "S3B"

    def test_size_mb_computed(self):
        pi = ProductInfo.from_odata(self._entry(ContentLength=1_048_576))
        assert pi.size_mb == pytest.approx(1.0, abs=0.01)

    def test_zero_content_length_gives_none(self):
        pi = ProductInfo.from_odata(self._entry(ContentLength=0))
        assert pi.size_mb is None

    def test_empty_footprint_gives_none(self):
        pi = ProductInfo.from_odata(self._entry(GeoFootprint={}))
        assert pi.footprint is None

    def test_online_flag(self):
        assert ProductInfo.from_odata(self._entry(Online=True)).online is True
        assert ProductInfo.from_odata(self._entry(Online=False)).online is False

    def test_sensing_start_parsed(self):
        pi = ProductInfo.from_odata(self._entry())
        assert pi.sensing_start.year == 2024
        assert pi.sensing_start.tzinfo is not None

    def test_timeliness_near_real_time(self):
        pi = ProductInfo.from_odata(self._entry(Name=REAL_NR))
        assert pi.timeliness == "NR"

    def test_timeliness_non_time_critical(self):
        pi = ProductInfo.from_odata(self._entry(Name=REAL_NT))
        assert pi.timeliness == "NT"

    def test_timeliness_short_time_critical(self):
        pi = ProductInfo.from_odata(self._entry(Name=_title(timeliness="ST")))
        assert pi.timeliness == "ST"

    def test_timeliness_unknown_is_empty_string(self):
        pi = ProductInfo.from_odata(self._entry())
        assert pi.timeliness == ""

    def test_timeliness_unrecognised_token_is_empty_string(self):
        pi = ProductInfo.from_odata(self._entry(Name=_title(timeliness="XX")))
        assert pi.timeliness == ""

    def test_timeliness_defaults_to_empty_when_constructed_directly(self):
        pi = ProductInfo(
            product_id="abc-123",
            title="S3A_OL_2_WFR____20240315T091500",
            sensing_start=datetime(2024, 3, 15, 9, 15, 0, tzinfo=timezone.utc),
            satellite="S3A",
            footprint=None,
            size_mb=None,
            online=True,
        )
        assert pi.timeliness == ""


class TestDedupeProducts:
    def test_empty_list(self):
        assert dedupe_products([]) == []

    def test_real_nr_nt_pair_keeps_only_nt(self):
        nr = ProductInfo.from_odata(
            {
                "Id": "nr",
                "Name": REAL_NR,
                "ContentDate": {"Start": "2026-03-15T09:44:07.123456Z"},
            }
        )
        nt = ProductInfo.from_odata(
            {
                "Id": "nt",
                "Name": REAL_NT,
                "ContentDate": {"Start": "2026-03-15T09:44:07.654321Z"},
            }
        )
        assert dedupe_products([nr, nt]) == [nt]

    def test_nr_only_is_kept(self):
        nr = _product("nr", timeliness="NR")
        assert dedupe_products([nr]) == [nr]

    def test_nt_listed_before_nr(self):
        nt = _product("nt", timeliness="NT", created="20240316T170000")
        nr = _product("nr", timeliness="NR", created="20240315T111500")
        assert dedupe_products([nt, nr]) == [nt]

    def test_nt_listed_after_nr(self):
        nr = _product("nr", timeliness="NR", created="20240315T111500")
        nt = _product("nt", timeliness="NT", created="20240316T170000")
        assert dedupe_products([nr, nt]) == [nt]

    def test_nt_preferred_even_when_nr_was_created_later(self):
        nt = _product("nt", timeliness="NT", created="20240316T170000")
        nr = _product("nr", timeliness="NR", created="20240320T000000")
        assert dedupe_products([nr, nt]) == [nt]
        assert dedupe_products([nt, nr]) == [nt]

    def test_st_preferred_over_nr(self):
        nr = _product("nr", timeliness="NR")
        st = _product("st", timeliness="ST")
        assert dedupe_products([nr, st]) == [st]
        assert dedupe_products([st, nr]) == [st]

    def test_nt_preferred_over_st_and_nr(self):
        nr = _product("nr", timeliness="NR")
        st = _product("st", timeliness="ST")
        nt = _product("nt", timeliness="NT")
        assert dedupe_products([st, nt, nr]) == [nt]

    def test_two_nt_keeps_latest_creation(self):
        old = _product("old", timeliness="NT", created="20240316T170000")
        new = _product("new", timeliness="NT", created="20250101T120000")
        assert dedupe_products([old, new]) == [new]
        assert dedupe_products([new, old]) == [new]

    def test_s3a_and_s3b_with_same_start_are_both_kept(self):
        a = _product("a", satellite="S3A")
        b = _product("b", satellite="S3B")
        assert dedupe_products([a, b]) == [a, b]

    def test_adjacent_frames_are_both_kept(self):
        first = _product("first", start="20240315T091500")
        second = _product("second", start="20240315T091800")
        assert dedupe_products([first, second]) == [first, second]

    def test_titles_without_timeliness_are_never_collapsed(self):
        first = _product("first", timeliness=None)
        second = _product("second", timeliness=None)
        assert dedupe_products([first, second]) == [first, second]

    def test_title_without_timeliness_not_collapsed_with_nt(self):
        bare = _product("bare", timeliness=None)
        nt = _product("nt", timeliness="NT")
        assert dedupe_products([bare, nt]) == [bare, nt]

    def test_titles_without_start_token_are_never_collapsed(self):
        def _no_start(product_id):
            return ProductInfo.from_odata(
                {
                    "Id": product_id,
                    "Name": "S3A_OL_2_WFR____MAR_O_NT_004.SEN3",
                    "ContentDate": {"Start": "2024-03-15T09:15:00.000Z"},
                }
            )

        first, second = _no_start("first"), _no_start("second")
        assert dedupe_products([first, second]) == [first, second]

    def test_malformed_start_token_is_not_a_start_token(self):
        def _malformed(product_id):
            return ProductInfo.from_odata(
                {
                    "Id": product_id,
                    "Name": "S3A_OL_2_WFR____20241345T091500_MAR_O_NT_004.SEN3",
                    "ContentDate": {"Start": "2024-03-15T09:15:00.000Z"},
                }
            )

        first, second = _malformed("first"), _malformed("second")
        assert dedupe_products([first, second]) == [first, second]

    def test_result_is_ascending_by_sensing_start(self):
        products = [
            _product("a-nr", start="20240315T091500", timeliness="NR"),
            _product("a-nt", start="20240315T091500", timeliness="NT"),
            _product("b-nt", start="20240315T091800", timeliness="NT"),
            _product("b-nr", start="20240315T091800", timeliness="NR"),
            _product("c-bare", start="20240315T092100", timeliness=None),
            _product("d-nr", start="20240316T090000", timeliness="NR"),
        ]
        result = dedupe_products(products)
        assert [p.product_id for p in result] == ["a-nt", "b-nt", "c-bare", "d-nr"]
        starts = [p.sensing_start for p in result]
        assert starts == sorted(starts)

    def test_timeliness_is_read_from_title_not_field(self):
        def _direct(product_id, timeliness):
            return ProductInfo(
                product_id=product_id,
                title=_title(timeliness=timeliness),
                sensing_start=datetime(2024, 3, 15, 9, 15, 0, tzinfo=timezone.utc),
                satellite="S3A",
                footprint=None,
                size_mb=None,
                online=True,
            )

        nr, nt = _direct("nr", "NR"), _direct("nt", "NT")
        assert dedupe_products([nr, nt]) == [nt]

    def test_missing_creation_token_loses_tie(self):
        def _named(product_id, name):
            return ProductInfo.from_odata(
                {
                    "Id": product_id,
                    "Name": name,
                    "ContentDate": {"Start": "2024-03-15T09:15:00.000Z"},
                }
            )

        bare = _named("bare", "S3A_OL_2_WFR____20240315T091500_MAR_O_NT_004.SEN3")
        full = _named("full", _title(timeliness="NT"))
        assert dedupe_products([bare, full]) == [full]
        assert dedupe_products([full, bare]) == [full]

    def test_exact_tie_keeps_first_listed(self):
        first = _product("first", timeliness="NT")
        second = _product("second", timeliness="NT")
        assert dedupe_products([first, second]) == [first]

    def test_input_list_is_not_mutated(self):
        nr = _product("nr", timeliness="NR")
        nt = _product("nt", timeliness="NT")
        products = [nr, nt]
        result = dedupe_products(products)
        assert products == [nr, nt]
        assert result is not products

    def test_returns_new_list_when_nothing_is_dropped(self):
        products = [_product("only")]
        result = dedupe_products(products)
        assert result == products
        assert result is not products


class TestSearchProducts:
    def _bbox(self):
        return BoundingBox(lon_min=10.0, lat_min=56.5, lon_max=13.0, lat_max=59.0)

    def _period(self):
        return TimePeriod(start_date=date(2024, 3, 15), end_date=date(2024, 3, 15))

    def _entry(
        self,
        product_id="abc-123",
        name=None,
        *,
        start="20240315T091500",
        timeliness=None,
        created="20240315T111500",
        satellite="S3A",
    ):
        if name is None:
            name = _title(
                start=start,
                timeliness=timeliness,
                created=created,
                satellite=satellite,
            )
        return {
            "Id": product_id,
            "Name": name,
            "ContentDate": {"Start": _iso(start)},
            "ContentLength": 0,
            "GeoFootprint": {},
            "Online": True,
        }

    def _mock_session(self, pages):
        mock_session = MagicMock()
        responses = []
        for i, entries in enumerate(pages):
            mock_resp = MagicMock()
            data = {"value": entries}
            if i < len(pages) - 1:
                data["@odata.nextLink"] = "https://next-page"
            mock_resp.json.return_value = data
            mock_resp.raise_for_status = MagicMock()
            responses.append(mock_resp)
        mock_session.get.side_effect = responses
        return mock_session

    def test_empty_response_returns_empty_list(self):
        mock_session = self._mock_session([[]])
        with patch("s3bloom.discovery.download._create_session", return_value=mock_session):
            products = search_products(self._bbox(), self._period())
        assert products == []

    def test_single_page(self):
        mock_session = self._mock_session([[self._entry()]])
        with patch("s3bloom.discovery.download._create_session", return_value=mock_session):
            products = search_products(self._bbox(), self._period())
        assert len(products) == 1
        assert products[0].product_id == "abc-123"

    def test_pagination_follows_next_link(self):
        mock_session = self._mock_session([
            [self._entry("id-1", start="20240315T091500", timeliness="NT")],
            [self._entry("id-2", start="20240315T091800", timeliness="NT")],
        ])
        with patch("s3bloom.discovery.download._create_session", return_value=mock_session):
            products = search_products(self._bbox(), self._period())
        assert [p.product_id for p in products] == ["id-1", "id-2"]
        assert mock_session.get.call_count == 2

    def test_nr_nt_pair_is_deduplicated(self):
        mock_session = self._mock_session([
            [
                self._entry("nr", timeliness="NR", created="20240315T111500"),
                self._entry("nt", timeliness="NT", created="20240316T170000"),
            ],
        ])
        with patch("s3bloom.discovery.download._create_session", return_value=mock_session):
            products = search_products(self._bbox(), self._period())
        assert [p.product_id for p in products] == ["nt"]
        assert products[0].timeliness == "NT"

    def test_duplicates_split_across_pages_are_deduplicated(self):
        mock_session = self._mock_session([
            [self._entry("nr", timeliness="NR")],
            [self._entry("nt", timeliness="NT")],
        ])
        with patch("s3bloom.discovery.download._create_session", return_value=mock_session):
            products = search_products(self._bbox(), self._period())
        assert [p.product_id for p in products] == ["nt"]

    def test_result_is_ascending_after_dedupe(self):
        mock_session = self._mock_session([
            [
                self._entry("a-nr", start="20240315T091500", timeliness="NR"),
                self._entry("a-nt", start="20240315T091500", timeliness="NT"),
                self._entry("b-nt", start="20240315T091800", timeliness="NT"),
                self._entry("b-nr", start="20240315T091800", timeliness="NR"),
            ],
        ])
        with patch("s3bloom.discovery.download._create_session", return_value=mock_session):
            products = search_products(self._bbox(), self._period())
        assert [p.product_id for p in products] == ["a-nt", "b-nt"]
        starts = [p.sensing_start for p in products]
        assert starts == sorted(starts)

    def test_logs_number_of_dropped_duplicates(self, caplog):
        mock_session = self._mock_session([
            [
                self._entry("nr", timeliness="NR"),
                self._entry("nt", timeliness="NT"),
            ],
        ])
        with (
            caplog.at_level(logging.INFO, logger="s3bloom.discovery.search"),
            patch("s3bloom.discovery.download._create_session", return_value=mock_session),
        ):
            search_products(self._bbox(), self._period())
        messages = [r.getMessage() for r in caplog.records if r.levelno == logging.INFO]
        assert any("1 duplicate" in m for m in messages)

    def test_does_not_log_duplicates_when_none_dropped(self, caplog):
        mock_session = self._mock_session([[self._entry("nt", timeliness="NT")]])
        with (
            caplog.at_level(logging.INFO, logger="s3bloom.discovery.search"),
            patch("s3bloom.discovery.download._create_session", return_value=mock_session),
        ):
            search_products(self._bbox(), self._period())
        assert not any("duplicate" in r.getMessage() for r in caplog.records)
