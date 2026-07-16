import pandas as pd

from src.asset_mapping import (
    ASSET_CLASSES,
    load_fund_class_map,
    holdings_to_class,
    funds_by_class,
)


def _write_snapshot(cache_dir) -> None:
    """tmp cache dizinine kucuk bir TEFAS snapshot parquet'i yaz."""
    df = pd.DataFrame({
        "fund_code": ["AHS", "BGL", "GMF", "XYZ"],
        "category":  ["Stock Fund", "Gold Fund", "Money Market Fund", "Fund of Funds"],
    })
    df.to_parquet(cache_dir / "snapshot_EMK_2026_05.parquet")


class TestLoadFundClassMap:
    def test_maps_fund_codes_to_classes(self, tmp_path):
        _write_snapshot(tmp_path)
        m = load_fund_class_map(str(tmp_path))
        assert m["AHS"] == "VEF"
        assert m["BGL"] == "ALT"
        assert m["GMF"] == "CASH"
        assert "XYZ" not in m          # "Fund of Funds" hicbir sinifa eslenmiyor
        assert m["VEF"] == "VEF"       # sinif kodu kendine eslenir

    def test_lowercase_and_case_insensitive_lookup(self, tmp_path):
        _write_snapshot(tmp_path)
        m = load_fund_class_map(str(tmp_path))
        # anahtarlar buyuk harf
        assert "AHS" in m and "ahs" not in m

    def test_no_snapshot_returns_only_class_codes(self, tmp_path):
        m = load_fund_class_map(str(tmp_path))  # bos dizin, snapshot yok
        assert m == {c: c for c in ASSET_CLASSES}


class TestHoldingsToClass:
    def test_aggregates_by_class_and_reports_unmapped(self, tmp_path):
        _write_snapshot(tmp_path)
        m = load_fund_class_map(str(tmp_path))
        class_tl, unmapped = holdings_to_class({"AHS": 100, "BGL": 50, "XYZ": 25}, m)
        assert class_tl == {"VEF": 100.0, "ALT": 50.0}
        assert unmapped == {"XYZ": 25.0}

    def test_same_class_multiple_funds_summed(self):
        m = {"AHS": "VEF", "IST": "VEF"}
        class_tl, unmapped = holdings_to_class({"AHS": 100, "IST": 200}, m)
        assert class_tl == {"VEF": 300.0}
        assert unmapped == {}

    def test_empty_holdings(self):
        class_tl, unmapped = holdings_to_class({}, {"AHS": "VEF"})
        assert class_tl == {}
        assert unmapped == {}


class TestFundsByClass:
    def test_groups_positive_holdings_only(self):
        m = {"AHS": "VEF", "BGL": "ALT", "IST": "VEF"}
        out = funds_by_class({"AHS": 100, "BGL": 50, "IST": 0}, m)
        assert out["VEF"] == ["AHS"]   # IST TL=0 oldugu icin haric
        assert out["ALT"] == ["BGL"]

    def test_unmapped_fund_excluded(self):
        m = {"AHS": "VEF"}
        out = funds_by_class({"AHS": 100, "XYZ": 50}, m)
        assert out == {"VEF": ["AHS"]}
