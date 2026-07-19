import json

import pandas as pd

from src.asset_mapping import (
    ASSET_CLASSES,
    MANUAL_CLASS_OVERRIDES,
    load_fund_class_map,
    load_user_overrides,
    save_user_override,
    holdings_to_class,
    funds_by_class,
)


def _write_snapshot(cache_dir) -> None:
    """tmp cache dizinine kucuk bir TEFAS snapshot parquet'i yaz."""
    df = pd.DataFrame({
        "fund_code": ["AHS", "BGL", "PPF", "GMF", "XYZ"],
        "category":  ["Stock Fund", "Gold Fund", "Money Market Fund",
                      "Fund of Funds ", "Fund of Funds"],
    })
    df.to_parquet(cache_dir / "snapshot_EMK_2026_05.parquet")


class TestLoadFundClassMap:
    def test_maps_fund_codes_to_classes(self, tmp_path):
        _write_snapshot(tmp_path)
        m = load_fund_class_map(str(tmp_path))
        assert m["AHS"] == "VEF"
        assert m["BGL"] == "ALT"
        assert m["PPF"] == "CASH"
        assert m["XYZ"] == "KCH"       # "Fund of Funds" artik KCH'ye eslenir (PLAN-12)
        assert m["VEF"] == "VEF"       # sinif kodu kendine eslenir

    def test_lowercase_and_case_insensitive_lookup(self, tmp_path):
        _write_snapshot(tmp_path)
        m = load_fund_class_map(str(tmp_path))
        # anahtarlar buyuk harf
        assert "AHS" in m and "ahs" not in m

    def test_no_snapshot_returns_class_codes_and_overrides(self, tmp_path):
        m = load_fund_class_map(str(tmp_path))  # bos dizin, snapshot yok
        expected = {c: c for c in ASSET_CLASSES}
        expected.update(MANUAL_CLASS_OVERRIDES)
        assert m == expected


class TestManualOverrides:
    def test_gmf_silver_fund_maps_to_alt(self, tmp_path):
        # GMF kategorisi "Fund of Funds " (yaniltici) — manuel istisna ALT'a esler
        _write_snapshot(tmp_path)
        m = load_fund_class_map(str(tmp_path))
        assert m["GMF"] == "ALT"

    def test_override_wins_over_category(self, tmp_path):
        # Kategori CASH derse bile istisna kazanmali
        df = pd.DataFrame({
            "fund_code": ["GMF"],
            "category":  ["Money Market Fund"],
        })
        df.to_parquet(tmp_path / "snapshot_EMK_2026_06.parquet")
        m = load_fund_class_map(str(tmp_path))
        assert m["GMF"] == "ALT"

    def test_override_available_without_snapshot(self, tmp_path):
        m = load_fund_class_map(str(tmp_path))
        assert m.get("GMF") == "ALT"


class TestExpandedCategories:
    """PLAN-12: genisletilmis ASSET_CATEGORY_MAP — daha once eslenmeyen
    TEFAS kategorileri (Standard, State Contribution, Fund of Funds, Initial,
    Life Cycle/Target, Lease Certificate) artik bir sinifa eslenir."""

    def _write_expanded_snapshot(self, cache_dir) -> None:
        df = pd.DataFrame({
            "fund_code": ["STD", "DVK", "FOF", "INI", "LCT", "LST", "PEQ"],
            "category": [
                "Standard Fund",
                "State Contribution Fund",
                "Fund of Funds",
                "Initial Fund",
                "Life Cycle/Target Fund",
                "Lease Certificate Participation Fund",
                "Participation Equity Fund",
            ],
        })
        df.to_parquet(cache_dir / "snapshot_EMK_2026_05.parquet")

    def test_standard_fund_maps_to_kts(self, tmp_path):
        self._write_expanded_snapshot(tmp_path)
        m = load_fund_class_map(str(tmp_path))
        assert m["STD"] == "KTS"

    def test_state_contribution_maps_to_kts(self, tmp_path):
        self._write_expanded_snapshot(tmp_path)
        m = load_fund_class_map(str(tmp_path))
        assert m["DVK"] == "KTS"

    def test_fund_of_funds_maps_to_kch(self, tmp_path):
        self._write_expanded_snapshot(tmp_path)
        m = load_fund_class_map(str(tmp_path))
        assert m["FOF"] == "KCH"

    def test_initial_fund_maps_to_cash(self, tmp_path):
        self._write_expanded_snapshot(tmp_path)
        m = load_fund_class_map(str(tmp_path))
        assert m["INI"] == "CASH"

    def test_life_cycle_target_maps_to_kch(self, tmp_path):
        self._write_expanded_snapshot(tmp_path)
        m = load_fund_class_map(str(tmp_path))
        assert m["LCT"] == "KCH"

    def test_lease_certificate_participation_maps_to_kts(self, tmp_path):
        self._write_expanded_snapshot(tmp_path)
        m = load_fund_class_map(str(tmp_path))
        assert m["LST"] == "KTS"

    def test_participation_equity_priority_preserved(self, tmp_path):
        # "equity" substring -> VEF ilk sirada kontrol edildigi icin kazanir
        self._write_expanded_snapshot(tmp_path)
        m = load_fund_class_map(str(tmp_path))
        assert m["PEQ"] == "VEF"


class TestUserOverrides:
    """PLAN-12: kalici kullanici override dosyasi (load_user_overrides /
    save_user_override)."""

    def test_missing_file_returns_empty_dict(self, tmp_path):
        path = str(tmp_path / "user_class_overrides.json")
        assert load_user_overrides(path) == {}

    def test_corrupt_file_returns_empty_dict(self, tmp_path):
        path = tmp_path / "user_class_overrides.json"
        path.write_text("{not valid json", encoding="utf-8")
        assert load_user_overrides(str(path)) == {}

    def test_invalid_class_values_filtered_out(self, tmp_path):
        path = tmp_path / "user_class_overrides.json"
        path.write_text(json.dumps({"ABC": "VEF", "FOO": "NOT_A_CLASS"}), encoding="utf-8")
        result = load_user_overrides(str(path))
        assert result == {"ABC": "VEF"}
        assert "FOO" not in result

    def test_keys_normalized_to_uppercase(self, tmp_path):
        path = tmp_path / "user_class_overrides.json"
        path.write_text(json.dumps({"abc": "VEF"}), encoding="utf-8")
        result = load_user_overrides(str(path))
        assert result == {"ABC": "VEF"}

    def test_save_valid_override_returns_true_and_persists(self, tmp_path):
        path = str(tmp_path / "user_class_overrides.json")
        assert save_user_override("ABC", "VEF", path=path) is True
        assert load_user_overrides(path) == {"ABC": "VEF"}

    def test_save_invalid_class_returns_false_and_no_file(self, tmp_path):
        path = tmp_path / "user_class_overrides.json"
        assert save_user_override("ABC", "NOT_A_CLASS", path=str(path)) is False
        assert not path.exists()

    def test_save_accumulates_multiple_overrides(self, tmp_path):
        path = str(tmp_path / "user_class_overrides.json")
        save_user_override("ABC", "VEF", path=path)
        save_user_override("DEF", "KTS", path=path)
        result = load_user_overrides(path)
        assert result == {"ABC": "VEF", "DEF": "KTS"}


class TestLoadFundClassMapOverridePriority:
    """PLAN-12: oncelik zinciri — kategori < MANUAL_CLASS_OVERRIDES <
    kullanici override dosyasi (kullanici dosyasi her zaman kazanir)."""

    def test_user_override_applied_without_snapshot(self, tmp_path):
        overrides_path = tmp_path / "user_class_overrides.json"
        overrides_path.write_text(json.dumps({"ABC": "VEF"}), encoding="utf-8")
        m = load_fund_class_map(str(tmp_path), user_overrides_path=str(overrides_path))
        assert m["ABC"] == "VEF"

    def test_user_override_applied_with_snapshot(self, tmp_path):
        _write_snapshot(tmp_path)
        overrides_path = tmp_path / "user_class_overrides.json"
        overrides_path.write_text(json.dumps({"ABC": "KTS"}), encoding="utf-8")
        m = load_fund_class_map(str(tmp_path), user_overrides_path=str(overrides_path))
        assert m["ABC"] == "KTS"
        assert m["AHS"] == "VEF"   # snapshot eslemesi de calismaya devam ediyor

    def test_user_override_wins_over_manual_class_overrides(self, tmp_path):
        # GMF normalde MANUAL_CLASS_OVERRIDES ile ALT'a sabitlenir; kullanici
        # dosyasi bunu bile ezebilmeli.
        _write_snapshot(tmp_path)
        overrides_path = tmp_path / "user_class_overrides.json"
        overrides_path.write_text(json.dumps({"GMF": "KCH"}), encoding="utf-8")
        m = load_fund_class_map(str(tmp_path), user_overrides_path=str(overrides_path))
        assert m["GMF"] == "KCH"

    def test_invalid_class_in_override_file_does_not_break_mapping(self, tmp_path):
        _write_snapshot(tmp_path)
        overrides_path = tmp_path / "user_class_overrides.json"
        overrides_path.write_text(json.dumps({"XYZ": "NOT_A_CLASS"}), encoding="utf-8")
        m = load_fund_class_map(str(tmp_path), user_overrides_path=str(overrides_path))
        # Gecersiz deger elenir; XYZ kategori eslemesinden gelen KCH'de kalir.
        assert m["XYZ"] == "KCH"


class TestHoldingsToClass:
    def test_aggregates_by_class_and_reports_unmapped(self, tmp_path):
        _write_snapshot(tmp_path)
        m = load_fund_class_map(str(tmp_path))
        # NOT: XYZ burada kullanilmiyor artik — "Fund of Funds" PLAN-12 ile
        # KCH'ye eslendigi icin XYZ artik unmapped degil (bkz. TestLoadFundClassMap).
        # ZZZ snapshot'ta hic yok -> gercekten unmapped kalir.
        class_tl, unmapped = holdings_to_class({"AHS": 100, "BGL": 50, "ZZZ": 25}, m)
        assert class_tl == {"VEF": 100.0, "ALT": 50.0}
        assert unmapped == {"ZZZ": 25.0}

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
