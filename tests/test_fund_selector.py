"""PLAN-13: sinif ici somut aday fon secici (suggest_funds_for_class).

Testler GERCEK data/ dizinlerini OKUMAZ: snapshot + predictions CSV
tmp_path'e yazilir, class_map explicit gecirilir (tests/test_asset_mapping.py
fixture deseni).
"""
import pandas as pd

from src.fund_selector import suggest_funds_for_class

# 6 fon, 3 sinif — class_map explicit verilir ki load_fund_class_map
# (ve dolayisiyla gercek data/user_class_overrides.json) hic okunmasin.
CLASS_MAP = {
    "AAA": "VEF", "BBB": "VEF", "FFF": "VEF",
    "CCC": "ALT", "DDD": "ALT",
    "EEE": "CASH",
}


def _write_snapshot(cache_dir) -> None:
    """tmp cache dizinine kucuk bir TEFAS snapshot parquet'i yaz."""
    df = pd.DataFrame({
        "fund_code": ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"],
        "fund_name": [
            "Alpha Hisse Senedi Fonu Cok Uzun Bir Unvan Ornegi",
            "Beta Hisse Fonu",
            "Gamma Altin Fonu",
            "Delta Altin Fonu",
            "Epsilon Para Piyasasi Fonu",
            "Zeta Hisse Fonu",
        ],
        "category": ["Stock Fund", "Stock Fund", "Gold Fund",
                     "Gold Fund", "Money Market Fund", "Stock Fund"],
        "risk": [6.0, 5.0, 3.0, None, 1.0, 4.0],
        "return_1y": [48.2, 44.0, 30.0, 25.0, 12.0, 55.0],
    })
    df.to_parquet(cache_dir / "snapshot_EMK_2026_06.parquet")


def _write_predictions(ml_dir, with_rank: bool = False) -> None:
    """tmp ml dizinine predictions CSV'si yaz (DDD ve EEE icin tahmin YOK)."""
    df = pd.DataFrame({
        "fund_code": ["AAA", "BBB", "FFF", "CCC"],
        "prediction_date": ["2026-07-01"] * 4,
        "predicted_fwd_return_3m": [0.123, 0.101, 0.150, 0.05],
        "model": ["lightgbm"] * 4,
    })
    if with_rank:
        # Rank'ta FFF > AAA > BBB sirasi korunur (getiriyle ayni yon)
        df["predicted_rank_3m"] = [0.70, 0.60, 0.95, 0.20]
    df.to_csv(ml_dir / "predictions_fwd_return_3m_20260701.csv", index=False)


def _dirs(tmp_path):
    cache = tmp_path / "cache"
    ml = tmp_path / "ml"
    cache.mkdir()
    ml.mkdir()
    return cache, ml


class TestMlScore:
    def test_ml_score_ordering_and_fields(self, tmp_path):
        cache, ml = _dirs(tmp_path)
        _write_snapshot(cache)
        _write_predictions(ml)
        out = suggest_funds_for_class(
            "VEF", cache_dir=str(cache), ml_dir=str(ml), class_map=CLASS_MAP
        )
        # ML tahminine gore azalan: FFF (0.150) > AAA (0.123) > BBB (0.101)
        assert [c["fund_code"] for c in out] == ["FFF", "AAA", "BBB"]
        assert all(c["score_basis"] == "ml_return" for c in out)
        assert out[0]["predicted_3m"] == 0.150
        assert out[0]["return_1y"] == 55.0
        assert out[0]["risk"] == 4.0
        # Cikti alanlari tam
        assert set(out[0]) == {
            "fund_code", "fund_name", "score_basis", "predicted_3m",
            "return_1y", "risk", "held",
        }

    def test_n_limit_respected(self, tmp_path):
        cache, ml = _dirs(tmp_path)
        _write_snapshot(cache)
        _write_predictions(ml)
        out = suggest_funds_for_class(
            "VEF", n=2, cache_dir=str(cache), ml_dir=str(ml), class_map=CLASS_MAP
        )
        assert [c["fund_code"] for c in out] == ["FFF", "AAA"]

    def test_only_requested_class_funds(self, tmp_path):
        cache, ml = _dirs(tmp_path)
        _write_snapshot(cache)
        _write_predictions(ml)
        out = suggest_funds_for_class(
            "ALT", cache_dir=str(cache), ml_dir=str(ml), class_map=CLASS_MAP
        )
        # ALT fonlarindan yalniz CCC'nin ML tahmini var (DDD NaN -> elenir);
        # hicbir VEF/CASH fonu sizmamali.
        assert [c["fund_code"] for c in out] == ["CCC"]

    def test_fund_name_truncated_to_40_chars(self, tmp_path):
        cache, ml = _dirs(tmp_path)
        _write_snapshot(cache)
        _write_predictions(ml)
        out = suggest_funds_for_class(
            "VEF", cache_dir=str(cache), ml_dir=str(ml), class_map=CLASS_MAP
        )
        by_code = {c["fund_code"]: c for c in out}
        assert len(by_code["AAA"]["fund_name"]) == 40


class TestRankPreferred:
    def test_rank_column_preferred_over_return(self, tmp_path):
        # predicted_rank_3m kolonu EKLENMIS CSV'de rank tercih edilir (PLAN-16 hazirligi)
        cache, ml = _dirs(tmp_path)
        _write_snapshot(cache)
        _write_predictions(ml, with_rank=True)
        out = suggest_funds_for_class(
            "VEF", cache_dir=str(cache), ml_dir=str(ml), class_map=CLASS_MAP
        )
        assert all(c["score_basis"] == "ml_rank" for c in out)
        # Rank'a gore azalan: FFF (0.95) > AAA (0.70) > BBB (0.60)
        assert [c["fund_code"] for c in out] == ["FFF", "AAA", "BBB"]
        # predicted_3m yine ham getiri tahminini tasir (gosterim icin)
        assert out[0]["predicted_3m"] == 0.150


class TestFallbacks:
    def test_no_ml_file_falls_back_to_return_1y(self, tmp_path):
        cache, ml = _dirs(tmp_path)
        _write_snapshot(cache)  # ml dizini bos
        out = suggest_funds_for_class(
            "ALT", cache_dir=str(cache), ml_dir=str(ml), class_map=CLASS_MAP
        )
        assert [c["fund_code"] for c in out] == ["CCC", "DDD"]  # 30.0 > 25.0
        assert all(c["score_basis"] == "return_1y" for c in out)
        assert all(c["predicted_3m"] is None for c in out)
        # NaN risk -> None
        assert out[1]["risk"] is None

    def test_no_snapshot_returns_empty(self, tmp_path):
        cache, ml = _dirs(tmp_path)  # cache bos
        _write_predictions(ml)
        out = suggest_funds_for_class(
            "VEF", cache_dir=str(cache), ml_dir=str(ml), class_map=CLASS_MAP
        )
        assert out == []

    def test_unknown_class_returns_empty(self, tmp_path):
        cache, ml = _dirs(tmp_path)
        _write_snapshot(cache)
        _write_predictions(ml)
        out = suggest_funds_for_class(
            "KTS", cache_dir=str(cache), ml_dir=str(ml), class_map=CLASS_MAP
        )
        assert out == []  # haritada KTS'ye eslenen fon yok


class TestHeldCodes:
    def test_held_codes_marked(self, tmp_path):
        cache, ml = _dirs(tmp_path)
        _write_snapshot(cache)
        _write_predictions(ml)
        out = suggest_funds_for_class(
            "VEF", cache_dir=str(cache), ml_dir=str(ml), class_map=CLASS_MAP,
            held_codes={"BBB"},
        )
        by_code = {c["fund_code"]: c["held"] for c in out}
        assert by_code == {"FFF": False, "AAA": False, "BBB": True}

    def test_held_codes_case_insensitive(self, tmp_path):
        cache, ml = _dirs(tmp_path)
        _write_snapshot(cache)
        _write_predictions(ml)
        out = suggest_funds_for_class(
            "VEF", cache_dir=str(cache), ml_dir=str(ml), class_map=CLASS_MAP,
            held_codes={"bbb"},
        )
        by_code = {c["fund_code"]: c["held"] for c in out}
        assert by_code["BBB"] is True

    def test_no_held_codes_all_false(self, tmp_path):
        cache, ml = _dirs(tmp_path)
        _write_snapshot(cache)
        _write_predictions(ml)
        out = suggest_funds_for_class(
            "VEF", cache_dir=str(cache), ml_dir=str(ml), class_map=CLASS_MAP
        )
        assert all(c["held"] is False for c in out)
