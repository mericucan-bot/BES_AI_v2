"""PLAN-15: 'uygulasaydin vs dokunmadin' karnesi testleri."""
import json
from pathlib import Path

import pandas as pd
import pytest

from src.counterfactual import build_tracks, _cost


def _write_snapshot(history_dir: Path, ym: str, date: str, total, weights,
                    class_weights, target_weights) -> None:
    snap = {
        "run_date": f"{date}T10:00:00+03:00",
        "portfolio_value": {"total_value": total, "weights": weights,
                            "class_weights": class_weights},
        "recommendation": {"target_weights": target_weights},
    }
    (history_dir / f"{ym}_snapshot.json").write_text(
        json.dumps(snap), encoding="utf-8")


def _write_nav(cache_dir: Path, prices: dict, dates: list) -> None:
    """prices: {fund_code: [p0,p1,p2]}; dates hizasinda."""
    rows = []
    for code, series in prices.items():
        for dt, px in zip(dates, series):
            rows.append({"fund_code": code, "date": pd.Timestamp(dt), "price": px})
    pd.DataFrame(rows).to_parquet(cache_dir / "nav_history.parquet")


def _write_catmap(cache_dir: Path, cats: dict) -> None:
    df = pd.DataFrame({"fund_code": list(cats), "category": list(cats.values())})
    df.to_parquet(cache_dir / "snapshot_EMK_2026_06.parquet")


def _setup(tmp_path):
    hist = tmp_path / "history"; hist.mkdir()
    cache = tmp_path / "cache"; cache.mkdir()
    return hist, cache


class TestBuildTracks:
    def test_needs_two_snapshots(self, tmp_path):
        hist, cache = _setup(tmp_path)
        _write_snapshot(hist, "2026_05", "2026-05-30", 100000,
                        {"F1": 1.0}, {"VEF": 1.0}, {"VEF": 1.0})
        df = build_tracks(str(hist), str(cache))
        assert df.empty

    def test_actual_track_from_fund_nav(self, tmp_path):
        hist, cache = _setup(tmp_path)
        dates = ["2026-05-30", "2026-06-30", "2026-07-30"]
        # F1 +10% sonra +10%, F2 duz sonra +5%
        _write_nav(cache, {"F1": [100, 110, 121], "F2": [100, 100, 105]}, dates)
        _write_catmap(cache, {"F1": "Stock Fund", "F2": "Gold Fund"})
        for ym, dt in zip(["2026_05", "2026_06", "2026_07"], dates):
            _write_snapshot(hist, ym, dt, 100000,
                            {"F1": 0.5, "F2": 0.5}, {"VEF": 0.5, "ALT": 0.5},
                            {"VEF": 0.5, "ALT": 0.5})
        df = build_tracks(str(hist), str(cache), slippage_pct=0.0)
        assert len(df) == 3
        # Ilk periyot: F1 +%10, F2 +%0 -> portfoy +%5
        assert df.iloc[1]["actual_ret"] == pytest.approx(0.05, abs=1e-6)
        assert df.iloc[1]["basis"].startswith("nav_fund")
        # Ikinci periyot: F1 +%10, F2 +%5 -> +%7.5
        assert df.iloc[2]["actual_ret"] == pytest.approx(0.075, abs=1e-6)

    def test_advised_track_with_rebalance_cost(self, tmp_path):
        hist, cache = _setup(tmp_path)
        dates = ["2026-05-30", "2026-06-30"]
        _write_nav(cache, {"F1": [100, 110], "F2": [100, 100]}, dates)
        _write_catmap(cache, {"F1": "Stock Fund", "F2": "Gold Fund"})
        # Baslangic sinifi 50/50, ilk hedef VEF 100% -> ilk rebalans maliyeti
        _write_snapshot(hist, "2026_05", dates[0], 100000,
                        {"F1": 0.5, "F2": 0.5}, {"VEF": 0.5, "ALT": 0.5},
                        {"VEF": 1.0})
        _write_snapshot(hist, "2026_06", dates[1], 100000,
                        {"F1": 0.5, "F2": 0.5}, {"VEF": 0.5, "ALT": 0.5},
                        {"VEF": 1.0})
        slip = 0.002
        df = build_tracks(str(hist), str(cache), slippage_pct=slip)
        # Ilk rebalans maliyeti: |1.0-0.5|(VEF) + |0-0.5|(ALT) = 1.0 turnover
        expected_start = 100000 * (1 - slip * 1.0)
        assert df.iloc[0]["advised_value"] == pytest.approx(expected_start, abs=0.01)
        # advised getiri = VEF sinifi (F1) getirisi = +%10
        assert df.iloc[1]["advised_ret"] == pytest.approx(0.10, abs=1e-6)

    def test_missing_nav_flat(self, tmp_path):
        hist, cache = _setup(tmp_path)   # nav_history yok
        dates = ["2026-05-30", "2026-06-30"]
        for ym, dt in zip(["2026_05", "2026_06"], dates):
            _write_snapshot(hist, ym, dt, 100000,
                            {"F1": 1.0}, {"VEF": 1.0}, {"VEF": 1.0})
        df = build_tracks(str(hist), str(cache))
        assert len(df) == 2
        assert "flat" in df.iloc[1]["basis"]
        # Deger sabit (getiri 0)
        assert df.iloc[1]["actual_value"] == pytest.approx(100000, abs=0.01)

    def test_start_row_equal(self, tmp_path):
        hist, cache = _setup(tmp_path)
        dates = ["2026-05-30", "2026-06-30"]
        _write_nav(cache, {"F1": [100, 110]}, dates)
        _write_catmap(cache, {"F1": "Stock Fund"})
        # Hedef = mevcut sinif (maliyetsiz baslangic)
        for ym, dt in zip(["2026_05", "2026_06"], dates):
            _write_snapshot(hist, ym, dt, 100000,
                            {"F1": 1.0}, {"VEF": 1.0}, {"VEF": 1.0})
        df = build_tracks(str(hist), str(cache), slippage_pct=0.002)
        assert df.iloc[0]["actual_value"] == pytest.approx(df.iloc[0]["advised_value"], abs=0.01)
        assert df.iloc[0]["basis"] == "start"


class TestCostHelper:
    def test_turnover_cost(self):
        # 50/50 -> 100/0 : turnover 1.0
        assert _cost({"A": 0.5, "B": 0.5}, {"A": 1.0}, 0.002) == pytest.approx(0.002)

    def test_no_change_zero_cost(self):
        assert _cost({"A": 1.0}, {"A": 1.0}, 0.002) == 0.0
