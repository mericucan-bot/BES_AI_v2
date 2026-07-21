"""PLAN-17: kisisel gecmis backtest (kendi fonlarinla 2 yil)."""
from unittest.mock import patch

import pandas as pd
import pytest

from src.personal_backtest import run_personal_backtest, _cost


def _write_nav(cache_dir, prices, dates):
    """prices: {fund_code: [p0,p1,...]}; dates hizasinda."""
    rows = []
    for code, series in prices.items():
        for dt, px in zip(dates, series):
            rows.append({"fund_code": code, "date": pd.Timestamp(dt), "price": px})
    pd.DataFrame(rows).to_parquet(cache_dir / "nav_history.parquet")


def _write_catmap(cache_dir, cats):
    df = pd.DataFrame({"fund_code": list(cats), "category": list(cats.values())})
    df.to_parquet(cache_dir / "snapshot_EMK_2026_06.parquet")


def _regime_stable(*_args, **_kwargs):
    return {"detected": "STABLE", "confidence": 0.7}


# BME ile hizali 3 tarih (2024-05-31, 2024-06-28, 2024-07-31)
_BME_DATES = list(pd.date_range("2024-05-31", "2024-07-31", freq="BME"))


class TestPersonalBacktest:
    def test_empty_without_nav(self, tmp_path):
        cache = tmp_path / "cache"
        cache.mkdir()
        with patch(
            "src.regime_engine.RegimeEngineV2.compute_composite_score",
            side_effect=_regime_stable,
        ):
            df = run_personal_backtest(
                {"F1": 50000, "F2": 50000}, cache_dir=str(cache)
            )
        assert df.empty

    def test_two_leg_tracks(self, tmp_path):
        cache = tmp_path / "cache"
        cache.mkdir()
        dates = _BME_DATES  # 3 BME tarihi
        # F1 +%10/+%10, F2 duz/+%5
        _write_nav(cache, {"F1": [100, 110, 121], "F2": [100, 100, 105]}, dates)
        _write_catmap(cache, {"F1": "Stock Fund", "F2": "Gold Fund"})

        with patch(
            "src.regime_engine.RegimeEngineV2.compute_composite_score",
            side_effect=_regime_stable,
        ):
            df = run_personal_backtest(
                {"F1": 50000, "F2": 50000},
                cache_dir=str(cache),
                initial_capital=100_000.0,
                slippage_pct=0.0,  # hold zinciri icin maliyet kapali
            )

        assert len(df) >= 3
        # Ilk periyot hold: 50/50 * (10%, 0%) = +5%
        assert df.iloc[1]["hold_ret"] == pytest.approx(0.05, abs=1e-6)
        assert df.iloc[1]["hold_value"] == pytest.approx(105_000.0, abs=1e-3)

        # Ikinci periyot: agirlik kaymasi sonrasi
        # w1=0.5*1.1/1.05, w2=0.5/1.05; ret = w1*0.1 + w2*0.05
        w1 = 0.5 * 1.1 / 1.05
        w2 = 0.5 / 1.05
        expected_r2 = w1 * 0.10 + w2 * 0.05
        assert df.iloc[2]["hold_ret"] == pytest.approx(expected_r2, abs=1e-6)
        # Buy-and-hold son: 50k*1.21 + 50k*1.05 = 113000
        assert df.iloc[2]["hold_value"] == pytest.approx(113_000.0, abs=1e-2)

        # advised sinif sepeti kullanir (STABLE: VEF/ALT/KTS/CASH)
        assert "advised_value" in df.columns
        assert df.iloc[1]["advised_ret"] is not None

    def test_rebalance_cost_applied(self, tmp_path):
        cache = tmp_path / "cache"
        cache.mkdir()
        dates = _BME_DATES[:2]  # 2 tarih yeterli degil; 3 lazim for range
        # 3 BME tarihi, duz fiyat (getiri 0) — sadece baslangic maliyeti net gorunsun
        dates = _BME_DATES
        _write_nav(cache, {"F1": [100, 100, 100], "F2": [100, 100, 100]}, dates)
        _write_catmap(cache, {"F1": "Stock Fund", "F2": "Gold Fund"})

        slip = 0.002
        # Baslangic sinif 50/50 VEF/ALT; STABLE hedef 0.3/0.3/0.3/0.1
        # turnover = |0.3-0.5|*2 + 0.3 + 0.1 = 0.8
        expected_turnover = 0.8
        expected_start_after = 100_000.0 * (1.0 - slip * expected_turnover)

        with patch(
            "src.regime_engine.RegimeEngineV2.compute_composite_score",
            side_effect=_regime_stable,
        ):
            df = run_personal_backtest(
                {"F1": 50000, "F2": 50000},
                cache_dir=str(cache),
                initial_capital=100_000.0,
                slippage_pct=slip,
            )

        assert not df.empty
        # Start satiri initial_capital; maliyet ilk getiri oncesi dusulur
        assert df.iloc[0]["advised_value"] == pytest.approx(100_000.0, abs=0.01)
        # Getiri 0 iken sonraki satir ~ expected_start_after (sonraki rebalans da
        # ayni hedefe gidebilir → ek maliyet ~0)
        assert df.iloc[1]["advised_value"] == pytest.approx(expected_start_after, abs=1.0)

        # Yardimci dogrulama
        assert _cost({"VEF": 0.5, "ALT": 0.5}, {"VEF": 0.3, "ALT": 0.3, "KTS": 0.3, "CASH": 0.1}, slip) == pytest.approx(
            slip * expected_turnover, abs=1e-9
        )

    def test_missing_fund_returns_flat(self, tmp_path):
        cache = tmp_path / "cache"
        cache.mkdir()
        dates = _BME_DATES
        _write_nav(cache, {"F1": [100, 110, 121], "F2": [100, 100, 105]}, dates)
        _write_catmap(cache, {"F1": "Stock Fund", "F2": "Gold Fund"})

        with patch(
            "src.regime_engine.RegimeEngineV2.compute_composite_score",
            side_effect=_regime_stable,
        ), patch(
            "src.backtest_engine.RealNavReturnProvider.fund_returns_between",
            return_value=None,
        ):
            df = run_personal_backtest(
                {"F1": 50000, "F2": 50000},
                cache_dir=str(cache),
                initial_capital=100_000.0,
                slippage_pct=0.0,
            )

        assert not df.empty
        # hold_ret 0, deger sabit, crash yok
        assert df.iloc[1]["hold_ret"] == pytest.approx(0.0, abs=1e-9)
        assert df.iloc[1]["hold_value"] == pytest.approx(100_000.0, abs=0.01)

    def test_start_row(self, tmp_path):
        cache = tmp_path / "cache"
        cache.mkdir()
        dates = _BME_DATES
        _write_nav(cache, {"F1": [100, 110, 121], "F2": [100, 100, 105]}, dates)
        _write_catmap(cache, {"F1": "Stock Fund", "F2": "Gold Fund"})

        with patch(
            "src.regime_engine.RegimeEngineV2.compute_composite_score",
            side_effect=_regime_stable,
        ):
            df = run_personal_backtest(
                {"F1": 50000, "F2": 50000},
                cache_dir=str(cache),
                initial_capital=100_000.0,
            )

        assert not df.empty
        assert df.iloc[0]["hold_value"] == pytest.approx(100_000.0, abs=0.01)
        assert df.iloc[0]["advised_value"] == pytest.approx(100_000.0, abs=0.01)
        assert df.iloc[0]["regime"] == "STABLE"
        assert pd.isna(df.iloc[0]["hold_ret"])
        assert pd.isna(df.iloc[0]["advised_ret"])
