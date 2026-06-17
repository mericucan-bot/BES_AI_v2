import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from src.backtest_engine import BacktestEngine, BacktestConfig, BacktestResult, MonthlyStep
from src.regime_engine import RegimeEngineV2


@pytest.fixture
def synthetic_data():
    """Deterministik test verisi."""
    dates = pd.date_range(start="2023-01-01", end="2026-03-31", freq="B")
    np.random.seed(42)
    n = len(dates)
    bist  = 10000 * np.cumprod(1 + np.random.normal(0.0005, 0.015, n))
    gold  = 2000  * np.cumprod(1 + np.random.normal(0.0003, 0.010, n))
    usdtry = 30   * np.cumprod(1 + np.random.normal(0.001,  0.008, n))
    return pd.DataFrame({"BIST": bist, "GOLD": gold, "USDTRY": usdtry}, index=dates)


@pytest.fixture
def mock_regime_stable():
    return {
        "detected": "STABLE",
        "confidence": 0.7,
        "scores": {"CRISIS": 0.1, "STABLE": 0.7, "RISK_ON": 0.15, "RATE_HIKE": 0.05},
        "data_quality": {"rows_count": 250},
    }


class TestBacktestConfig:
    def test_default_config(self):
        config = BacktestConfig()
        assert config.initial_capital == 100_000
        assert config.use_learning is False

    def test_custom_config(self):
        config = BacktestConfig(start_date="2025-01-01", initial_capital=50_000)
        assert config.start_date == "2025-01-01"
        assert config.initial_capital == 50_000


class TestBacktestDates:
    def test_generate_rebalance_dates(self):
        config = BacktestConfig(start_date="2024-01-01", end_date="2024-07-01")
        engine = BacktestEngine(config)
        dates = engine._generate_rebalance_dates()
        assert len(dates) >= 5
        for d in dates:
            # Business month end = son is gunu, her zaman ay sonuna yakin (>= 28. gun)
            assert d.day >= 28

    def test_too_short_range(self):
        config = BacktestConfig(start_date="2024-01-15", end_date="2024-01-20")
        engine = BacktestEngine(config)
        dates = engine._generate_rebalance_dates()
        assert len(dates) <= 1


class TestBacktestAntiLookahead:
    def test_as_of_date_used_in_regime_detection(self, synthetic_data, mock_regime_stable):
        """RegimeEngine as_of_date ile cagrilmali — look-ahead korumasi."""
        config = BacktestConfig(start_date="2024-06-01", end_date="2024-09-01")
        engine = BacktestEngine(config)

        as_of_dates_called = []

        def tracking_compute(**kwargs):
            as_of_dates_called.append(kwargs.get("as_of_date"))
            return mock_regime_stable

        with patch.object(engine.regime_engine, "compute_composite_score",
                          side_effect=tracking_compute), \
             patch.object(engine.regime_engine, "fetch_live_data",
                          return_value=synthetic_data), \
             patch("src.backtest_engine.MacroEngine") as mock_macro:
            mock_macro.return_value.get_macro_snapshot.return_value = {"tcmb_rate_change": 0}
            engine.run()

        for date in as_of_dates_called:
            assert date is not None, "compute_composite_score as_of_date=None ile cagrildi — LOOK-AHEAD RISKI!"


class TestBacktestRun:
    def test_run_produces_steps(self, synthetic_data, mock_regime_stable):
        config = BacktestConfig(start_date="2024-06-01", end_date="2025-01-01")
        engine = BacktestEngine(config)

        with patch.object(engine.regime_engine, "compute_composite_score",
                          return_value=mock_regime_stable), \
             patch.object(engine.regime_engine, "fetch_live_data",
                          return_value=synthetic_data), \
             patch("src.backtest_engine.MacroEngine") as mock_macro:
            mock_macro.return_value.get_macro_snapshot.return_value = {"tcmb_rate_change": 0}
            result = engine.run()

        assert result.months_count >= 5
        assert len(result.steps) >= 5
        assert result.steps[-1].portfolio_value > 0

    def test_portfolio_value_changes(self, synthetic_data, mock_regime_stable):
        config = BacktestConfig(
            start_date="2024-06-01", end_date="2025-01-01", initial_capital=100_000
        )
        engine = BacktestEngine(config)

        with patch.object(engine.regime_engine, "compute_composite_score",
                          return_value=mock_regime_stable), \
             patch.object(engine.regime_engine, "fetch_live_data",
                          return_value=synthetic_data), \
             patch("src.backtest_engine.MacroEngine") as mock_macro:
            mock_macro.return_value.get_macro_snapshot.return_value = {"tcmb_rate_change": 0}
            result = engine.run()

        assert result.steps[-1].portfolio_value != config.initial_capital

    def test_benchmark_tracked_separately(self, synthetic_data, mock_regime_stable):
        config = BacktestConfig(start_date="2024-06-01", end_date="2025-01-01")
        engine = BacktestEngine(config)

        with patch.object(engine.regime_engine, "compute_composite_score",
                          return_value=mock_regime_stable), \
             patch.object(engine.regime_engine, "fetch_live_data",
                          return_value=synthetic_data), \
             patch("src.backtest_engine.MacroEngine") as mock_macro:
            mock_macro.return_value.get_macro_snapshot.return_value = {"tcmb_rate_change": 0}
            result = engine.run()

        assert result.steps[-1].benchmark_value > 0


class TestBacktestMetrics:
    def test_metrics_calculated(self, synthetic_data, mock_regime_stable):
        config = BacktestConfig(start_date="2024-06-01", end_date="2025-06-01")
        engine = BacktestEngine(config)

        with patch.object(engine.regime_engine, "compute_composite_score",
                          return_value=mock_regime_stable), \
             patch.object(engine.regime_engine, "fetch_live_data",
                          return_value=synthetic_data), \
             patch("src.backtest_engine.MacroEngine") as mock_macro:
            mock_macro.return_value.get_macro_snapshot.return_value = {"tcmb_rate_change": 0}
            result = engine.run()

        assert 0 <= result.win_rate <= 1
        assert result.max_drawdown <= 0
        assert result.months_count > 0

    def test_sharpe_ratio_is_float(self, synthetic_data, mock_regime_stable):
        """Sharpe negatif olabilir — TR'de risk-free %36+"""
        config = BacktestConfig(start_date="2024-06-01", end_date="2025-06-01")
        engine = BacktestEngine(config)

        with patch.object(engine.regime_engine, "compute_composite_score",
                          return_value=mock_regime_stable), \
             patch.object(engine.regime_engine, "fetch_live_data",
                          return_value=synthetic_data), \
             patch("src.backtest_engine.MacroEngine") as mock_macro:
            mock_macro.return_value.get_macro_snapshot.return_value = {"tcmb_rate_change": 0}
            result = engine.run()

        assert isinstance(result.sharpe_ratio, float)

    def test_win_rate_in_range(self, synthetic_data, mock_regime_stable):
        config = BacktestConfig(start_date="2024-06-01", end_date="2025-06-01")
        engine = BacktestEngine(config)

        with patch.object(engine.regime_engine, "compute_composite_score",
                          return_value=mock_regime_stable), \
             patch.object(engine.regime_engine, "fetch_live_data",
                          return_value=synthetic_data), \
             patch("src.backtest_engine.MacroEngine") as mock_macro:
            mock_macro.return_value.get_macro_snapshot.return_value = {"tcmb_rate_change": 0}
            result = engine.run()

        assert 0 <= result.win_rate <= 1


class TestBacktestOutput:
    def test_to_dataframe(self, synthetic_data, mock_regime_stable):
        config = BacktestConfig(start_date="2024-06-01", end_date="2025-01-01")
        engine = BacktestEngine(config)

        with patch.object(engine.regime_engine, "compute_composite_score",
                          return_value=mock_regime_stable), \
             patch.object(engine.regime_engine, "fetch_live_data",
                          return_value=synthetic_data), \
             patch("src.backtest_engine.MacroEngine") as mock_macro:
            mock_macro.return_value.get_macro_snapshot.return_value = {"tcmb_rate_change": 0}
            result = engine.run()

        df = engine.to_dataframe(result)
        assert not df.empty
        assert "portfolio_value" in df.columns
        assert "regime" in df.columns
        assert len(df) == result.months_count

    def test_print_summary(self, synthetic_data, mock_regime_stable):
        config = BacktestConfig(start_date="2024-06-01", end_date="2025-01-01")
        engine = BacktestEngine(config)

        with patch.object(engine.regime_engine, "compute_composite_score",
                          return_value=mock_regime_stable), \
             patch.object(engine.regime_engine, "fetch_live_data",
                          return_value=synthetic_data), \
             patch("src.backtest_engine.MacroEngine") as mock_macro:
            mock_macro.return_value.get_macro_snapshot.return_value = {"tcmb_rate_change": 0}
            result = engine.run()

        summary = engine.print_summary(result)
        assert "BACKTEST SONUCLARI" in summary
        assert "CAGR" in summary
        assert "Sharpe" in summary
        assert "Win Rate" in summary

    def test_empty_result_summary(self):
        engine = BacktestEngine()
        result = BacktestResult(config=BacktestConfig(), steps=[])
        summary = engine.print_summary(result)
        assert "sonuc yok" in summary.lower()


class TestBacktestCosts:
    def test_costs_reduce_portfolio_value(self, synthetic_data):
        """Transaction cost'lar portfoy degerini dusturmeli."""
        from src.cost_model import CostConfig

        # Rejim degisimi => turnover => maliyet farki gorulur
        regimes = [
            {"detected": "STABLE",   "confidence": 0.7, "scores": {}, "data_quality": {"rows_count": 250}},
            {"detected": "RISK_ON",  "confidence": 0.7, "scores": {}, "data_quality": {"rows_count": 250}},
            {"detected": "STABLE",   "confidence": 0.7, "scores": {}, "data_quality": {"rows_count": 250}},
            {"detected": "RISK_ON",  "confidence": 0.7, "scores": {}, "data_quality": {"rows_count": 250}},
            {"detected": "STABLE",   "confidence": 0.7, "scores": {}, "data_quality": {"rows_count": 250}},
            {"detected": "RISK_ON",  "confidence": 0.7, "scores": {}, "data_quality": {"rows_count": 250}},
        ]

        config_high = BacktestConfig(
            start_date="2024-06-01", end_date="2025-01-01",
            cost_config=CostConfig(slippage_pct=0.01),
        )
        config_zero = BacktestConfig(
            start_date="2024-06-01", end_date="2025-01-01",
            cost_config=CostConfig(slippage_pct=0.0),
        )

        with patch.object(RegimeEngineV2, "compute_composite_score",
                          side_effect=regimes * 3), \
             patch.object(RegimeEngineV2, "fetch_live_data",
                          return_value=synthetic_data), \
             patch("src.backtest_engine.MacroEngine") as mock_macro:
            mock_macro.return_value.get_macro_snapshot.return_value = {"tcmb_rate_change": 0}
            result_high = BacktestEngine(config_high).run()

        with patch.object(RegimeEngineV2, "compute_composite_score",
                          side_effect=regimes * 3), \
             patch.object(RegimeEngineV2, "fetch_live_data",
                          return_value=synthetic_data), \
             patch("src.backtest_engine.MacroEngine") as mock_macro:
            mock_macro.return_value.get_macro_snapshot.return_value = {"tcmb_rate_change": 0}
            result_zero = BacktestEngine(config_zero).run()

        # Yuksek maliyet → portfoy degeri daha dusuk olmali
        assert result_high.total_cost_pct > result_zero.total_cost_pct
        assert result_high.steps[-1].portfolio_value < result_zero.steps[-1].portfolio_value


class TestBacktestVisualization:
    def test_to_dataframe_columns(self, synthetic_data, mock_regime_stable):
        config = BacktestConfig(start_date="2024-06-01", end_date="2025-01-01")
        engine = BacktestEngine(config)

        with patch.object(engine.regime_engine, "compute_composite_score",
                          return_value=mock_regime_stable), \
             patch.object(engine.regime_engine, "fetch_live_data",
                          return_value=synthetic_data), \
             patch("src.backtest_engine.MacroEngine") as mock_macro:
            mock_macro.return_value.get_macro_snapshot.return_value = {"tcmb_rate_change": 0}
            result = engine.run()

        df = engine.to_dataframe(result)
        required_cols = ["regime", "confidence", "portfolio_return", "benchmark_return",
                         "alpha", "net_alpha", "cost_pct", "portfolio_value", "benchmark_value"]
        for col in required_cols:
            assert col in df.columns, f"Eksik sutun: {col}"

    def test_to_dataframe_index_is_datetime(self, synthetic_data, mock_regime_stable):
        config = BacktestConfig(start_date="2024-06-01", end_date="2025-01-01")
        engine = BacktestEngine(config)

        with patch.object(engine.regime_engine, "compute_composite_score",
                          return_value=mock_regime_stable), \
             patch.object(engine.regime_engine, "fetch_live_data",
                          return_value=synthetic_data), \
             patch("src.backtest_engine.MacroEngine") as mock_macro:
            mock_macro.return_value.get_macro_snapshot.return_value = {"tcmb_rate_change": 0}
            result = engine.run()

        df = engine.to_dataframe(result)
        assert isinstance(df.index, pd.DatetimeIndex)

    def test_summary_contains_all_sections(self, synthetic_data, mock_regime_stable):
        config = BacktestConfig(start_date="2024-06-01", end_date="2025-01-01")
        engine = BacktestEngine(config)

        with patch.object(engine.regime_engine, "compute_composite_score",
                          return_value=mock_regime_stable), \
             patch.object(engine.regime_engine, "fetch_live_data",
                          return_value=synthetic_data), \
             patch("src.backtest_engine.MacroEngine") as mock_macro:
            mock_macro.return_value.get_macro_snapshot.return_value = {"tcmb_rate_change": 0}
            result = engine.run()

        summary = engine.print_summary(result)
        assert "Toplam Getiri" in summary
        assert "CAGR" in summary
        assert "Sharpe" in summary
        assert "Max Drawdown" in summary
        assert "Win Rate" in summary
        assert "Rejim Dagilimi" in summary
        assert "Equity Curve" in summary


class TestRealNavProviderTz:
    """returns_asof tz-aware girdiyle (pipeline run_date gibi) cokmemeli."""

    def _provider(self, tmp_path):
        import pandas as pd
        from src.backtest_engine import RealNavReturnProvider
        df = pd.DataFrame({
            "date": ["2026-06-16"] * 3,
            "fund_code": ["AAA", "BBB", "CCC"],
            "category": ["Stock Fund", "Gold Fund", "Money Market Fund"],
            "return_1m": [5.0, -1.0, 3.0],
        })
        df.to_parquet(tmp_path / "snapshot_EMK_20260616.parquet")
        return RealNavReturnProvider(cache_dir=str(tmp_path))

    def test_tz_aware_timestamp_does_not_raise(self, tmp_path):
        from datetime import datetime
        from zoneinfo import ZoneInfo
        import pandas as pd
        prov = self._provider(tmp_path)
        ts = pd.Timestamp(datetime(2026, 6, 16, tzinfo=ZoneInfo("Europe/Istanbul")))
        res = prov.returns_asof(ts)
        assert isinstance(res, dict)
        assert res["VEF"] == pytest.approx(0.05)  # Stock Fund return_1m/100

    def test_tz_naive_still_works(self, tmp_path):
        import pandas as pd
        prov = self._provider(tmp_path)
        res = prov.returns_asof(pd.Timestamp("2026-06-16"))
        assert isinstance(res, dict)


class TestRealNavHistory:
    """nav_history.parquet'ten tam-dönem gerçek getiri (returns_between)."""

    def _cache(self, tmp_path):
        import pandas as pd
        # Kategori haritası (snapshot): AAA=Stock(VEF), BBB=Gold(ALT)
        snap = pd.DataFrame({
            "date": ["2026-06-16"] * 2,
            "fund_code": ["AAA", "BBB"],
            "fund_name": ["A", "B"],
            "category": ["Stock Fund", "Gold Fund"],
            "return_1m": [1.0, 2.0],
        })
        snap.to_parquet(tmp_path / "snapshot_EMK_20260616.parquet")
        # Günlük NAV: AAA 100->110 (+%10), BBB 50->45 (-%10)
        nav = pd.DataFrame({
            "fund_code": ["AAA","AAA","AAA","BBB","BBB","BBB"],
            "date": ["2025-01-02","2025-01-15","2025-01-31"]*2,
            "price": [100.0,105.0,110.0, 50.0,48.0,45.0],
        })
        nav.to_parquet(tmp_path / "nav_history.parquet")
        return str(tmp_path)

    def _provider(self, tmp_path):
        from src.backtest_engine import RealNavReturnProvider
        return RealNavReturnProvider(cache_dir=self._cache(tmp_path))

    def test_nav_history_loaded(self, tmp_path):
        p = self._provider(tmp_path)
        assert p.has_nav_history()
        assert "AAA" in p._asset_funds["VEF"]
        assert "BBB" in p._asset_funds["ALT"]

    def test_returns_between_exact(self, tmp_path):
        p = self._provider(tmp_path)
        r = p.returns_between("2025-01-02", "2025-01-31")
        assert r["VEF"] == pytest.approx(0.10)   # 100->110
        assert r["ALT"] == pytest.approx(-0.10)  # 50->45

    def test_returns_between_asof(self, tmp_path):
        # Aralik disi/asof: 2025-01-20 -> en son <= o tarih (105)
        p = self._provider(tmp_path)
        r = p.returns_between("2025-01-02", "2025-01-20")
        assert r["VEF"] == pytest.approx(0.05)   # 100->105

    def test_before_data_returns_none(self, tmp_path):
        p = self._provider(tmp_path)
        assert p.returns_between("2020-01-01", "2020-02-01") is None
