import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch
from src.regime_engine import RegimeEngineV2


class TestRegimeEngineNormalization:
    """Normalize fonksiyonlarının matematiksel doğruluğu."""

    def setup_method(self):
        self.engine = RegimeEngineV2()

    def test_normalize_drawdown_zero(self):
        assert self.engine._normalize_drawdown(0.0) == 0.0

    def test_normalize_drawdown_max(self):
        assert self.engine._normalize_drawdown(-0.30) == pytest.approx(1.0)

    def test_normalize_drawdown_extreme_clipped(self):
        assert self.engine._normalize_drawdown(-0.50) == pytest.approx(1.0)

    def test_normalize_drawdown_negative_input_handled(self):
        assert self.engine._normalize_drawdown(-0.15) == pytest.approx(0.5)

    def test_normalize_volatility_low(self):
        assert self.engine._normalize_volatility(0.15) == 0.0

    def test_normalize_volatility_high(self):
        assert self.engine._normalize_volatility(0.60) == pytest.approx(1.0)

    def test_normalize_momentum_symmetry(self):
        pos = self.engine._normalize_momentum(0.10)
        neg = self.engine._normalize_momentum(-0.10)
        assert pos == neg


class TestRegimeEngineProbabilities:
    """Softmax ve olasılık çıktıları."""

    def setup_method(self):
        self.engine = RegimeEngineV2()

    def test_probabilities_sum_to_one(self):
        scores = {"CRISIS": 0.7, "RISK_ON": 0.2, "RATE_HIKE": 0.4, "STABLE": 0.5}
        probs = self.engine._scores_to_probabilities(scores)
        assert sum(probs.values()) == pytest.approx(1.0, abs=1e-6)

    def test_probabilities_all_positive(self):
        scores = {"CRISIS": 0.7, "RISK_ON": 0.2, "RATE_HIKE": 0.4, "STABLE": 0.5}
        probs = self.engine._scores_to_probabilities(scores)
        assert all(p > 0 for p in probs.values())

    def test_highest_score_highest_probability(self):
        scores = {"CRISIS": 0.9, "RISK_ON": 0.1, "RATE_HIKE": 0.1, "STABLE": 0.1}
        probs = self.engine._scores_to_probabilities(scores)
        assert max(probs, key=probs.get) == "CRISIS"


class TestRegimeEngineWithMockData:
    """yfinance'i mock'layarak deterministik test."""

    def test_compute_score_with_normal_data(self, synthetic_market_data):
        engine = RegimeEngineV2()

        with patch.object(engine, "fetch_live_data", return_value=synthetic_market_data):
            result = engine.compute_composite_score()

        assert "detected" in result
        assert "confidence" in result
        assert "probabilities" in result
        assert "scores" in result
        assert "metrics" in result
        assert "data_quality" in result

        assert 0 <= result["confidence"] <= 1
        assert sum(result["probabilities"].values()) == pytest.approx(1.0, abs=1e-3)
        assert all(0 <= s <= 1 for s in result["scores"].values())
        assert result["detected"] in ["CRISIS", "RISK_ON", "RATE_HIKE", "STABLE"]

    def test_crisis_data_detects_crisis_or_high_dd(self, crisis_market_data):
        engine = RegimeEngineV2()

        with patch.object(engine, "fetch_live_data", return_value=crisis_market_data):
            result = engine.compute_composite_score()

        assert abs(result["metrics"]["dd"]) > 0.10, \
            f"Kriz datasında drawdown az: {result['metrics']['dd']}"

        assert result["probabilities"]["CRISIS"] > result["probabilities"]["STABLE"], \
            f"Kriz datasında CRISIS olasılığı düşük: {result['probabilities']}"


class TestRegimeEngineLookAheadBias:
    """Look-ahead bias prevention — kritik test."""

    def test_as_of_date_passed_to_fetch(self, synthetic_market_data):
        engine = RegimeEngineV2()
        cutoff = pd.Timestamp("2024-09-01")

        with patch.object(engine, "fetch_live_data") as mock_fetch:
            mock_fetch.return_value = synthetic_market_data[synthetic_market_data.index < cutoff]
            result = engine.compute_composite_score(as_of_date=cutoff)

            mock_fetch.assert_called_once()
            call_kwargs = mock_fetch.call_args.kwargs
            assert call_kwargs.get("as_of_date") == cutoff or cutoff in mock_fetch.call_args.args

        as_of = pd.Timestamp(result["data_quality"]["as_of"])
        assert as_of < cutoff, f"Look-ahead leak: {as_of} >= {cutoff}"


class TestRegimeEngineDataQuality:
    """Veri kalitesi raporlama."""

    def test_data_quality_fields_present(self, synthetic_market_data):
        engine = RegimeEngineV2()

        with patch.object(engine, "fetch_live_data", return_value=synthetic_market_data):
            result = engine.compute_composite_score()

        dq = result["data_quality"]
        assert "rows_count" in dq
        assert "missing_pct" in dq
        assert "as_of" in dq
        assert dq["rows_count"] == len(synthetic_market_data)
        assert 0 <= dq["missing_pct"] <= 1


class TestRegimeEngineEdgeCases:
    """Hatalı/eksik veri durumları."""

    def test_insufficient_data_warns_not_raises(self):
        """fetch_live_data < 60 satırda ARTIK fırlatmaz: uyarır ve veriyi döner.

        (Eski sözleşme ValueError fırlatıyordu; caller'lar kendi guard'larını
        uyguladığı için tek eşik compute_composite_score'ta — fetch_live_data
        yetersiz veride sadece warning basar. Bkz. regime_engine.py yorumu.)
        """
        engine = RegimeEngineV2()
        # Sadece 3 satırlık yfinance yanıtı simüle et
        tiny_df = pd.DataFrame(
            {"Close": [100.0, 101.0, 102.0]},
            index=pd.date_range("2024-01-01", periods=3),
        )
        with patch("yfinance.download", return_value=tiny_df):
            result = engine.fetch_live_data()  # fırlatmamalı
        assert isinstance(result, pd.DataFrame)
        assert len(result) <= 3
