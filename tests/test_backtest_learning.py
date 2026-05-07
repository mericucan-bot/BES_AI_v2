import json
import pytest
from pathlib import Path

from src.backtest_engine import BacktestEngine, BacktestConfig, BacktestResult, MonthlyStep
from src.learning_engine import LearningEngineV2


@pytest.fixture
def sample_backtest_result():
    """Test icin minimal backtest sonucu."""
    steps = [
        MonthlyStep(
            date="2024-07-31", regime="STABLE", confidence=0.7,
            regime_scores={}, target_weights={"VEF": 0.3, "ALT": 0.3, "KTS": 0.3, "CASH": 0.1},
            previous_weights={}, portfolio_return=0.03, benchmark_return=0.02,
            alpha=0.01, net_alpha=0.008, rebalance_cost_pct=0.002,
            turnover_pct=0.2, portfolio_value=103000, benchmark_value=102000,
            data_quality_rows=250,
        ),
        MonthlyStep(
            date="2024-08-30", regime="CRISIS", confidence=0.8,
            regime_scores={}, target_weights={"ALT": 0.6, "KTS": 0.3, "CASH": 0.1},
            previous_weights={}, portfolio_return=-0.01, benchmark_return=-0.03,
            alpha=0.02, net_alpha=0.018, rebalance_cost_pct=0.002,
            turnover_pct=0.3, portfolio_value=101970, benchmark_value=98940,
            data_quality_rows=250,
        ),
        MonthlyStep(
            date="2024-09-30", regime="STABLE", confidence=0.65,
            regime_scores={}, target_weights={"VEF": 0.3, "ALT": 0.3, "KTS": 0.3, "CASH": 0.1},
            previous_weights={}, portfolio_return=0.02, benchmark_return=0.015,
            alpha=0.005, net_alpha=0.003, rebalance_cost_pct=0.002,
            turnover_pct=0.15, portfolio_value=104009, benchmark_value=100424,
            data_quality_rows=250,
        ),
    ]
    return BacktestResult(config=BacktestConfig(), steps=steps, months_count=3)


class TestExportToLearningHistory:
    def test_creates_file(self, tmp_path, sample_backtest_result):
        engine = BacktestEngine()
        output = tmp_path / "learning.json"

        n = engine.export_to_learning_history(sample_backtest_result, str(output))

        assert output.exists()
        assert n == 3

        with open(output) as f:
            data = json.load(f)
        assert len(data) == 3
        assert data[0]["regime"] == "STABLE"
        assert data[1]["regime"] == "CRISIS"

    def test_preserves_existing_observations(self, tmp_path, sample_backtest_result):
        engine = BacktestEngine()
        output = tmp_path / "learning.json"

        existing = [{"date": "2024-06-30", "regime": "RISK_ON", "weights_used": {},
                     "monthly_return": 0.05, "alpha_vs_benchmark": 0.02}]
        with open(output, "w") as f:
            json.dump(existing, f)

        engine.export_to_learning_history(sample_backtest_result, str(output))

        with open(output) as f:
            data = json.load(f)
        assert len(data) == 4  # 1 eski + 3 yeni
        assert data[0]["date"] == "2024-06-30"

    def test_deduplicates_by_date(self, tmp_path, sample_backtest_result):
        engine = BacktestEngine()
        output = tmp_path / "learning.json"

        engine.export_to_learning_history(sample_backtest_result, str(output))
        n = engine.export_to_learning_history(sample_backtest_result, str(output))

        assert n == 0
        with open(output) as f:
            data = json.load(f)
        assert len(data) == 3


class TestLearningIntegration:
    def test_learning_engine_reads_exported_data(self, tmp_path, sample_backtest_result):
        engine = BacktestEngine()
        history_path = tmp_path / "learning.json"

        engine.export_to_learning_history(sample_backtest_result, str(history_path))

        le = LearningEngineV2(history_path=str(history_path))
        stats = le.get_regime_stats()

        assert stats["STABLE"]["n"] == 2
        assert stats["CRISIS"]["n"] == 1

    def test_enough_observations_triggers_learning(self, tmp_path):
        """7+ gozlem birikince statik prior yerine ogrenilmis agirliklar kullanilir."""
        history_path = tmp_path / "learning.json"

        observations = []
        for i in range(7):
            observations.append({
                "date": f"2024-{i+1:02d}-28",
                "regime": "STABLE",
                "weights_used": {"VEF": 0.25 + i*0.02, "ALT": 0.25, "KTS": 0.25 - i*0.01, "CASH": 0.25 - i*0.01},
                "monthly_return": 0.02 + i*0.005,
                "alpha_vs_benchmark": 0.005 + i*0.002,
            })

        with open(history_path, "w") as f:
            json.dump(observations, f)

        le = LearningEngineV2(history_path=str(history_path))
        weights = le.get_optimized_weights("STABLE")

        static = LearningEngineV2.STATIC_PRIORS["STABLE"]
        assert weights != static, "7 gozlemden sonra ogrenilmis agirliklar kullanilmali"
        assert abs(sum(weights.values()) - 1.0) < 0.01
