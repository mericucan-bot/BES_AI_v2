import json
import pytest
from pathlib import Path
from unittest.mock import patch
import pandas as pd
import numpy as np

from src.pipeline import MonthlyPipeline


@pytest.fixture
def pipeline_dirs(tmp_path):
    portfolio_path = tmp_path / "portfolio.json"
    history_dir   = tmp_path / "history"
    learning_path  = tmp_path / "learning.json"

    portfolio_path.write_text(json.dumps({
        "holdings_tl": {"VEF": 30000, "ALT": 25000, "KTS": 20000, "KCH": 15000, "CASH": 10000}
    }), encoding="utf-8")
    history_dir.mkdir()

    return {
        "portfolio_path": str(portfolio_path),
        "history_dir":    str(history_dir),
        "learning_path":  str(learning_path),
    }


@pytest.fixture
def mock_regime_result():
    return {
        "detected": "STABLE",
        "confidence": 0.65,
        "scores": {"CRISIS": 0.2, "STABLE": 0.65, "RISK_ON": 0.1, "RATE_HIKE": 0.05},
        "probabilities": {"CRISIS": 0.2, "STABLE": 0.5, "RISK_ON": 0.2, "RATE_HIKE": 0.1},
        "metrics": {"dd": -0.05, "vol": 0.18, "usd_mom": 0.02, "gold_mom": 0.03, "bist_60d_return": 0.04},
        "data_quality": {"rows_count": 250, "missing_pct": 0.0, "as_of": "2026-04-29"},
    }


class TestPipelineFirstRun:
    def test_run_no_previous_snapshot(self, pipeline_dirs, mock_regime_result):
        pipeline = MonthlyPipeline(**pipeline_dirs)

        with patch.object(pipeline.regime_engine, "compute_composite_score",
                          return_value=mock_regime_result):
            result = pipeline.run()

        assert result["status"] == "SUCCESS"
        assert result["portfolio_value"]["total_value"] == 100000
        assert result["regime"]["detected"] == "STABLE"
        assert result["previous_evaluation"] is None
        assert "actions" in result["recommendation"]

    def test_snapshot_file_created(self, pipeline_dirs, mock_regime_result):
        pipeline = MonthlyPipeline(**pipeline_dirs)

        with patch.object(pipeline.regime_engine, "compute_composite_score",
                          return_value=mock_regime_result):
            result = pipeline.run()

        snapshot_path = Path(result["snapshot_path"])
        assert snapshot_path.exists()
        with open(snapshot_path, encoding="utf-8") as f:
            saved = json.load(f)
        assert saved["regime"]["detected"] == "STABLE"
        assert saved["status"] == "SUCCESS"


class TestPipelineSecondRun:
    def _make_prev_snapshot(self, history_dir: Path) -> None:
        prev = {
            "run_date": "2026-03-30T10:00:00+03:00",
            "portfolio_value": {"total_value": 95000, "weights": {}, "date": "2026-03-30"},
            "regime": {"detected": "CRISIS"},
            "recommendation": {"target_weights": {"ALT": 0.6, "KTS": 0.3, "CASH": 0.1}},
        }
        (history_dir / "2026_03_snapshot.json").write_text(json.dumps(prev), encoding="utf-8")

    def test_evaluates_previous_observation(self, pipeline_dirs, mock_regime_result):
        self._make_prev_snapshot(Path(pipeline_dirs["history_dir"]))
        pipeline = MonthlyPipeline(**pipeline_dirs)

        synth_dates = pd.date_range(end="2026-04-29", periods=60, freq="B")
        synth_data  = pd.DataFrame({
            "BIST":   np.linspace(8000, 8400, 60),
            "USDTRY": np.linspace(33, 34, 60),
            "GOLD":   np.linspace(2100, 2200, 60),
        }, index=synth_dates)

        with patch.object(pipeline.regime_engine, "compute_composite_score",
                          return_value=mock_regime_result), \
             patch.object(pipeline.regime_engine, "fetch_live_data",
                          return_value=synth_data):
            result = pipeline.run()

        assert result["status"] == "SUCCESS"
        ev = result["previous_evaluation"]
        assert ev is not None
        assert ev["previous_value"] == 95000
        assert ev["current_value"]  == 100000
        assert ev["monthly_return"] == pytest.approx(0.0526, abs=0.001)
        assert ev["status"] in ["WIN", "LOSS", "NEUTRAL"]

    def test_observation_recorded_in_learning_history(self, pipeline_dirs, mock_regime_result):
        self._make_prev_snapshot(Path(pipeline_dirs["history_dir"]))
        pipeline = MonthlyPipeline(**pipeline_dirs)

        with patch.object(pipeline.regime_engine, "compute_composite_score",
                          return_value=mock_regime_result), \
             patch.object(pipeline.regime_engine, "fetch_live_data",
                          return_value=pd.DataFrame()):  # benchmark = 0
            pipeline.run()

        learning_path = Path(pipeline_dirs["learning_path"])
        assert learning_path.exists()
        with open(learning_path, encoding="utf-8") as f:
            history = json.load(f)
        assert len(history) == 1
        assert history[0]["regime"] == "CRISIS"


class TestPipelineRecommendations:
    def test_hold_below_threshold(self, pipeline_dirs):
        pipeline = MonthlyPipeline(**pipeline_dirs)
        recs = pipeline._generate_recommendations(
            current_weights={"VEF": 0.30},
            target_weights={"VEF": 0.3009},  # 90 TL fark: 0.0009 * 100_000 = 90 < 100
            total_value=100000,
            min_threshold_tl=100,
        )
        assert recs[0]["action"] == "HOLD"

    def test_buy_and_sell_above_threshold(self, pipeline_dirs):
        pipeline = MonthlyPipeline(**pipeline_dirs)
        recs = pipeline._generate_recommendations(
            current_weights={"VEF": 0.30, "ALT": 0.40},
            target_weights={"VEF": 0.40, "ALT": 0.30},
            total_value=100000,
            min_threshold_tl=100,
        )
        actions = {r["asset"]: r["action"] for r in recs}
        assert actions["VEF"] == "BUY"
        assert actions["ALT"] == "SELL"


class TestPipelineErrors:
    def test_missing_portfolio(self, tmp_path):
        pipeline = MonthlyPipeline(
            portfolio_path=str(tmp_path / "nonexistent.json"),
            history_dir=str(tmp_path / "history"),
            learning_path=str(tmp_path / "learning.json"),
        )
        result = pipeline.run()
        assert result["status"] == "ERROR"
        assert "Portföy" in result["message"]

    def test_empty_holdings(self, tmp_path):
        portfolio_path = tmp_path / "empty.json"
        portfolio_path.write_text(json.dumps({"holdings_tl": {}}), encoding="utf-8")

        pipeline = MonthlyPipeline(
            portfolio_path=str(portfolio_path),
            history_dir=str(tmp_path / "history"),
            learning_path=str(tmp_path / "learning.json"),
        )
        result = pipeline.run()
        assert result["status"] == "ERROR"


class TestPipelineIdempotency:
    def test_same_month_overwrites_snapshot(self, pipeline_dirs, mock_regime_result):
        """Ayni ay iki kere calistirilirsa onceki snapshot uzerine yazilir."""
        pipeline = MonthlyPipeline(**pipeline_dirs)

        with patch.object(pipeline.regime_engine, "compute_composite_score",
                          return_value=mock_regime_result):
            r1 = pipeline.run()
            r2 = pipeline.run()

        assert r1["status"] == "SUCCESS"
        assert r2["status"] == "SUCCESS"
        assert r1["snapshot_path"] == r2["snapshot_path"]
        snapshots = list(Path(pipeline_dirs["history_dir"]).glob("*_snapshot.json"))
        assert len(snapshots) == 1
