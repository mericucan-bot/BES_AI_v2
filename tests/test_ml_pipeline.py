import json
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import patch

from src.ml_pipeline import MLPipeline


@pytest.fixture
def mock_fund_navs():
    """3 fonluk 300-gunluk gunluk NAV verisi (test icin)."""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=300, freq="B")
    navs = {}
    for code in ["FUND1", "FUND2", "FUND3"]:
        drift = np.random.uniform(0.0003, 0.001)
        vol = np.random.uniform(0.01, 0.02)
        prices = 100 * np.cumprod(1 + np.random.normal(drift, vol, 300))
        navs[code] = pd.Series(prices, index=dates)
    return navs


@pytest.fixture
def mock_monthly_navs():
    """3 fonluk 24-aylik sentetik aylik NAV (TEFAS gercek verisi gibi)."""
    np.random.seed(42)
    dates = pd.date_range("2024-01-31", periods=24, freq="BME")
    navs = {}
    for code in ["AEA", "IPB", "GAE"]:
        drift = np.random.uniform(0.005, 0.015)
        vol = np.random.uniform(0.02, 0.05)
        prices = 100 * np.cumprod(1 + np.random.normal(drift, vol, 24))
        navs[code] = pd.Series(prices, index=dates)
    return navs


@pytest.fixture
def mock_market_data():
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=300, freq="B")
    return pd.DataFrame(
        {
            "BIST": 10000 * np.cumprod(1 + np.random.normal(0.0005, 0.015, 300)),
            "USDTRY": 30 * np.cumprod(1 + np.random.normal(0.001, 0.008, 300)),
            "GOLD": 2000 * np.cumprod(1 + np.random.normal(0.0003, 0.010, 300)),
        },
        index=dates,
    )


@pytest.fixture
def macro_patch():
    return {"cpi_yoy": 0.306, "current_policy_rate": 42.5}


class TestMLPipelineInit:
    def test_init_creates_output_dir(self, tmp_path):
        ml = MLPipeline(output_dir=str(tmp_path / "ml"))
        assert ml.output_dir.exists()


class TestBuildFeatures:
    def test_daily_nav_uses_feature_engineer(self, tmp_path, mock_fund_navs, mock_market_data, macro_patch):
        ml = MLPipeline(output_dir=str(tmp_path / "ml"))
        with patch.object(ml.macro_engine, "get_macro_snapshot", return_value=macro_patch):
            dataset = ml.build_features(mock_fund_navs, mock_market_data, sample_frequency="weekly")
        assert not dataset.empty
        assert dataset["fund_code"].nunique() == 3
        assert "fwd_return_3m" in dataset.columns

    def test_monthly_nav_uses_snapshot_builder(self, tmp_path, mock_monthly_navs, mock_market_data, macro_patch):
        ml = MLPipeline(output_dir=str(tmp_path / "ml"))
        with patch.object(ml.macro_engine, "get_macro_snapshot", return_value=macro_patch):
            dataset = ml.build_features(mock_monthly_navs, mock_market_data, sample_frequency="monthly")
        assert not dataset.empty
        assert dataset["fund_code"].nunique() == 3
        assert "fwd_return_3m" in dataset.columns

    def test_auto_detects_monthly(self, tmp_path, mock_monthly_navs, mock_market_data, macro_patch):
        ml = MLPipeline(output_dir=str(tmp_path / "ml"))
        with patch.object(ml.macro_engine, "get_macro_snapshot", return_value=macro_patch):
            dataset = ml.build_features(mock_monthly_navs, mock_market_data, sample_frequency="auto")
        assert not dataset.empty

    def test_auto_detects_daily(self, tmp_path, mock_fund_navs, mock_market_data, macro_patch):
        ml = MLPipeline(output_dir=str(tmp_path / "ml"))
        with patch.object(ml.macro_engine, "get_macro_snapshot", return_value=macro_patch):
            dataset = ml.build_features(mock_fund_navs, mock_market_data, sample_frequency="auto")
        assert not dataset.empty


class TestTrainModels:
    def test_trains_successfully(self, tmp_path, mock_fund_navs, mock_market_data, macro_patch):
        ml = MLPipeline(output_dir=str(tmp_path / "ml"))
        with patch.object(ml.macro_engine, "get_macro_snapshot", return_value=macro_patch):
            dataset = ml.build_features(mock_fund_navs, mock_market_data, sample_frequency="weekly")
        results = ml.train_models(dataset)
        assert len(results) > 0
        for name, r in results.items():
            assert r.avg_mae >= 0

    def test_saves_predictor(self, tmp_path, mock_fund_navs, mock_market_data, macro_patch):
        ml = MLPipeline(output_dir=str(tmp_path / "ml"))
        with patch.object(ml.macro_engine, "get_macro_snapshot", return_value=macro_patch):
            dataset = ml.build_features(mock_fund_navs, mock_market_data, sample_frequency="weekly")
        ml.train_models(dataset)
        loaded = ml._load_predictor("fwd_return_3m")
        assert loaded is not None
        assert loaded.is_fitted


class TestFullPipeline:
    def test_full_pipeline_mock(self, tmp_path, mock_fund_navs, mock_market_data, macro_patch):
        ml = MLPipeline(
            output_dir=str(tmp_path / "ml"),
            cache_dir=str(tmp_path / "cache"),
        )
        with (
            patch.object(ml, "collect_fund_data", return_value=mock_fund_navs),
            patch.object(ml, "collect_market_data", return_value=mock_market_data),
            patch.object(ml.macro_engine, "get_macro_snapshot", return_value=macro_patch),
        ):
            result = ml.run_full_pipeline()

        assert result["status"] == "SUCCESS"
        assert result["fund_count"] == 3
        assert result["best_model"] in ["ridge", "random_forest", "xgboost"]
        assert (tmp_path / "ml" / "latest_run_summary.json").exists()
        assert (tmp_path / "ml" / "latest_dataset.parquet").exists()

    def test_full_pipeline_monthly_mock(self, tmp_path, mock_monthly_navs, mock_market_data, macro_patch):
        ml = MLPipeline(
            output_dir=str(tmp_path / "ml"),
            cache_dir=str(tmp_path / "cache"),
        )
        with (
            patch.object(ml, "collect_fund_data", return_value=mock_monthly_navs),
            patch.object(ml, "collect_market_data", return_value=mock_market_data),
            patch.object(ml.macro_engine, "get_macro_snapshot", return_value=macro_patch),
        ):
            result = ml.run_full_pipeline()

        assert result["status"] == "SUCCESS"

    def test_empty_fund_data_returns_error(self, tmp_path):
        ml = MLPipeline(output_dir=str(tmp_path / "ml"))
        with patch.object(ml, "collect_fund_data", return_value={}):
            result = ml.run_full_pipeline()
        assert result["status"] == "ERROR"

    def test_summary_json_valid(self, tmp_path, mock_fund_navs, mock_market_data, macro_patch):
        ml = MLPipeline(output_dir=str(tmp_path / "ml"), cache_dir=str(tmp_path / "cache"))
        with (
            patch.object(ml, "collect_fund_data", return_value=mock_fund_navs),
            patch.object(ml, "collect_market_data", return_value=mock_market_data),
            patch.object(ml.macro_engine, "get_macro_snapshot", return_value=macro_patch),
        ):
            ml.run_full_pipeline()

        with open(tmp_path / "ml" / "latest_run_summary.json") as f:
            summary = json.load(f)

        assert "best_model" in summary
        assert "model_comparison" in summary
        assert "top_features" in summary


class TestDetectFrequency:
    def test_detects_daily(self, mock_fund_navs):
        ml = MLPipeline()
        freq = ml._detect_frequency(mock_fund_navs)
        assert freq == "weekly"

    def test_detects_monthly(self, mock_monthly_navs):
        ml = MLPipeline()
        freq = ml._detect_frequency(mock_monthly_navs)
        assert freq == "monthly"

    def test_empty_returns_weekly(self):
        ml = MLPipeline()
        freq = ml._detect_frequency({})
        assert freq == "weekly"
