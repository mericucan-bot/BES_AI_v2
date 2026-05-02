import pytest
import numpy as np
import pandas as pd
from src.ml_model import BESPredictor, ModelConfig, ModelResult


@pytest.fixture
def synthetic_dataset():
    """ML testi icin deterministik dataset."""
    np.random.seed(42)
    n = 200
    dates = pd.date_range("2024-01-01", periods=n, freq="W-FRI")

    X = pd.DataFrame(
        {
            "return_1m": np.random.normal(0.02, 0.05, n),
            "return_3m": np.random.normal(0.06, 0.10, n),
            "vol_1m": np.abs(np.random.normal(0.20, 0.05, n)),
            "vol_3m": np.abs(np.random.normal(0.18, 0.04, n)),
            "momentum_1m_3m": np.random.normal(0, 0.03, n),
            "sharpe_3m": np.random.normal(0.5, 0.8, n),
            "bist_return_1m": np.random.normal(0.01, 0.06, n),
            "usdtry_return_1m": np.random.normal(0.02, 0.03, n),
            "beta_bist_63d": np.random.normal(0.8, 0.3, n),
            "cpi_yoy": 0.306,
        },
        index=dates,
    )

    y = (
        0.3 * X["return_1m"]
        + 0.2 * X["momentum_1m_3m"]
        - 0.1 * X["vol_1m"]
        + np.random.normal(0, 0.03, n)
    )
    y.name = "fwd_return_3m"
    return X, y


class TestModelConfig:
    def test_default_config(self):
        config = ModelConfig()
        assert config.target == "fwd_return_3m"
        assert config.min_train_samples == 30

    def test_custom_target(self):
        config = ModelConfig(target="fwd_return_12m")
        assert config.target == "fwd_return_12m"


class TestWalkForwardSplits:
    def test_creates_splits(self, synthetic_dataset):
        X, y = synthetic_dataset
        predictor = BESPredictor()
        splits = predictor._create_walk_forward_splits(X, y)
        assert len(splits) > 0

    def test_no_leakage(self, synthetic_dataset):
        X, y = synthetic_dataset
        predictor = BESPredictor()
        splits = predictor._create_walk_forward_splits(X, y)
        for train_idx, test_idx in splits:
            assert train_idx[-1] < test_idx[0], "LEAKAGE: train test'ten sonra!"

    def test_expanding_window(self, synthetic_dataset):
        X, y = synthetic_dataset
        predictor = BESPredictor()
        splits = predictor._create_walk_forward_splits(X, y)
        if len(splits) >= 2:
            assert len(splits[0][0]) <= len(splits[1][0])

    def test_insufficient_data(self):
        X = pd.DataFrame({"f1": [1, 2, 3]}, index=pd.date_range("2024-01-01", periods=3))
        y = pd.Series([0.1, 0.2, 0.3], index=X.index)
        predictor = BESPredictor()
        splits = predictor._create_walk_forward_splits(X, y)
        assert len(splits) == 0


class TestTrainAndEvaluate:
    def test_ridge_trains(self, synthetic_dataset):
        X, y = synthetic_dataset
        predictor = BESPredictor()
        result = predictor.train_and_evaluate(X, y, model_name="ridge")
        assert result is not None
        assert result.avg_mae > 0
        assert 0.0 <= result.avg_directional_accuracy <= 1.0

    def test_random_forest_trains(self, synthetic_dataset):
        X, y = synthetic_dataset
        predictor = BESPredictor()
        result = predictor.train_and_evaluate(X, y, model_name="random_forest")
        assert result is not None
        assert result.feature_importance is not None
        assert len(result.feature_importance) > 0

    def test_xgboost_trains(self, synthetic_dataset):
        X, y = synthetic_dataset
        predictor = BESPredictor()
        result = predictor.train_and_evaluate(X, y, model_name="xgboost")
        assert result is not None

    def test_fold_results_populated(self, synthetic_dataset):
        X, y = synthetic_dataset
        predictor = BESPredictor()
        result = predictor.train_and_evaluate(X, y, model_name="ridge")
        assert len(result.fold_results) > 0
        for fold in result.fold_results:
            assert fold.train_size >= 30
            assert fold.test_size > 0
            assert fold.mae >= 0


class TestPredict:
    def test_predict_after_train(self, synthetic_dataset):
        X, y = synthetic_dataset
        predictor = BESPredictor()
        predictor.train_and_evaluate(X, y, model_name="ridge")
        preds = predictor.predict(X.tail(10), model_name="ridge")
        assert preds is not None
        assert len(preds) == 10

    def test_predict_without_train_fails(self, synthetic_dataset):
        X, _ = synthetic_dataset
        predictor = BESPredictor()
        preds = predictor.predict(X.tail(10), model_name="ridge")
        assert preds is None


class TestCompareModels:
    def test_compare_returns_multiple(self, synthetic_dataset):
        X, y = synthetic_dataset
        predictor = BESPredictor()
        results = predictor.compare_models(X, y)
        assert len(results) >= 2
        assert "ridge" in results
        assert "random_forest" in results


class TestAntiLeakage:
    def test_no_future_data_in_train(self, synthetic_dataset):
        X, y = synthetic_dataset
        predictor = BESPredictor()
        splits = predictor._create_walk_forward_splits(X, y)
        for train_idx, test_idx in splits:
            train_max = X.index[train_idx[-1]]
            test_min = X.index[test_idx[0]]
            assert train_max < test_min, f"LEAKAGE: train={train_max}, test={test_min}"


class TestPrintSummary:
    def test_summary_contains_key_fields(self, synthetic_dataset):
        X, y = synthetic_dataset
        predictor = BESPredictor()
        result = predictor.train_and_evaluate(X, y, model_name="ridge")
        summary = predictor.print_summary(result)
        assert "MODEL: ridge" in summary
        assert "MAE" in summary
        assert "DirAcc" in summary
