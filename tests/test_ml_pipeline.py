import json
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import patch

from src.ml_pipeline import MLPipeline


def _write_nav_parquet(cache_dir, fund_days, seed=42):
    """Sahte nav_history.parquet yaz (gercek gunluk NAV semasi).

    fund_days: {fund_code: gun_sayisi}. Fiyat = 100*(1+kucuk getiri).cumprod().
    Returns: yazilan parquet yolu (Path).
    """
    np.random.seed(seed)
    frames = []
    for code, n in fund_days.items():
        dates = pd.date_range("2024-01-01", periods=n, freq="B")
        prices = 100 * np.cumprod(1 + np.random.normal(0.0005, 0.01, n))
        frames.append(pd.DataFrame({
            "fund_code": code,
            "fund_name": f"{code} Test Fonu",
            "date": dates,
            "price": prices,
        }))
    df = pd.concat(frames, ignore_index=True)
    path = Path(cache_dir) / "nav_history.parquet"
    df.to_parquet(path, index=False)
    return path


@pytest.fixture
def mock_fund_navs():
    """3 fonluk 600-gunluk gunluk NAV verisi (test icin).

    NOT: Tarih-bazli purge'lu walk-forward (3 ay = 63 gun embargo) icin
    yeterli gecmis gerekir; kisa seriler tum gozlemleri purge eder. Gercek
    TEFAS verisinde 2+ yil mevcut oldugundan bu fikstur o kosulu temsil eder.
    """
    np.random.seed(42)
    n = 600
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    navs = {}
    for code in ["FUND1", "FUND2", "FUND3"]:
        drift = np.random.uniform(0.0003, 0.001)
        vol = np.random.uniform(0.01, 0.02)
        prices = 100 * np.cumprod(1 + np.random.normal(drift, vol, n))
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
    n = 600
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {
            "BIST": 10000 * np.cumprod(1 + np.random.normal(0.0005, 0.015, n)),
            "USDTRY": 30 * np.cumprod(1 + np.random.normal(0.001, 0.008, n)),
            "GOLD": 2000 * np.cumprod(1 + np.random.normal(0.0003, 0.010, n)),
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

    def test_multi_target_pipeline(self, tmp_path, mock_fund_navs, mock_market_data, macro_patch):
        """targets listesi verildiginde pipeline basarili olmali."""
        ml = MLPipeline(
            output_dir=str(tmp_path / "ml"),
            cache_dir=str(tmp_path / "cache"),
        )
        with (
            patch.object(ml, "collect_fund_data", return_value=mock_fund_navs),
            patch.object(ml, "collect_market_data", return_value=mock_market_data),
            patch.object(ml.macro_engine, "get_macro_snapshot", return_value=macro_patch),
        ):
            result = ml.run_full_pipeline(targets=["fwd_return_3m"])

        assert result["status"] == "SUCCESS"
        assert result["targets"] == ["fwd_return_3m"]
        assert result["target"] == "fwd_return_3m"
        assert (tmp_path / "ml" / "latest_run_summary_fwd_return_3m.json").exists()

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


class TestCollectFundDataNav:
    """collect_fund_data artik birincil kaynak olarak nav_history.parquet kullanir."""

    def test_collect_fund_data_nav_reads_parquet(self, tmp_path):
        _write_nav_parquet(tmp_path, {"FUND1": 200, "FUND2": 200, "FUND3": 200})
        ml = MLPipeline(output_dir=str(tmp_path / "ml"), cache_dir=str(tmp_path))

        navs = ml.collect_fund_data()

        assert set(navs.keys()) == {"FUND1", "FUND2", "FUND3"}
        for code, s in navs.items():
            assert isinstance(s, pd.Series)
            assert isinstance(s.index, pd.DatetimeIndex)
            assert len(s) == 200

    def test_collect_fund_data_no_nav_no_synthetic(self, tmp_path):
        """nav_history yok + allow_synthetic=False -> {} ve fetch_monthly_series cagrilmaz."""
        ml = MLPipeline(output_dir=str(tmp_path / "ml"), cache_dir=str(tmp_path))
        with (
            patch.object(ml.collector, "update_nav_history", return_value=0),
            patch.object(ml.collector, "fetch_monthly_series") as mock_fetch,
        ):
            result = ml.collect_fund_data()

        assert result == {}
        mock_fetch.assert_not_called()

    def test_min_days_filter(self, tmp_path):
        """126 gunden kisa seriler elenir."""
        _write_nav_parquet(tmp_path, {"LONGFUND": 200, "SHORTFUND": 50})
        ml = MLPipeline(output_dir=str(tmp_path / "ml"), cache_dir=str(tmp_path))

        navs = ml.collect_fund_data()

        assert "LONGFUND" in navs
        assert "SHORTFUND" not in navs

    def test_full_pipeline_on_real_shaped_data(self, tmp_path, macro_patch):
        """Sahte-gercekci gunluk NAV parquet'i -> FeatureEngineer yolu (haftalik, 63g fwd)."""
        _write_nav_parquet(tmp_path, {"FUND1": 320, "FUND2": 320, "FUND3": 320})
        ml = MLPipeline(output_dir=str(tmp_path / "ml"), cache_dir=str(tmp_path))

        navs = ml.collect_fund_data()
        assert len(navs) == 3

        np.random.seed(7)
        dates = pd.date_range("2024-01-01", periods=320, freq="B")
        market = pd.DataFrame(
            {
                "BIST": 10000 * np.cumprod(1 + np.random.normal(0.0005, 0.015, 320)),
                "USDTRY": 30 * np.cumprod(1 + np.random.normal(0.001, 0.008, 320)),
                "GOLD": 2000 * np.cumprod(1 + np.random.normal(0.0003, 0.010, 320)),
            },
            index=dates,
        )

        with patch.object(ml.macro_engine, "get_macro_snapshot", return_value=macro_patch):
            dataset = ml.build_features(navs, market)

        assert not dataset.empty
        assert "fwd_return_3m" in dataset.columns
        # FeatureEngineer yolu haftalik (Cuma) ornekler
        assert (dataset.index.dayofweek == 4).all()

    def test_legacy_12m_target_scale(self, tmp_path, mock_monthly_navs, mock_market_data, macro_patch):
        """Legacy snapshot modunda fwd_return_12m = nav.pct_change(12).shift(-12) (100x kucuk DEGIL)."""
        ml = MLPipeline(output_dir=str(tmp_path / "ml"), cache_dir=str(tmp_path / "cache"))
        with patch.object(ml.macro_engine, "get_macro_snapshot", return_value=macro_patch):
            dataset = ml._build_snapshot_features(mock_monthly_navs, mock_market_data)

        code = "AEA"
        expected = mock_monthly_navs[code].pct_change(12).shift(-12)
        actual = (
            dataset[dataset["fund_code"] == code]["fwd_return_12m"].reindex(expected.index)
        )

        np.testing.assert_allclose(
            actual.to_numpy(dtype=float),
            expected.to_numpy(dtype=float),
            rtol=1e-9,
            equal_nan=True,
        )
        # /100 bug'inda tum degerler ~100x kucuk olurdu; gercek 12A getirisi buyuk
        assert np.nanmax(np.abs(actual.to_numpy(dtype=float))) > 0.01
