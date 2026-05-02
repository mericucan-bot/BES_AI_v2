import pytest
import numpy as np
import pandas as pd
from src.feature_engineer import FeatureEngineer


@pytest.fixture
def sample_nav():
    """300 gunluk deterministik NAV serisi."""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=300, freq="B")
    prices = 100 * np.cumprod(1 + np.random.normal(0.0005, 0.015, 300))
    return pd.Series(prices, index=dates, name="NAV")


@pytest.fixture
def sample_market_data():
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


class TestFundFeatures:
    def test_creates_return_features(self, sample_nav):
        fe = FeatureEngineer()
        df = fe.create_fund_features(sample_nav, "TEST")
        assert "return_1m" in df.columns
        assert "return_3m" in df.columns
        assert "return_1y" in df.columns

    def test_creates_volatility_features(self, sample_nav):
        fe = FeatureEngineer()
        df = fe.create_fund_features(sample_nav, "TEST")
        assert "vol_1m" in df.columns
        assert "vol_3m" in df.columns

    def test_creates_momentum_features(self, sample_nav):
        fe = FeatureEngineer()
        df = fe.create_fund_features(sample_nav, "TEST")
        assert "momentum_1m_3m" in df.columns
        assert "zscore_1m" in df.columns

    def test_creates_sharpe_features(self, sample_nav):
        fe = FeatureEngineer()
        df = fe.create_fund_features(sample_nav, "TEST")
        assert "sharpe_3m" in df.columns

    def test_no_nav_leakage(self, sample_nav):
        fe = FeatureEngineer()
        df = fe.create_fund_features(sample_nav, "TEST")
        assert "nav" not in df.columns
        assert "daily_return" not in df.columns

    def test_fund_code_preserved(self, sample_nav):
        fe = FeatureEngineer()
        df = fe.create_fund_features(sample_nav, "MY_FUND")
        assert (df["fund_code"] == "MY_FUND").all()


class TestMacroFeatures:
    def test_creates_bist_features(self, sample_market_data):
        fe = FeatureEngineer()
        df = fe.create_macro_features(
            bist_prices=sample_market_data["BIST"],
            usdtry_prices=sample_market_data["USDTRY"],
            gold_prices=sample_market_data["GOLD"],
        )
        assert "bist_return_1m" in df.columns
        assert "bist_vol_1m" in df.columns
        assert "bist_drawdown" in df.columns

    def test_creates_usdtry_features(self, sample_market_data):
        fe = FeatureEngineer()
        df = fe.create_macro_features(
            bist_prices=None,
            usdtry_prices=sample_market_data["USDTRY"],
            gold_prices=None,
        )
        assert "usdtry_return_1m" in df.columns
        assert "usdtry_momentum" in df.columns

    def test_cpi_static_feature(self):
        fe = FeatureEngineer()
        dates = pd.date_range("2024-01-01", periods=10, freq="B")
        bist = pd.Series(range(10), index=dates, dtype=float)
        df = fe.create_macro_features(bist, None, None, cpi_yoy=0.306)
        assert (df["cpi_yoy"] == 0.306).all()
        assert "monthly_inflation" in df.columns


class TestInteractionFeatures:
    def test_rolling_beta(self, sample_nav, sample_market_data):
        fe = FeatureEngineer()
        fund_ret = sample_nav.pct_change()
        bist_ret = sample_market_data["BIST"].pct_change()
        df = fe.create_fund_macro_interaction(fund_ret, bist_ret, "bist", window=63)
        assert "beta_bist_63d" in df.columns
        assert "corr_bist_63d" in df.columns


class TestTargetVariables:
    def test_target_is_forward_looking(self, sample_nav):
        fe = FeatureEngineer()
        df = fe.create_target_variables(sample_nav)
        # Son deger NaN olmali (gelecek veri yok)
        assert df[fe.TARGET_3M].iloc[-1] != df[fe.TARGET_3M].iloc[-1]  # NaN check
        # Ilk deger NaN olmamali (300 gun > 63 gun)
        assert not df[fe.TARGET_3M].iloc[0] != df[fe.TARGET_3M].iloc[0]

    def test_direction_binary(self, sample_nav):
        fe = FeatureEngineer()
        df = fe.create_target_variables(sample_nav)
        valid = df["fwd_direction_3m"].dropna()
        assert set(valid.unique()).issubset({0.0, 1.0})


class TestBuildDataset:
    def test_build_basic(self, sample_nav, sample_market_data):
        fe = FeatureEngineer()
        fund_navs = {"FUND1": sample_nav}
        dataset = fe.build_dataset(fund_navs, sample_market_data, sample_frequency="weekly")
        assert not dataset.empty
        assert "fund_code" in dataset.columns
        assert fe.TARGET_3M in dataset.columns

    def test_skips_short_series(self, sample_market_data):
        fe = FeatureEngineer()
        short_nav = pd.Series([1, 2, 3], index=pd.date_range("2024-01-01", periods=3))
        fund_navs = {"SHORT": short_nav}
        dataset = fe.build_dataset(fund_navs, sample_market_data)
        assert dataset.empty

    def test_weekly_sampling(self, sample_nav, sample_market_data):
        fe = FeatureEngineer()
        dataset_daily = fe.build_dataset(
            {"F": sample_nav}, sample_market_data, sample_frequency="daily"
        )
        dataset_weekly = fe.build_dataset(
            {"F": sample_nav}, sample_market_data, sample_frequency="weekly"
        )
        assert len(dataset_weekly) < len(dataset_daily)


class TestGetCleanFeatures:
    def test_separates_x_and_y(self, sample_nav, sample_market_data):
        fe = FeatureEngineer()
        dataset = fe.build_dataset(
            {"F": sample_nav}, sample_market_data, sample_frequency="daily"
        )
        X, y_3m, y_12m = fe.get_clean_features(dataset)
        assert fe.TARGET_3M not in X.columns
        assert fe.TARGET_12M not in X.columns
        assert "fund_code" not in X.columns
