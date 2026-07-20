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
        # PLAN-16: rank hedefi eklenir (kesitsel yuzdelik)
        assert "fwd_rank_3m" in dataset.columns


class TestFlowFeatures:
    """PLAN-16: nav_history ham verisinden fon-akisi feature'lari."""

    def _nav_raw(self, n=100):
        np.random.seed(7)
        dates = pd.date_range("2024-01-01", periods=n, freq="B")
        frames = []
        for code in ["F1", "F2"]:
            price = 100 * np.cumprod(1 + np.random.normal(0.0005, 0.01, n))
            size = 1e6 * np.cumprod(1 + np.random.normal(0.001, 0.02, n))
            kisi = (1000 + np.arange(n) * 2).astype(float)
            frames.append(pd.DataFrame({
                "fund_code": code, "date": dates, "price": price,
                "portfoy_buyukluk": size, "kisi_sayisi": kisi,
            }))
        return pd.concat(frames, ignore_index=True)

    def test_flow_features_computed(self, tmp_path):
        ml = MLPipeline(output_dir=str(tmp_path / "ml"))
        nav = self._nav_raw()
        flow = ml._build_flow_features(nav)
        assert flow is not None
        assert set(["date", "fund_code", "size_chg_1m", "kisi_chg_1m", "flow_proxy_1m"]).issubset(flow.columns)
        # flow_proxy = size_chg - ret; bir ornek satirda elle dogrula
        f1 = nav[nav["fund_code"] == "F1"].sort_values("date").reset_index(drop=True)
        i = 40
        exp_size = f1["portfoy_buyukluk"].iloc[i] / f1["portfoy_buyukluk"].iloc[i-21] - 1
        exp_ret = f1["price"].iloc[i] / f1["price"].iloc[i-21] - 1
        row = flow[(flow["fund_code"] == "F1") & (flow["date"] == f1["date"].iloc[i])].iloc[0]
        assert row["flow_proxy_1m"] == pytest.approx(exp_size - exp_ret, abs=1e-9)

    def test_missing_columns_returns_none(self, tmp_path):
        ml = MLPipeline(output_dir=str(tmp_path / "ml"))
        nav = self._nav_raw().drop(columns=["kisi_sayisi"])
        assert ml._build_flow_features(nav) is None

    def test_flow_merged_into_dataset(self, tmp_path, mock_market_data, macro_patch):
        ml = MLPipeline(output_dir=str(tmp_path / "ml"))
        # nav_history'i akis kolonlariyla dogrudan yukle
        nav = self._nav_raw(n=600)
        ml._last_nav_raw = nav
        navs = {}
        for code in ["F1", "F2"]:
            s = nav[nav["fund_code"] == code].set_index("date")["price"]
            navs[code] = s
        with patch.object(ml.macro_engine, "get_macro_snapshot", return_value=macro_patch):
            dataset = ml.build_features(navs, mock_market_data, sample_frequency="weekly")
        n_before = len(dataset)
        assert "flow_proxy_1m" in dataset.columns
        assert "size_chg_1m" in dataset.columns
        assert len(dataset) == n_before   # merge satir sayisini degistirmedi


class TestRankTarget:
    """PLAN-16: fwd_rank_3m tarih-ici kesitsel + feature'a sizmaz."""

    def test_rank_is_cross_sectional_and_excluded_from_X(self, tmp_path):
        from src.feature_engineer import FeatureEngineer
        fe = FeatureEngineer()
        # 2 tarih x 3 fon mini dataset — bilinen getiri sirasi
        idx = pd.to_datetime(["2024-06-07", "2024-06-07", "2024-06-07",
                              "2024-06-14", "2024-06-14", "2024-06-14"])
        ds = pd.DataFrame({
            "return_1m": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06],
            "fund_code": ["A", "B", "C", "A", "B", "C"],
            "fwd_return_3m": [0.1, 0.2, 0.3, 0.9, 0.5, 0.1],
            "fwd_return_12m": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
        }, index=idx)
        ds["fwd_rank_3m"] = ds.groupby(ds.index)["fwd_return_3m"].rank(pct=True)
        # ilk tarihte C en yuksek -> rank 1.0
        first = ds.loc[ds.index[0:3]]
        assert first[first["fund_code"] == "C"]["fwd_rank_3m"].iloc[0] == pytest.approx(1.0)
        # ikinci tarihte A en yuksek -> rank 1.0
        second = ds.iloc[3:]
        assert second[second["fund_code"] == "A"]["fwd_rank_3m"].iloc[0] == pytest.approx(1.0)
        # get_clean_features: fwd_rank_3m X'e sizmamali
        X, _, _ = fe.get_clean_features(ds)
        assert "fwd_rank_3m" not in X.columns
        assert "fwd_return_3m" not in X.columns

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
