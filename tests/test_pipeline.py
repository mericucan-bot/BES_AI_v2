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


@pytest.fixture(autouse=True)
def _stub_fund_selector():
    """PLAN-13: pipeline BUY onerilerinde aday fon secici cagiriyor; secici
    default ml_dir="data/ml" glob'una gider. Testler GERCEK data/ml dizinini
    okumasin diye tum modulde stub'lanir (testler kendi patch'iyle ezebilir;
    import run() icinde oldugundan patch hedefi src.fund_selector)."""
    with patch("src.fund_selector.suggest_funds_for_class", return_value=[]):
        yield


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
        # PLAN-14: onemlilik skoru her kosumda uretilir
        assert result["significance"]["level"] in {"quiet", "notable", "action"}
        assert 0 <= result["significance"]["score"] <= 100

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


class TestPeriodDays:
    """return_1m tabanli hesaplar icin ~1 ay periyot gate'i."""

    def _now(self):
        from datetime import datetime
        from zoneinfo import ZoneInfo
        return datetime(2026, 6, 30, tzinfo=ZoneInfo("Europe/Istanbul"))

    def test_monthly_gap(self):
        # ~1 ay onceki snapshot -> gate acik (20..45 arasi)
        d = MonthlyPipeline._period_days("2026-05-31T10:00:00+03:00", self._now())
        assert d == 30
        assert 20 <= d <= 45

    def test_two_month_gap_outside_window(self):
        d = MonthlyPipeline._period_days("2026-04-30T10:00:00+03:00", self._now())
        assert d == 61
        assert not (20 <= d <= 45)

    def test_naive_prev_date_no_tz_error(self):
        # tz-naive onceki tarih, tz-aware current: tarih-bazli oldugu icin patlamaz
        d = MonthlyPipeline._period_days("2026-05-31", self._now())
        assert d == 30

    def test_unparseable_returns_none(self):
        assert MonthlyPipeline._period_days("not-a-date", self._now()) is None
        assert MonthlyPipeline._period_days(None, self._now()) is None


class TestPipelineFundClassMapping:
    """PLAN-03: gercek TEFAS fon kodlu portfoy -> oneriler SINIF uzayinda uretilir."""

    def _write_snapshot(self, cache_dir: Path) -> None:
        # NOT: "GMF" bilerek kullanilmiyor — MANUAL_CLASS_OVERRIDES'ta global
        # istisnasi var; notr para piyasasi ornegi olarak PPF kullanilir.
        df = pd.DataFrame({
            "fund_code": ["AHS", "BGL", "PPF"],
            "category":  ["Stock Fund", "Gold Fund", "Money Market Fund"],
        })
        df.to_parquet(cache_dir / "snapshot_EMK_2026_05.parquet")

    def _make_pipeline(self, tmp_path, holdings, with_snapshot):
        portfolio_path = tmp_path / "portfolio.json"
        history_dir    = tmp_path / "history"
        learning_path  = tmp_path / "learning.json"
        cache_dir      = tmp_path / "cache"
        history_dir.mkdir()
        cache_dir.mkdir()
        portfolio_path.write_text(json.dumps({"holdings_tl": holdings}), encoding="utf-8")
        if with_snapshot:
            self._write_snapshot(cache_dir)
        return MonthlyPipeline(
            portfolio_path=str(portfolio_path),
            history_dir=str(history_dir),
            learning_path=str(learning_path),
            tefas_cache_dir=str(cache_dir),
        )

    def test_real_fund_codes_map_to_classes(self, tmp_path, mock_regime_result):
        from src.asset_mapping import ASSET_CLASSES

        pipeline = self._make_pipeline(
            tmp_path, {"AHS": 30000, "BGL": 30000, "PPF": 40000}, with_snapshot=True
        )
        with patch.object(pipeline.regime_engine, "compute_composite_score",
                          return_value=mock_regime_result):
            result = pipeline.run()

        assert result["status"] == "SUCCESS"
        actions = result["recommendation"]["actions"]
        assert actions, "sinif bazinda oneri uretilmeli"
        # Tum oneriler SINIF kodu; hicbir gercek fon kodu (ozellikle SAT) yok
        assert all(r["asset"] in ASSET_CLASSES for r in actions)
        assert not any(r["asset"] in {"AHS", "BGL", "PPF"} for r in actions)
        # class_weights snapshot'a yazildi; fon-bazli weights korundu
        assert result["portfolio_value"]["class_weights"] == {"VEF": 0.3, "ALT": 0.3, "CASH": 0.4}
        assert set(result["portfolio_value"]["weights"]) == {"AHS", "BGL", "PPF"}
        # Her oneriye o siniftaki gercek fon ipucu eklendi
        by_asset = {r["asset"]: r for r in actions}
        assert by_asset["CASH"]["funds_in_class"] == ["PPF"]

    def test_no_snapshot_no_recommendations(self, tmp_path, mock_regime_result):
        pipeline = self._make_pipeline(
            tmp_path, {"AHS": 30000, "BGL": 30000, "PPF": 40000}, with_snapshot=False
        )
        with patch.object(pipeline.regime_engine, "compute_composite_score",
                          return_value=mock_regime_result):
            result = pipeline.run()

        assert result["status"] == "SUCCESS"
        # Sessizce "her seyi sat" URETME — oneri yok + not mevcut
        assert result["recommendation"]["actions"] == []
        assert result["recommendation_note"]
        assert set(result["portfolio_value"]["unmapped_tl"]) == {"AHS", "BGL", "PPF"}

    def test_demo_class_codes_unchanged(self, tmp_path, mock_regime_result):
        """Sinif kodlu (demo) portfoy davranisi degismez — kendine eslenir."""
        from src.asset_mapping import ASSET_CLASSES

        pipeline = self._make_pipeline(
            tmp_path,
            {"VEF": 30000, "ALT": 25000, "KTS": 20000, "KCH": 15000, "CASH": 10000},
            with_snapshot=True,
        )
        with patch.object(pipeline.regime_engine, "compute_composite_score",
                          return_value=mock_regime_result):
            result = pipeline.run()

        assert result["status"] == "SUCCESS"
        assert "unmapped_tl" not in result["portfolio_value"]
        assert result["recommendation_note"] is None
        assert all(r["asset"] in ASSET_CLASSES for r in result["recommendation"]["actions"])


class TestPipelineCandidateFunds:
    """PLAN-13: BUY onerilerine sinif ici somut aday fonlar (candidate_funds)."""

    _FAKE_CANDIDATES = [
        {"fund_code": "KTB", "fund_name": "Kamu Borclanma Fonu",
         "score_basis": "ml_return", "predicted_3m": 0.123,
         "return_1y": 48.2, "risk": 3.0, "held": False},
        {"fund_code": "AEK", "fund_name": "Baska Kamu Fonu",
         "score_basis": "ml_return", "predicted_3m": 0.101,
         "return_1y": 44.0, "risk": 3.0, "held": False},
    ]

    def _make_pipeline(self, tmp_path):
        # TestPipelineFundClassMapping deseni: AHS->VEF 0.3, BGL->ALT 0.3,
        # PPF->CASH 0.4; STABLE hedefi KTS=0.30 icerdiginden KTS kesin BUY olur.
        portfolio_path = tmp_path / "portfolio.json"
        history_dir    = tmp_path / "history"
        learning_path  = tmp_path / "learning.json"
        cache_dir      = tmp_path / "cache"
        history_dir.mkdir()
        cache_dir.mkdir()
        portfolio_path.write_text(json.dumps({
            "holdings_tl": {"AHS": 30000, "BGL": 30000, "PPF": 40000}
        }), encoding="utf-8")
        df = pd.DataFrame({
            "fund_code": ["AHS", "BGL", "PPF"],
            "category":  ["Stock Fund", "Gold Fund", "Money Market Fund"],
        })
        df.to_parquet(cache_dir / "snapshot_EMK_2026_05.parquet")
        return MonthlyPipeline(
            portfolio_path=str(portfolio_path),
            history_dir=str(history_dir),
            learning_path=str(learning_path),
            tefas_cache_dir=str(cache_dir),
        )

    def test_buy_recommendations_get_candidate_funds(self, tmp_path, mock_regime_result):
        pipeline = self._make_pipeline(tmp_path)
        with patch.object(pipeline.regime_engine, "compute_composite_score",
                          return_value=mock_regime_result), \
             patch("src.fund_selector.suggest_funds_for_class",
                   return_value=self._FAKE_CANDIDATES) as mock_sel:
            result = pipeline.run()

        assert result["status"] == "SUCCESS"
        actions = result["recommendation"]["actions"]
        buys = [a for a in actions if a["action"] == "BUY"]
        assert buys, "en az bir BUY onerisi beklenir (KTS)"
        for b in buys:
            assert b["candidate_funds"] == self._FAKE_CANDIDATES
        # BUY olmayanlara candidate_funds EKLENMEZ (deferred HOLD haric —
        # onlar limit filtresinden once BUY idi)
        for a in actions:
            if a["action"] == "SELL":
                assert "candidate_funds" not in a
        # Secici dogru parametrelerle cagrildi: kullanicinin fonlari held_codes
        _, kwargs = mock_sel.call_args
        assert kwargs["held_codes"] == {"AHS", "BGL", "PPF"}
        assert kwargs["cache_dir"] == pipeline.tefas_cache_dir
        assert kwargs["class_map"].get("AHS") == "VEF"

    def test_selector_exception_pipeline_still_success(self, tmp_path, mock_regime_result):
        pipeline = self._make_pipeline(tmp_path)
        with patch.object(pipeline.regime_engine, "compute_composite_score",
                          return_value=mock_regime_result), \
             patch("src.fund_selector.suggest_funds_for_class",
                   side_effect=RuntimeError("selector patladi")):
            result = pipeline.run()

        assert result["status"] == "SUCCESS"
        assert result["recommendation"]["actions"], "oneriler yine uretilmeli"


class TestPipelineLearningDedupAndWeights:
    """PLAN-06: ayni-ay dedup + gerceklesen sinif agirliklariyla ogrenme."""

    def _make_prev_snapshot(self, history_dir: Path, class_weights=None) -> None:
        pv = {"total_value": 95000, "weights": {}, "date": "2026-03-30"}
        if class_weights is not None:
            pv["class_weights"] = class_weights
        prev = {
            "run_date": "2026-03-30T10:00:00+03:00",
            "portfolio_value": pv,
            "regime": {"detected": "CRISIS"},
            "recommendation": {"target_weights": {"ALT": 0.6, "KTS": 0.3, "CASH": 0.1}},
        }
        (history_dir / "2026_03_snapshot.json").write_text(json.dumps(prev), encoding="utf-8")

    def test_pipeline_rerun_no_duplicate(self, pipeline_dirs, mock_regime_result):
        """Ayni ay 2 kez kosu -> onceki ay icin TEK gozlem (duplicate yok)."""
        self._make_prev_snapshot(Path(pipeline_dirs["history_dir"]))
        pipeline = MonthlyPipeline(**pipeline_dirs)

        with patch.object(pipeline.regime_engine, "compute_composite_score",
                          return_value=mock_regime_result), \
             patch.object(pipeline.regime_engine, "fetch_live_data",
                          return_value=pd.DataFrame()):
            pipeline.run()
            pipeline.run()  # ayni ay ikinci kosu

        with open(Path(pipeline_dirs["learning_path"]), encoding="utf-8") as f:
            history = json.load(f)
        from_prev = [h for h in history if h.get("source_id") == "2026_03_snapshot.json"]
        assert len(from_prev) == 1
        assert len(history) == 1

    def test_weights_used_prefers_actual(self, pipeline_dirs, mock_regime_result):
        """class_weights varsa gozlem ONU kullanir (onerilen target'i degil)."""
        self._make_prev_snapshot(
            Path(pipeline_dirs["history_dir"]),
            class_weights={"VEF": 0.7, "CASH": 0.3},
        )
        pipeline = MonthlyPipeline(**pipeline_dirs)
        with patch.object(pipeline.regime_engine, "compute_composite_score",
                          return_value=mock_regime_result), \
             patch.object(pipeline.regime_engine, "fetch_live_data",
                          return_value=pd.DataFrame()):
            result = pipeline.run()

        with open(Path(pipeline_dirs["learning_path"]), encoding="utf-8") as f:
            history = json.load(f)
        assert len(history) == 1
        assert history[0]["weights_used"] == {"VEF": 0.7, "CASH": 0.3}
        assert result["previous_evaluation"]["weights_basis"] == "actual_class"

    def test_weights_used_falls_back_to_target(self, pipeline_dirs, mock_regime_result):
        """class_weights YOKSA onerilen target_weights'a duser."""
        self._make_prev_snapshot(Path(pipeline_dirs["history_dir"]), class_weights=None)
        pipeline = MonthlyPipeline(**pipeline_dirs)
        with patch.object(pipeline.regime_engine, "compute_composite_score",
                          return_value=mock_regime_result), \
             patch.object(pipeline.regime_engine, "fetch_live_data",
                          return_value=pd.DataFrame()):
            result = pipeline.run()

        with open(Path(pipeline_dirs["learning_path"]), encoding="utf-8") as f:
            history = json.load(f)
        assert len(history) == 1
        assert history[0]["weights_used"] == {"ALT": 0.6, "KTS": 0.3, "CASH": 0.1}
        assert result["previous_evaluation"]["weights_basis"] == "recommended_target"
