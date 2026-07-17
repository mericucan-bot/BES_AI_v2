import json
import pytest
from src.learning_engine import LearningEngineV2


class TestLearningEngineInit:
    def test_empty_history_returns_static_prior(self, temp_history_path):
        engine = LearningEngineV2(history_path=str(temp_history_path))
        weights = engine.get_optimized_weights("CRISIS")
        assert weights == LearningEngineV2.STATIC_PRIORS["CRISIS"]

    def test_unknown_regime_falls_back_to_stable(self, temp_history_path):
        engine = LearningEngineV2(history_path=str(temp_history_path))
        weights = engine.get_optimized_weights("UNKNOWN_REGIME")
        assert weights == LearningEngineV2.STATIC_PRIORS["STABLE"]


class TestLearningEngineRecording:
    def test_observation_persists_to_disk(self, temp_history_path):
        engine = LearningEngineV2(history_path=str(temp_history_path))
        engine.record_observation(
            date="2024-01-01",
            regime="CRISIS",
            weights_used={"ALT": 0.6, "KTS": 0.3, "CASH": 0.1},
            monthly_return=0.02,
            alpha_vs_benchmark=0.01,
        )
        assert temp_history_path.exists()
        with open(temp_history_path) as f:
            data = json.load(f)
        assert len(data) == 1
        assert data[0]["regime"] == "CRISIS"

    def test_multiple_observations_accumulate(self, temp_history_path, sample_observations):
        engine = LearningEngineV2(history_path=str(temp_history_path))
        for obs in sample_observations:
            engine.record_observation(**obs)
        with open(temp_history_path) as f:
            data = json.load(f)
        assert len(data) == len(sample_observations)

    def test_record_observation_leaves_no_tmp(self, temp_history_path):
        """PLAN-09: atomic_write_text kullanimi sonrasi gecici (.tmp) dosya kalmamali."""
        engine = LearningEngineV2(history_path=str(temp_history_path))
        engine.record_observation(
            date="2024-01-01",
            regime="CRISIS",
            weights_used={"ALT": 0.6, "KTS": 0.3, "CASH": 0.1},
            monthly_return=0.02,
            alpha_vs_benchmark=0.01,
        )
        assert temp_history_path.exists()
        assert list(temp_history_path.parent.glob("*.tmp")) == []
        with open(temp_history_path, encoding="utf-8") as f:
            data = json.load(f)
        assert len(data) == 1
        assert data[0]["regime"] == "CRISIS"


class TestLearningEngineLearning:
    def test_below_threshold_uses_static(self, temp_history_path, sample_observations):
        engine = LearningEngineV2(history_path=str(temp_history_path))
        for obs in sample_observations[:5]:  # eşik 6, 5 gözlem yetmez
            engine.record_observation(**obs)
        weights = engine.get_optimized_weights("CRISIS")
        assert weights == LearningEngineV2.STATIC_PRIORS["CRISIS"]

    def test_above_threshold_returns_learned(self, temp_history_path, sample_observations):
        engine = LearningEngineV2(history_path=str(temp_history_path))
        for obs in sample_observations:  # 8 gözlem, hepsi pozitif alpha
            engine.record_observation(**obs)
        weights = engine.get_optimized_weights("CRISIS")
        assert weights != LearningEngineV2.STATIC_PRIORS["CRISIS"]
        assert sum(weights.values()) == pytest.approx(1.0, abs=1e-3)

    def test_all_negative_alpha_falls_back(self, temp_history_path):
        engine = LearningEngineV2(history_path=str(temp_history_path))
        for i in range(8):
            engine.record_observation(
                date=f"2024-{i + 1:02d}-01",
                regime="CRISIS",
                weights_used={"ALT": 0.6, "KTS": 0.3, "CASH": 0.1},
                monthly_return=-0.01,
                alpha_vs_benchmark=-0.005,
            )
        weights = engine.get_optimized_weights("CRISIS")
        assert weights == LearningEngineV2.STATIC_PRIORS["CRISIS"]


class TestLearningEngineConfidence:
    def test_no_observations_zero_confidence(self, temp_history_path):
        engine = LearningEngineV2(history_path=str(temp_history_path))
        assert engine.calculate_confidence_score("CRISIS") == 0.0

    def test_confidence_in_range(self, temp_history_path, sample_observations):
        engine = LearningEngineV2(history_path=str(temp_history_path))
        for obs in sample_observations:
            engine.record_observation(**obs)
        conf = engine.calculate_confidence_score("CRISIS")
        assert 0 <= conf <= 1

    def test_higher_n_higher_confidence(self, temp_history_path):
        engine = LearningEngineV2(history_path=str(temp_history_path))

        for i in range(3):
            engine.record_observation(
                date=f"2024-{i + 1:02d}-01", regime="CRISIS",
                weights_used={"ALT": 0.6, "KTS": 0.3, "CASH": 0.1},
                monthly_return=0.02, alpha_vs_benchmark=0.01,
            )
        conf_3 = engine.calculate_confidence_score("CRISIS")

        for i in range(3, 9):
            engine.record_observation(
                date=f"2024-{i + 1:02d}-01", regime="CRISIS",
                weights_used={"ALT": 0.6, "KTS": 0.3, "CASH": 0.1},
                monthly_return=0.02, alpha_vs_benchmark=0.01,
            )
        conf_9 = engine.calculate_confidence_score("CRISIS")

        assert conf_9 > conf_3, f"Daha fazla gözlemle confidence artmalı: {conf_3} → {conf_9}"


class TestLearningEngineStats:
    def test_stats_structure(self, temp_history_path, sample_observations):
        engine = LearningEngineV2(history_path=str(temp_history_path))
        for obs in sample_observations:
            engine.record_observation(**obs)
        stats = engine.get_regime_stats()
        assert "CRISIS" in stats
        assert "n" in stats["CRISIS"]
        assert stats["CRISIS"]["n"] == len(sample_observations)

    def test_empty_regime_in_stats(self, temp_history_path):
        engine = LearningEngineV2(history_path=str(temp_history_path))
        stats = engine.get_regime_stats()
        for regime in ["CRISIS", "RISK_ON", "RATE_HIKE", "STABLE"]:
            assert regime in stats
            assert stats[regime]["n"] == 0


class TestShrinkage:
    def test_learned_weights_shrunk_toward_prior(self, temp_history_path):
        """Kucuk orneklemde ogrenilmis agirlik prior'a dogru cekilmeli (overfit'i azalt)."""
        engine = LearningEngineV2(history_path=str(temp_history_path))
        # RISK_ON icin 6 pozitif gozlem, hepsi %100 VEF (konsantre/overfit egilimi)
        for i in range(6):
            engine.record_observation(
                date=f"2024-0{i+1}-01",
                regime="RISK_ON",
                weights_used={"VEF": 1.0},
                monthly_return=0.05,
                alpha_vs_benchmark=0.02,
            )
        w = engine.get_optimized_weights("RISK_ON")
        prior_vef = LearningEngineV2.STATIC_PRIORS["RISK_ON"]["VEF"]
        # Saf ogrenilmis %100 olurdu; shrinkage ile prior(<1) ile 1.0 arasinda olmali
        assert prior_vef < w["VEF"] < 1.0, f"VEF shrink edilmedi: {w}"
        assert abs(sum(w.values()) - 1.0) < 1e-9  # normalize

    def test_more_observations_less_shrinkage(self, temp_history_path):
        """n buyudukce ogrenilmise daha cok yaklasilmali (lambda artar)."""
        def vef_after(n):
            p = temp_history_path.parent / f"hist_{n}.json"
            eng = LearningEngineV2(history_path=str(p))
            for i in range(n):
                eng.record_observation(
                    date=f"2024-01-{i+1:02d}", regime="RISK_ON",
                    weights_used={"VEF": 1.0}, monthly_return=0.05, alpha_vs_benchmark=0.02,
                )
            return eng.get_optimized_weights("RISK_ON")["VEF"]
        # 24 gozlem, 6 gozlemden daha az shrink (VEF 1.0'a daha yakin)
        assert vef_after(24) > vef_after(6)


class TestStaticOnly:
    def test_static_only_ignores_history_file(self, temp_history_path):
        """static_only=True: diskte kazanan gozlemler olsa bile yoksayilmali
        (backtest'in use_learning=False dalinin look-ahead'siz calismasi icin)."""
        winning_obs = [
            {
                "date": f"2024-{i + 1:02d}-01",
                "regime": "STABLE",
                "weights_used": {"VEF": 0.30, "ALT": 0.30, "KTS": 0.30, "CASH": 0.10},
                "monthly_return": 0.02 + i * 0.001,
                "alpha_vs_benchmark": 0.01 + i * 0.001,
            }
            for i in range(7)
        ]
        with open(temp_history_path, "w", encoding="utf-8") as f:
            json.dump(winning_obs, f)

        engine = LearningEngineV2(history_path=str(temp_history_path), static_only=True)
        assert engine.history == []
        weights = engine.get_optimized_weights("STABLE")
        assert weights == LearningEngineV2.STATIC_PRIORS["STABLE"]

        before = temp_history_path.read_text(encoding="utf-8")
        engine.record_observation(
            date="2024-08-01",
            regime="STABLE",
            weights_used={"VEF": 1.0},
            monthly_return=0.05,
            alpha_vs_benchmark=0.03,
        )
        after = temp_history_path.read_text(encoding="utf-8")
        assert before == after, "record_observation static_only modunda dosyayi degistirmemeli"
        assert engine.history == []


class TestRecordObservationSourceId:
    """PLAN-06: source_id ile ayni-kaynak dedup (replace) davranisi."""

    def test_record_observation_replaces_same_source(self, temp_history_path):
        engine = LearningEngineV2(history_path=str(temp_history_path))
        engine.record_observation(
            date="2026-06-30", regime="STABLE",
            weights_used={"VEF": 0.3, "ALT": 0.3, "KTS": 0.3, "CASH": 0.1},
            monthly_return=0.02, alpha_vs_benchmark=0.01,
            source_id="2026_06_snapshot.json",
        )
        engine.record_observation(
            date="2026-06-30", regime="STABLE",
            weights_used={"VEF": 0.3, "ALT": 0.3, "KTS": 0.3, "CASH": 0.1},
            monthly_return=0.03, alpha_vs_benchmark=0.02,
            source_id="2026_06_snapshot.json",
        )
        # Ayni source -> ikinci gozlem birincinin YERINE gecer
        assert len(engine.history) == 1
        assert engine.history[0]["alpha_vs_benchmark"] == pytest.approx(0.02)
        # Diske de yansimali
        with open(temp_history_path) as f:
            data = json.load(f)
        assert len(data) == 1
        assert data[0]["alpha_vs_benchmark"] == pytest.approx(0.02)

    def test_record_observation_without_source_appends(self, temp_history_path):
        engine = LearningEngineV2(history_path=str(temp_history_path))
        engine.record_observation(
            date="2026-06-30", regime="STABLE",
            weights_used={"VEF": 1.0},
            monthly_return=0.02, alpha_vs_benchmark=0.01,
        )
        engine.record_observation(
            date="2026-06-30", regime="STABLE",
            weights_used={"VEF": 1.0},
            monthly_return=0.03, alpha_vs_benchmark=0.02,
        )
        # source_id yok -> eski davranis: iki kayit birikir
        assert len(engine.history) == 2

    def test_source_id_never_deletes_legacy_records(self, temp_history_path):
        """Eski (source_id'siz) kayitlar hicbir kosulda silinmemeli/degismemeli."""
        legacy = [{
            "date": "2024-01-01", "regime": "CRISIS",
            "weights_used": {"ALT": 0.6, "KTS": 0.3, "CASH": 0.1},
            "monthly_return": 0.01, "alpha_vs_benchmark": 0.005,
        }]
        with open(temp_history_path, "w", encoding="utf-8") as f:
            json.dump(legacy, f)

        engine = LearningEngineV2(history_path=str(temp_history_path))
        engine.record_observation(
            date="2026-06-30", regime="STABLE", weights_used={"VEF": 1.0},
            monthly_return=0.02, alpha_vs_benchmark=0.01,
            source_id="2026_06_snapshot.json",
        )
        engine.record_observation(
            date="2026-06-30", regime="STABLE", weights_used={"VEF": 1.0},
            monthly_return=0.03, alpha_vs_benchmark=0.02,
            source_id="2026_06_snapshot.json",
        )
        # 1 legacy (dokunulmadi) + 1 source (replace edildi)
        assert len(engine.history) == 2
        legacy_kept = [h for h in engine.history if h.get("source_id") is None]
        assert len(legacy_kept) == 1
        assert legacy_kept[0] == legacy[0]
