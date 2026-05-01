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
