import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

logger = logging.getLogger(__name__)


class LearningEngineV2:
    # Statik fallback prior'lar (yeterli gecmis veri yoksa)
    STATIC_PRIORS = {
        "CRISIS":    {"ALT": 0.60, "KTS": 0.30, "CASH": 0.10},
        "RISK_ON":   {"VEF": 0.50, "KCH": 0.40, "CASH": 0.10},
        "RATE_HIKE": {"KTS": 0.70, "ALT": 0.20, "CASH": 0.10},
        "STABLE":    {"VEF": 0.30, "ALT": 0.30, "KTS": 0.30, "CASH": 0.10},
    }

    MIN_OBSERVATIONS = 6  # Bir rejim icin en az kac gozlem olmali ki "ogrenildi" sayilsin

    def __init__(self, history_path: str = "data/learning_history.json"):
        self.history_path = Path(history_path)
        self.history: List[Dict] = self._load_history()

    def _load_history(self) -> List[Dict]:
        """Gecmis gozlemleri yukle.

        Format:
        [
          {
            "date": "2024-03-31",
            "regime": "CRISIS",
            "weights_used": {"ALT": 0.60, ...},
            "monthly_return": 0.024,
            "alpha_vs_benchmark": 0.011
          },
          ...
        ]
        """
        if not self.history_path.exists():
            logger.info(f"History dosyasi yok, bos baslatiliyor: {self.history_path}")
            return []
        try:
            with open(self.history_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.error(f"History yuklenemedi: {e}")
            return []

    def record_observation(
        self,
        date: str,
        regime: str,
        weights_used: Dict[str, float],
        monthly_return: float,
        alpha_vs_benchmark: float,
    ) -> None:
        """Yeni bir performans gozlemi kaydet."""
        self.history.append({
            "date": date,
            "regime": regime,
            "weights_used": weights_used,
            "monthly_return": float(monthly_return),
            "alpha_vs_benchmark": float(alpha_vs_benchmark),
        })
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.history_path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, ensure_ascii=False, indent=2)
        logger.info(f"Gozlem kaydedildi: {date} / {regime} / alpha={alpha_vs_benchmark:.4f}")

    def get_optimized_weights(self, regime: str) -> Dict[str, float]:
        """
        Eger regime icin yeterli gecmis gozlem varsa alpha-agirlikli ortalama don.
        Yoksa STATIC_PRIORS fallback.
        """
        regime_obs = [h for h in self.history if h["regime"] == regime]

        if len(regime_obs) < self.MIN_OBSERVATIONS:
            logger.info(
                f"{regime} icin yetersiz gozlem ({len(regime_obs)}/{self.MIN_OBSERVATIONS}), "
                f"static prior kullaniliyor"
            )
            return self.STATIC_PRIORS.get(regime, self.STATIC_PRIORS["STABLE"])

        positive_obs = [h for h in regime_obs if h["alpha_vs_benchmark"] > 0]

        if not positive_obs:
            logger.warning(f"{regime} icin pozitif alpha gozlem yok, static prior kullaniliyor")
            return self.STATIC_PRIORS.get(regime, self.STATIC_PRIORS["STABLE"])

        # Alpha-weighted ortalama: daha yuksek alpha ureten agirlik setlerine daha fazla pay
        total_alpha = sum(h["alpha_vs_benchmark"] for h in positive_obs)
        learned_weights: Dict[str, float] = {}

        for obs in positive_obs:
            weight_factor = obs["alpha_vs_benchmark"] / total_alpha
            for asset, w in obs["weights_used"].items():
                learned_weights[asset] = learned_weights.get(asset, 0) + weight_factor * w

        # Toplam 1 olacak sekilde normalize et
        total_w = sum(learned_weights.values())
        if total_w > 0:
            learned_weights = {k: v / total_w for k, v in learned_weights.items()}

        logger.info(f"{regime} icin ogrenilmis agirliklar kullaniliyor (n={len(positive_obs)})")
        return learned_weights

    def calculate_confidence_score(self, regime: str) -> float:
        """
        Bu rejim icin tahminlerimize ne kadar guveniyoruz?
        Win rate + ornek buyuklugu kombinasyonu.
        """
        regime_obs = [h for h in self.history if h["regime"] == regime]
        n = len(regime_obs)

        if n == 0:
            return 0.0

        wins = sum(1 for h in regime_obs if h["alpha_vs_benchmark"] > 0)
        win_rate = wins / n

        # Sample size confidence: n arttikca 1'e yaklasir
        # 12 gozlem = ~1 yil = tam guven esigi
        sample_confidence = min(n / 12, 1.0)

        return round(win_rate * sample_confidence, 3)

    def get_regime_stats(self) -> Dict[str, Dict]:
        """Her rejim icin ozet istatistikler. Dashboard'da gostermek icin."""
        stats = {}
        for regime in self.STATIC_PRIORS.keys():
            obs = [h for h in self.history if h["regime"] == regime]
            if not obs:
                stats[regime] = {"n": 0, "win_rate": None, "avg_alpha": None, "confidence": 0.0}
                continue
            wins = sum(1 for h in obs if h["alpha_vs_benchmark"] > 0)
            stats[regime] = {
                "n": len(obs),
                "win_rate": round(wins / len(obs), 3),
                "avg_alpha": round(float(np.mean([h["alpha_vs_benchmark"] for h in obs])), 4),
                "confidence": self.calculate_confidence_score(regime),
            }
        return stats


if __name__ == "__main__":
    import tempfile
    from src.logging_config import configure_logging, get_logger as _get_logger
    configure_logging()
    _log = _get_logger(__name__)

    # Gecici dosya ile test (gercek data/ klasorunu kirletmez)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        tmp_path = f.name

    engine = LearningEngineV2(history_path=tmp_path)

    # Yetersiz veri -> static prior donmeli
    w = engine.get_optimized_weights("CRISIS")
    _log.info(f"Ilk cagri (bos history): {w}")
    assert w == LearningEngineV2.STATIC_PRIORS["CRISIS"], "Static prior eslesmiyor"

    # 7 pozitif gozlem ekle
    for i in range(7):
        engine.record_observation(
            date=f"2024-{i + 1:02d}-01",
            regime="CRISIS",
            weights_used={"ALT": 0.50 + i * 0.02, "KTS": 0.40 - i * 0.02, "CASH": 0.10},
            monthly_return=0.02,
            alpha_vs_benchmark=0.01 + i * 0.001,
        )

    # Artik ogrenilmis agirliklar donmeli
    w_learned = engine.get_optimized_weights("CRISIS")
    _log.info(f"Ogrenilmis agirliklar: {w_learned}")
    assert abs(sum(w_learned.values()) - 1.0) < 0.001, "Toplam 1 olmali"

    # Confidence
    conf = engine.calculate_confidence_score("CRISIS")
    _log.info(f"Confidence: {conf}")
    assert 0 <= conf <= 1, "Confidence 0-1 arasinda olmali"

    # Stats
    _log.info(f"Stats: {engine.get_regime_stats()}")

    os.unlink(tmp_path)
    _log.info("Tum testler gecti")
