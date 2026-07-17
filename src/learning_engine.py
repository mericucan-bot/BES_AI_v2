import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

from src.io_utils import atomic_write_text

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
    SHRINKAGE_K = 12      # Shrinkage gucu: n=K gozlemde ogrenilmis/prior yari-yari

    def __init__(self, history_path: str = "data/learning_history.json",
                 static_only: bool = False):
        """static_only=True: gecmis YUKLENMEZ ve kaydedilmez — her zaman
        STATIC_PRIORS doner (backtest'in look-ahead'siz statik modu icin)."""
        self.static_only = static_only
        self.history_path = Path(history_path)
        self.history: List[Dict] = [] if static_only else self._load_history()

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
        source_id: Optional[str] = None,
    ) -> None:
        """Yeni bir performans gozlemi kaydet.

        source_id: gozlemin kaynagi (orn. onceki snapshot dosya adi). Ayni
        source_id ile ikinci cagri ESKISININ YERINE GECER (ayni ay iki kez
        kosulursa duplicate olusmaz). None ise eski davranis (sadece ekle).
        """
        if self.static_only:
            logger.warning("static_only modunda gozlem kaydedilmez — atlandi")
            return
        if source_id:
            before = len(self.history)
            self.history = [h for h in self.history if h.get("source_id") != source_id]
            if len(self.history) < before:
                logger.info(f"Ayni kaynakli eski gozlem degistirildi: {source_id}")
        obs = {
            "date": date,
            "regime": regime,
            "weights_used": weights_used,
            "monthly_return": float(monthly_return),
            "alpha_vs_benchmark": float(alpha_vs_benchmark),
        }
        if source_id:
            obs["source_id"] = source_id
        self.history.append(obs)
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(self.history_path, json.dumps(self.history, ensure_ascii=False, indent=2))
        logger.info(f"Gozlem kaydedildi: {date} / {regime} / alpha={alpha_vs_benchmark:.4f}")

    # Risk profiline göre varlık ağırlığı çarpanları
    _RISK_MULTIPLIERS = {
        "muhafazakar": {"VEF": 0.50, "KCH": 0.70, "ALT": 1.00, "KTS": 1.40, "CASH": 1.50},
        "dengeli":     {"VEF": 1.00, "KCH": 1.00, "ALT": 1.00, "KTS": 1.00, "CASH": 1.00},
        "agresif":     {"VEF": 1.30, "KCH": 1.30, "ALT": 1.00, "KTS": 0.70, "CASH": 0.50},
    }

    def get_optimized_weights(self, regime: str, risk_profile: Optional[str] = None) -> Dict[str, float]:
        """
        Eger regime icin yeterli gecmis gozlem varsa alpha-agirlikli ortalama don.
        Yoksa STATIC_PRIORS fallback.
        risk_profile: "muhafazakar" | "dengeli" | "agresif" | None
        """
        regime_obs = [h for h in self.history if h["regime"] == regime]

        if len(regime_obs) < self.MIN_OBSERVATIONS:
            logger.info(
                f"{regime} icin yetersiz gozlem ({len(regime_obs)}/{self.MIN_OBSERVATIONS}), "
                f"static prior kullaniliyor"
            )
            base = self.STATIC_PRIORS.get(regime, self.STATIC_PRIORS["STABLE"])
        else:
            positive_obs = [h for h in regime_obs if h["alpha_vs_benchmark"] > 0]
            win_rate = len(positive_obs) / len(regime_obs) if regime_obs else 0.0

            # Win rate < %50 ise ogrenilmis agirliklara guvenmek yerine prior'a don.
            # Floating point esit-sifir riskine karsi 1e-9 esigi.
            total_alpha = sum(h["alpha_vs_benchmark"] for h in positive_obs) if positive_obs else 0.0

            if not positive_obs:
                logger.warning(f"{regime} icin pozitif alpha gozlem yok, static prior kullaniliyor")
                base = self.STATIC_PRIORS.get(regime, self.STATIC_PRIORS["STABLE"])
            elif win_rate < 0.5:
                logger.info(
                    f"{regime} icin win_rate dusuk ({len(positive_obs)}/{len(regime_obs)}), "
                    f"overfit riski — static prior kullaniliyor"
                )
                base = self.STATIC_PRIORS.get(regime, self.STATIC_PRIORS["STABLE"])
            elif total_alpha <= 1e-9:
                logger.warning(f"{regime} icin toplam alpha ~0, static prior kullaniliyor")
                base = self.STATIC_PRIORS.get(regime, self.STATIC_PRIORS["STABLE"])
            else:
                # Alpha-weighted ortalama: daha yuksek alpha ureten agirlik setlerine daha fazla pay
                learned_weights: Dict[str, float] = {}

                for obs in positive_obs:
                    weight_factor = obs["alpha_vs_benchmark"] / total_alpha
                    for asset, w in obs["weights_used"].items():
                        learned_weights[asset] = learned_weights.get(asset, 0) + weight_factor * w

                total_w = sum(learned_weights.values())
                if total_w > 0:
                    learned_weights = {k: v / total_w for k, v in learned_weights.items()}

                # SHRINKAGE: kucuk orneklemde ogrenilmis agirliklar overfit eder.
                # Prior'a dogru cek: lambda = n/(n+K). n buyudukce ogrenilmise,
                # kucukken prior'a yaklasir (K gozlemde yari-yari). Boylece az
                # veriyle asiri-konsantre/gurultulu agirliklar yumusatilir.
                n = len(positive_obs)
                lam = n / (n + self.SHRINKAGE_K)
                prior = self.STATIC_PRIORS.get(regime, self.STATIC_PRIORS["STABLE"])
                assets = set(learned_weights) | set(prior)
                blended = {
                    a: lam * learned_weights.get(a, 0.0) + (1 - lam) * prior.get(a, 0.0)
                    for a in assets
                }
                total_b = sum(blended.values())
                if total_b > 0:
                    blended = {k: v / total_b for k, v in blended.items()}

                logger.info(
                    f"{regime} icin ogrenilmis agirliklar (n={n}, shrinkage λ={lam:.2f} "
                    f"-> prior'a {(1-lam)*100:.0f}% cekildi)"
                )
                base = blended

        # Risk profiline göre çarpan uygula ve yeniden normalize et
        if risk_profile and risk_profile in self._RISK_MULTIPLIERS:
            multipliers = self._RISK_MULTIPLIERS[risk_profile]
            adjusted = {k: v * multipliers.get(k, 1.0) for k, v in base.items()}
            total_adj = sum(adjusted.values())
            if total_adj > 0:
                adjusted = {k: v / total_adj for k, v in adjusted.items()}
            logger.info(f"Risk profili '{risk_profile}' uygulandı")
            return adjusted

        return base

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
            alphas = [h["alpha_vs_benchmark"] for h in obs]
            stats[regime] = {
                "n": len(obs),
                "win_rate": round(wins / len(obs), 3),
                "avg_alpha": round(sum(alphas) / len(alphas), 4),
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
