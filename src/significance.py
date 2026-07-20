"""Aylik kosum sonucunun 'onemlilik' skoru — bildirim yogunlugunu belirler.

Pasif kullanici icin: onemli bir sey yoksa tek satirlik 'her sey yolunda';
varsa nedenleri basa yazan tam rapor. Skor deterministik (birikimli, 0-100).
"""
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class SignificanceConfig:
    drift_notable: float = 0.15   # hedeften sinif bazinda mutlak sapma esigi
    drift_action: float = 0.25
    turnover_notable: float = 0.20
    concentration_notable: float = 0.50   # tek sinif payi bu esigi asarsa notable
    concentration_action: float = 0.70    # bu esigi asarsa action
    level_action: int = 50        # skor esikleri
    level_notable: int = 25


# Rejim kodu -> okunabilir TR etiket (reason metinleri icin)
_REGIME_TR = {
    "STABLE": "Sakin", "CRISIS": "Kriz",
    "RISK_ON": "Yukselis", "RATE_HIKE": "Faiz Artisi",
}


def compute_significance(
    regime_result: Dict,
    evaluation: Optional[Dict],
    class_weights: Dict[str, float],
    target_weights: Dict[str, float],
    cost_analysis: Optional[Dict],
    config: Optional[SignificanceConfig] = None,
) -> Dict:
    """
    Returns: {"score": 0-100, "level": "quiet"|"notable"|"action",
              "reasons": [str, ...]}
    """
    cfg = config or SignificanceConfig()
    score = 0
    reasons: List[str] = []

    detected = regime_result.get("detected") if regime_result else None

    # 1) Rejim degisti (onceki degerlendirme mevcut VE farkli)
    prev_regime = evaluation.get("previous_regime") if evaluation else None
    if prev_regime and detected and prev_regime != detected:
        score += 40
        reasons.append(
            f"Rejim degisti: {_REGIME_TR.get(prev_regime, prev_regime)} -> "
            f"{_REGIME_TR.get(detected, detected)}"
        )

    # 2) Mevcut rejim CRISIS
    if detected == "CRISIS":
        score += 40
        reasons.append("Kriz rejimi tespit edildi")

    # 3) Hedeften sinif bazinda maksimum sapma (class_weights bos degilse)
    if class_weights:
        assets = set(class_weights) | set(target_weights or {})
        max_drift = 0.0
        max_asset = None
        for a in assets:
            d = abs(class_weights.get(a, 0.0) - (target_weights or {}).get(a, 0.0))
            if d > max_drift:
                max_drift, max_asset = d, a
        if max_drift >= cfg.drift_action:
            score += 30
            reasons.append(f"Hedeften belirgin sapma: %{max_drift*100:.0f} ({max_asset})")
        elif max_drift >= cfg.drift_notable:
            score += 15
            reasons.append(f"Hedeften sapma: %{max_drift*100:.0f} ({max_asset})")
    else:
        # Portfoy sinif haritasi eksik — kullanici dikkatine deger
        score += 15
        reasons.append("Portfoy sinif haritasi eksik")

    # 4) Onerilen degisim buyuklugu (turnover)
    turnover = (cost_analysis or {}).get("turnover_pct", 0.0) or 0.0
    if turnover >= cfg.turnover_notable:
        score += 10
        reasons.append(f"Onerilen degisim portfoyun %{turnover*100:.0f}'i")

    # 4b) Konsantrasyon riski — tek sinifin asiri agirligi (class_weights doluysa)
    if class_weights:
        top_asset = max(class_weights, key=class_weights.get)
        top_w = class_weights[top_asset]
        if top_w >= cfg.concentration_action:
            score += 25
            reasons.append(
                f"Konsantrasyon riski: portföyün %{top_w*100:.0f}'i tek sınıfta ({top_asset})"
            )
        elif top_w >= cfg.concentration_notable:
            score += 12
            reasons.append(
                f"Yüksek yoğunlaşma: %{top_w*100:.0f} ({top_asset}) — çeşitlendirmeyi düşün"
            )

    # 5) Anomaliler (high tercih; yoksa medium)
    anomalies = (regime_result or {}).get("anomalies") or []
    high = [a for a in anomalies if a.get("severity") == "high"]
    medium = [a for a in anomalies if a.get("severity") == "medium"]
    if high:
        score += 20
        reasons.append(high[0].get("message", "Yuksek onemli piyasa anomalisi"))
    elif medium:
        score += 10
        reasons.append(medium[0].get("message", "Orta onemli piyasa hareketi"))

    score = max(0, min(100, score))

    if score >= cfg.level_action:
        level = "action"
    elif score >= cfg.level_notable:
        level = "notable"
    else:
        level = "quiet"

    return {"score": score, "level": level, "reasons": reasons}
