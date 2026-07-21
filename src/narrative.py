"""Aylik pipeline sonucunu 5 cumlelik Turkce anlatiya cevirir (Claude API).
API anahtari/ag yoksa deterministik sablon ozete duser — akis asla durmaz.

LLM anlati icin opsiyonel: pip install anthropic + ANTHROPIC_API_KEY.
"""
import logging
import os
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# Ucuz, kisa Turkce ozet icin yeterli (claude-api skill onerisi).
_MODEL = "claude-haiku-4-5"

_REGIME_TR = {
    "STABLE": "Sakin Piyasa", "CRISIS": "Kriz Modu",
    "RISK_ON": "Yukselis", "RATE_HIKE": "Faiz Artisi",
}
_TR_MONTHS = {
    1: "Ocak", 2: "Subat", 3: "Mart", 4: "Nisan", 5: "Mayis", 6: "Haziran",
    7: "Temmuz", 8: "Agustos", 9: "Eylul", 10: "Ekim", 11: "Kasim", 12: "Aralik",
}


def _month_label() -> str:
    from datetime import datetime
    now = datetime.now()
    return f"{_TR_MONTHS[now.month]} {now.year}"


def _actionable(pipeline_result: Dict) -> list:
    acts = (pipeline_result.get("recommendation", {}) or {}).get("actions", []) or []
    return [a for a in acts if a.get("action") in ("BUY", "SELL")]


def _template_summary(pipeline_result: Dict) -> str:
    """LLM'siz deterministik ozet — her zaman calisir (ag/anahtar gerektirmez)."""
    if not pipeline_result or pipeline_result.get("status") != "SUCCESS":
        return f"{_month_label()} analizi tamamlanamadi; ayrintilar dashboard'da."
    regime = pipeline_result.get("regime", {}).get("detected", "STABLE")
    regime_tr = _REGIME_TR.get(regime, regime)
    total = pipeline_result.get("portfolio_value", {}).get("total_value", 0)
    acts = _actionable(pipeline_result)
    sig = pipeline_result.get("significance") or {}
    reasons = sig.get("reasons") or []

    parts = [f"{_month_label()} itibariyla piyasa {regime_tr} modunda."]
    parts.append(f"Portfoyunuzun toplam degeri {total:,.0f} TL.".replace(",", "."))
    if acts:
        parts.append(f"Bu ay {len(acts)} degisiklik onerisi var.")
    else:
        parts.append("Bu ay portfoyde degisiklik gerekmiyor.")
    if reasons and sig.get("level") in ("notable", "action"):
        parts.append(reasons[0] + ".")
    return " ".join(parts)


def _build_prompt(pipeline_result: Dict, ml_summary: Optional[Dict],
                  max_sentences: int) -> str:
    import json
    regime = pipeline_result.get("regime", {})
    ozet = {
        "ay": _month_label(),
        "rejim": _REGIME_TR.get(regime.get("detected"), regime.get("detected")),
        "rejim_guven": round(regime.get("confidence", 0), 2),
        "portfoy_deger_tl": pipeline_result.get("portfolio_value", {}).get("total_value"),
        "onemlilik": pipeline_result.get("significance", {}),
        "aksiyonlar": [
            {"sinif": a.get("asset"), "islem": a.get("action"),
             "tutar_tl": round(a.get("diff_tl", 0))}
            for a in _actionable(pipeline_result)
        ][:3],
    }
    real = pipeline_result.get("real_portfolio") or {}
    if real.get("real_total_return") is not None:
        ozet["reel_getiri"] = real["real_total_return"]
    if ml_summary and ml_summary.get("status") == "SUCCESS":
        ozet["ai_model_ic"] = ml_summary.get("best_ic")
    return (
        f"Sen bir BES yatirim danismanisin. Asagidaki aylik analizi bir yatirimciya "
        f"{max_sentences} cumlelik SADE Turkce ile ozetle. Yatirim tavsiyesi verme; "
        f"ne oldugunu ve neden bu onerinin ciktigini acikla. Sayilari abartma, "
        f"markdown kullanma, yalniz duz metin.\n\nAnaliz (JSON):\n"
        + json.dumps(ozet, ensure_ascii=False, default=str)
    )


def _call_claude(prompt: str, api_key: str) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    resp = client.messages.create(
        model=_MODEL,
        max_tokens=400,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(b.text for b in resp.content if b.type == "text")


def generate_narrative(
    pipeline_result: Dict,
    ml_summary: Optional[Dict] = None,
    max_sentences: int = 5,
) -> str:
    """Turkce anlati ozeti. API yoksa _template_summary fallback."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return _template_summary(pipeline_result)
    try:
        prompt = _build_prompt(pipeline_result, ml_summary, max_sentences)
        text = _call_claude(prompt, api_key)
        return text.strip() or _template_summary(pipeline_result)
    except Exception as e:
        logger.warning(f"LLM anlati uretilemedi, sablona dusuldu: {e}")
        return _template_summary(pipeline_result)
