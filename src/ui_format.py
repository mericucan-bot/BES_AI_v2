"""
UI sunum yardimcilari: rejim aciklamasi, guven metni, TL/aksiyon formatlama.

Saf fonksiyonlar (Streamlit bagimsiz) — hem app.py hem testler kullanir.
"""
from typing import Dict


def explain_regime(regime: str) -> Dict:
    explanations = {
        "STABLE": {
            "symbol": "◎",
            "label": "Sakin Piyasa",
            "color": "blue",
            "border": "#3b82f6",
            "summary": "Piyasalarda belirgin bir yön yok. Dengeli bir dağılım mantıklı.",
            "action": "Portföyünü dengeli tut, panik yapma.",
            "detail": "Volatilite düşük, BIST belirgin bir trend göstermiyor, döviz sakin. Bu ortamda aşırı agresif veya defansif olmaya gerek yok.",
        },
        "CRISIS": {
            "symbol": "▼",
            "label": "Kriz Modu",
            "color": "red",
            "border": "#ef4444",
            "summary": "Piyasalarda ciddi düşüş veya belirsizlik var. Korunma öncelikli.",
            "action": "Altın ve sabit getirili fonlara ağırlık ver, hisse oranını azalt.",
            "detail": "BIST'te sert düşüş, dövizde hızlı yükseliş veya yüksek volatilite tespit edildi. Bu dönemlerde sermayeyi korumak önceliklidir.",
        },
        "RISK_ON": {
            "symbol": "▲",
            "label": "Yükseliş Trendi",
            "color": "green",
            "border": "#22c55e",
            "summary": "Piyasalar yukarı yönlü. Hisse ağırlığını artırma fırsatı.",
            "action": "Hisse ve karma fonlara ağırlık ver.",
            "detail": "BIST güçlü momentum gösteriyor, volatilite makul seviyelerde. Tarihsel olarak bu dönemlerde hisse fonları iyi performans gösterir.",
        },
        "RATE_HIKE": {
            "symbol": "≡",
            "label": "Faiz Artışı Dönemi",
            "color": "orange",
            "border": "#f59e0b",
            "summary": "Merkez bankası faiz artırıyor. Sabit getirili fonlar öne çıkıyor.",
            "action": "Kamu borçlanma (KTS) fonlarına ağırlık ver.",
            "detail": "TCMB politika faizini artırma eğiliminde. Yüksek faiz ortamında tahvil/bono fonları cazip getiri sunuyor.",
        },
    }
    return explanations.get(regime, explanations["STABLE"])


def confidence_to_text(confidence: float) -> str:
    if confidence >= 0.80:
        return "Yüksek güven — sinyal güçlü"
    elif confidence >= 0.60:
        return "Orta güven — makul sinyal"
    elif confidence >= 0.40:
        return "Düşük güven — belirsiz ortam"
    else:
        return "Çok düşük güven — dikkatli ol"


def format_tl(value: float) -> str:
    return f"{value:,.0f} TL".replace(",", ".")


def action_text(action: str) -> str:
    return {"BUY": "EKLE", "SELL": "AZALT", "HOLD": "DEĞİŞTİRME"}.get(action, action)
