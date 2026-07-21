"""Telegram bildirimi — onemli aylarda (notable/action) anlik uyari.

Kurulum: @BotFather'dan bot olustur -> TELEGRAM_BOT_TOKEN; kendi chat id'n
icin bota /start yazip https://api.telegram.org/bot<TOKEN>/getUpdates
ciktisindaki chat.id'yi al -> TELEGRAM_CHAT_ID. Ikisi .env'e yazilir.
"""
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

_API = "https://api.telegram.org/bot{token}/sendMessage"
_MAX_LEN = 3900   # Telegram siniri 4096; guvenli pay

_TR_MONTHS = {
    1: "Ocak", 2: "Şubat", 3: "Mart", 4: "Nisan",
    5: "Mayıs", 6: "Haziran", 7: "Temmuz", 8: "Ağustos",
    9: "Eylül", 10: "Ekim", 11: "Kasım", 12: "Aralık",
}

_REGIME_TR = {
    "STABLE": "Sakin Piyasa",
    "CRISIS": "Kriz Modu",
    "RISK_ON": "Yükseliş",
    "RATE_HIKE": "Faiz Artışı",
}


class TelegramNotifier:
    def __init__(self, bot_token: Optional[str] = None, chat_id: Optional[str] = None):
        self.bot_token = bot_token or os.getenv("TELEGRAM_BOT_TOKEN")
        self.chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID")
        self.is_configured = bool(self.bot_token and self.chat_id)

    def send(self, text: str) -> bool:
        """Duz metin gonder (parse_mode YOK — MarkdownV2 escape tuzagina girme).
        Yapilandirilmamis/ag hatasi -> False, ASLA exception."""
        if not self.is_configured:
            return False
        try:
            import requests
            r = requests.post(
                _API.format(token=self.bot_token),
                json={"chat_id": self.chat_id, "text": text[:_MAX_LEN]},
                timeout=15,
            )
            ok = r.status_code == 200
            if not ok:
                logger.warning(
                    f"Telegram gonderim hatasi: {r.status_code} {r.text[:200]}"
                )
            return ok
        except Exception as e:
            logger.warning(f"Telegram hatasi: {e}")
            return False


def _tr_month_year(dt: Optional[datetime] = None) -> str:
    d = dt or datetime.now()
    return f"{_TR_MONTHS[d.month]} {d.year}"


def build_alert_message(pipeline_result: Dict) -> Optional[str]:
    """significance level notable/action ise kisa uyari metni; quiet/None -> None."""
    if not pipeline_result:
        return None
    sig = pipeline_result.get("significance") or {}
    level = sig.get("level")
    if level not in ("notable", "action"):
        return None

    score = sig.get("score", 0)
    reasons = sig.get("reasons") or []
    total = (
        pipeline_result.get("portfolio_value", {}) or {}
    ).get("total_value", 0)
    detected = (pipeline_result.get("regime") or {}).get("detected", "?")
    regime_tr = _REGIME_TR.get(detected, detected)

    lines = [
        f"⚠️ BES AI — {_tr_month_year()}",
        f"Önemlilik: {score}/100 ({level})",
    ]
    for r in reasons:
        lines.append(f"• {r}")
    lines.append(f"Portföy: {total:,.0f} TL | Rejim: {regime_tr}")

    actions = (pipeline_result.get("recommendation") or {}).get("actions") or []
    first = next(
        (a for a in actions if a.get("action") in ("BUY", "SELL")),
        None,
    )
    if first:
        sign = "+" if first.get("action") == "BUY" else ""
        diff = first.get("diff_tl", 0)
        lines.append(
            f"İlk aksiyon: {first.get('action')} {first.get('asset')} "
            f"{sign}{diff:,.0f} TL"
        )

    lines.append("Detay: aylık e-posta / dashboard")

    try:
        from src.data_health import check_data_health
        health = check_data_health()
        if not health.get("ok"):
            warnings = health.get("warnings") or []
            if warnings:
                lines.append(f"⚙️ {warnings[0]}")
    except Exception:
        pass

    return "\n".join(lines)


def build_multi_message(all_results: List[Dict]) -> Optional[str]:
    """Coklu portfoy: her portfoyden tek satir; hicbiri notable+ degilse None.
    all_results: [{"slug","name","result"}] (main.run_all_portfolios ciktisi)."""
    if not all_results:
        return None

    interesting = []
    for item in all_results:
        result = item.get("result") or {}
        sig = result.get("significance") or {}
        level = sig.get("level")
        if level in ("notable", "action"):
            interesting.append(item)

    if not interesting:
        return None

    lines = [f"⚠️ BES AI — {_tr_month_year()} | {len(all_results)} portföy"]
    for item in all_results:
        name = item.get("name") or item.get("slug") or "?"
        result = item.get("result") or {}
        if result.get("status") != "SUCCESS":
            lines.append(f"{name}: HATA")
            continue
        sig = result.get("significance") or {}
        level = sig.get("level") or "quiet"
        score = sig.get("score", 0)
        total = (result.get("portfolio_value") or {}).get("total_value", 0)
        lines.append(f"{name}: {score}/100 ({level}) — {total:,.0f} TL")
        if level == "action":
            reasons = sig.get("reasons") or []
            if reasons:
                lines.append(f"  • {reasons[0]}")

    return "\n".join(lines)
