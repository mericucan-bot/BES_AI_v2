"""
E-posta bildirim tercihleri: yukle/kaydet + e-posta dogrulama.

Streamlit'ten bagimsiz saf mantik (disk JSON + regex) — unit-test edilebilir.
"""
import json
import re
from pathlib import Path
from typing import Union

from src.io_utils import atomic_write_text

NOTIF_PREFS_PATH = Path("data/notification_prefs.json")
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def is_valid_email(addr: str) -> bool:
    return bool(addr) and bool(_EMAIL_RE.match(addr.strip()))


def load_notification_prefs(path: Union[str, Path] = NOTIF_PREFS_PATH) -> dict:
    defaults = {
        "email_enabled": False,
        "email_address": "",
        "on_regime_change": True,
        "weekly_summary": False,
        "critical_signal": True,
    }
    try:
        path = Path(path)
        if path.exists():
            saved = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(saved, dict):
                return defaults
            # Tip-safe merge: bozuk veri uygulamayi cokertmesin
            for key, default_val in defaults.items():
                if key not in saved:
                    continue
                try:
                    if isinstance(default_val, bool):
                        defaults[key] = bool(saved[key])
                    elif isinstance(default_val, str):
                        defaults[key] = str(saved[key])
                except (TypeError, ValueError):
                    pass
    except Exception:
        pass
    return defaults


def save_notification_prefs(prefs: dict, path: Union[str, Path] = NOTIF_PREFS_PATH) -> bool:
    try:
        atomic_write_text(
            Path(path),
            json.dumps(prefs, ensure_ascii=False, indent=2),
        )
        return True
    except Exception:
        return False
