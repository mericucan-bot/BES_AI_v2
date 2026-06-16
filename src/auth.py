"""
Uygulama sifre korumasi ve sunucu-tarafi brute-force throttle.

Streamlit'ten bagimsiz saf mantik — bu yuzden unit-test edilebilir.
Sifre secret/env'den; throttle durumu diskte JSON (tum oturum/tablar ortak).
app.py bu fonksiyonlari import eder; st.secrets'i get_app_password'a parametre gecer.
"""
import json
import os
import time
from pathlib import Path
from typing import Mapping, Optional, Tuple, Union

from src.io_utils import atomic_write_text

# Sunucu tarafi rate-limit parametreleri
AUTH_THROTTLE_PATH = Path("data/auth_throttle.json")
AUTH_WINDOW_SEC = 300   # 5 dakikalik kayar pencere
AUTH_MAX_FAILED = 5     # Pencere icinde 5 hata = lockout
AUTH_LOCKOUT_SEC = 60   # Lockout suresi


def get_app_password(secrets: Optional[Mapping] = None) -> str:
    """Secret/env'den sifreyi al. Tanimli degilse bos string doner.

    secrets: st.secrets gibi Mapping (opsiyonel). Streamlit bagimliligini
    cagirana birakir; burada yalnizca 'APP_PASSWORD' anahtari okunur.
    """
    try:
        if secrets is not None and "APP_PASSWORD" in secrets:
            return secrets["APP_PASSWORD"]
    except Exception:
        pass
    return os.environ.get("APP_PASSWORD", "")


def load_auth_throttle(path: Union[str, Path] = AUTH_THROTTLE_PATH) -> dict:
    path = Path(path)
    if not path.exists():
        return {"failed_attempts": [], "lockout_until": 0.0}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return {
            "failed_attempts": [float(t) for t in data.get("failed_attempts", [])],
            "lockout_until": float(data.get("lockout_until", 0.0)),
        }
    except Exception:
        return {"failed_attempts": [], "lockout_until": 0.0}


def save_auth_throttle(state: dict, path: Union[str, Path] = AUTH_THROTTLE_PATH) -> None:
    try:
        atomic_write_text(Path(path), json.dumps(state))
    except Exception:
        pass


def record_failed_attempt(path: Union[str, Path] = AUTH_THROTTLE_PATH) -> Tuple[int, float]:
    """Bir hatali deneme kaydet. Donus: (kalan_deneme, lockout_until)."""
    now = time.time()
    state = load_auth_throttle(path)
    # Eski denemeleri pencereden cikar
    state["failed_attempts"] = [
        t for t in state["failed_attempts"] if now - t < AUTH_WINDOW_SEC
    ]
    state["failed_attempts"].append(now)
    if len(state["failed_attempts"]) >= AUTH_MAX_FAILED:
        state["lockout_until"] = now + AUTH_LOCKOUT_SEC
        state["failed_attempts"] = []
    save_auth_throttle(state, path)
    remaining = max(0, AUTH_MAX_FAILED - len(state["failed_attempts"]))
    return remaining, state["lockout_until"]


def reset_auth_throttle(path: Union[str, Path] = AUTH_THROTTLE_PATH) -> None:
    save_auth_throttle({"failed_attempts": [], "lockout_until": 0.0}, path)
