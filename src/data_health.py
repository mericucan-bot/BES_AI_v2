"""Veri tazelik/saglik kontrolu — sessiz veri curumasini yakalar."""
import glob
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class HealthThresholds:
    nav_stale_days: int = 10        # nav_history bu kadar eskiyse uyar
    snapshot_stale_days: int = 45   # son tefas snapshot
    macro_stale_days: int = 3       # makro cache
    ml_stale_days: int = 45         # ml ozeti


def _age_days(dt: datetime, now: Optional[datetime] = None) -> int:
    """dt'den (now varsayilan: simdi) bugune kadar gecen tam gun sayisi.
    Negatif fark (saat kaymasi vb.) 0'a sabitlenir."""
    now = now or datetime.now()
    return max(0, (now - dt).days)


def _check_nav_history(cache_dir: str, max_age_days: int) -> Dict:
    """nav_history.parquet'teki en guncel 'date' kolonunun yasi."""
    name = "nav_history"
    try:
        path = os.path.join(cache_dir, "nav_history.parquet")
        if not os.path.exists(path):
            return {
                "name": name, "status": "missing",
                "detail": "⚠️ NAV geçmişi (nav_history.parquet) bulunamadı",
                "age_days": None,
            }

        import pandas as pd  # agir bagimlilik; yalniz gerektiginde yukle

        df = pd.read_parquet(path)
        if df.empty or "date" not in df.columns:
            return {
                "name": name, "status": "missing",
                "detail": "⚠️ NAV geçmişi boş veya bozuk (date kolonu yok)",
                "age_days": None,
            }
        last_date = pd.to_datetime(df["date"]).max()
        age = _age_days(last_date.to_pydatetime())
        if age > max_age_days:
            return {
                "name": name, "status": "stale",
                "detail": f"⚠️ NAV verisi {age} gündür güncellenmedi",
                "age_days": age,
            }
        return {
            "name": name, "status": "ok",
            "detail": f"NAV verisi {age} gün önce güncellendi",
            "age_days": age,
        }
    except Exception as e:
        logger.warning(f"nav_history saglik kontrolu hatasi: {e}")
        return {
            "name": name, "status": "missing",
            "detail": "⚠️ NAV geçmişi okunamadı (bozuk dosya)",
            "age_days": None,
        }


def _check_tefas_snapshot(cache_dir: str, max_age_days: int) -> Dict:
    """En guncel snapshot_EMK_*.parquet dosyasinin mtime yasi."""
    name = "tefas_snapshot"
    try:
        files = sorted(glob.glob(os.path.join(cache_dir, "snapshot_EMK_*.parquet")))
        if not files:
            return {
                "name": name, "status": "missing",
                "detail": "⚠️ TEFAS snapshot dosyası bulunamadı",
                "age_days": None,
            }
        mtime = datetime.fromtimestamp(os.path.getmtime(files[-1]))
        age = _age_days(mtime)
        if age > max_age_days:
            return {
                "name": name, "status": "stale",
                "detail": f"⚠️ TEFAS snapshot {age} gündür güncellenmedi",
                "age_days": age,
            }
        return {
            "name": name, "status": "ok",
            "detail": f"TEFAS snapshot {age} gün önce güncellendi",
            "age_days": age,
        }
    except Exception as e:
        logger.warning(f"tefas snapshot saglik kontrolu hatasi: {e}")
        return {
            "name": name, "status": "missing",
            "detail": "⚠️ TEFAS snapshot kontrol edilemedi",
            "age_days": None,
        }


def _check_macro_cache(macro_dir: str, max_age_days: int) -> Dict:
    """macro_dir/*.json icindeki en yeni 'fetched_at' (yoksa dosya mtime'i)."""
    name = "macro_cache"
    try:
        files = glob.glob(os.path.join(macro_dir, "*.json"))
        if not files:
            return {
                "name": name, "status": "missing",
                "detail": "⚠️ Makro cache dosyası bulunamadı",
                "age_days": None,
            }
        newest: Optional[datetime] = None
        for fp in files:
            ts = None
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                fetched_at = payload.get("fetched_at")
                if fetched_at:
                    ts = datetime.fromisoformat(fetched_at)
            except Exception:
                ts = None
            if ts is None:
                try:
                    ts = datetime.fromtimestamp(os.path.getmtime(fp))
                except Exception:
                    ts = None
            if ts is not None and (newest is None or ts > newest):
                newest = ts
        if newest is None:
            return {
                "name": name, "status": "missing",
                "detail": "⚠️ Makro cache tazeliği okunamadı",
                "age_days": None,
            }
        age = _age_days(newest)
        if age > max_age_days:
            return {
                "name": name, "status": "stale",
                "detail": f"⚠️ Makro veri {age} gündür güncellenmedi",
                "age_days": age,
            }
        return {
            "name": name, "status": "ok",
            "detail": f"Makro veri {age} gün önce güncellendi",
            "age_days": age,
        }
    except Exception as e:
        logger.warning(f"macro cache saglik kontrolu hatasi: {e}")
        return {
            "name": name, "status": "missing",
            "detail": "⚠️ Makro cache kontrol edilemedi",
            "age_days": None,
        }


def _check_ml_summary(ml_dir: str, max_age_days: int) -> Dict:
    """ml_dir/latest_run_summary.json icindeki 'run_date' yasi."""
    name = "ml_summary"
    try:
        path = os.path.join(ml_dir, "latest_run_summary.json")
        if not os.path.exists(path):
            return {
                "name": name, "status": "missing",
                "detail": "⚠️ ML özeti (latest_run_summary.json) bulunamadı",
                "age_days": None,
            }
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        run_date = payload.get("run_date")
        if not run_date:
            return {
                "name": name, "status": "missing",
                "detail": "⚠️ ML özetinde run_date alanı yok",
                "age_days": None,
            }
        dt = datetime.fromisoformat(run_date)
        age = _age_days(dt)
        if age > max_age_days:
            return {
                "name": name, "status": "stale",
                "detail": f"⚠️ ML özeti {age} gündür yenilenmedi",
                "age_days": age,
            }
        return {
            "name": name, "status": "ok",
            "detail": f"ML özeti {age} gün önce yenilendi",
            "age_days": age,
        }
    except Exception as e:
        logger.warning(f"ml summary saglik kontrolu hatasi: {e}")
        return {
            "name": name, "status": "missing",
            "detail": "⚠️ ML özeti kontrol edilemedi (bozuk dosya)",
            "age_days": None,
        }


def check_data_health(
    cache_dir: str = "data/tefas_cache",
    macro_dir: str = "data/cache",
    ml_dir: str = "data/ml",
    thresholds: Optional[HealthThresholds] = None,
) -> Dict:
    """
    Returns: {"ok": bool, "checks": [{"name","status","detail","age_days"}], "warnings": [str]}
    status: "ok" | "stale" | "missing". ok = hicbir check "missing"/"stale" degil.

    Yalnizca yerel dosya varligi/icerigi/mtime okur — aga cikmaz. Modul hicbir
    sekilde exception firlatmaz: her kontrol kendi icinde yakalanir (ve bu
    fonksiyon da yine de her cagriyi ayrica try/except'ler); biri patlarsa o
    kontrol "missing" olarak isaretlenir, digerleri calismaya devam eder.
    """
    th = thresholds or HealthThresholds()
    checks: List[Dict] = []

    _checklist = (
        ("nav_history", _check_nav_history, cache_dir, th.nav_stale_days),
        ("tefas_snapshot", _check_tefas_snapshot, cache_dir, th.snapshot_stale_days),
        ("macro_cache", _check_macro_cache, macro_dir, th.macro_stale_days),
        ("ml_summary", _check_ml_summary, ml_dir, th.ml_stale_days),
    )
    for fallback_name, check_fn, arg_dir, max_age in _checklist:
        try:
            checks.append(check_fn(arg_dir, max_age))
        except Exception as e:
            # _check_* fonksiyonlari kendi try/except'ini icerir; buraya
            # normalde dusulmez ama beklenmedik bir hata bile modulu patlatmasin.
            logger.warning(f"{fallback_name} kontrolu beklenmedik sekilde patladi: {e}")
            checks.append({
                "name": fallback_name, "status": "missing",
                "detail": f"⚠️ {fallback_name} kontrol edilemedi",
                "age_days": None,
            })

    warnings = [c["detail"] for c in checks if c["status"] in ("stale", "missing")]
    return {"ok": not warnings, "checks": checks, "warnings": warnings}
