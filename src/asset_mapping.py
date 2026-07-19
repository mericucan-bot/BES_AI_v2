"""Fon kodu -> soyut varlik sinifi (VEF/KTS/ALT/KCH/CASH) eslemesi.

Kaynak: en guncel TEFAS snapshot'indaki 'category' alani.
Sinif kodlarinin kendisi (demo portfoyler) kendine eslenir.
"""
import glob
import logging
import os
from typing import Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

ASSET_CLASSES = ("VEF", "KTS", "ALT", "KCH", "CASH")

# Soyut varlik sinifi -> TEFAS 'category' (kucuk-harf substring) eslemesi.
# NOT: backtest_engine.py'deki kopyadan TASINDI (tek kaynak burasi).
# SIRA ONEMLI: ilk eslesen sinif kazanir (dict sirasi = kontrol sirasi).
# "participation equity" VEF'e, "participation variable" KCH'ye burada yakalanir.
ASSET_CATEGORY_MAP: Dict[str, List[str]] = {
    "VEF":  ["stock fund", "equity", "index fund"],
    "ALT":  ["gold", "precious metals"],
    "KCH":  ["mixed fund", "variable fund", "fund of funds",
             "life cycle", "target fund"],
    # Standart/Devlet Katkisi fonlari mevzuat geregi agirlikla kamu borclanma
    # senedi tasir -> KTS. Kira sertifikasi/sukuk = faizsiz borclanma -> KTS.
    "KTS":  ["debt instruments", "government bonds", "govt. bonds",
             "standard fund", "state contribution", "lease certificate", "sukuk"],
    # Baslangic fonlari dusuk riskli/likit -> CASH
    "CASH": ["money market", "initial"],
}

# Kod bazli manuel istisnalar — kategori alani yaniltici olan fonlar icin.
# Kategori eslemesinden SONRA uygulanir (her zaman kazanir).
# GMF: "GUMUS FON SEPETI" — gumus = kiymetli maden, ama TEFAS kategorisi
# "Fund of Funds" oldugundan otomatik eslenemiyor (kullanici onayi: 2026-07-18).
MANUAL_CLASS_OVERRIDES: Dict[str, str] = {
    "GMF": "ALT",
}

USER_OVERRIDES_PATH = "data/user_class_overrides.json"


def load_user_overrides(path: str = USER_OVERRIDES_PATH) -> Dict[str, str]:
    """Kullanicinin app'ten kaydettigi {FON_KODU: sinif} istisnalari.
    Dosya yoksa/bozuksa bos dict. Yalniz gecerli siniflar kabul edilir."""
    try:
        import json
        from pathlib import Path as _P
        p = _P(path)
        if not p.exists():
            return {}
        raw = json.loads(p.read_text(encoding="utf-8"))
        return {str(k).upper(): v for k, v in raw.items() if v in ASSET_CLASSES}
    except Exception:
        return {}


def save_user_override(code: str, asset_class: str,
                       path: str = USER_OVERRIDES_PATH) -> bool:
    """Tek bir fon istisnasini kalici kaydet (atomik). Gecersiz sinif -> False."""
    if asset_class not in ASSET_CLASSES:
        return False
    try:
        import json
        from src.io_utils import atomic_write_text
        cur = load_user_overrides(path)
        cur[str(code).upper()] = asset_class
        atomic_write_text(path, json.dumps(cur, ensure_ascii=False, indent=2))
        return True
    except Exception:
        return False


def load_fund_class_map(
    cache_dir: str = "data/tefas_cache",
    user_overrides_path: str = USER_OVERRIDES_PATH,
) -> Dict[str, str]:
    """En guncel snapshot'tan {FON_KODU: sinif} haritasi. Sinif kodlari
    kendine eslenir; snapshot yoksa yalniz sinif kodlari doner.
    Oncelik zinciri: kategori eslemesi < MANUAL_CLASS_OVERRIDES <
    kullanici override dosyasi (en son kazanir)."""
    mapping: Dict[str, str] = {c: c for c in ASSET_CLASSES}
    files = sorted(glob.glob(os.path.join(cache_dir, "snapshot_EMK_*.parquet")))
    if not files:
        logger.warning(f"Fon-sinif haritasi: snapshot yok ({cache_dir})")
        mapping.update(MANUAL_CLASS_OVERRIDES)
        mapping.update(load_user_overrides(user_overrides_path))
        return mapping
    try:
        df = pd.read_parquet(files[-1])
    except Exception as e:
        logger.warning(f"Fon-sinif haritasi: snapshot okunamadi: {e}")
        mapping.update(MANUAL_CLASS_OVERRIDES)
        mapping.update(load_user_overrides(user_overrides_path))
        return mapping
    if "fund_code" not in df.columns or "category" not in df.columns:
        mapping.update(MANUAL_CLASS_OVERRIDES)
        mapping.update(load_user_overrides(user_overrides_path))
        return mapping
    for code, cat in zip(df["fund_code"], df["category"]):
        cat_l = str(cat).lower()
        for asset, subs in ASSET_CATEGORY_MAP.items():
            if any(s in cat_l for s in subs):
                mapping[str(code).upper()] = asset
                break
    # Oncelik zinciri: kategori < MANUAL_CLASS_OVERRIDES < kullanici override dosyasi
    mapping.update(MANUAL_CLASS_OVERRIDES)
    mapping.update(load_user_overrides(user_overrides_path))
    return mapping


def holdings_to_class(
    holdings_tl: Dict[str, float],
    mapping: Dict[str, str],
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """TL bakiyeleri sinif bazinda topla.
    Returns: (sinif_tl, eslenmeyen_tl) — ikisi de {kod: TL}."""
    class_tl: Dict[str, float] = {}
    unmapped: Dict[str, float] = {}
    for code, tl in (holdings_tl or {}).items():
        cls = mapping.get(str(code).upper())
        if cls:
            class_tl[cls] = class_tl.get(cls, 0.0) + float(tl)
        else:
            unmapped[str(code)] = float(tl)
    return class_tl, unmapped


def funds_by_class(
    holdings_tl: Dict[str, float],
    mapping: Dict[str, str],
) -> Dict[str, List[str]]:
    """{sinif: [kullanicinin o siniftaki fon kodlari]} (TL>0 olanlar)."""
    out: Dict[str, List[str]] = {}
    for code, tl in (holdings_tl or {}).items():
        if tl <= 0:
            continue
        cls = mapping.get(str(code).upper())
        if cls:
            out.setdefault(cls, []).append(str(code))
    return out
