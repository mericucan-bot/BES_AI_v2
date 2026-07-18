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
ASSET_CATEGORY_MAP: Dict[str, List[str]] = {
    "VEF":  ["stock fund", "equity", "index fund"],
    "KTS":  ["debt instruments", "government bonds", "govt. bonds"],
    "ALT":  ["gold", "precious metals"],
    "KCH":  ["mixed fund", "variable fund"],
    "CASH": ["money market"],
}

# Kod bazli manuel istisnalar — kategori alani yaniltici olan fonlar icin.
# Kategori eslemesinden SONRA uygulanir (her zaman kazanir).
# GMF: "GUMUS FON SEPETI" — gumus = kiymetli maden, ama TEFAS kategorisi
# "Fund of Funds" oldugundan otomatik eslenemiyor (kullanici onayi: 2026-07-18).
MANUAL_CLASS_OVERRIDES: Dict[str, str] = {
    "GMF": "ALT",
}


def load_fund_class_map(cache_dir: str = "data/tefas_cache") -> Dict[str, str]:
    """En guncel snapshot'tan {FON_KODU: sinif} haritasi. Sinif kodlari
    kendine eslenir; snapshot yoksa yalniz sinif kodlari doner."""
    mapping: Dict[str, str] = {c: c for c in ASSET_CLASSES}
    files = sorted(glob.glob(os.path.join(cache_dir, "snapshot_EMK_*.parquet")))
    if not files:
        logger.warning(f"Fon-sinif haritasi: snapshot yok ({cache_dir})")
        mapping.update(MANUAL_CLASS_OVERRIDES)
        return mapping
    try:
        df = pd.read_parquet(files[-1])
    except Exception as e:
        logger.warning(f"Fon-sinif haritasi: snapshot okunamadi: {e}")
        mapping.update(MANUAL_CLASS_OVERRIDES)
        return mapping
    if "fund_code" not in df.columns or "category" not in df.columns:
        mapping.update(MANUAL_CLASS_OVERRIDES)
        return mapping
    for code, cat in zip(df["fund_code"], df["category"]):
        cat_l = str(cat).lower()
        for asset, subs in ASSET_CATEGORY_MAP.items():
            if any(s in cat_l for s in subs):
                mapping[str(code).upper()] = asset
                break
    # Manuel istisnalar kategori sonucunu ezer (snapshot olsun olmasin gecerli)
    mapping.update(MANUAL_CLASS_OVERRIDES)
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
