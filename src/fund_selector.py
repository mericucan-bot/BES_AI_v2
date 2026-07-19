"""Sinif bazli oneriye somut fon adaylari uretir (ML skoru + gercek getiri).

Yalniz YEREL dosyalari okur (ag cagrisi YOK):
- En guncel TEFAS snapshot'i: data/tefas_cache/snapshot_EMK_*.parquet
- En guncel ML tahminleri:     data/ml/predictions_fwd_return_3m_*.csv
Skor onceligi: predicted_rank_3m (varsa) > predicted_fwd_return_3m >
snapshot return_1y (ML dosyasi yoksa fallback).
"""
import glob
import logging
import os
from typing import Dict, List, Optional, Set

import pandas as pd

logger = logging.getLogger(__name__)


def suggest_funds_for_class(
    asset_class: str,
    n: int = 3,
    cache_dir: str = "data/tefas_cache",
    ml_dir: str = "data/ml",
    class_map: Optional[Dict[str, str]] = None,
    held_codes: Optional[Set[str]] = None,
) -> List[Dict]:
    """
    Verilen sinif icin en iyi n aday fon.

    Skor onceligi: predicted_rank_3m (varsa) > predicted_fwd_return_3m >
    snapshot return_1y (ML dosyasi yoksa fallback).
    Returns: [{fund_code, fund_name, score_basis, predicted_3m, return_1y,
               risk, held}] — skora gore azalan. Veri yoksa [].

    Her adim try/except'li: hicbir hata yukari sizmaz, loglayip [] doner.
    """
    try:
        # 1. En guncel snapshot'i oku (yoksa [])
        files = sorted(glob.glob(os.path.join(cache_dir, "snapshot_EMK_*.parquet")))
        if not files:
            logger.info(f"Aday fon: snapshot yok ({cache_dir})")
            return []
        try:
            snap = pd.read_parquet(files[-1])
        except Exception as e:
            logger.warning(f"Aday fon: snapshot okunamadi: {e}")
            return []
        if "fund_code" not in snap.columns:
            logger.warning("Aday fon: snapshot'ta fund_code kolonu yok")
            return []

        # 2. Sinif haritasi -> yalniz istenen sinifin fonlari. (5 sinif sozde-kodu
        #    zaten snapshot'ta fon olarak yok, ozel filtre gerekmez.)
        if class_map is None:
            try:
                from src.asset_mapping import load_fund_class_map
                class_map = load_fund_class_map(cache_dir)
            except Exception as e:
                logger.warning(f"Aday fon: sinif haritasi yuklenemedi: {e}")
                class_map = {}
        cand = snap[snap["fund_code"].map(
            lambda c: class_map.get(str(c).upper()) == asset_class
        )].copy()
        if cand.empty:
            return []

        # 3. En guncel predictions CSV'yi oku (varsa) ve fund_code uzerinden merge et.
        score_col = None
        score_basis = "return_1y"
        try:
            pred_files = sorted(glob.glob(
                os.path.join(ml_dir, "predictions_fwd_return_3m_*.csv")
            ))
            if pred_files:
                preds = pd.read_csv(pred_files[-1])
                if "fund_code" in preds.columns:
                    keep = ["fund_code"]
                    for col in ("predicted_fwd_return_3m", "predicted_rank_3m"):
                        if col in preds.columns:
                            keep.append(col)
                    cand = cand.merge(preds[keep], on="fund_code", how="left")
                    # Rank varsa tercih et; yoksa ham getiri tahmini
                    if "predicted_rank_3m" in cand.columns:
                        score_col = "predicted_rank_3m"
                        score_basis = "ml_rank"
                    elif "predicted_fwd_return_3m" in cand.columns:
                        score_col = "predicted_fwd_return_3m"
                        score_basis = "ml_return"
        except Exception as e:
            logger.warning(f"Aday fon: ML tahmin dosyasi okunamadi: {e}")

        # 4. ML yoksa return_1y fallback
        if score_col is None:
            if "return_1y" not in cand.columns:
                return []
            score_col = "return_1y"
            score_basis = "return_1y"

        # Skoru NaN olanlari at, azalan sirala, ilk n
        cand = cand[cand[score_col].notna()]
        if cand.empty:
            return []
        cand = cand.sort_values(score_col, ascending=False).head(max(0, int(n)))

        # 5. Cikti alanlari
        held_upper = {str(c).upper() for c in held_codes} if held_codes else set()
        out: List[Dict] = []
        for _, row in cand.iterrows():
            code = str(row["fund_code"])
            name = row.get("fund_name")
            name = "" if name is None or pd.isna(name) else str(name)
            pred = row.get("predicted_fwd_return_3m")
            r1y = row.get("return_1y")
            risk = row.get("risk")
            out.append({
                "fund_code": code,
                "fund_name": name[:40],
                "score_basis": score_basis,
                "predicted_3m": None if pred is None or pd.isna(pred) else float(pred),
                "return_1y": None if r1y is None or pd.isna(r1y) else float(r1y),
                "risk": None if risk is None or pd.isna(risk) else float(risk),
                "held": code.upper() in held_upper,
            })
        return out
    except Exception as e:
        logger.warning(f"Aday fon onerisi uretilemedi ({asset_class}): {e}")
        return []
