"""Oneri karnesi: 'uygulasaydin' vs 'dokunmadin' serilerini snapshot +
gercek NAV verisinden uretir. Ek durum dosyasi tutmaz (deterministik).

- actual (dokunmadin): gercek fon bakiyelerinin NAV ile revalue zinciri
  (katki/cikislardan arindirilmis piyasa getirisi).
- advised (uygulasaydin): her snapshot'ta o ayin target_weights'ine rebalans
  edilen sinif-portfoyunun zinciri (rebalans maliyeti dusulmus).

Streamlit'siz saf pandas — app yalnizca cagirip cizer.
"""
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


def _cost(w_from: Dict[str, float], w_to: Dict[str, float], slip: float) -> float:
    """Rebalans maliyeti (oran): her bacak slippage — cost_model ile ayni semantik."""
    assets = set(w_from) | set(w_to)
    turnover = sum(abs(w_to.get(a, 0.0) - w_from.get(a, 0.0)) for a in assets)
    return slip * turnover


def _load_snapshots(history_dir: str) -> List[Dict]:
    rows = []
    for p in sorted(Path(history_dir).glob("*_snapshot.json")):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            logger.warning(f"Karne: snapshot okunamadi ({p.name}): {e}")
            continue
        pv = d.get("portfolio_value", {})
        rows.append({
            "date": (d.get("run_date") or "")[:10],
            "total_value": pv.get("total_value"),
            "weights": pv.get("weights") or {},
            "class_weights": pv.get("class_weights") or {},
            "target_weights": d.get("recommendation", {}).get("target_weights") or {},
            "eval_market": (d.get("previous_evaluation") or {}).get("market_return"),
        })
    # Gecerli deger + tarih olanlar
    return [r for r in rows if r["date"] and r["total_value"]]


def build_tracks(
    history_dir: str = "data/history",
    cache_dir: str = "data/tefas_cache",
    slippage_pct: float = 0.002,
) -> pd.DataFrame:
    """
    Returns: DataFrame(date, actual_value, advised_value, actual_ret,
                       advised_ret, basis) — kronolojik. <2 snapshot -> bos df.
    basis: actual getirisinin kaynagi (nav_fund|nav_class|eval_market|flat),
           advised veri eksikse "|adv_flat" ekli.
    """
    snaps = _load_snapshots(history_dir)
    if len(snaps) < 2:
        return pd.DataFrame()

    from src.performance_tracker import PerformanceTracker
    tracker = PerformanceTracker()
    try:
        from src.backtest_engine import RealNavReturnProvider
        provider = RealNavReturnProvider(cache_dir=cache_dir)
    except Exception as e:
        logger.warning(f"Karne: NAV provider kurulamadi: {e}")
        provider = None

    def _cls_returns(d0, d1):
        if provider is None or not provider.has_nav_history():
            return None
        try:
            return provider.returns_between(d0, d1)
        except Exception:
            return None

    v_act = float(snaps[0]["total_value"])
    v_adv = float(snaps[0]["total_value"])
    w_adv = dict(snaps[0]["target_weights"])
    # Ilk rebalans maliyeti (mevcut -> ilk hedef)
    v_adv *= 1 - _cost(snaps[0]["class_weights"], w_adv, slippage_pct)

    rows = [{
        "date": snaps[0]["date"], "actual_value": round(v_act, 2),
        "advised_value": round(v_adv, 2), "actual_ret": float("nan"),
        "advised_ret": float("nan"), "basis": "start",
    }]

    for s0, s1 in zip(snaps[:-1], snaps[1:]):
        d0, d1 = s0["date"], s1["date"]
        cls_ret = _cls_returns(d0, d1)

        # --- actual getiri (oncelik zinciri) ---
        actual_ret, basis = None, None
        if provider is not None and provider.has_nav_history() and s0["weights"]:
            try:
                fr = provider.fund_returns_between(list(s0["weights"]), d0, d1)
                if fr:
                    prev_tl = {c: s0["total_value"] * w for c, w in s0["weights"].items()}
                    res = tracker.revalue_holdings(prev_tl, fr)
                    if res is not None:
                        actual_ret, basis = res["market_return"], "nav_fund"
            except Exception:
                pass
        if actual_ret is None and cls_ret is not None and s0["class_weights"]:
            actual_ret = sum(cw * cls_ret.get(c, 0.0) for c, cw in s0["class_weights"].items())
            basis = "nav_class"
        if actual_ret is None and isinstance(s1["eval_market"], (int, float)):
            actual_ret, basis = float(s1["eval_market"]), "eval_market"
        if actual_ret is None:
            actual_ret, basis = 0.0, "flat"
            logger.warning(f"Karne: {d0}->{d1} actual getiri verisi yok, notr")

        # --- advised getiri ---
        if cls_ret is not None and w_adv:
            advised_ret = sum(w_adv.get(c, 0.0) * cls_ret.get(c, 0.0) for c in w_adv)
        else:
            advised_ret = actual_ret
            basis = basis + "|adv_flat"

        v_act *= 1 + actual_ret
        v_adv *= 1 + advised_ret

        # --- rebalans (d1 aninda): surukleme -> yeni hedef ---
        if cls_ret is not None and w_adv and (1 + advised_ret) != 0:
            w_drift = {c: w_adv.get(c, 0.0) * (1 + cls_ret.get(c, 0.0)) / (1 + advised_ret)
                       for c in w_adv}
        else:
            w_drift = dict(w_adv)
        w_new = dict(s1["target_weights"]) or w_adv
        v_adv *= 1 - _cost(w_drift, w_new, slippage_pct)
        w_adv = w_new

        rows.append({
            "date": d1, "actual_value": round(v_act, 2),
            "advised_value": round(v_adv, 2), "actual_ret": round(actual_ret, 6),
            "advised_ret": round(advised_ret, 6), "basis": basis,
        })

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df.reset_index(drop=True)
