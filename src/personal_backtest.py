"""Kullanicinin GUNCEL fonlariyla gecmis backtest: 'kendi fonlarini tuttun' vs
'sistemi kullansaydin' — nav_history'nin tum donemi uzerinde aylik walk-forward.
Deterministik; ek durum dosyasi tutmaz."""
import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _cost(w_from: Dict[str, float], w_to: Dict[str, float], slip: float) -> float:
    """Rebalans maliyeti (oran): her bacak slippage — counterfactual ile ayni semantik."""
    assets = set(w_from) | set(w_to)
    turnover = sum(abs(w_to.get(a, 0.0) - w_from.get(a, 0.0)) for a in assets)
    return slip * turnover


def run_personal_backtest(
    holdings_tl: Dict[str, float],
    cache_dir: str = "data/tefas_cache",
    initial_capital: float = 100_000.0,
    slippage_pct: float = 0.002,
    use_learning: bool = False,
) -> pd.DataFrame:
    """
    Returns: DataFrame(date, hold_value, advised_value, regime, hold_ret,
                       advised_ret) — aylik. Veri yetersizse BOS df.
    """
    empty = pd.DataFrame(
        columns=["date", "hold_value", "advised_value", "regime", "hold_ret", "advised_ret"]
    )

    if not holdings_tl:
        return empty

    from src.backtest_engine import RealNavReturnProvider
    provider = RealNavReturnProvider(cache_dir=cache_dir)
    if not provider.has_nav_history():
        logger.warning("personal_backtest: nav_history yok veya kullanilamaz")
        return empty

    nav_index = provider._nav_pivot.index
    d_min = nav_index.min()
    d_max = nav_index.max()
    rebal_dates = pd.date_range(d_min, d_max, freq="BME")
    if len(rebal_dates) < 2:
        logger.warning("personal_backtest: yetersiz aylik tarih (<2)")
        return empty

    from src.asset_mapping import load_fund_class_map, holdings_to_class
    fund_class_map = load_fund_class_map(cache_dir)
    codes = [c for c in holdings_tl if holdings_tl[c] > 0]
    if not codes:
        return empty

    total_tl = sum(float(holdings_tl[c]) for c in codes)
    if total_tl <= 0:
        return empty
    w_fund = {c: float(holdings_tl[c]) / total_tl for c in codes}

    class_tl, _unmapped = holdings_to_class(
        {c: holdings_tl[c] for c in codes}, fund_class_map
    )
    class_total = sum(class_tl.values())
    w_class_start = (
        {k: v / class_total for k, v in class_tl.items()} if class_total > 0 else {}
    )

    from src.regime_engine import RegimeEngineV2
    from src.learning_engine import LearningEngineV2
    regime_engine = RegimeEngineV2()
    learner = LearningEngineV2(static_only=not use_learning)

    def _detect_regime(as_of) -> str:
        try:
            out = regime_engine.compute_composite_score(as_of_date=pd.Timestamp(as_of))
            return out.get("detected") or "STABLE"
        except Exception as e:
            logger.warning(f"personal_backtest: rejim hatasi ({as_of}): {e} — STABLE")
            return "STABLE"

    d0 = rebal_dates[0]
    regime0 = _detect_regime(d0)
    w_adv = dict(learner.get_optimized_weights(regime0))

    hold_value = float(initial_capital)
    advised_value = float(initial_capital)

    rows: List[Dict] = [{
        "date": d0,
        "hold_value": hold_value,
        "advised_value": advised_value,
        "regime": regime0,
        "hold_ret": np.nan,
        "advised_ret": np.nan,
    }]

    # Ilk rebalans maliyeti (start satirindan sonra, ilk getiri doneminden once)
    init_cost = _cost(w_class_start, w_adv, slippage_pct)
    if init_cost > 0:
        advised_value *= (1.0 - init_cost)

    for i in range(len(rebal_dates) - 1):
        d_start = rebal_dates[i]
        d_end = rebal_dates[i + 1]

        # hold: gercek fon getirileri
        fr = provider.fund_returns_between(codes, d_start, d_end)
        if fr:
            hold_ret = sum(w_fund.get(c, 0.0) * fr.get(c, 0.0) for c in codes)
        else:
            logger.info(f"personal_backtest: fon getirisi yok {d_start.date()}->{d_end.date()}")
            hold_ret = 0.0
            fr = {}

        # advised: sinif sepeti
        cls_ret = provider.returns_between(d_start, d_end)
        if cls_ret:
            advised_ret = sum(w_adv.get(a, 0.0) * cls_ret.get(a, 0.0) for a in w_adv)
        else:
            advised_ret = hold_ret

        hold_value *= (1.0 + hold_ret)
        advised_value *= (1.0 + advised_ret)

        # Buy-and-hold agirlik kaymasi
        if fr and abs(1.0 + hold_ret) > 1e-12:
            for c in list(w_fund.keys()):
                w_fund[c] = w_fund[c] * (1.0 + fr.get(c, 0.0)) / (1.0 + hold_ret)

        # Rejim (d_end as_of) + yeni hedef
        regime = _detect_regime(d_end)
        w_new = dict(learner.get_optimized_weights(regime))

        # Suruklenmis advised agirliklari -> rebalans maliyeti
        if cls_ret and abs(1.0 + advised_ret) > 1e-12:
            w_drifted = {
                a: w_adv.get(a, 0.0) * (1.0 + cls_ret.get(a, 0.0)) / (1.0 + advised_ret)
                for a in set(w_adv) | set(cls_ret)
            }
        else:
            w_drifted = dict(w_adv)

        rb_cost = _cost(w_drifted, w_new, slippage_pct)
        if rb_cost > 0:
            advised_value *= (1.0 - rb_cost)
        w_adv = w_new

        rows.append({
            "date": d_end,
            "hold_value": hold_value,
            "advised_value": advised_value,
            "regime": regime,
            "hold_ret": hold_ret,
            "advised_ret": advised_ret,
        })

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df.reset_index(drop=True)
