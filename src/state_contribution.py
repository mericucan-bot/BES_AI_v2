"""BES devlet katkisi (%30) optimizasyonu — tavan = yillik brut asgari ucretin %30'u."""
import logging
import os
from dataclasses import dataclass
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# 2026 tahmini brut asgari ucret (aylik, TL). Guncellenebilir; resmi rakam
# aciklaninca degistirilmeli. Env ile override: BES_MIN_WAGE_MONTHLY.
DEFAULT_MIN_WAGE_MONTHLY_2026 = 30000.0
MATCH_RATE = 0.30


@dataclass
class ContributionConfig:
    match_rate: float = MATCH_RATE
    min_wage_monthly: float = DEFAULT_MIN_WAGE_MONTHLY_2026


def _resolve_min_wage(config: Optional[ContributionConfig]) -> float:
    env = os.environ.get("BES_MIN_WAGE_MONTHLY")
    if env:
        try:
            return float(env)
        except ValueError:
            pass
    return (config or ContributionConfig()).min_wage_monthly


def analyze_contribution(
    monthly_contribution_tl: Optional[float],
    config: Optional[ContributionConfig] = None,
) -> Dict:
    """
    Returns:
      {
        "monthly_contribution": float|None,
        "annual_contribution": float|None,
        "annual_match": float,          # bu yil kazanilan devlet katkisi
        "max_annual_match": float,      # tavan
        "match_gap": float,             # kacirilanolan (max - current), >=0
        "at_cap": bool,
        "suggested_extra_monthly": float,  # tavani doldurmak icin ek aylik
        "utilization_pct": float,       # 0-1
      }
    Katki None/0 ise annual_match=0, tum tavan kacirilir.
    """
    cfg = config or ContributionConfig()
    min_wage_monthly = _resolve_min_wage(config)
    match_rate = cfg.match_rate

    min_wage_annual = min_wage_monthly * 12.0
    max_annual_match = min_wage_annual * match_rate

    annual_contribution = (monthly_contribution_tl or 0) * 12.0
    annual_match = min(annual_contribution * match_rate, max_annual_match)
    match_gap = max(0.0, max_annual_match - annual_match)
    at_cap = match_gap < 1.0

    # Tavanı dolduran yillik katki = min_wage_annual
    cap_annual_contribution = (
        max_annual_match / match_rate if match_rate > 0 else min_wage_annual
    )
    suggested_extra_monthly = max(
        0.0, (cap_annual_contribution - annual_contribution) / 12.0
    )

    utilization_pct = (
        annual_match / max_annual_match if max_annual_match > 0 else 0.0
    )

    return {
        "monthly_contribution": monthly_contribution_tl,
        "annual_contribution": annual_contribution if monthly_contribution_tl is not None else None,
        "annual_match": annual_match,
        "max_annual_match": max_annual_match,
        "match_gap": match_gap,
        "at_cap": at_cap,
        "suggested_extra_monthly": suggested_extra_monthly,
        "utilization_pct": utilization_pct,
    }
