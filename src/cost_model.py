import logging
from typing import Dict, List, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class CostConfig:
    """
    BES/TEFAS fon islem maliyet parametreleri.
    Varsayilan degerler BES ortalamasina gore ayarlanmistir.
    """
    switch_fee_pct: float = 0.0
    exit_load_pct: float = 0.0
    slippage_pct: float = 0.002
    max_monthly_switches: int = 6
    min_switch_amount_tl: float = 100
    # BES SISTEM-ICI fon degisikliklerinde stopaj YOKTUR; stopaj yalniz sistemden
    # CIKISTA ve elde tutma suresine gore alinir. Aylik rebalance kapsaminda 0 dogru.
    stopaj_pct: float = 0.0
    # Yillik fon yonetim gideri (tasima maliyeti). Gercek-NAV getirilerinde zaten
    # net oldugu icin yalniz proxy modunda uygulanir — bkz. holding_cost_pct.
    management_fee_annual_pct: float = 0.018


class TransactionCostModel:
    """
    Rebalance islemlerinin maliyetini hesaplar.

    Kullanim:
    1. Pipeline'da rebalance onerisi uretildikten sonra maliyet hesapla
    2. Alpha'dan maliyet dus → net alpha
    3. Maliyet cok yuksekse rebalance'i atla (minimum threshold)
    """

    def __init__(self, config: Optional[CostConfig] = None):
        self.config = config or CostConfig()

    def calculate_rebalance_cost(
        self,
        recommendations: List[Dict],
        total_value: float,
    ) -> Dict:
        """
        Bir rebalance planinin toplam maliyetini hesapla.

        recommendations: pipeline._generate_recommendations() ciktisi
        total_value: toplam portfoy degeri
        """
        breakdown = []
        total_cost = 0.0
        turnover = 0.0
        switch_count = 0

        for rec in recommendations:
            action = rec.get("action", "HOLD")
            diff_tl = abs(rec.get("diff_tl", 0))

            if action == "HOLD" or diff_tl < self.config.min_switch_amount_tl:
                continue

            switch_count += 1
            turnover += diff_tl

            slippage = diff_tl * self.config.slippage_pct
            fee = diff_tl * self.config.switch_fee_pct
            exit_load = diff_tl * self.config.exit_load_pct if action == "SELL" else 0.0

            item_cost = slippage + fee + exit_load
            total_cost += item_cost

            breakdown.append({
                "asset": rec.get("asset", "?"),
                "action": action,
                "amount_tl": round(diff_tl, 2),
                "slippage_tl": round(slippage, 2),
                "fee_tl": round(fee, 2),
                "exit_load_tl": round(exit_load, 2),
                "total_cost_tl": round(item_cost, 2),
            })

        total_cost_pct = total_cost / total_value if total_value > 0 else 0
        turnover_pct = turnover / total_value if total_value > 0 else 0
        exceeds_limit = switch_count > self.config.max_monthly_switches

        if exceeds_limit:
            logger.warning(
                f"Rebalance {switch_count} fon degisikligi gerektiriyor, "
                f"aylik limit {self.config.max_monthly_switches}. "
                f"Bazi islemler ertelenmeli."
            )

        logger.info(
            f"Rebalance maliyeti: {total_cost:,.2f} TL (%{total_cost_pct*100:.3f}), "
            f"turnover: {turnover:,.2f} TL (%{turnover_pct*100:.1f}), "
            f"{switch_count} islem"
        )

        return {
            "total_cost_tl": round(total_cost, 2),
            "total_cost_pct": round(total_cost_pct, 6),
            "turnover_tl": round(turnover, 2),
            "turnover_pct": round(turnover_pct, 4),
            "switch_count": switch_count,
            "exceeds_monthly_limit": exceeds_limit,
            "cost_breakdown": breakdown,
            "cost_effective": True,
        }

    def holding_cost_pct(self, period_months: float = 1.0) -> float:
        """
        Donemsel TASIMA maliyeti (fon yonetim gideri), oran olarak.
        annual_fee * (period_months/12).

        ONEMLI: TEFAS return_1m gibi GERCEK fon getirileri zaten yonetim
        gideri DUSULMUS (net) gelir — gercek-NAV modunda bu ek olarak
        UYGULANMAMALI (cift sayim olur). Yalniz proxy/sentetik getirilerde
        (fee iceremeyen) uygulanir.
        """
        return self.config.management_fee_annual_pct * (period_months / 12.0)

    def calculate_net_alpha(
        self,
        gross_alpha: float,
        rebalance_cost_pct: float,
    ) -> Dict:
        """
        Brut alpha'dan islem maliyetini duserek net alpha hesapla.
        """
        net = gross_alpha - rebalance_cost_pct

        if gross_alpha != 0:
            cost_ratio = abs(rebalance_cost_pct / gross_alpha)
        else:
            cost_ratio = float("inf") if rebalance_cost_pct > 0 else 0

        cost_effective = cost_ratio < 0.50 if gross_alpha > 0 else True

        result = {
            "gross_alpha": round(gross_alpha, 6),
            "cost_pct": round(rebalance_cost_pct, 6),
            "net_alpha": round(net, 6),
            "cost_ratio": round(cost_ratio, 4) if cost_ratio != float("inf") else None,
            "cost_effective": cost_effective,
        }

        if not cost_effective:
            logger.warning(
                f"Rebalance maliyet-etkin degil: brut a={gross_alpha:.2%}, "
                f"maliyet={rebalance_cost_pct:.2%}, oran={cost_ratio:.1%}"
            )

        return result

    def filter_recommendations_by_limit(
        self,
        recommendations: List[Dict],
    ) -> List[Dict]:
        """
        Aylik fon degisikligi limitini asiyorsa, en buyuk diff'lere oncelik ver.
        Limit asiminda kucuk islemler HOLD olarak isaretlenir (deferred=True).
        """
        actionable = [r for r in recommendations if r.get("action") != "HOLD"]
        holds = [r for r in recommendations if r.get("action") == "HOLD"]

        if len(actionable) <= self.config.max_monthly_switches:
            return recommendations

        sorted_actions = sorted(actionable, key=lambda x: -abs(x.get("diff_tl", 0)))
        kept = sorted_actions[:self.config.max_monthly_switches]
        deferred = sorted_actions[self.config.max_monthly_switches:]

        for item in deferred:
            item["action"] = "HOLD"
            item["deferred"] = True
            logger.info(f"Islem ertelendi (limit): {item['asset']} {item['diff_tl']:+,.0f} TL")

        return kept + deferred + holds
