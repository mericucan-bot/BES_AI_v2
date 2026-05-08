import logging
import os
import json
from datetime import datetime
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class PerformanceTracker:
    def __init__(self, history_dir="data/history/"):
        self.history_dir = history_dir

    def calculate_current_portfolio_value(self, holdings_tl):
        total_value = sum(holdings_tl.values())
        weights = {k: (v / total_value) for k, v in holdings_tl.items()}
        return {"total_value": round(total_value, 2), "weights": weights, "date": datetime.now().isoformat()}

    def calculate_real_return(
        self,
        nominal_return: float,
        cpi_yoy: Optional[float],
        period_months: int = 1,
    ) -> Dict:
        """
        Fisher denklemi ile reel getiri hesapla.

        nominal_return: Donemsel nominal getiri (oran, 0.05 = %5)
        cpi_yoy: Yillik TUFE orani (oran, 0.306 = %30.6)
        period_months: Donem uzunlugu (ay)

        Fisher: (1 + nominal) / (1 + inflation_period) - 1
        """
        result = {
            "nominal_return": round(nominal_return, 6),
            "cpi_yoy": cpi_yoy,
            "inflation_period": None,
            "real_return": None,
            "inflation_drag": None,
        }

        if cpi_yoy is None or cpi_yoy <= -1:
            logger.warning("CPI verisi yok veya gecersiz, reel getiri hesaplanamıyor")
            return result

        try:
            inflation_period = (1 + cpi_yoy) ** (period_months / 12) - 1
        except (ValueError, OverflowError) as e:
            logger.error(f"Enflasyon hesaplama hatasi: {e}")
            return result

        real_return = (1 + nominal_return) / (1 + inflation_period) - 1
        inflation_drag = real_return - nominal_return

        result["inflation_period"] = round(inflation_period, 6)
        result["real_return"] = round(real_return, 6)
        result["inflation_drag"] = round(inflation_drag, 6)

        logger.debug(
            f"Reel getiri: nominal={nominal_return:.2%} - enflasyon={inflation_period:.2%} "
            f"= reel={real_return:.2%}"
        )
        return result

    def calculate_real_portfolio_value(
        self,
        current_value: float,
        initial_value: float,
        initial_date: str,
        current_date: str,
        cpi_yoy: Optional[float],
    ) -> Dict:
        """
        Portfoyun bugunku reel degerini hesapla.

        "100.000 TL yatirdim, su an 120.000 TL, ama reel olarak ne kadar kazandim?"
        """
        from datetime import datetime as dt

        result = {
            "nominal_value": current_value,
            "real_value": None,
            "nominal_total_return": None,
            "real_total_return": None,
            "months_elapsed": None,
        }

        if initial_value <= 0:
            return result

        nominal_total_return = (current_value - initial_value) / initial_value
        result["nominal_total_return"] = round(nominal_total_return, 6)

        try:
            d1 = dt.fromisoformat(initial_date.replace("Z", "+00:00"))
            d2 = dt.fromisoformat(current_date.replace("Z", "+00:00"))
            months = (d2.year - d1.year) * 12 + (d2.month - d1.month)
            months = max(months, 1)
        except (ValueError, TypeError):
            months = 1

        result["months_elapsed"] = months

        if cpi_yoy is None:
            return result

        real_calc = self.calculate_real_return(nominal_total_return, cpi_yoy, months)
        result["real_total_return"] = real_calc["real_return"]

        if real_calc["real_return"] is not None:
            result["real_value"] = round(initial_value * (1 + real_calc["real_return"]), 2)

        return result

    def get_portfolio_history(self, history_dir: str = "data/history") -> "pd.DataFrame":
        """
        Tüm aylık snapshot'lardan portföy değer geçmişini çıkar.

        Returns: DataFrame(date, total_value, regime, confidence,
                           monthly_return, real_return, real_value, snapshot_file)
        """
        import pandas as pd
        from pathlib import Path

        history_path = Path(history_dir)
        snapshots = sorted(history_path.glob("*_snapshot.json"))

        if not snapshots:
            logger.info("Snapshot dosyası yok, boş DataFrame dönüyor")
            return pd.DataFrame()

        rows = []
        for snap_path in snapshots:
            try:
                with open(snap_path, encoding="utf-8") as f:
                    data = json.load(f)

                run_date   = data.get("run_date", "")[:10]
                pv         = data.get("portfolio_value", {})
                regime_obj = data.get("regime", {})
                regime     = regime_obj.get("detected", "?")
                confidence = regime_obj.get("confidence", 0)

                prev_eval    = data.get("previous_evaluation") or {}
                monthly_return = prev_eval.get("monthly_return")

                real_metrics = prev_eval.get("real_metrics") or {}
                real_return  = real_metrics.get("real_return")

                real_pf    = data.get("real_portfolio") or {}
                real_value = real_pf.get("real_value")

                rows.append({
                    "date":          run_date,
                    "total_value":   pv.get("total_value", 0),
                    "regime":        regime,
                    "confidence":    confidence,
                    "monthly_return": monthly_return,
                    "real_return":   real_return,
                    "real_value":    real_value,
                    "snapshot_file": snap_path.name,
                })
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"Snapshot okuma hatası ({snap_path.name}): {e}")
                continue

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

        logger.info(f"Portföy geçmişi: {len(df)} snapshot yüklendi")
        return df
