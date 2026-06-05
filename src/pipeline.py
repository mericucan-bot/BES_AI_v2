import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from zoneinfo import ZoneInfo

from src.regime_engine import RegimeEngineV2
from src.learning_engine import LearningEngineV2
from src.performance_tracker import PerformanceTracker
from src.cost_model import TransactionCostModel, CostConfig

logger = logging.getLogger(__name__)
TR_TZ = ZoneInfo("Europe/Istanbul")


class MonthlyPipeline:
    """
    Aylik BES portfoy analiz pipeline'i.

    Akis:
    1. Mevcut piyasa rejimi tespiti (RegimeEngine)
    2. Hedef agirliklar (LearningEngine'den, statik veya ogrenilmis)
    3. Mevcut portfoy degerlemesi (PerformanceTracker)
    4. Rebalance onerileri (BUY/SELL/HOLD)
    5. ONCEKI AY DEGERLENDIRMESI -> LearningEngine'e gozlem
    6. Aylik snapshot kaydi (data/history/YYYY_MM_snapshot.json)
    7. JSON rapor uretimi
    """

    def __init__(
        self,
        portfolio_path: str = "data/my_portfolio.json",
        history_dir: str = "data/history",
        learning_path: str = "data/learning_history.json",
    ):
        self.portfolio_path = Path(portfolio_path)
        self.history_dir = Path(history_dir)
        self.learning_path = Path(learning_path)
        self.history_dir.mkdir(parents=True, exist_ok=True)

        self.regime_engine = RegimeEngineV2()
        self.learning_engine = LearningEngineV2(history_path=str(self.learning_path))
        self.tracker = PerformanceTracker(history_dir=str(self.history_dir) + "/")
        self.cost_model = TransactionCostModel()

    def _load_portfolio(self) -> Optional[Dict]:
        if not self.portfolio_path.exists():
            logger.error(f"Portfoy dosyasi bulunamadi: {self.portfolio_path}")
            return None
        try:
            with open(self.portfolio_path, encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"Portfoy JSON parse hatasi: {e}")
            return None

    def _save_snapshot(self, snapshot: Dict, run_date: datetime) -> Path:
        filename = f"{run_date.strftime('%Y_%m')}_snapshot.json"
        path = self.history_dir / filename

        if path.exists():
            # Veri kaybini onle: eskisini timestamp'li .bak olarak yedekle
            backup_name = (
                f"{run_date.strftime('%Y_%m')}_snapshot."
                f"bak_{run_date.strftime('%H%M%S')}.json"
            )
            backup_path = self.history_dir / backup_name
            try:
                path.rename(backup_path)
                logger.warning(
                    f"Bu ay icin snapshot zaten vardi, yedeklendi: {backup_name}"
                )
            except OSError as e:
                logger.error(f"Snapshot yedeklenemedi, uzerine yazilacak: {e}")

        with open(path, "w", encoding="utf-8") as f:
            json.dump(snapshot, f, ensure_ascii=False, indent=2, default=str)
        logger.info(f"Snapshot kaydedildi: {path}")
        return path

    def _evaluate_previous_observation(
        self, current_value: float, current_date: datetime
    ) -> Optional[Dict]:
        """
        Onceki snapshot varsa, bu ay portfoyun ne yaptigini degerlendir.
        LearningEngine'e gozlem olarak kaydet.

        ONEMLI: Bu fonksiyon snapshot YAZILMADAN ONCE cagrilmali,
        yoksa bugunku snapshot'i kendi gecmisi olarak okuruz.
        """
        snapshots = sorted(self.history_dir.glob("*_snapshot.json"))

        # Bu ayin snapshot'i varsa onu haric tut (idempotent re-run icin)
        current_filename = f"{current_date.strftime('%Y_%m')}_snapshot.json"
        snapshots = [s for s in snapshots if s.name != current_filename]

        if not snapshots:
            logger.info("Onceki snapshot yok, ilk calistirma — degerlendirme atlandi")
            return None

        prev_path = snapshots[-1]
        try:
            with open(prev_path, encoding="utf-8") as f:
                prev = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            logger.error(f"Onceki snapshot okunamadi ({prev_path}): {e}")
            return None

        prev_value = prev.get("portfolio_value", {}).get("total_value")
        prev_regime = prev.get("regime", {}).get("detected")
        prev_weights = prev.get("recommendation", {}).get("target_weights", {})
        prev_run_date = prev.get("run_date")

        if not prev_value or not prev_regime:
            logger.warning(f"Onceki snapshot eksik veri ({prev_path.name}), atlaniyor")
            return None

        monthly_return = (current_value - prev_value) / prev_value
        benchmark_return = self._calculate_bist_benchmark(prev_run_date, current_date)
        alpha = monthly_return - benchmark_return

        # Onceki rebalance maliyetini gross alpha'dan dus → net alpha
        prev_cost_pct = prev.get("recommendation", {}).get("cost_analysis", {}).get("total_cost_pct", 0.0)
        net_alpha = alpha - prev_cost_pct

        self.learning_engine.record_observation(
            date=current_date.strftime("%Y-%m-%d"),
            regime=prev_regime,
            weights_used=prev_weights,
            monthly_return=monthly_return,
            alpha_vs_benchmark=net_alpha,
        )

        status = "WIN" if net_alpha > 0.005 else "LOSS" if net_alpha < -0.005 else "NEUTRAL"
        evaluation = {
            "previous_snapshot": prev_path.name,
            "previous_run_date": prev_run_date,
            "previous_value": prev_value,
            "current_value": current_value,
            "monthly_return": round(monthly_return, 4),
            "benchmark_return": round(benchmark_return, 4),
            "gross_alpha": round(alpha, 4),
            "rebalance_cost_pct": round(prev_cost_pct, 6),
            "net_alpha": round(net_alpha, 4),
            "alpha_vs_benchmark": round(net_alpha, 4),
            "previous_regime": prev_regime,
            "status": status,
        }
        logger.info(
            f"Onceki ay degerlendirmesi: {status} "
            f"(getiri={monthly_return:.2%}, bench={benchmark_return:.2%}, "
            f"brut α={alpha:+.2%}, maliyet={prev_cost_pct:.4%}, net α={net_alpha:+.2%})"
        )
        return evaluation

    def _calculate_bist_benchmark(
        self, prev_date_str: Optional[str], current_date: datetime
    ) -> float:
        """
        Iki tarih arasinda BIST 100 getirisi.
        Hata durumunda 0.0 doner (cash benchmark gibi davranir).
        """
        if not prev_date_str:
            return 0.0
        try:
            prev_date = datetime.fromisoformat(prev_date_str.replace("Z", "+00:00"))
            data = self.regime_engine.fetch_live_data(as_of_date=current_date)
            if data.empty or "BIST" not in data.columns:
                return 0.0

            bist = data["BIST"].dropna()
            if len(bist) < 2:
                return 0.0

            # "pad" (ffill) semantik olarak dogru: prev_date >= ilgili veri noktasi
            # olacak sekilde geriye dogru en yakini bulur — look-ahead bias yok.
            prev_idx = bist.index.get_indexer([prev_date], method="pad")[0]
            if prev_idx < 0 or prev_idx >= len(bist):
                return 0.0

            prev_bist = float(bist.iloc[prev_idx])
            curr_bist = float(bist.iloc[-1])
            return (curr_bist - prev_bist) / prev_bist
        except Exception as e:
            logger.warning(f"BIST benchmark hesaplanamadi, 0 kullaniliyor: {e}")
            return 0.0

    def _generate_recommendations(
        self,
        current_weights: Dict[str, float],
        target_weights: Dict[str, float],
        total_value: float,
        min_threshold_tl: float = 100,
    ) -> List[Dict]:
        """Rebalance aksiyonlarini uret."""
        recommendations = []
        all_assets = set(list(current_weights.keys()) + list(target_weights.keys()))

        for asset in all_assets:
            curr_w = current_weights.get(asset, 0)
            target_w = target_weights.get(asset, 0)
            diff_w = target_w - curr_w
            diff_tl = diff_w * total_value

            if abs(diff_tl) < min_threshold_tl:
                action = "HOLD"
            elif diff_tl > 0:
                action = "BUY"
            else:
                action = "SELL"

            recommendations.append({
                "asset": asset,
                "current_weight": round(curr_w, 4),
                "target_weight": round(target_w, 4),
                "diff_tl": round(diff_tl, 2),
                "action": action,
            })

        return sorted(recommendations, key=lambda x: (x["action"] == "HOLD", -abs(x["diff_tl"])))

    def run(self) -> Dict:
        """Tam pipeline'i calistir."""
        run_date = datetime.now(TR_TZ)
        logger.info(f"=== Pipeline basladi: {run_date.isoformat()} ===")

        # 1. Portfoy yukle
        portfolio = self._load_portfolio()
        if not portfolio:
            return {"status": "ERROR", "message": "Portföy yüklenemedi", "run_date": run_date.isoformat()}

        holdings = portfolio.get("holdings_tl", {})
        if not holdings:
            return {"status": "ERROR", "message": "holdings_tl boş", "run_date": run_date.isoformat()}

        # 2. Portfoy degerlemesi
        portfolio_value = self.tracker.calculate_current_portfolio_value(holdings)
        logger.info(f"Toplam deger: {portfolio_value['total_value']:,.2f} TL")

        # 3. ONCE onceki gozlemi degerlendir (snapshot YAZILMADAN!)
        evaluation = self._evaluate_previous_observation(
            current_value=portfolio_value["total_value"],
            current_date=run_date,
        )

        # 4. Rejim tespiti
        try:
            regime_result = self.regime_engine.compute_composite_score()
        except Exception as e:
            logger.exception(f"Rejim tespiti basarisiz: {e}")
            return {"status": "ERROR", "message": f"Rejim hatasi: {e}", "run_date": run_date.isoformat()}

        detected_regime = regime_result["detected"]

        # 4b. Onceki ay degerlendirmesine reel getiri ekle (macro artik mevcut)
        if evaluation and evaluation.get("monthly_return") is not None:
            cpi_yoy = regime_result.get("macro", {}).get("cpi_yoy")
            real_calc = self.tracker.calculate_real_return(
                nominal_return=evaluation["monthly_return"],
                cpi_yoy=cpi_yoy,
                period_months=1,
            )
            evaluation["real_metrics"] = real_calc
        logger.info(f"Tespit edilen rejim: {detected_regime} (guven: {regime_result['confidence']:.1%})")

        # 5. Hedef agirliklar
        target_weights = self.learning_engine.get_optimized_weights(detected_regime)
        regime_confidence = self.learning_engine.calculate_confidence_score(detected_regime)

        # 6. Oneriler
        recommendations = self._generate_recommendations(
            current_weights=portfolio_value["weights"],
            target_weights=target_weights,
            total_value=portfolio_value["total_value"],
        )

        # 6b. Aylik limit filtresi
        recommendations = self.cost_model.filter_recommendations_by_limit(recommendations)

        # 6c. Maliyet hesabi
        cost_analysis = self.cost_model.calculate_rebalance_cost(
            recommendations=recommendations,
            total_value=portfolio_value["total_value"],
        )

        # 7. Ilk snapshot'tan bu yana reel toplam getiri
        first_snapshot = self._get_first_snapshot()
        cpi_yoy = regime_result.get("macro", {}).get("cpi_yoy")
        real_portfolio = None
        if first_snapshot:
            _initial_v = first_snapshot.get("total_value")
            _initial_d = first_snapshot.get("date")
            # Tum alanlar mevcut degilse %0 yerine None don — UI "veri yok" gostersin
            if _initial_v is not None and _initial_d:
                real_portfolio = self.tracker.calculate_real_portfolio_value(
                    current_value=portfolio_value["total_value"],
                    initial_value=_initial_v,
                    initial_date=_initial_d,
                    current_date=run_date.isoformat(),
                    cpi_yoy=cpi_yoy,
                )
            else:
                logger.warning("Ilk snapshot eksik veri, reel getiri hesaplanamadi")

        # 8. Snapshot olustur
        snapshot = {
            "status": "SUCCESS",
            "run_date": run_date.isoformat(),
            "portfolio_value": portfolio_value,
            "regime": regime_result,
            "recommendation": {
                "target_weights": target_weights,
                "regime_learning_confidence": regime_confidence,
                "actions": recommendations,
                "cost_analysis": cost_analysis,
            },
            "previous_evaluation": evaluation,
            "real_portfolio": real_portfolio,
        }

        # 9. Diske kaydet
        snapshot_path = self._save_snapshot(snapshot, run_date)
        snapshot["snapshot_path"] = str(snapshot_path)

        logger.info(f"=== Pipeline tamamlandi: {run_date.isoformat()} ===")
        return snapshot

    def _get_first_snapshot(self) -> Optional[Dict]:
        """En eski snapshot'tan baslangic degerini al."""
        snapshots = sorted(self.history_dir.glob("*_snapshot.json"))
        if not snapshots:
            return None
        try:
            with open(snapshots[0], encoding="utf-8") as f:
                data = json.load(f)
            pv = data.get("portfolio_value", {})
            return {
                "total_value": pv.get("total_value"),
                "date": data.get("run_date"),
            }
        except (OSError, json.JSONDecodeError, KeyError):
            return None
