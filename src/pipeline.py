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
from src.io_utils import atomic_write_text

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
        tefas_cache_dir: str = "data/tefas_cache",
    ):
        self.portfolio_path = Path(portfolio_path)
        self.history_dir = Path(history_dir)
        self.learning_path = Path(learning_path)
        self.tefas_cache_dir = tefas_cache_dir
        self.history_dir.mkdir(parents=True, exist_ok=True)

        self.regime_engine = RegimeEngineV2()
        self.learning_engine = LearningEngineV2(history_path=str(self.learning_path))
        self.tracker = PerformanceTracker(history_dir=str(self.history_dir) + "/")
        self.cost_model = TransactionCostModel()

        # Fon-bazli yeniden degerleme icin TEFAS collector (yalniz yerel cache okur).
        # Yoksa pipeline yine calisir; piyasa getirisi hesaplanamaz, fallback olur.
        try:
            from src.data_collector import TEFASCollector
            self.collector = TEFASCollector()
        except Exception as e:
            logger.warning(f"TEFAS collector baslatilamadi, piyasa getirisi devre disi: {e}")
            self.collector = None

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

        atomic_write_text(path, json.dumps(snapshot, ensure_ascii=False, indent=2, default=str))
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

        prev_pv = prev.get("portfolio_value", {})
        prev_value = prev_pv.get("total_value")
        prev_fund_weights = prev_pv.get("weights", {})  # fon kodu -> agirlik
        prev_regime = prev.get("regime", {}).get("detected")
        prev_weights = prev.get("recommendation", {}).get("target_weights", {})
        prev_run_date = prev.get("run_date")

        if not prev_value or not prev_regime:
            logger.warning(f"Onceki snapshot eksik veri ({prev_path.name}), atlaniyor")
            return None

        # Kullanicinin bildirdigi toplam degisim (katki/cikislari da icerir)
        nominal_monthly_return = (current_value - prev_value) / prev_value

        # return_1m tabanli hesaplar (piyasa getirisi + karma benchmark) yalnizca
        # ~1 aylik degerlendirme periyodunda gecerli. Pipeline aylik kosmuyorsa
        # (orn. 2 ay arayla) 1-aylik getiriyi kullanmak periyodu yanlis temsil eder
        # — bu durumda periyot-dogru olan nominal/BIST'e duser. Periyot
        # cozumlenemezse (eski/eksik tarih) mevcut davranis korunur (is_monthly=True).
        elapsed_days = self._period_days(prev_run_date, current_date)
        is_monthly = elapsed_days is None or (20 <= elapsed_days <= 45)
        if elapsed_days is not None and not is_monthly:
            logger.info(
                f"Degerlendirme periyodu ~1 ay degil ({elapsed_days}g) — return_1m "
                f"tabanli piyasa getirisi/karma benchmark atlandi, nominal/BIST kullanildi"
            )

        # PIYASA getirisi: onceki fon bakiyelerini gerceklesen fon getirileriyle
        # yeniden degerle (katki/cikislardan arindirilmis — alfa icin daha dogru).
        # Once gercek gunluk NAV'dan TAM DONEM (her periyotta gecerli), yoksa
        # ~aylik snapshot return_1m'i (yalniz is_monthly).
        market_return = self._compute_market_return(
            prev_value, prev_fund_weights, prev_run_date, current_date, is_monthly
        )
        if market_return is not None:
            logger.info(
                f"Piyasa getirisi: {market_return:+.2%} (nominal toplam: {nominal_monthly_return:+.2%})"
            )

        # Alfa ve ogrenme sinyali piyasa getirisi uzerinden (varsa); yoksa nominal
        return_for_alpha = market_return if market_return is not None else nominal_monthly_return
        monthly_return = nominal_monthly_return  # gosterim/geriye uyum icin

        # Benchmark: cok-varlikli BES portfoyu icin %100 BIST adil degil. Once
        # gercek-NAV tam-donem karma BES sepeti, yoksa ~aylik snapshot, o da yoksa
        # periyot-dogru BIST 100.
        benchmark_return, benchmark_basis = self._resolve_benchmark(
            prev_run_date, current_date, is_monthly
        )
        alpha = return_for_alpha - benchmark_return

        # Onceki rebalance maliyetini gross alpha'dan dus → net alpha
        prev_cost_pct = prev.get("recommendation", {}).get("cost_analysis", {}).get("total_cost_pct", 0.0)
        net_alpha = alpha - prev_cost_pct

        # Gercekleseni tercih et (PLAN-03 sonrasi snapshot'larda var); yoksa
        # onerilen hedefe geri dus (eski snapshot'lar icin).
        actual_class_weights = prev_pv.get("class_weights") or {}
        weights_for_learning = actual_class_weights if actual_class_weights else prev_weights

        self.learning_engine.record_observation(
            date=current_date.strftime("%Y-%m-%d"),
            regime=prev_regime,
            weights_used=weights_for_learning,
            monthly_return=return_for_alpha,
            alpha_vs_benchmark=net_alpha,
            source_id=prev_path.name,
        )

        status = "WIN" if net_alpha > 0.005 else "LOSS" if net_alpha < -0.005 else "NEUTRAL"
        evaluation = {
            "previous_snapshot": prev_path.name,
            "previous_run_date": prev_run_date,
            "previous_value": prev_value,
            "current_value": current_value,
            "monthly_return": round(monthly_return, 4),
            "market_return": round(market_return, 4) if market_return is not None else None,
            "return_basis": "market" if market_return is not None else "nominal",
            "benchmark_return": round(benchmark_return, 4),
            "benchmark_basis": benchmark_basis,
            "gross_alpha": round(alpha, 4),
            "rebalance_cost_pct": round(prev_cost_pct, 6),
            "net_alpha": round(net_alpha, 4),
            "alpha_vs_benchmark": round(net_alpha, 4),
            "weights_basis": "actual_class" if actual_class_weights else "recommended_target",
            "previous_regime": prev_regime,
            "status": status,
        }
        logger.info(
            f"Onceki ay degerlendirmesi: {status} "
            f"(getiri={monthly_return:.2%}, bench={benchmark_return:.2%}, "
            f"brut α={alpha:+.2%}, maliyet={prev_cost_pct:.4%}, net α={net_alpha:+.2%})"
        )
        return evaluation

    @staticmethod
    def _period_days(prev_date_str: Optional[str], current_date: datetime) -> Optional[int]:
        """Onceki snapshot ile su an arasi gun sayisi (tz-guvenli; yalniz tarih).
        Cozumlenemezse None."""
        if not prev_date_str:
            return None
        try:
            prev_dt = datetime.fromisoformat(prev_date_str.replace("Z", "+00:00"))
            return abs((current_date.date() - prev_dt.date()).days)
        except (ValueError, TypeError, AttributeError):
            return None

    def _get_nav_provider(self):
        """RealNavReturnProvider'i tek sefer kur ve onbellekle (yoksa None)."""
        if not hasattr(self, "_nav_provider_cached"):
            try:
                from src.backtest_engine import RealNavReturnProvider
                p = RealNavReturnProvider()
                self._nav_provider_cached = p if p.has_data() else None
            except Exception as e:
                logger.warning(f"NAV provider kurulamadi: {e}")
                self._nav_provider_cached = None
        return self._nav_provider_cached

    def _compute_market_return(
        self,
        prev_total: float,
        prev_fund_weights: Dict[str, float],
        prev_date: Optional[str],
        current_date: datetime,
        is_monthly: bool,
    ) -> Optional[float]:
        """
        Portfoyun PIYASA getirisi (katki/cikislardan arindirilmis), oran.
        Oncelik: (1) gercek gunluk NAV'dan TAM DONEM getirisi — her periyotta
        gecerli; (2) ~aylik snapshot return_1m (yalniz is_monthly). Yoksa None.
        """
        if not prev_fund_weights or not prev_total:
            return None
        codes = list(prev_fund_weights.keys())
        prev_holdings_tl = {c: prev_total * w for c, w in prev_fund_weights.items()}

        # 1) NAV tam-donem
        nav = self._get_nav_provider()
        if nav is not None and nav.has_nav_history() and prev_date:
            try:
                fr = nav.fund_returns_between(codes, prev_date, current_date)
                if fr:
                    res = self.tracker.revalue_holdings(prev_holdings_tl, fr)
                    if res is not None:
                        return res["market_return"]
            except Exception as e:
                logger.warning(f"NAV tam-donem getiri hatasi: {e}")

        # 2) Snapshot return_1m (yalniz ~aylik)
        if is_monthly and self.collector is not None:
            try:
                fr = self.collector.get_fund_returns(codes=codes, period="return_1m")
                if fr:
                    res = self.tracker.revalue_holdings(prev_holdings_tl, fr)
                    if res is not None:
                        return res["market_return"]
            except Exception as e:
                logger.warning(f"return_1m piyasa getirisi hatasi: {e}")
        return None

    def _resolve_benchmark(
        self, prev_date: Optional[str], current_date: datetime, is_monthly: bool
    ):
        """
        Karma BES benchmark getirisi + bazi. Oncelik: (1) gercek-NAV tam-donem
        esit-agirlik sepet; (2) ~aylik snapshot sepet; (3) periyot-dogru BIST 100.
        Donus: (oran, bazi).
        """
        from src.backtest_engine import DEFAULT_BENCHMARK_WEIGHTS

        def _blend(real):
            return float(sum(w * real.get(a, 0.0) for a, w in DEFAULT_BENCHMARK_WEIGHTS.items()))

        nav = self._get_nav_provider()
        # 1) NAV tam-donem
        if nav is not None and nav.has_nav_history() and prev_date:
            try:
                real = nav.returns_between(prev_date, current_date)
                if real is not None:
                    return _blend(real), "blended_nav"
            except Exception as e:
                logger.warning(f"NAV tam-donem benchmark hatasi: {e}")
        # 2) Snapshot return_1m sepet (yalniz ~aylik)
        if is_monthly and nav is not None:
            try:
                import pandas as pd
                real = nav.returns_asof(pd.Timestamp(current_date))
                if real is not None:
                    return _blend(real), "blended_1m"
            except Exception as e:
                logger.warning(f"Snapshot benchmark hatasi: {e}")
        # 3) Periyot-dogru BIST 100
        return self._calculate_bist_benchmark(prev_date, current_date), "bist100"

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

        # 2b. Fon kodu -> varlik sinifi eslemesi (oneriler sinif uzayinda uretilir).
        # Gercek portfoy TEFAS fon kodlari icerir (AHS/BGL/...); hedef agirliklar
        # ise soyut sinif kodlari (VEF/KTS/...). Karsilastirma sinif uzayinda yapilir;
        # portfolio_value["weights"] fon-bazli KALIR (piyasa getirisi icin gerekli).
        from src.asset_mapping import load_fund_class_map, holdings_to_class, funds_by_class
        fund_class_map = load_fund_class_map(self.tefas_cache_dir)
        class_tl, unmapped_tl = holdings_to_class(holdings, fund_class_map)
        mapped_total = sum(class_tl.values())
        class_weights = (
            {k: v / mapped_total for k, v in class_tl.items()} if mapped_total > 0 else {}
        )
        portfolio_value["class_weights"] = {k: round(v, 4) for k, v in class_weights.items()}
        if unmapped_tl:
            portfolio_value["unmapped_tl"] = unmapped_tl
            logger.warning(
                f"Sinifa eslenemeyen fonlar (onerilerde HARIC): {list(unmapped_tl)} "
                f"— guncel snapshot cekilirse (auto_refresh_cache) eslenebilir"
            )

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

        # 6. Oneriler — SINIF uzayinda (fon kodlari degil). Kenar durum: hicbir fon
        # sinifa eslenemediyse (snapshot yok) sessizce "her seyi sat" URETME.
        recommendation_note = None
        if not class_weights and unmapped_tl:
            recommendations = []
            recommendation_note = "fon-sinif haritasi yok, oneri uretilemedi"
            logger.error(
                "Fon-sinif haritasi bos ve tum fonlar eslenemedi — oneri uretilmedi. "
                "Guncel TEFAS snapshot'i cekilmeli (snapshot_EMK_*.parquet)."
            )
        else:
            recommendations = self._generate_recommendations(
                current_weights=class_weights,
                target_weights=target_weights,
                total_value=mapped_total,
            )
            # Her oneriye kullanicinin o siniftaki gercek fonlarini ipucu ekle
            _fbc = funds_by_class(holdings, fund_class_map)
            for rec in recommendations:
                rec["funds_in_class"] = _fbc.get(rec["asset"], [])

            # 6d. AL onerilerine somut aday fonlar ekle (hata pipeline'i durdurmaz)
            try:
                from src.fund_selector import suggest_funds_for_class
                _held = {str(c).upper() for c in holdings}
                for rec in recommendations:
                    if rec.get("action") == "BUY":
                        rec["candidate_funds"] = suggest_funds_for_class(
                            rec["asset"], cache_dir=self.tefas_cache_dir,
                            class_map=fund_class_map, held_codes=_held,
                        )
            except Exception as e:
                logger.warning(f"Aday fon onerisi uretilemedi: {e}")

        # 6b. Aylik limit filtresi
        recommendations = self.cost_model.filter_recommendations_by_limit(recommendations)

        # 6c. Maliyet hesabi
        cost_analysis = self.cost_model.calculate_rebalance_cost(
            recommendations=recommendations,
            total_value=portfolio_value["total_value"],
        )

        # 6e. Onemlilik skoru (bildirim yogunlugu icin)
        from src.significance import compute_significance
        significance = compute_significance(
            regime_result=regime_result,
            evaluation=evaluation,
            class_weights=class_weights,
            target_weights=target_weights,
            cost_analysis=cost_analysis,
        )
        logger.info(
            f"Onemlilik: {significance['score']}/100 ({significance['level']})"
        )

        # 6f. Devlet katkisi durumu (BES %30 match / tavan)
        from src.state_contribution import analyze_contribution
        _monthly_contrib = portfolio.get("monthly_contribution_tl")
        state_contribution = analyze_contribution(_monthly_contrib)
        logger.info(
            f"Devlet katkisi: yilda {state_contribution['annual_match']:,.0f} TL "
            f"(tavan {state_contribution['max_annual_match']:,.0f}, "
            f"at_cap={state_contribution['at_cap']})"
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
            "recommendation_note": recommendation_note,
            "previous_evaluation": evaluation,
            "real_portfolio": real_portfolio,
            "significance": significance,
            "state_contribution": state_contribution,
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
