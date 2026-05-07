import logging
from collections import Counter
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.regime_engine import RegimeEngineV2
from src.learning_engine import LearningEngineV2
from src.cost_model import TransactionCostModel, CostConfig
from src.macro_engine import MacroEngine

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """Backtest parametreleri."""
    start_date: str = "2024-01-01"
    end_date: str = "2026-04-01"
    rebalance_frequency: str = "monthly"
    initial_capital: float = 100_000.0
    benchmark_weights: Optional[Dict[str, float]] = None
    cost_config: Optional[CostConfig] = None
    use_learning: bool = False


@dataclass
class MonthlyStep:
    """Bir aylik adimin sonuclari."""
    date: str
    regime: str
    confidence: float
    regime_scores: Dict[str, float]
    target_weights: Dict[str, float]
    previous_weights: Dict[str, float]
    portfolio_return: float
    benchmark_return: float
    alpha: float
    net_alpha: float
    rebalance_cost_pct: float
    turnover_pct: float
    portfolio_value: float
    benchmark_value: float
    data_quality_rows: int


@dataclass
class BacktestResult:
    """Backtest sonuclari."""
    config: BacktestConfig
    steps: List[MonthlyStep]
    total_return: float = 0.0
    benchmark_total_return: float = 0.0
    cagr: float = 0.0
    benchmark_cagr: float = 0.0
    volatility: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    benchmark_max_drawdown: float = 0.0
    win_rate: float = 0.0
    avg_alpha: float = 0.0
    avg_net_alpha: float = 0.0
    total_cost_pct: float = 0.0
    avg_turnover: float = 0.0
    months_count: int = 0


class BacktestEngine:
    """
    Walk-forward backtest engine.

    Her ayda:
    1. decision_date itibariyle rejim tespit et (look-ahead yok)
    2. Hedef agirliklar belirle
    3. BIR SONRAKI ayda gerceklesen getiriyi hesapla
    4. Maliyet dus

    Anti-lookahead garantileri:
    - compute_composite_score(as_of_date=decision_date) her zaman kullanilir
    - Getiri bir sonraki periyotta gerceklenir (karar ani != getiri ani)
    - LearningEngine statik prior kullanir (use_learning=False varsayilan)
    """

    def __init__(self, config: Optional[BacktestConfig] = None):
        self.config = config or BacktestConfig()
        self.regime_engine = RegimeEngineV2()
        if self.config.use_learning:
            self.learning_engine = LearningEngineV2(history_path="data/learning_history.json")
            logger.info("Ogrenilmis agirliklar kullanilacak (use_learning=True)")
        else:
            self.learning_engine = LearningEngineV2()
            logger.info("Statik prior agirliklar kullanilacak (use_learning=False)")
        self.cost_model = TransactionCostModel(self.config.cost_config or CostConfig())

    def _generate_rebalance_dates(self) -> List[pd.Timestamp]:
        """Aylik rebalance tarihlerini uret (her ayin son is gunu)."""
        start = pd.Timestamp(self.config.start_date)
        end = pd.Timestamp(self.config.end_date)
        try:
            dates = pd.date_range(start=start, end=end, freq="BME")
        except Exception:
            dates = pd.date_range(start=start, end=end, freq="BM")
        return list(dates)

    def _get_asset_returns(
        self,
        weights: Dict[str, float],
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
    ) -> float:
        """
        Verilen agirliklarla portfoyun bir sonraki aydaki getirisini hesapla.

        BES fon proxy'leri:
        - VEF (Hisse Fonu)      → BIST 100 getirisi
        - KTS (Kamu Borc)       → Sabit getiri (yillik %40/12)
        - ALT (Altin Fonu)      → Altin getirisi
        - KCH (Karma)           → BIST*0.5 + GOLD*0.3 + sabit*0.2
        - CASH (Para Piyasasi)  → Sabit getiri (dusuk)
        """
        try:
            market_data = self.regime_engine.fetch_live_data(as_of_date=end_date)

            if market_data.empty or len(market_data) < 20:
                logger.warning(f"Yetersiz market data: {start_date} -> {end_date}")
                return 0.0

            bist_ret = self._safe_return(market_data, "BIST", start_date, end_date)
            gold_ret = self._safe_return(market_data, "GOLD", start_date, end_date)

            monthly_fixed = 0.40 / 12
            low_fixed = 0.35 / 12

            asset_returns = {
                "VEF":  bist_ret,
                "KTS":  monthly_fixed,
                "ALT":  gold_ret,
                "KCH":  bist_ret * 0.5 + gold_ret * 0.3 + monthly_fixed * 0.2,
                "CASH": low_fixed,
            }

            return float(sum(
                weights.get(asset, 0) * asset_returns.get(asset, 0)
                for asset in set(list(weights.keys()) + list(asset_returns.keys()))
            ))

        except Exception as e:
            logger.error(f"Getiri hesaplama hatasi ({start_date}->{end_date}): {e}")
            return 0.0

    def _safe_return(
        self,
        data: pd.DataFrame,
        column: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> float:
        """Guvenli donemsel getiri hesaplama."""
        if column not in data.columns:
            return 0.0
        series = data[column].dropna()
        if series.empty:
            return 0.0

        start_vals = series[series.index >= start]
        end_vals = series[series.index <= end]

        if start_vals.empty or end_vals.empty:
            return 0.0

        start_price = float(start_vals.iloc[0])
        end_price = float(end_vals.iloc[-1])

        if start_price == 0:
            return 0.0
        return (end_price - start_price) / start_price

    def _weights_to_recommendations(
        self,
        current: Dict[str, float],
        target: Dict[str, float],
        total_value: float,
    ) -> List[Dict]:
        """Agirlik farkini recommendation formatina cevir."""
        recs = []
        for asset in set(list(current.keys()) + list(target.keys())):
            diff_tl = (target.get(asset, 0) - current.get(asset, 0)) * total_value
            action = "BUY" if diff_tl > 100 else "SELL" if diff_tl < -100 else "HOLD"
            recs.append({"asset": asset, "action": action, "diff_tl": diff_tl})
        return recs

    def run(self) -> BacktestResult:
        """Walk-forward backtest calistir."""
        logger.info(f"=== Backtest basladi: {self.config.start_date} -> {self.config.end_date} ===")

        rebalance_dates = self._generate_rebalance_dates()
        if len(rebalance_dates) < 2:
            logger.error("Yetersiz tarih araligi, en az 2 rebalance tarihi gerekli")
            return BacktestResult(config=self.config, steps=[])

        # Macro'yu bir kez cek, tum adimlar icin yeniden kullan (N API cagrisini onler)
        try:
            macro_data = MacroEngine().get_macro_snapshot()
            logger.info("Backtest macro snapshot alindi (tekrar kullanilacak)")
        except Exception as e:
            logger.warning(f"Macro engine hatasi, fallback: {e}")
            macro_data = {"tcmb_rate_change": 0}

        benchmark_weights = self.config.benchmark_weights or {
            "VEF": 0.25, "KTS": 0.25, "ALT": 0.25, "CASH": 0.25,
        }

        steps: List[MonthlyStep] = []
        portfolio_value = self.config.initial_capital
        benchmark_value = self.config.initial_capital
        current_weights: Dict[str, float] = {}

        for i in range(len(rebalance_dates) - 1):
            decision_date = rebalance_dates[i]
            next_date = rebalance_dates[i + 1]

            logger.debug(f"Adim {i+1}: karar={decision_date.date()}, getiri -> {next_date.date()}")

            # 1. Rejim tespiti — as_of_date ile look-ahead korumasi
            try:
                regime_result = self.regime_engine.compute_composite_score(
                    macro_data=macro_data,
                    as_of_date=decision_date,
                )
            except Exception as e:
                logger.warning(f"Rejim tespiti basarisiz ({decision_date.date()}): {e}, STABLE varsayiliyor")
                regime_result = {
                    "detected": "STABLE",
                    "confidence": 0.0,
                    "scores": {},
                    "data_quality": {"rows_count": 0},
                }

            detected = regime_result["detected"]
            confidence = regime_result.get("confidence", 0)

            # 2. Hedef agirliklar (statik prior)
            target_weights = self.learning_engine.get_optimized_weights(detected)

            # 3. Maliyet hesabi
            if current_weights:
                recs = self._weights_to_recommendations(current_weights, target_weights, portfolio_value)
                cost_result = self.cost_model.calculate_rebalance_cost(recs, portfolio_value)
                rebalance_cost_pct = cost_result["total_cost_pct"]
                turnover_pct = cost_result["turnover_pct"]
            else:
                rebalance_cost_pct = 0.0
                turnover_pct = 0.0

            previous_weights = dict(current_weights)
            current_weights = dict(target_weights)

            # 4. Bir sonraki ayin getirisi (gerceklesen)
            portfolio_return = self._get_asset_returns(target_weights, decision_date, next_date)
            benchmark_return = self._get_asset_returns(benchmark_weights, decision_date, next_date)

            # 5. Net alpha ve portfoy degeri
            gross_alpha = portfolio_return - benchmark_return
            net_alpha = gross_alpha - rebalance_cost_pct

            portfolio_value *= (1 + portfolio_return - rebalance_cost_pct)
            benchmark_value *= (1 + benchmark_return)

            steps.append(MonthlyStep(
                date=str(decision_date.date()),
                regime=detected,
                confidence=confidence,
                regime_scores=regime_result.get("scores", {}),
                target_weights=target_weights,
                previous_weights=previous_weights,
                portfolio_return=round(portfolio_return, 6),
                benchmark_return=round(benchmark_return, 6),
                alpha=round(gross_alpha, 6),
                net_alpha=round(net_alpha, 6),
                rebalance_cost_pct=round(rebalance_cost_pct, 6),
                turnover_pct=round(turnover_pct, 4),
                portfolio_value=round(portfolio_value, 2),
                benchmark_value=round(benchmark_value, 2),
                data_quality_rows=regime_result.get("data_quality", {}).get("rows_count", 0),
            ))

            logger.info(
                f"[{decision_date.date()}] {detected} (guven:{confidence:.0%}) | "
                f"AI:{portfolio_return:+.2%} vs Bench:{benchmark_return:+.2%} | "
                f"a:{gross_alpha:+.2%} net:{net_alpha:+.2%} | "
                f"Portfoy:{portfolio_value:,.0f} TL"
            )

        result = self._calculate_metrics(steps)

        logger.info(
            f"=== Backtest tamamlandi: {result.months_count} ay, "
            f"AI:{result.total_return:+.2%} vs Bench:{result.benchmark_total_return:+.2%}, "
            f"Sharpe:{result.sharpe_ratio:.2f}, MaxDD:{result.max_drawdown:.2%} ==="
        )
        return result

    def _calculate_metrics(self, steps: List[MonthlyStep]) -> BacktestResult:
        result = BacktestResult(config=self.config, steps=steps)
        if not steps:
            return result

        result.months_count = len(steps)
        monthly_returns = [s.portfolio_return - s.rebalance_cost_pct for s in steps]
        bench_returns = [s.benchmark_return for s in steps]
        alphas = [s.alpha for s in steps]
        net_alphas = [s.net_alpha for s in steps]

        result.total_return = steps[-1].portfolio_value / self.config.initial_capital - 1
        result.benchmark_total_return = steps[-1].benchmark_value / self.config.initial_capital - 1

        years = result.months_count / 12
        if years > 0:
            result.cagr = (1 + result.total_return) ** (1 / years) - 1
            result.benchmark_cagr = (1 + result.benchmark_total_return) ** (1 / years) - 1

        if len(monthly_returns) > 1:
            result.volatility = float(np.std(monthly_returns, ddof=1) * np.sqrt(12))

        # Sharpe (risk-free = %36 yillik — Turkiye faiz ortami)
        if result.volatility > 0:
            result.sharpe_ratio = (result.cagr - 0.36) / result.volatility

        # Max Drawdown
        equity = [self.config.initial_capital]
        for r in monthly_returns:
            equity.append(equity[-1] * (1 + r))
        peak = equity[0]
        max_dd = 0.0
        for val in equity:
            peak = max(peak, val)
            max_dd = min(max_dd, (val - peak) / peak)
        result.max_drawdown = max_dd

        bench_curve = [self.config.initial_capital]
        for r in bench_returns:
            bench_curve.append(bench_curve[-1] * (1 + r))
        peak = bench_curve[0]
        bench_dd = 0.0
        for val in bench_curve:
            peak = max(peak, val)
            bench_dd = min(bench_dd, (val - peak) / peak)
        result.benchmark_max_drawdown = bench_dd

        wins = sum(1 for a in alphas if a > 0)
        result.win_rate = wins / len(alphas) if alphas else 0
        result.avg_alpha = float(np.mean(alphas)) if alphas else 0
        result.avg_net_alpha = float(np.mean(net_alphas)) if net_alphas else 0
        result.total_cost_pct = sum(s.rebalance_cost_pct for s in steps)
        result.avg_turnover = float(np.mean([s.turnover_pct for s in steps]))

        return result

    def export_to_learning_history(
        self,
        result: BacktestResult,
        output_path: str = "data/learning_history.json",
    ) -> int:
        """
        Backtest sonuclarini learning_history.json formatina cevir ve kaydet.
        Returns: yazilan yeni gozlem sayisi.
        """
        import json
        from pathlib import Path

        observations = []
        for step in result.steps:
            observations.append({
                "date": step.date,
                "regime": step.regime,
                "weights_used": step.target_weights,
                "monthly_return": step.portfolio_return,
                "alpha_vs_benchmark": step.net_alpha,
            })

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        existing = []
        if path.exists():
            try:
                with open(path, encoding="utf-8") as f:
                    existing = json.load(f)
            except (json.JSONDecodeError, OSError):
                existing = []

        existing_dates = {obs["date"] for obs in existing}
        new_obs = [obs for obs in observations if obs["date"] not in existing_dates]

        combined = existing + new_obs
        combined.sort(key=lambda x: x["date"])

        with open(path, "w", encoding="utf-8") as f:
            json.dump(combined, f, ensure_ascii=False, indent=2, default=str)

        logger.info(
            f"Learning history guncellendi: {len(new_obs)} yeni gozlem eklendi, "
            f"toplam {len(combined)} gozlem ({path})"
        )
        return len(new_obs)

    def to_dataframe(self, result: BacktestResult) -> pd.DataFrame:
        """Backtest sonuclarini DataFrame'e cevir."""
        if not result.steps:
            return pd.DataFrame()
        rows = [
            {
                "date": s.date,
                "regime": s.regime,
                "confidence": s.confidence,
                "portfolio_return": s.portfolio_return,
                "benchmark_return": s.benchmark_return,
                "alpha": s.alpha,
                "net_alpha": s.net_alpha,
                "cost_pct": s.rebalance_cost_pct,
                "turnover_pct": s.turnover_pct,
                "portfolio_value": s.portfolio_value,
                "benchmark_value": s.benchmark_value,
            }
            for s in result.steps
        ]
        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"])
        return df.set_index("date")

    def print_summary(self, result: BacktestResult) -> str:
        """Insan-okunabilir backtest ozeti."""
        if not result.steps:
            return "Backtest sonuc yok."

        lines = []
        lines.append("=" * 64)
        lines.append("BACKTEST SONUCLARI")
        lines.append("=" * 64)
        lines.append(f"Donem          : {self.config.start_date} -> {self.config.end_date} ({result.months_count} ay)")
        lines.append(f"Baslangic      : {self.config.initial_capital:,.0f} TL")
        lines.append("")
        lines.append(f"{'':15s} {'AI Portfoy':>12s} {'Benchmark':>12s}")
        lines.append(f"{'-'*15} {'-'*12} {'-'*12}")
        lines.append(f"{'Son Deger':15s} {result.steps[-1].portfolio_value:>12,.0f} {result.steps[-1].benchmark_value:>12,.0f}")
        lines.append(f"{'Toplam Getiri':15s} {result.total_return:>12.2%} {result.benchmark_total_return:>12.2%}")
        lines.append(f"{'CAGR':15s} {result.cagr:>12.2%} {result.benchmark_cagr:>12.2%}")
        lines.append(f"{'Volatilite':15s} {result.volatility:>12.2%} {'—':>12s}")
        lines.append(f"{'Sharpe':15s} {result.sharpe_ratio:>12.2f} {'—':>12s}")
        lines.append(f"{'Max Drawdown':15s} {result.max_drawdown:>12.2%} {result.benchmark_max_drawdown:>12.2%}")
        lines.append("")
        lines.append(f"Win Rate       : {result.win_rate:.1%} ({sum(1 for s in result.steps if s.alpha > 0)}/{result.months_count} ay)")
        lines.append(f"Ort. Alpha     : {result.avg_alpha:+.2%} (brut), {result.avg_net_alpha:+.2%} (net)")
        lines.append(f"Toplam Maliyet : {result.total_cost_pct:.3%}")
        lines.append(f"Ort. Turnover  : {result.avg_turnover:.1%}")
        lines.append("")

        regime_counts = Counter(s.regime for s in result.steps)
        lines.append("Rejim Dagilimi:")
        for regime, count in sorted(regime_counts.items()):
            pct = count / result.months_count * 100
            avg_a = np.mean([s.alpha for s in result.steps if s.regime == regime])
            lines.append(f"  {regime:12s}: {count:2d} ay ({pct:4.1f}%) | ort. a={avg_a:+.2%}")

        lines.append("")
        lines.append("Equity Curve (aylik):")
        values = [s.portfolio_value for s in result.steps]
        min_val = min(values)
        max_val = max(values)
        val_range = max_val - min_val if max_val != min_val else 1
        width = 40
        for step in result.steps:
            bar_len = int((step.portfolio_value - min_val) / val_range * width)
            bar = "#" * bar_len
            r_char = step.regime[0]
            lines.append(f"  {step.date[:7]} [{r_char}] {bar} {step.portfolio_value:,.0f}")

        lines.append("=" * 64)
        return "\n".join(lines)
