import logging
import math
import time
import pandas as pd
import numpy as np
import yfinance as yf
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

# BIST kritik — eksikse hesaplama anlamsiz. Digerleri yumusak.
REQUIRED_SYMBOLS = ("BIST",)


class RegimeEngineV2:
    def __init__(self):
        # Analiz edilecek semboller: BIST 100, USD/TRY, Ons Altin
        self.symbols = {
            'BIST': 'XU100.IS',
            'USDTRY': 'USDTRY=X',
            'GOLD': 'GC=F'
        }

    # ------------------------------------------------------------------
    # Yardimci: guvenli erisim
    # ------------------------------------------------------------------

    def _safe_get_last(self, series: pd.Series, metric_name: str) -> float:
        """NaN-safe son deger alma, log'lu."""
        if series.empty or series.isna().all():
            logger.warning(f"{metric_name} icin yeterli veri yok, 0 donuluyor")
            return 0.0
        last = series.dropna().iloc[-1]
        return float(last)

    def _compare_with_index(self, index: pd.DatetimeIndex, ts: pd.Timestamp) -> pd.Series:
        """Timezone farkindan bagimsiz index < ts karsilastirmasi."""
        if index.tz is not None and ts.tzinfo is None:
            ts = ts.tz_localize(index.tz)
        elif index.tz is None and ts.tzinfo is not None:
            ts = ts.tz_localize(None)
        return index < ts

    # ------------------------------------------------------------------
    # Normalizasyon helpers — tum ciktilar [0, 1]
    # TODO: bu esikler simdilik sezgisel; backtest kalibrasyonu ayri prompt
    # ------------------------------------------------------------------

    def _normalize_drawdown(self, dd: float) -> float:
        """%0 dusus = 0, %30+ dusus = 1."""
        return float(np.clip(abs(dd) / 0.30, 0, 1))

    def _normalize_volatility(self, vol: float) -> float:
        """Yillik vol %20 normal, %60+ kriz seviyesi."""
        return float(np.clip((vol - 0.20) / 0.40, 0, 1))

    def _normalize_momentum(self, mom: float, threshold: float = 0.15) -> float:
        """Mutlak momentum degerinin threshold'a orani."""
        return float(np.clip(abs(mom) / threshold, 0, 1))

    # ------------------------------------------------------------------
    # Soft decision
    # ------------------------------------------------------------------

    def _scores_to_probabilities(
        self, scores: Dict[str, float], temperature: float = 0.5
    ) -> Dict[str, float]:
        """Skorlari olasiliga cevir. Dusuk temperature = daha keskin karar."""
        exp_scores = {k: math.exp(v / temperature) for k, v in scores.items()}
        total = sum(exp_scores.values())
        return {k: v / total for k, v in exp_scores.items()}

    # ------------------------------------------------------------------
    # Veri cekme
    # ------------------------------------------------------------------

    def fetch_live_data(
        self,
        as_of_date: Optional[pd.Timestamp] = None,
        lookback_days: int = 365,
    ) -> pd.DataFrame:
        """
        as_of_date: Verinin kesilecegi tarih (None = bugun)
        lookback_days: as_of_date'ten geriye kac gun veri cekilecek
        """
        data = {}
        # yfinance Yahoo'yu scrape eder, rate-limit yer; gecici hatalarda 3 deneme
        max_attempts = 3
        backoff_seconds = (0.5, 1.0, 2.0)

        for name, ticker in self.symbols.items():
            last_error: Optional[Exception] = None
            for attempt in range(max_attempts):
                try:
                    if as_of_date is None:
                        df = yf.download(ticker, period="1y", interval="1d", progress=False)
                    else:
                        start = as_of_date - pd.Timedelta(days=lookback_days)
                        end = as_of_date  # exclusive: yfinance end dahil degil
                        df = yf.download(ticker, start=start, end=end, interval="1d", progress=False)

                    if df.empty:
                        raise ValueError(f"{ticker} icin bos veri dondu")

                    if isinstance(df.columns, pd.MultiIndex):
                        close = df['Close'][ticker]
                    else:
                        close = df['Close']

                    if close.isna().all():
                        raise ValueError(f"{ticker} icin tum degerler NaN")

                    logger.info(f"{name} ({ticker}): {len(close)} satir veri cekildi (deneme {attempt + 1})")
                    data[name] = close
                    last_error = None
                    break

                except Exception as e:
                    last_error = e
                    if attempt < max_attempts - 1:
                        logger.warning(
                            f"{name} ({ticker}) deneme {attempt + 1}/{max_attempts} basarisiz: {e}, "
                            f"{backoff_seconds[attempt]}s sonra tekrar"
                        )
                        time.sleep(backoff_seconds[attempt])
                    else:
                        logger.error(
                            f"{name} ({ticker}) verisi {max_attempts} denemeden sonra cekilemedi: {e}"
                        )

        # Kritik sembol(ler) eksikse fallback — caller anlamli hata mesaji uretebilsin
        missing_required = [s for s in REQUIRED_SYMBOLS if s not in data]
        if missing_required:
            logger.error(f"Kritik sembol(ler) cekilemedi: {missing_required}")
            return pd.DataFrame(columns=["BIST", "USDTRY", "GOLD"])

        if not data:
            logger.error("Hiçbir piyasa verisi çekilemedi")
            return pd.DataFrame(columns=["BIST", "USDTRY", "GOLD"])

        result = pd.DataFrame(data).ffill()

        # Cift guven: as_of_date'ten sonraki hicbir veri sizmasin
        if as_of_date is not None:
            mask = self._compare_with_index(result.index, as_of_date)
            result = result[mask]

        if len(result) < 60:
            raise ValueError(
                f"Yetersiz veri: {len(result)} satir mevcut, minimum 60 islem gunu gerekli"
            )

        return result

    # ------------------------------------------------------------------
    # Ana hesaplama
    # ------------------------------------------------------------------

    def compute_composite_score(
        self,
        macro_data: Optional[Dict[str, Any]] = None,
        as_of_date: Optional[pd.Timestamp] = None,
    ) -> Dict[str, Any]:
        """Canli veriyi ceker ve rejim skorunu hesaplar."""
        if macro_data is None:
            try:
                from src.macro_engine import MacroEngine
                macro_data = MacroEngine().get_macro_snapshot()
                logger.info(f"Macro data otomatik cekildi: {macro_data.get('data_quality')}")
            except Exception as e:
                logger.warning(f"Macro engine hatasi, fallback degerler: {e}")
                macro_data = {"tcmb_rate_change": 0}

        market_data = self.fetch_live_data(as_of_date=as_of_date)

        missing_required = [s for s in REQUIRED_SYMBOLS if s not in market_data.columns]
        insufficient = market_data.empty or len(market_data) < 5

        if insufficient or missing_required:
            if missing_required:
                error_msg = (
                    f"Kritik piyasa verisi alınamadı ({', '.join(missing_required)}). "
                    "Yahoo Finance kaynağı geçici olarak ulaşılamıyor olabilir."
                )
            else:
                error_msg = "Yetersiz piyasa verisi, en az 5 işlem günü gerekli."

            logger.warning(f"{error_msg} — varsayılan sonuç dönülüyor")
            return {
                "detected": "STABLE",
                "confidence": 0.0,
                "scores": {"CRISIS": 0, "RISK_ON": 0, "RATE_HIKE": 0, "STABLE": 0},
                "probabilities": {"CRISIS": 0.25, "RISK_ON": 0.25, "RATE_HIKE": 0.25, "STABLE": 0.25},
                "metrics": {"dd": 0, "vol": 0, "usd_mom": 0, "gold_mom": 0, "bist_60d_return": 0},
                "data_quality": {
                    "rows_count": len(market_data),
                    "missing_pct": 1.0,
                    "as_of": "N/A",
                    "error": error_msg,
                },
                "macro": macro_data if macro_data else {},
            }

        # --- Ham metrikler ---

        # Drawdown: min_periods=20 -> ilk 20 gun NaN, yaniltici "max=ilk_gun" olmaz
        bist_rolling_max = market_data['BIST'].rolling(window=252, min_periods=20).max()
        bist_dd = self._safe_get_last(
            (market_data['BIST'] / bist_rolling_max) - 1,
            "BIST Drawdown"
        )

        # Volatilite: en az 20 gecerli gun sart
        vol_series = market_data['BIST'].pct_change().tail(20).dropna()
        if len(vol_series) < 20:
            logger.warning(
                f"BIST Volatilite icin 20 gunluk gecerli veri yok ({len(vol_series)} gun), 0 donuluyor"
            )
            vol = 0.0
        else:
            vol = float(vol_series.std() * np.sqrt(252))
            if np.isnan(vol):
                logger.warning("BIST Volatilite hesaplanamadi, 0 donuluyor")
                vol = 0.0

        usd_mom = self._safe_get_last(market_data['USDTRY'].pct_change(20), "USD/TRY Momentum")
        gold_mom = self._safe_get_last(market_data['GOLD'].pct_change(20), "Gold Momentum")
        bist_60d_return = self._safe_get_last(market_data['BIST'].pct_change(60), "BIST 60-gun Momentum")

        tcmb_trend = float(macro_data.get('tcmb_rate_change', 0))

        # --- Normalizasyon ---
        dd_score = self._normalize_drawdown(bist_dd)
        vol_score = self._normalize_volatility(vol)
        usd_mom_score = self._normalize_momentum(usd_mom, threshold=0.15)
        gold_mom_score = self._normalize_momentum(gold_mom, threshold=0.15)  # noqa: F841

        # 5pp artis (0.05) tam sinyal; clip ile [0,1]'de tut
        rate_hike_signal = float(np.clip(max(0.0, tcmb_trend / 0.05), 0, 1))

        # --- Skorlama: her rejim [0, 1] araliginda, kiyaslanabilir ---
        # TODO: agirliklar backtest kalibrasyonuyla guncellenecek
        scores_raw = {
            "CRISIS":    0.5 * dd_score
                         + 0.3 * vol_score
                         + 0.2 * (usd_mom_score if usd_mom > 0 else 0),
            "RATE_HIKE": 0.7 * rate_hike_signal
                         + 0.3 * (usd_mom_score if usd_mom > 0 else 0),
            "RISK_ON":   0.6 * float(np.clip(bist_60d_return / 0.20, 0, 1))
                         + 0.4 * (1 - vol_score),
            "STABLE":    1.0 - max(dd_score, vol_score, usd_mom_score),
        }

        scores = {k: float(np.clip(v, 0, 1)) for k, v in scores_raw.items()}

        # --- Karar ---
        probs = self._scores_to_probabilities(scores)
        detected = max(scores, key=scores.get)
        confidence = scores[detected]

        data_quality = {
            "rows_count": len(market_data),
            "missing_pct": float(market_data.isna().sum().sum() / market_data.size),
            "as_of": str(market_data.index[-1].date()),
        }

        anomalies = self.detect_anomalies(market_data)

        return {
            "detected": detected,
            "confidence": confidence,
            "scores": scores,
            "probabilities": probs,
            "metrics": {
                "dd": float(bist_dd),
                "vol": float(vol),
                "usd_mom": float(usd_mom),
                "gold_mom": float(gold_mom),
                "bist_60d_return": float(bist_60d_return),
            },
            "data_quality": data_quality,
            "macro": macro_data,
            "anomalies": anomalies,
        }

    def detect_anomalies(self, market_data: "pd.DataFrame") -> list:
        """
        Z-score tabanli anomali tespiti: BIST, USD/TRY, Altin icin.
        Ayrica BIST'te ardisik yukselis/dusus kontrolu yapar.

        Returns: list of dicts with keys type, asset, severity, message
        """
        anomalies = []
        if market_data is None or len(market_data) < 10:
            return anomalies

        asset_tr = {"BIST": "BIST 100", "USDTRY": "USD/TRY", "GOLD": "Altın"}

        for col in ["BIST", "USDTRY", "GOLD"]:
            if col not in market_data.columns:
                continue
            returns = market_data[col].pct_change().dropna()
            if len(returns) < 10:
                continue

            window = min(60, len(returns) - 1)
            hist   = returns.iloc[-window - 1:-1]
            last_r = float(returns.iloc[-1])
            mu     = float(hist.mean())
            sigma  = float(hist.std())

            if sigma < 1e-8:
                continue

            z = (last_r - mu) / sigma
            name = asset_tr.get(col, col)
            normal_range = sigma * 2 * 100  # ±% cinsinden

            if abs(z) > 3:
                anomalies.append({
                    "type":     "spike",
                    "asset":    col,
                    "severity": "high",
                    "z_score":  round(z, 2),
                    "message":  (
                        f"🚨 {name}'te olağandışı hareket: "
                        f"Günlük %{last_r*100:+.2f} değişim "
                        f"(normal aralık: ±%{normal_range:.2f})"
                    ),
                })
            elif abs(z) > 2:
                anomalies.append({
                    "type":     "spike",
                    "asset":    col,
                    "severity": "medium",
                    "z_score":  round(z, 2),
                    "message":  (
                        f"⚡ {name}'te sert hareket: "
                        f"%{last_r*100:+.2f} (normalin {abs(z):.1f}× üzerinde)"
                    ),
                })

        # BIST ardisik hareket kontrolu (5+ gun)
        if "BIST" in market_data.columns:
            bist_ret = market_data["BIST"].pct_change().dropna()
            if len(bist_ret) >= 5:
                last5 = bist_ret.iloc[-5:]
                if (last5 < 0).all():
                    anomalies.append({
                        "type":     "streak",
                        "asset":    "BIST",
                        "severity": "medium",
                        "message":  "📉 BIST 5 gündür art arda düşüyor",
                    })
                elif (last5 > 0).all():
                    anomalies.append({
                        "type":     "streak",
                        "asset":    "BIST",
                        "severity": "low",
                        "message":  "📈 BIST 5 gündür art arda yükseliyor",
                    })

        # Volatilite ani artisi kontrolu
        if "BIST" in market_data.columns:
            bist_ret = market_data["BIST"].pct_change().dropna()
            if len(bist_ret) >= 10:
                vol_5d  = float(bist_ret.iloc[-5:].std())
                vol_60d = float(bist_ret.iloc[-60:].std()) if len(bist_ret) >= 60 else float(bist_ret.std())
                if vol_60d > 1e-8 and vol_5d > vol_60d * 2:
                    anomalies.append({
                        "type":     "vol_spike",
                        "asset":    "BIST",
                        "severity": "medium",
                        "message":  (
                            f"⚡ Volatilite ani artışı: Son 5 gün oynaklığı "
                            f"(%{vol_5d*100:.2f}) uzun dönem ortalamasının "
                            f"{vol_5d/vol_60d:.1f}× katı"
                        ),
                    })

        return anomalies

    def detect_regime_change_risk(self, result: Dict, history_dir: str = "data/history") -> Dict:
        """
        Mevcut rejim skorlarından rejim degisikligi riskini hesapla.

        Returns:
            risk_level:     "low" | "medium" | "high"
            risk_score:     0-100 (100 = en yuksek risk)
            potential_next: ikinci en yuksek skorlu rejim
            message:        kullaniciya gosterilecek metin
            recent_regimes: son 3 ayin rejim listesi (varsa)
        """
        scores = result.get("scores", {})
        detected = result.get("detected", "STABLE")

        if len(scores) < 2:
            return {"risk_level": "low", "risk_score": 0, "potential_next": None, "message": "", "recent_regimes": []}

        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_score     = sorted_scores[0][1]
        second_regime = sorted_scores[1][0]
        second_score  = sorted_scores[1][1]
        gap = top_score - second_score

        risk_score = int(max(0, min(100, 100 - gap * 200)))

        if gap < 0.10:
            risk_level = "high"
        elif gap < 0.25:
            risk_level = "medium"
        else:
            risk_level = "low"

        # Son 3 ayin rejim gecmisini oku
        recent_regimes: list = []
        try:
            import json as _j
            from pathlib import Path as _P
            snap_files = sorted(_P(history_dir).glob("*.json"))[-3:]
            for sf in snap_files:
                d = _j.loads(sf.read_text(encoding="utf-8"))
                reg = d.get("regime", {})
                det = reg.get("detected") if isinstance(reg, dict) else None
                date_str = d.get("run_date", sf.stem)[:7]
                if det:
                    recent_regimes.append(f"{date_str}:{det}")
        except Exception:
            pass

        regime_tr = {"CRISIS": "Kriz", "RISK_ON": "Yükseliş", "RATE_HIKE": "Faiz Artışı", "STABLE": "Sakin"}
        next_tr   = regime_tr.get(second_regime, second_regime)

        if risk_level == "high":
            message = (
                f"⚠️ Rejim Değişikliği Riski Yüksek — **{next_tr}** rejimine geçiş olasılığı artıyor. "
                "Portföyünü gözden geçir."
            )
        elif risk_level == "medium":
            message = f"🔔 Rejim geçiş sinyalleri var — **{next_tr}** olasılığı yükseliyor."
        else:
            message = ""

        return {
            "risk_level":     risk_level,
            "risk_score":     risk_score,
            "potential_next": second_regime,
            "message":        message,
            "recent_regimes": recent_regimes,
        }


if __name__ == "__main__":
    from src.logging_config import configure_logging
    configure_logging()
    engine = RegimeEngineV2()

    # Test 1: Bugun icin
    logger.info("=== Bugun ===")
    r1 = engine.compute_composite_score()
    logger.info(f"Rejim: {r1['detected']}, As of: {r1['data_quality']['as_of']}")
    logger.info(f"Guven: {r1['confidence']:.2%}")
    logger.info(f"Olasiliklar: { {k: f'{v:.3f}' for k, v in r1['probabilities'].items()} }")

    assert 0 <= r1['confidence'] <= 1, "Confidence 0-1 arasinda olmali"
    assert abs(sum(r1['probabilities'].values()) - 1.0) < 0.001, "Olasiliklar 1'e toplanmali"
    logger.info("Confidence ve probability assertionlari gecti")

    # Test 2: Gecmis bir tarih icin (look-ahead bias testi)
    past_date = pd.Timestamp("2024-06-15")
    logger.info(f"=== {past_date.date()} ===")
    r2 = engine.compute_composite_score(as_of_date=past_date)
    logger.info(f"Rejim: {r2['detected']}, As of: {r2['data_quality']['as_of']}")
    logger.info(f"Guven: {r2['confidence']:.2%}")
    logger.info(f"Olasiliklar: { {k: f'{v:.3f}' for k, v in r2['probabilities'].items()} }")

    assert pd.Timestamp(r2['data_quality']['as_of']) < past_date, \
        "LOOK-AHEAD BIAS! as_of tarihi past_date'ten once olmali"
    logger.info("Look-ahead bias testi gecti")
