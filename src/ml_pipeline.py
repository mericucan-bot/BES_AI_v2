"""
ML Pipeline — TEFAS BES fon getiri tahmini.

VERİ NOTU: TEFAS API gunluk NAV degil, ay-sonu aggregate getiriler dondurur
(return_1m, return_3m, return_1y...). Bu pipeline iki mod destekler:
- snapshot_mode: Aylik TEFAS snapshot'larindan feature uretir (gercek TEFAS verisi)
- nav_mode: Gunluk NAV serisinden FeatureEngineer kullanir (mock/test verisi)
Mod, veri frekansina gore otomatik secilir.
"""
import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.data_collector import TEFASCollector, POPULAR_BES_FUNDS
from src.feature_engineer import FeatureEngineer
from src.ml_model import BESPredictor, ModelConfig, ModelResult
from src.regime_engine import RegimeEngineV2
from src.macro_engine import MacroEngine

logger = logging.getLogger(__name__)

_TARGET_3M = "fwd_return_3m"
_TARGET_12M = "fwd_return_12m"


class MLPipeline:
    """
    Uctan uca ML egitim pipeline'i.

    Adimlar:
    1. collect_fund_data  — TEFAS'tan snapshot serisi cek
    2. collect_market_data — yfinance'tan BIST/USD/Altin
    3. build_features     — Feature engineering (aylik veya gunluk moda gore)
    4. train_models       — Model egitimi ve karsilastirma
    5. generate_predictions — Tum fonlar icin tahmin
    6. save_results       — Sonuclari kaydet
    """

    def __init__(
        self,
        output_dir: str = "data/ml",
        cache_dir: str = "data/tefas_cache",
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.collector = TEFASCollector(cache_dir=cache_dir)
        self.feature_eng = FeatureEngineer()
        self.regime_engine = RegimeEngineV2()
        self.macro_engine = MacroEngine()

    # ------------------------------------------------------------------
    # 1. Veri toplama
    # ------------------------------------------------------------------

    def collect_fund_data(
        self,
        fund_codes: Optional[List[str]] = None,
        lookback_days: int = 730,
        max_funds: Optional[int] = None,
    ) -> Dict[str, pd.Series]:
        """
        TEFAS'tan BES fon snapshot serisini cek ve aylık NAV olarak dondutur.

        TEFAS API gunluk fiyat degil, ay-sonu aggregate getiriler (return_1m vb.)
        dondurdugundan, return_1m bileşik buyume ile sentetik NAV olusturulur.
        Ortalama 24 aylık veri noktası (2 yillik lookback).

        max_funds: None ise tum fund_codes islenir, sayi verilirse ilk N fon alinir.
        Returns: {fund_code: Series(index=ay_sonu_tarihi, values=sentetik_nav)}
        """
        if fund_codes is None:
            fund_codes = list(POPULAR_BES_FUNDS.keys())

        if max_funds and len(fund_codes) > max_funds:
            fund_codes = fund_codes[:max_funds]
            logger.info(f"Fon sayisi {max_funds} ile sinirlandirildi")

        end = datetime.now()
        start = end - timedelta(days=lookback_days)
        start_str = start.strftime("%Y-%m-%d")
        end_str = end.strftime("%Y-%m-%d")

        logger.info(
            f"TEFAS veri toplama: {len(fund_codes)} fon, "
            f"{start_str} -> {end_str} (aylık snapshot modu)"
        )

        # Tüm fonlar icin tek seferde aylık seri cek
        all_data = self.collector.fetch_monthly_series(
            start=start_str,
            end=end_str,
        )

        if all_data.empty:
            logger.error("TEFAS'tan hicbir veri alinamadi")
            return {}

        fund_navs: Dict[str, pd.Series] = {}
        failed: List[str] = []

        for code in fund_codes:
            fund_df = (
                all_data[all_data["fund_code"] == code.upper()]
                .sort_values("date")
                .copy()
            )

            if len(fund_df) < 6:
                logger.warning(f"  {code}: yetersiz snapshot ({len(fund_df)} ay), atlandi")
                failed.append(code)
                continue

            if "return_1m" not in fund_df.columns or fund_df["return_1m"].isna().all():
                logger.warning(f"  {code}: return_1m verisi yok, atlandi")
                failed.append(code)
                continue

            returns_pct = fund_df.set_index("date")["return_1m"].dropna()
            if len(returns_pct) < 6:
                failed.append(code)
                continue

            # Bileşik NAV: 100 * prod(1 + r/100)
            nav = (1 + returns_pct / 100).cumprod() * 100
            fund_navs[code] = nav
            logger.debug(f"  {code}: {len(nav)} aylık nokta")

        logger.info(
            f"Fon verisi tamamlandi: {len(fund_navs)} basarili, {len(failed)} basarisiz"
        )
        if failed:
            logger.warning(f"Basarisiz fonlar: {failed}")

        return fund_navs

    def collect_market_data(self, lookback_days: int = 760) -> pd.DataFrame:
        """yfinance'tan gunluk piyasa verisi cek (varsayilan 760 gun ~ 2.5 yil)."""
        logger.info(f"Piyasa verisi cekiliyor (yfinance, {lookback_days} gun)...")
        market = self.regime_engine.fetch_live_data(
            as_of_date=pd.Timestamp.now(),
            lookback_days=lookback_days,
        )
        logger.info(f"Piyasa verisi: {len(market)} gun, sutunlar: {market.columns.tolist()}")
        return market

    # ------------------------------------------------------------------
    # 2. Feature engineering
    # ------------------------------------------------------------------

    def _detect_frequency(self, fund_navs: Dict[str, pd.Series]) -> str:
        """Veri frekansini otomatik tespit et (aylik mi gunluk mu)."""
        if not fund_navs:
            return "weekly"
        first_nav = next(iter(fund_navs.values()))
        if len(first_nav) < 2:
            return "weekly"
        gaps = pd.Series(first_nav.index).diff().dropna()
        median_gap_days = gaps.median().days
        return "monthly" if median_gap_days > 20 else "weekly"

    def build_features(
        self,
        fund_navs: Dict[str, pd.Series],
        market_data: pd.DataFrame,
        sample_frequency: str = "auto",
    ) -> pd.DataFrame:
        """
        Fon ve piyasa verisinden ML feature matrisi uret.

        Mod otomatik: aylik nav -> snapshot feature builder,
                       gunluk nav -> FeatureEngineer (test/mock).
        """
        if sample_frequency == "auto":
            sample_frequency = self._detect_frequency(fund_navs)

        logger.info(f"Feature engineering modu: {sample_frequency} ({len(fund_navs)} fon)")

        if sample_frequency == "monthly":
            return self._build_snapshot_features(fund_navs, market_data)

        # Gunluk mod — FeatureEngineer kullan
        macro = self.macro_engine.get_macro_snapshot()
        dataset = self.feature_eng.build_dataset(
            fund_navs=fund_navs,
            market_data=market_data,
            macro_snapshot=macro,
            sample_frequency=sample_frequency,
            min_nav_length=126,
        )
        logger.info(f"Dataset (FeatureEngineer): {dataset.shape}")
        return dataset

    def _build_snapshot_features(
        self,
        fund_navs: Dict[str, pd.Series],
        market_data: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Aylik NAV (return_1m bileşigi) serisinden feature matrisi uret.

        FeatureEngineer gunluk pencere boyutlari yerine aylik pencereler kullanir:
        - vol_3m = son 3 ayin return_1m std'si
        - momentum = kisa vs uzun vade karsilastirmasi
        - sharpe = aylik Sharpe
        - Target = bir sonraki ayin return_1m (normalize)
        """
        macro = self.macro_engine.get_macro_snapshot()
        cpi = float(macro.get("cpi_yoy") or 0.30) if macro else 0.30
        annual_rate = float(macro.get("current_policy_rate") or 42.5) / 100 if macro else 0.425
        monthly_rf = (1 + annual_rate) ** (1 / 12) - 1

        # Aylik piyasa verisi (BME = business month end)
        mkt_monthly = market_data.resample("BME").last()
        mkt_bist = mkt_monthly["BIST"].pct_change() * 100 if "BIST" in mkt_monthly else None
        mkt_usd = mkt_monthly["USDTRY"].pct_change() * 100 if "USDTRY" in mkt_monthly else None
        mkt_gold = mkt_monthly["GOLD"].pct_change() * 100 if "GOLD" in mkt_monthly else None

        all_rows = []

        for code, nav in fund_navs.items():
            if nav is None or len(nav) < 6:
                continue

            df = pd.DataFrame(index=nav.index)

            # Aylik getiri yüzdesi
            ret = nav.pct_change() * 100
            df["return_1m"] = ret
            df["return_3m"] = nav.pct_change(3) * 100
            df["return_6m"] = nav.pct_change(6) * 100
            df["return_12m"] = nav.pct_change(12) * 100

            # Rolling volatilite (aylik)
            df["vol_3m"] = ret.rolling(3, min_periods=2).std()
            df["vol_6m"] = ret.rolling(6, min_periods=3).std()

            # Momentum
            df["momentum_1m_3m"] = ret - ret.rolling(3, min_periods=2).mean()
            df["momentum_3m_6m"] = (
                ret.rolling(3, min_periods=2).mean()
                - ret.rolling(6, min_periods=3).mean()
            )

            # Z-score (kendi gecmisine gore)
            roll_mean = ret.rolling(12, min_periods=6).mean()
            roll_std = ret.rolling(12, min_periods=6).std()
            df["zscore_1m"] = (ret - roll_mean) / roll_std.replace(0, np.nan)

            # Rolling Sharpe (aylik, yilliklandirilmis)
            excess = ret / 100 - monthly_rf
            rm = excess.rolling(6, min_periods=3).mean()
            rs = excess.rolling(6, min_periods=3).std()
            df["sharpe_6m"] = (rm / rs.replace(0, np.nan)) * np.sqrt(12)

            # Drawdown
            roll_max = nav.rolling(12, min_periods=3).max()
            df["drawdown"] = (nav / roll_max) - 1
            df["drawdown_6m"] = df["drawdown"].rolling(6).min()

            # Makro (aylik, nearest reindex)
            for col, series in [
                ("bist_return_1m", mkt_bist),
                ("usdtry_return_1m", mkt_usd),
                ("gold_return_1m", mkt_gold),
            ]:
                if series is not None:
                    df[col] = series.reindex(nav.index, method="nearest")

            df["cpi_yoy"] = cpi
            df["policy_rate"] = annual_rate

            # Target: bir sonraki ayin normalized getirisi
            df[_TARGET_3M] = (ret / 100).shift(-1)
            df[_TARGET_12M] = (nav.pct_change(12) / 100).shift(-12)

            df["fund_code"] = code
            all_rows.append(df)

        if not all_rows:
            logger.error("Hicbir fon icin snapshot feature uretilemedi")
            return pd.DataFrame()

        dataset = pd.concat(all_rows)

        nan_pct = dataset.drop(columns=["fund_code", _TARGET_3M, _TARGET_12M], errors="ignore").isna().mean()
        high_nan = nan_pct[nan_pct > 0.5].index.tolist()
        if high_nan:
            logger.warning(f"%50+ NaN nedeniyle dusurulacak: {high_nan}")

        logger.info(
            f"Snapshot dataset: {len(dataset)} satir, "
            f"{dataset['fund_code'].nunique()} fon, "
            f"{dataset.shape[1]} sutun"
        )
        return dataset

    # ------------------------------------------------------------------
    # 3. Model egitimi
    # ------------------------------------------------------------------

    def train_models(
        self,
        dataset: pd.DataFrame,
        target: str = _TARGET_3M,
    ) -> Dict[str, ModelResult]:
        """Tum modelleri egit ve karsilastir."""
        logger.info(f"Model egitimi basliyor (target: {target})")

        X, y_3m, y_12m = self.feature_eng.get_clean_features(dataset)
        y = y_3m if target == _TARGET_3M else y_12m

        valid_n = int((X.notna().all(axis=1) & y.notna()).sum())
        logger.info(f"Gecerli egitim satiri: {valid_n}/{len(X)}")

        # Az veri icin config'i otomatik kucult
        if valid_n < 80:
            test_size = max(3, valid_n // 8)
            min_train = max(10, valid_n // 4)
            config = ModelConfig(
                target=target,
                min_train_samples=min_train,
                test_size_weeks=test_size,
                n_splits=3,
            )
            logger.warning(
                f"Az veri ({valid_n} satir) — kucultulmus config: "
                f"min_train={min_train}, test_size={test_size}"
            )
        else:
            config = ModelConfig(target=target)

        predictor = BESPredictor(config)

        if valid_n < 20:
            logger.warning(
                f"Cok az egitim verisi ({valid_n} satir). "
                f"Gercek TEFAS verisiyle cok daha fazla veri elde edilir."
            )

        results = predictor.compare_models(X, y)

        best_name = max(results, key=lambda k: results[k].avg_ic) if results else None
        if best_name:
            logger.info(f"En iyi model: {best_name} (IC={results[best_name].avg_ic:.3f})")
            self._save_predictor(predictor, target)

        return results

    # ------------------------------------------------------------------
    # 4. Tahmin
    # ------------------------------------------------------------------

    def generate_predictions(
        self,
        dataset: pd.DataFrame,
        model_name: str = "xgboost",
        target: str = _TARGET_3M,
    ) -> pd.DataFrame:
        """Tum fonlar icin en son tarihli tahminleri uret."""
        predictor = self._load_predictor(target)
        if predictor is None:
            logger.error("Kayitli model bulunamadi, once train calistirin")
            return pd.DataFrame()

        X, _, _ = self.feature_eng.get_clean_features(dataset)

        rows = []
        for code in dataset["fund_code"].unique():
            mask = dataset["fund_code"] == code
            fund_X = X[mask].dropna(how="any")
            if fund_X.empty:
                continue

            last_row = fund_X.iloc[[-1]]
            pred = predictor.predict(last_row, model_name=model_name)
            if pred is not None and not pred.empty and not pd.isna(pred.iloc[0]):
                rows.append({
                    "fund_code": code,
                    "prediction_date": str(last_row.index[0].date()),
                    f"predicted_{target}": round(float(pred.iloc[0]), 4),
                    "model": model_name,
                })

        if not rows:
            return pd.DataFrame()

        pred_df = pd.DataFrame(rows).sort_values(f"predicted_{target}", ascending=False)
        logger.info(f"Tahmin uretildi: {len(pred_df)} fon")
        return pred_df

    # ------------------------------------------------------------------
    # 5. Model persistence
    # ------------------------------------------------------------------

    def _save_predictor(self, predictor: BESPredictor, target: str) -> None:
        # joblib: sklearn modelleri icin konvansiyonel + numpy'da daha verimli.
        # NOT: joblib/pickle GUVENLIK siniri DEGILDIR — yalniz GUVENILIR kaynaktan
        # uretilen model dosyalari yuklenmeli (untrusted .joblib = kod calistirma riski).
        import joblib
        path = self.output_dir / f"predictor_{target}.joblib"
        try:
            joblib.dump(predictor, path)
            logger.info(f"Model kaydedildi: {path}")
        except Exception as e:
            logger.error(f"Model kaydetme hatasi: {e}")

    def _load_predictor(self, target: str) -> Optional[BESPredictor]:
        import joblib
        path = self.output_dir / f"predictor_{target}.joblib"
        if not path.exists():
            # Geriye uyum: eski .pkl varsa onu da kabul et
            legacy = self.output_dir / f"predictor_{target}.pkl"
            if legacy.exists():
                path = legacy
            else:
                return None
        try:
            return joblib.load(path)
        except Exception as e:
            logger.error(f"Model yukleme hatasi: {e}")
            return None

    # ------------------------------------------------------------------
    # 6. Tam pipeline
    # ------------------------------------------------------------------

    def run_full_pipeline(
        self,
        fund_codes: Optional[List[str]] = None,
        targets: Optional[List[str]] = None,
        max_funds: Optional[int] = None,
    ) -> Dict:
        """
        Uctan uca ML pipeline'i calistir.

        targets: ["fwd_return_3m"] veya ["fwd_return_3m", "fwd_return_12m"].
                 None → sadece 3M (mevcut davranis).

        Returns: {status, fund_count, dataset_shape, best_model, predictions, ...}
                 Top-level anahtarlar her zaman birincil (3M) target'i yansitir.
        """
        if targets is None:
            targets = [_TARGET_3M]

        t0 = time.time()
        logger.info("=" * 60)
        logger.info(f"ML PIPELINE BASLADI (targets: {targets})")
        logger.info("=" * 60)

        fund_navs = self.collect_fund_data(fund_codes, max_funds=max_funds)
        if not fund_navs:
            return {"status": "ERROR", "message": "Hicbir fon verisi cekilemedi"}

        market_data = self.collect_market_data()

        dataset = self.build_features(fund_navs, market_data)
        if dataset.empty:
            return {"status": "ERROR", "message": "Feature matrisi bos"}

        dataset_path = self.output_dir / "latest_dataset.parquet"
        dataset.to_parquet(dataset_path)
        logger.info(f"Dataset kaydedildi: {dataset_path}")

        all_model_results: Dict[str, Dict] = {}
        all_predictions: Dict[str, pd.DataFrame] = {}

        for target in targets:
            logger.info(f"\n{'='*40}")
            logger.info(f"Target: {target}")
            logger.info(f"{'='*40}")

            model_results = self.train_models(dataset, target=target)
            if not model_results:
                logger.warning(f"{target} icin model egitimi basarisiz, atlandi")
                continue

            all_model_results[target] = model_results

            best_name = max(model_results, key=lambda k: model_results[k].avg_ic)
            predictions = self.generate_predictions(dataset, model_name=best_name, target=target)
            all_predictions[target] = predictions

            if not predictions.empty:
                pred_path = (
                    self.output_dir
                    / f"predictions_{target}_{datetime.now().strftime('%Y%m%d')}.csv"
                )
                predictions.to_csv(pred_path, index=False)
                logger.info(f"Tahminler kaydedildi: {pred_path}")

            # Her target icin ayri summary kaydet
            best = model_results[best_name]
            target_summary = {
                "status": "SUCCESS",
                "run_date": datetime.now().isoformat(),
                "target": target,
                "best_model": best_name,
                "best_ic": best.avg_ic,
                "best_mae": best.avg_mae,
                "best_dir_acc": best.avg_directional_accuracy,
                "fund_count": len(fund_navs),
                "model_comparison": {
                    name: {
                        "mae": r.avg_mae,
                        "rmse": r.avg_rmse,
                        "dir_acc": r.avg_directional_accuracy,
                        "ic": r.avg_ic,
                    }
                    for name, r in model_results.items()
                },
                "top_features": (
                    dict(list(best.feature_importance.items())[:10])
                    if best.feature_importance
                    else {}
                ),
                "predictions_count": len(predictions),
            }
            target_summary_path = self.output_dir / f"latest_run_summary_{target}.json"
            with open(target_summary_path, "w", encoding="utf-8") as f:
                json.dump(target_summary, f, indent=2, ensure_ascii=False, default=str)

        if not all_model_results:
            return {"status": "ERROR", "message": "Hicbir target icin model egitimi basarisiz"}

        # Top-level summary: birincil target (genellikle 3M) yansitir — geri uyumluluk
        primary = targets[0]
        primary_results = all_model_results[primary]
        primary_best_name = max(primary_results, key=lambda k: primary_results[k].avg_ic)
        primary_best = primary_results[primary_best_name]
        primary_preds = all_predictions.get(primary, pd.DataFrame())

        summary = {
            "status": "SUCCESS",
            "run_date": datetime.now().isoformat(),
            "run_time_sec": round(time.time() - t0, 1),
            "fund_count": len(fund_navs),
            "dataset_shape": list(dataset.shape),
            "target": primary,
            "targets": targets,
            "best_model": primary_best_name,
            "best_ic": primary_best.avg_ic,
            "best_mae": primary_best.avg_mae,
            "best_dir_acc": primary_best.avg_directional_accuracy,
            "model_comparison": {
                name: {
                    "mae": r.avg_mae,
                    "rmse": r.avg_rmse,
                    "dir_acc": r.avg_directional_accuracy,
                    "ic": r.avg_ic,
                }
                for name, r in primary_results.items()
            },
            "top_features": (
                dict(list(primary_best.feature_importance.items())[:10])
                if primary_best.feature_importance
                else {}
            ),
            "predictions_count": len(primary_preds),
        }

        summary_path = self.output_dir / "latest_run_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

        logger.info("=" * 60)
        logger.info(f"ML PIPELINE TAMAMLANDI ({summary['run_time_sec']}s)")
        for tgt, results in all_model_results.items():
            bn = max(results, key=lambda k: results[k].avg_ic)
            b  = results[bn]
            logger.info(
                f"  {tgt}: {bn} | IC={b.avg_ic:.3f} | DirAcc={b.avg_directional_accuracy:.1%}"
            )
        logger.info("=" * 60)

        summary["predictions"] = primary_preds
        return summary
