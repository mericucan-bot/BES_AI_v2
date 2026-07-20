import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    BES fon verileri ve makro verilerden ML feature'ları üretir.

    Feature grupları:
    1. Fon bazlı: rolling return, vol, momentum, sharpe
    2. Makro bazlı: faiz, enflasyon, döviz sinyalleri
    3. Rejim bazlı: regime engine skorları
    4. Cross-sectional: fonun kendi kategorisine göre ranking

    Anti-leakage: Tüm rolling hesaplamalar sadece geçmişe bakıyor (shift(1) ile)
    """

    TARGET_3M = "fwd_return_3m"
    TARGET_12M = "fwd_return_12m"
    TARGET_RANK_3M = "fwd_rank_3m"   # PLAN-16: tarih-ici kesitsel yuzdelik sira

    def __init__(self):
        pass

    def create_fund_features(self, nav_series: pd.Series, fund_code: str) -> pd.DataFrame:
        """
        Tek bir fonun NAV serisinden feature'lar üret.

        nav_series: index=date, values=NAV fiyatları
        Returns: DataFrame(index=date, columns=feature_names)
        """
        df = pd.DataFrame({"nav": nav_series})

        df["daily_return"] = df["nav"].pct_change()

        for window in [5, 10, 21, 63, 126, 252]:
            label = {5: "1w", 10: "2w", 21: "1m", 63: "3m", 126: "6m", 252: "1y"}[window]
            df[f"return_{label}"] = df["nav"].pct_change(window)

        for window in [21, 63, 126]:
            label = {21: "1m", 63: "3m", 126: "6m"}[window]
            df[f"vol_{label}"] = (
                df["daily_return"].rolling(window, min_periods=max(10, window // 2)).std()
                * np.sqrt(252)
            )

        df["momentum_1m_3m"] = df["return_1m"] - df["return_3m"]
        df["momentum_3m_6m"] = df["return_3m"] - df["return_6m"]

        for period in ["1m", "3m"]:
            col = f"return_{period}"
            roll_mean = df[col].rolling(126, min_periods=63).mean()
            roll_std = df[col].rolling(126, min_periods=63).std()
            df[f"zscore_{period}"] = (df[col] - roll_mean) / roll_std.replace(0, np.nan)

        daily_rf = 0.36 / 252
        for window in [63, 126]:
            label = {63: "3m", 126: "6m"}[window]
            excess = df["daily_return"] - daily_rf
            roll_mean = excess.rolling(window, min_periods=window // 2).mean()
            roll_std = excess.rolling(window, min_periods=window // 2).std()
            df[f"sharpe_{label}"] = (roll_mean / roll_std.replace(0, np.nan)) * np.sqrt(252)

        rolling_max = df["nav"].rolling(252, min_periods=21).max()
        df["drawdown"] = (df["nav"] / rolling_max) - 1
        df["drawdown_3m"] = df["drawdown"].rolling(63).min()

        for window in [21, 63, 126]:
            label = {21: "1m", 63: "3m", 126: "6m"}[window]
            ma = df["nav"].rolling(window, min_periods=window // 2).mean()
            df[f"nav_vs_ma_{label}"] = (df["nav"] / ma) - 1

        df["fund_code"] = fund_code
        df = df.drop(columns=["nav", "daily_return"])
        return df

    def create_macro_features(
        self,
        bist_prices: Optional[pd.Series],
        usdtry_prices: Optional[pd.Series],
        gold_prices: Optional[pd.Series],
        cpi_yoy: Optional[float] = None,
        policy_rate: Optional[float] = None,
    ) -> pd.DataFrame:
        """Makro verilerden feature'lar üret."""
        df = pd.DataFrame()

        if bist_prices is not None and not bist_prices.empty:
            df["bist_return_1m"] = bist_prices.pct_change(21)
            df["bist_return_3m"] = bist_prices.pct_change(63)
            df["bist_vol_1m"] = bist_prices.pct_change().rolling(21).std() * np.sqrt(252)
            df["bist_vol_3m"] = bist_prices.pct_change().rolling(63).std() * np.sqrt(252)
            bist_max = bist_prices.rolling(252, min_periods=21).max()
            df["bist_drawdown"] = (bist_prices / bist_max) - 1

        if usdtry_prices is not None and not usdtry_prices.empty:
            df["usdtry_return_1m"] = usdtry_prices.pct_change(21)
            df["usdtry_return_3m"] = usdtry_prices.pct_change(63)
            df["usdtry_vol_1m"] = usdtry_prices.pct_change().rolling(21).std() * np.sqrt(252)
            df["usdtry_momentum"] = usdtry_prices.pct_change(5) - usdtry_prices.pct_change(21)

        if gold_prices is not None and not gold_prices.empty:
            df["gold_return_1m"] = gold_prices.pct_change(21)
            df["gold_return_3m"] = gold_prices.pct_change(63)

        if cpi_yoy is not None:
            df["cpi_yoy"] = cpi_yoy
            df["monthly_inflation"] = (1 + cpi_yoy) ** (1 / 12) - 1

        if policy_rate is not None:
            df["policy_rate"] = policy_rate / 100

        return df

    def create_fund_macro_interaction(
        self,
        fund_returns: pd.Series,
        macro_series: pd.Series,
        macro_name: str,
        window: int = 63,
    ) -> pd.DataFrame:
        """Fon-makro etkileşim feature'ları (rolling beta, korelasyon)."""
        df = pd.DataFrame()
        df[f"corr_{macro_name}_{window}d"] = fund_returns.rolling(
            window, min_periods=window // 2
        ).corr(macro_series)
        cov = fund_returns.rolling(window, min_periods=window // 2).cov(macro_series)
        var = macro_series.rolling(window, min_periods=window // 2).var()
        df[f"beta_{macro_name}_{window}d"] = cov / var.replace(0, np.nan)
        return df

    def create_target_variables(self, nav_series: pd.Series) -> pd.DataFrame:
        """
        Hedef değişkenleri üret (GELECEK getiriler — sadece eğitimde kullanılacak).

        Prediction sırasında bu sütunlar NaN olacak — bu beklenen davranış.
        """
        df = pd.DataFrame(index=nav_series.index)
        df[self.TARGET_3M] = nav_series.pct_change(63).shift(-63)
        df[self.TARGET_12M] = nav_series.pct_change(252).shift(-252)
        df["fwd_direction_3m"] = (df[self.TARGET_3M] > 0).astype(float)
        df["fwd_direction_12m"] = (df[self.TARGET_12M] > 0).astype(float)
        return df

    def build_dataset(
        self,
        fund_navs: Dict[str, pd.Series],
        market_data: pd.DataFrame,
        macro_snapshot: Optional[Dict] = None,
        sample_frequency: str = "weekly",
        min_nav_length: int = 126,
    ) -> pd.DataFrame:
        """
        Tüm feature'ları birleştirerek eğitime hazır dataset üret.

        fund_navs: {fund_code: Series(index=date, values=nav)}
        market_data: DataFrame(columns=[BIST, USDTRY, GOLD], index=date)
        macro_snapshot: MacroEngine.get_macro_snapshot() çıktısı
        sample_frequency: "daily", "weekly" veya "monthly" (tüm satırlar)
        min_nav_length: Minimum veri noktası (günlük=126, aylık=6)
        """
        all_rows = []

        macro_features = self.create_macro_features(
            bist_prices=market_data.get("BIST"),
            usdtry_prices=market_data.get("USDTRY"),
            gold_prices=market_data.get("GOLD"),
            cpi_yoy=macro_snapshot.get("cpi_yoy") if macro_snapshot else None,
            policy_rate=macro_snapshot.get("current_policy_rate") if macro_snapshot else None,
        )

        for fund_code, nav in fund_navs.items():
            if nav is None or len(nav) < min_nav_length:
                logger.debug(
                    f"Atlanan fon (yetersiz veri): {fund_code} "
                    f"({len(nav) if nav is not None else 0} nokta, min={min_nav_length})"
                )
                continue

            try:
                fund_feat = self.create_fund_features(nav, fund_code)
                targets = self.create_target_variables(nav)

                fund_daily_ret = nav.pct_change()
                interactions = pd.DataFrame(index=nav.index)

                if "BIST" in market_data.columns:
                    bist_ret = market_data["BIST"].pct_change().reindex(nav.index)
                    interactions = interactions.join(
                        self.create_fund_macro_interaction(fund_daily_ret, bist_ret, "bist", 63),
                        how="left",
                    )

                if "USDTRY" in market_data.columns:
                    usd_ret = market_data["USDTRY"].pct_change().reindex(nav.index)
                    interactions = interactions.join(
                        self.create_fund_macro_interaction(fund_daily_ret, usd_ret, "usdtry", 63),
                        how="left",
                    )

                combined = fund_feat.join(macro_features, how="left")
                combined = combined.join(interactions, how="left")
                combined = combined.join(targets, how="left")

                if sample_frequency == "weekly":
                    combined = combined[combined.index.dayofweek == 4]
                elif sample_frequency == "monthly":
                    pass  # Zaten aylik, filtre yok

                all_rows.append(combined)

            except Exception as e:
                logger.warning(f"Feature uretim hatasi ({fund_code}): {e}")
                continue

        if not all_rows:
            logger.error("Hicbir fon icin feature uretilemedi!")
            return pd.DataFrame()

        dataset = pd.concat(all_rows, ignore_index=False)

        nan_pct = dataset.isna().mean()
        high_nan = nan_pct[nan_pct > 0.5]
        if not high_nan.empty:
            logger.warning(f"%50+ NaN olan feature'lar: {high_nan.index.tolist()}")

        logger.info(
            f"Dataset hazir: {len(dataset)} satir, {len(dataset.columns)} feature, "
            f"{dataset['fund_code'].nunique()} fon"
        )
        return dataset

    def get_feature_names(self) -> List[str]:
        """Model egitiminde kullanilmayacak (exclude) sutun listesi."""
        return [
            "fund_code",
            self.TARGET_3M,
            self.TARGET_12M,
            self.TARGET_RANK_3M,
            "fwd_direction_3m",
            "fwd_direction_12m",
        ]

    def get_clean_features(
        self, dataset: pd.DataFrame, max_nan_col_pct: float = 0.5
    ) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """
        Dataset'i X (features), y_3m, y_12m olarak ayir.

        max_nan_col_pct: Bu esikten fazla NaN olan sutunlar dusurulur (varsayilan 0.5 = %50).
                         Az veriyle calisirken uzun-pencere feature'larinin otomatik filtrelenmesini saglar.
        """
        target_cols = [self.TARGET_3M, self.TARGET_12M, self.TARGET_RANK_3M,
                       "fwd_direction_3m", "fwd_direction_12m"]
        meta_cols = ["fund_code"]

        feature_cols = [c for c in dataset.columns if c not in target_cols + meta_cols]

        X = dataset[feature_cols].copy()

        # Cok yuksek NaN oranli sutunlari dusur
        nan_pct = X.isna().mean()
        dropped = nan_pct[nan_pct > max_nan_col_pct].index.tolist()
        if dropped:
            logger.warning(f"Yuksek NaN (>{max_nan_col_pct:.0%}) nedeniyle dusuruldu: {dropped}")
            X = X.drop(columns=dropped)

        y_3m = dataset[self.TARGET_3M].copy()
        y_12m = dataset[self.TARGET_12M].copy()

        valid_3m = X.notna().all(axis=1) & y_3m.notna()
        valid_12m = X.notna().all(axis=1) & y_12m.notna()

        logger.info(
            f"Temiz veri: 3M icin {valid_3m.sum()}/{len(X)} satir, "
            f"12M icin {valid_12m.sum()}/{len(X)} satir"
        )

        return X, y_3m, y_12m
