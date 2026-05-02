import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    logger.warning("xgboost yuklu degil, sadece RandomForest ve Ridge kullanilacak")

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    logger.info("shap yuklu degil, feature importance SHAP olmadan hesaplanacak")


@dataclass
class ModelConfig:
    """Model egitim parametreleri."""
    target: str = "fwd_return_3m"
    min_train_samples: int = 30
    n_splits: int = 5
    test_size_weeks: int = 8
    random_state: int = 42

    xgb_params: Dict = field(default_factory=lambda: {
        "n_estimators": 200,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
    })

    rf_params: Dict = field(default_factory=lambda: {
        "n_estimators": 200,
        "max_depth": 6,
        "min_samples_leaf": 5,
        "max_features": "sqrt",
    })


@dataclass
class FoldResult:
    """Tek bir walk-forward fold'un sonuclari."""
    fold_idx: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    train_size: int
    test_size: int
    mae: float
    rmse: float
    directional_accuracy: float
    ic: float
    predictions: List[float]
    actuals: List[float]


@dataclass
class ModelResult:
    """Tam model egitim sonuclari."""
    model_name: str
    target: str
    config: ModelConfig
    fold_results: List[FoldResult]
    avg_mae: float = 0.0
    avg_rmse: float = 0.0
    avg_directional_accuracy: float = 0.0
    avg_ic: float = 0.0
    feature_importance: Optional[Dict[str, float]] = None
    total_train_samples: int = 0
    total_test_samples: int = 0
    training_time_sec: float = 0.0


class BESPredictor:
    """
    BES fon getiri tahmin modeli.

    Walk-forward (expanding window) validation ile egitilir.
    Anti-leakage: Her fold'da test seti her zaman egitimden sonra gelir.

    Kullanim:
    1. FeatureEngineer.build_dataset ile feature matrisi uret
    2. train_and_evaluate ile egit + degerlendir
    3. predict ile yeni veri uzerinde tahmin yap
    """

    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or ModelConfig()
        self.models: Dict = {}
        self.scaler = StandardScaler()
        self.feature_names: List[str] = []
        self.is_fitted = False

    def _create_walk_forward_splits(
        self, X: pd.DataFrame, y: pd.Series
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Walk-forward (expanding window) split'ler uret.

          |--TRAIN--|--TEST--|
          |----TRAIN----|--TEST--|
          |------TRAIN------|--TEST--|

        Anti-leakage garantisi: train her zaman test'ten once.
        """
        valid_mask = X.notna().all(axis=1) & y.notna()
        valid_indices = np.where(valid_mask)[0]

        min_needed = self.config.min_train_samples + self.config.test_size_weeks
        if len(valid_indices) < min_needed:
            logger.warning(
                f"Yetersiz veri: {len(valid_indices)} satir, minimum {min_needed} gerekli"
            )
            return []

        splits = []
        total = len(valid_indices)
        test_size = self.config.test_size_weeks

        for i in range(self.config.n_splits):
            test_end = total - (i * test_size)
            test_start = test_end - test_size

            if test_start < self.config.min_train_samples:
                break

            train_idx = valid_indices[:test_start]
            test_idx = valid_indices[test_start:test_end]

            if len(train_idx) >= self.config.min_train_samples and len(test_idx) > 0:
                splits.append((train_idx, test_idx))

        splits.reverse()
        logger.info(f"Walk-forward: {len(splits)} fold olusturuldu")
        return splits

    def _evaluate_fold(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> Tuple[float, float, float, float]:
        """MAE, RMSE, directional accuracy, IC hesapla."""
        from scipy.stats import spearmanr

        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))

        if len(y_true) > 0:
            directional_acc = float(np.sum(np.sign(y_true) == np.sign(y_pred))) / len(y_true)
        else:
            directional_acc = 0.0

        if len(y_true) > 2:
            ic, _ = spearmanr(y_true, y_pred)
            ic = 0.0 if np.isnan(ic) else float(ic)
        else:
            ic = 0.0

        return mae, rmse, directional_acc, ic

    def _build_model(self, model_name: str):
        """Model instance olustur."""
        if model_name == "xgboost":
            if HAS_XGBOOST:
                return xgb.XGBRegressor(
                    **self.config.xgb_params,
                    random_state=self.config.random_state,
                    verbosity=0,
                )
            logger.warning("XGBoost yuklu degil, RandomForest fallback")
            return RandomForestRegressor(
                **self.config.rf_params,
                random_state=self.config.random_state,
                n_jobs=-1,
            )
        if model_name == "random_forest":
            return RandomForestRegressor(
                **self.config.rf_params,
                random_state=self.config.random_state,
                n_jobs=-1,
            )
        if model_name == "ridge":
            return Ridge(alpha=1.0)
        raise ValueError(f"Bilinmeyen model: {model_name}")

    def train_and_evaluate(
        self, X: pd.DataFrame, y: pd.Series, model_name: str = "xgboost"
    ) -> Optional[ModelResult]:
        """
        Walk-forward validation ile model egit ve degerlendir.

        model_name: "xgboost" | "random_forest" | "ridge"
        """
        t0 = time.time()
        logger.info(f"=== {model_name} egitimi basliyor ({self.config.target}) ===")

        numeric_cols = [c for c in X.columns if X[c].dtype.kind in "fiub"]
        X_num = X[numeric_cols].copy()
        self.feature_names = numeric_cols

        splits = self._create_walk_forward_splits(X_num, y)
        if not splits:
            logger.error("Walk-forward split olusturulamadi, yetersiz veri")
            return None

        fold_results: List[FoldResult] = []
        last_model = None
        last_scaler = StandardScaler()

        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            X_tr = X_num.iloc[train_idx].values
            y_tr = y.iloc[train_idx].values
            X_te = X_num.iloc[test_idx].values
            y_te = y.iloc[test_idx].values

            # Son NaN temizleme
            tr_ok = ~(np.isnan(X_tr).any(axis=1) | np.isnan(y_tr))
            te_ok = ~(np.isnan(X_te).any(axis=1) | np.isnan(y_te))
            X_tr, y_tr = X_tr[tr_ok], y_tr[tr_ok]
            X_te, y_te = X_te[te_ok], y_te[te_ok]

            if len(X_tr) < self.config.min_train_samples or len(X_te) == 0:
                continue

            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_te_s = scaler.transform(X_te)

            model = self._build_model(model_name)
            if model_name == "xgboost" and HAS_XGBOOST:
                model.fit(
                    X_tr_s, y_tr,
                    eval_set=[(X_te_s, y_te)],
                    verbose=False,
                )
            else:
                model.fit(X_tr_s, y_tr)

            y_pred = model.predict(X_te_s)
            mae, rmse, dir_acc, ic = self._evaluate_fold(y_te, y_pred)

            train_dates = X_num.index[train_idx]
            test_dates = X_num.index[test_idx]

            fold_results.append(FoldResult(
                fold_idx=fold_idx,
                train_start=str(train_dates[0].date()),
                train_end=str(train_dates[-1].date()),
                test_start=str(test_dates[0].date()),
                test_end=str(test_dates[-1].date()),
                train_size=len(X_tr),
                test_size=len(X_te),
                mae=round(mae, 6),
                rmse=round(rmse, 6),
                directional_accuracy=round(dir_acc, 4),
                ic=round(ic, 4),
                predictions=y_pred.tolist(),
                actuals=y_te.tolist(),
            ))
            last_model = model
            last_scaler = scaler

            logger.info(
                f"  Fold {fold_idx}: train={len(X_tr)}, test={len(X_te)}, "
                f"MAE={mae:.4f}, RMSE={rmse:.4f}, DirAcc={dir_acc:.1%}, IC={ic:.3f}"
            )

        if not fold_results:
            logger.error("Hicbir fold tamamlanamadi")
            return None

        self.models[model_name] = last_model
        self.scaler = last_scaler
        self.is_fitted = True

        result = ModelResult(
            model_name=model_name,
            target=self.config.target,
            config=self.config,
            fold_results=fold_results,
            avg_mae=round(float(np.mean([f.mae for f in fold_results])), 6),
            avg_rmse=round(float(np.mean([f.rmse for f in fold_results])), 6),
            avg_directional_accuracy=round(
                float(np.mean([f.directional_accuracy for f in fold_results])), 4
            ),
            avg_ic=round(float(np.mean([f.ic for f in fold_results])), 4),
            total_train_samples=sum(f.train_size for f in fold_results),
            total_test_samples=sum(f.test_size for f in fold_results),
            training_time_sec=round(time.time() - t0, 2),
        )
        result.feature_importance = self._get_feature_importance(last_model, X_num, last_scaler)

        logger.info(
            f"=== {model_name} tamamlandi: MAE={result.avg_mae:.4f}, "
            f"DirAcc={result.avg_directional_accuracy:.1%}, IC={result.avg_ic:.3f} ==="
        )
        return result

    def _get_feature_importance(
        self, model, X: pd.DataFrame, scaler: StandardScaler
    ) -> Dict[str, float]:
        """Feature importance: SHAP (tree models only) > built-in > coef (normalized)."""
        names = self.feature_names
        if not names or model is None:
            return {}

        if HAS_SHAP and hasattr(model, "feature_importances_"):
            try:
                sample = min(100, len(X))
                Xs = scaler.transform(X.iloc[:sample][names].values)
                explainer = shap.TreeExplainer(model)
                shap_vals = explainer.shap_values(Xs)
                imp = np.abs(shap_vals).mean(axis=0)
                total = imp.sum() or 1.0
                return dict(sorted(zip(names, (imp / total).tolist()), key=lambda x: -x[1]))
            except Exception as e:
                logger.warning(f"SHAP basarisiz, built-in kullaniliyor: {e}")

        if hasattr(model, "feature_importances_"):
            imp = model.feature_importances_
            return dict(sorted(zip(names, imp.tolist()), key=lambda x: -x[1]))

        if hasattr(model, "coef_"):
            imp = np.abs(model.coef_)
            total = imp.sum() or 1.0
            return dict(sorted(zip(names, (imp / total).tolist()), key=lambda x: -x[1]))

        return {}

    def predict(self, X_new: pd.DataFrame, model_name: str = "xgboost") -> Optional[pd.Series]:
        """Egitilmis model ile tahmin yap."""
        if not self.is_fitted or model_name not in self.models:
            logger.error(f"Model egitilmemis veya bulunamadi: {model_name}")
            return None

        missing = set(self.feature_names) - set(X_new.columns)
        if missing:
            logger.error(f"Eksik feature'lar: {missing}")
            return None

        X_clean = X_new[self.feature_names].copy()
        nan_rows = X_clean.isna().any(axis=1)

        X_scaled = self.scaler.transform(X_clean.fillna(0).values)
        preds = self.models[model_name].predict(X_scaled)

        result = pd.Series(preds, index=X_new.index, name=f"pred_{self.config.target}")
        result[nan_rows] = np.nan
        return result

    def compare_models(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Dict[str, ModelResult]:
        """Tum modelleri egit ve karsilastir."""
        results: Dict[str, ModelResult] = {}
        for name in ["ridge", "random_forest", "xgboost"]:
            logger.info(f"\n--- {name} ---")
            r = self.train_and_evaluate(X, y, model_name=name)
            if r:
                results[name] = r

        if results:
            rows = [
                {
                    "Model": n,
                    "MAE": r.avg_mae,
                    "RMSE": r.avg_rmse,
                    "DirAcc": f"{r.avg_directional_accuracy:.1%}",
                    "IC": r.avg_ic,
                    "Sure (s)": r.training_time_sec,
                }
                for n, r in results.items()
            ]
            logger.info(f"\nModel karsilastirma:\n{pd.DataFrame(rows).to_string(index=False)}")

        return results

    def print_summary(self, result: ModelResult) -> str:
        """Insan-okunabilir model ozeti."""
        lines = [
            "=" * 60,
            f"MODEL: {result.model_name} | TARGET: {result.target}",
            "=" * 60,
            f"Fold sayisi     : {len(result.fold_results)}",
            f"Toplam egitim   : {result.total_train_samples} ornek",
            f"Toplam test     : {result.total_test_samples} ornek",
            f"Egitim suresi   : {result.training_time_sec:.1f} saniye",
            "",
            f"Ort. MAE        : {result.avg_mae:.4f}",
            f"Ort. RMSE       : {result.avg_rmse:.4f}",
            f"Ort. DirAcc     : {result.avg_directional_accuracy:.1%}",
            f"Ort. IC         : {result.avg_ic:.3f}",
            "",
        ]

        if result.feature_importance:
            lines.append("Top 10 Feature:")
            for i, (feat, imp) in enumerate(list(result.feature_importance.items())[:10]):
                bar = "#" * max(1, int(imp * 50))
                lines.append(f"  {i+1:2d}. {feat:25s} {imp:.4f}  {bar}")

        lines.append("=" * 60)
        return "\n".join(lines)
