"""
Model Training & AutoML Benchmark Module
Trains, tunes, evaluates, and selects best ML models for classification and regression.
"""

from typing import Dict, List, Optional, Tuple, Any
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, r2_score, mean_squared_error, mean_absolute_error
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline

# Optional gradient boosting libraries
try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    HAS_LGB = True
except ImportError:
    HAS_LGB = False


class ModelTrainer:
    """AutoML Multi-Model Trainer and Benchmark Suite."""

    @staticmethod
    def get_candidate_models(task_type: str, class_weight: Optional[str] = "balanced", random_state: int = 42) -> Dict[str, Any]:
        """Returns dictionary of candidate models for classification or regression."""
        models = {}

        if task_type == "classification":
            models["Random Forest"] = RandomForestClassifier(
                n_estimators=150, max_depth=10, random_state=random_state,
                class_weight=class_weight, n_jobs=-1
            )
            models["Gradient Boosting"] = GradientBoostingClassifier(
                n_estimators=120, max_depth=5, learning_rate=0.08, random_state=random_state
            )
            models["Logistic Regression"] = LogisticRegression(
                max_iter=1000, class_weight=class_weight, random_state=random_state
            )

            if HAS_XGB:
                models["XGBoost"] = XGBClassifier(
                    n_estimators=150, max_depth=5, learning_rate=0.08,
                    random_state=random_state, eval_metric="logloss", n_jobs=-1
                )

            if HAS_LGB:
                models["LightGBM"] = LGBMClassifier(
                    n_estimators=150, max_depth=6, learning_rate=0.08,
                    random_state=random_state, class_weight=class_weight, n_jobs=-1, verbose=-1
                )

        else: # Regression
            models["Random Forest"] = RandomForestRegressor(
                n_estimators=150, max_depth=12, random_state=random_state, n_jobs=-1
            )
            models["Gradient Boosting"] = GradientBoostingRegressor(
                n_estimators=120, max_depth=5, learning_rate=0.08, random_state=random_state
            )
            models["Ridge Regression"] = Ridge(alpha=1.0, random_state=random_state)

            if HAS_XGB:
                models["XGBoost"] = XGBRegressor(
                    n_estimators=150, max_depth=5, learning_rate=0.08,
                    random_state=random_state, n_jobs=-1
                )

            if HAS_LGB:
                models["LightGBM"] = LGBMRegressor(
                    n_estimators=150, max_depth=6, learning_rate=0.08,
                    random_state=random_state, n_jobs=-1, verbose=-1
                )

        return models

    @classmethod
    def train_and_benchmark(
        cls,
        X_df: pd.DataFrame,
        y_series: pd.Series,
        preprocessor: Any,
        task_type: str = "classification",
        test_size: float = 0.2,
        random_state: int = 42,
        selected_model_names: Optional[List[str]] = None,
        max_train_samples: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Trains and benchmarks all selected models with memory-safe scaling.
        Supports large-scale datasets via configurable stratified sampling.
        """
        # Clean target nulls
        valid_idx = y_series.dropna().index
        X_clean = X_df.loc[valid_idx]
        y_clean = y_series.loc[valid_idx]

        # Optional subsampling for huge datasets (e.g. 500k+ rows)
        if max_train_samples and len(X_clean) > max_train_samples:
            strat_sub = y_clean if (task_type == "classification" and y_clean.nunique() <= 10) else None
            X_clean, _, y_clean, _ = train_test_split(
                X_clean, y_clean, train_size=max_train_samples, random_state=random_state, stratify=strat_sub
            )

        # Stratified split for classification
        stratify = y_clean if (task_type == "classification" and y_clean.nunique() <= 10) else None

        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            X_clean, y_clean, test_size=test_size, random_state=random_state, stratify=stratify
        )

        # Fit preprocessor on train
        preprocessor.fit(X_train_raw, y_train)
        X_train_proc = preprocessor.transform(X_train_raw)
        X_test_proc = preprocessor.transform(X_test_raw)

        # Cast to float32 to conserve RAM
        if isinstance(X_train_proc, np.ndarray):
            X_train_proc = X_train_proc.astype(np.float32)
            X_test_proc = X_test_proc.astype(np.float32)

        candidate_models = cls.get_candidate_models(task_type=task_type, random_state=random_state)
        # `is not None` (not a truthiness check): an explicit empty list means
        # "the user deselected every model" and must filter down to zero
        # candidates, not silently fall back to training the full default set.
        if selected_model_names is not None:
            candidate_models = {k: v for k, v in candidate_models.items() if k in selected_model_names}

        if not candidate_models:
            raise ValueError(
                "Eğitilecek hiçbir model seçilmedi. Lütfen en az bir model seçip tekrar deneyin."
            )

        # For binary classification, track the positive/"success" class so we can
        # score models on how well they find it specifically. A model can look
        # strong on weighted-average metrics (dominated by the majority class)
        # while still being close to useless at catching the minority "success"
        # cases — weighted F1 alone can hide that, so we compute a positive-class
        # F1 too and use it (not the weighted score) to pick the best model.
        is_binary = task_type == "classification" and y_clean.nunique() == 2
        pos_label = sorted(y_clean.unique())[-1] if is_binary else None

        benchmark_results = []
        trained_models = {}
        predictions_dict = {}

        for name, model in candidate_models.items():
            try:
                # Handle XGBoost scale_pos_weight for binary classification
                if name == "XGBoost" and task_type == "classification" and y_train.nunique() == 2:
                    neg_count = (y_train == 0).sum()
                    pos_count = (y_train == 1).sum()
                    if pos_count > 0:
                        model.set_params(scale_pos_weight=neg_count / pos_count)

                model.fit(X_train_proc, y_train)
                y_pred = model.predict(X_test_proc)
                trained_models[name] = model

                if task_type == "classification":
                    acc = accuracy_score(y_test, y_pred)
                    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
                    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
                    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

                    # Positive-class F1 (binary only): how well the model actually
                    # finds the "success" cases, undiluted by majority-class performance.
                    pos_f1 = None
                    if is_binary:
                        try:
                            pos_f1 = f1_score(y_test, y_pred, pos_label=pos_label, average="binary", zero_division=0)
                        except Exception:
                            pos_f1 = None

                    # ROC-AUC if probabilities available
                    roc_auc = None
                    if hasattr(model, "predict_proba"):
                        try:
                            y_proba = model.predict_proba(X_test_proc)
                            if y_clean.nunique() == 2:
                                roc_auc = roc_auc_score(y_test, y_proba[:, 1])
                            else:
                                roc_auc = roc_auc_score(y_test, y_proba, multi_class="ovr")
                        except Exception:
                            roc_auc = None

                    benchmark_results.append({
                        "Model": name,
                        "Accuracy": round(acc, 4),
                        "F1-Score": round(f1, 4),
                        "F1 (Başarı Sınıfı)": round(pos_f1, 4) if pos_f1 is not None else "N/A",
                        "Precision": round(prec, 4),
                        "Recall": round(rec, 4),
                        "ROC-AUC": round(roc_auc, 4) if roc_auc is not None else "N/A"
                    })
                    predictions_dict[name] = {
                        "y_pred": y_pred,
                        "cm": confusion_matrix(y_test, y_pred),
                        "report": classification_report(y_test, y_pred, output_dict=True, zero_division=0)
                    }

                else: # Regression
                    r2 = r2_score(y_test, y_pred)
                    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                    mae = mean_absolute_error(y_test, y_pred)

                    benchmark_results.append({
                        "Model": name,
                        "R2 Score": round(r2, 4),
                        "RMSE": round(rmse, 4),
                        "MAE": round(mae, 4)
                    })
                    predictions_dict[name] = {
                        "y_pred": y_pred,
                        "residuals": (y_test - y_pred).values
                    }

            except Exception as e:
                print(f"Error training {name}: {str(e)}")

        results_df = pd.DataFrame(benchmark_results)

        if results_df.empty:
            raise ValueError(
                "Hiçbir model başarıyla eğitilemedi. Veri setini, hedef sütunu ve seçilen "
                "özellikleri kontrol edip tekrar deneyin."
            )

        # Select best model. For binary classification we rank by positive-class
        # (success) F1 rather than the weighted-average F1: a weighted score can
        # rank a model highly just because it's good at the easy majority class
        # while barely ever catching a real "success" — see "F1 (Başarı Sınıfı)".
        if task_type == "classification":
            if is_binary and "F1 (Başarı Sınıfı)" in results_df.columns and (results_df["F1 (Başarı Sınıfı)"] != "N/A").any():
                ranked = results_df[results_df["F1 (Başarı Sınıfı)"] != "N/A"]
                best_model_name = ranked.sort_values(by="F1 (Başarı Sınıfı)", ascending=False).iloc[0]["Model"]
            else:
                best_model_name = results_df.sort_values(by="F1-Score", ascending=False).iloc[0]["Model"]
        else:
            best_model_name = results_df.sort_values(by="R2 Score", ascending=False).iloc[0]["Model"]

        best_estimator = trained_models[best_model_name]

        # Assemble end-to-end inference pipeline
        full_pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", best_estimator)
        ])

        return {
            "benchmark_table": results_df,
            "best_model_name": best_model_name,
            "best_estimator": best_estimator,
            "full_pipeline": full_pipeline,
            "trained_models": trained_models,
            "predictions_dict": predictions_dict,
            "test_data": {
                "X_test_raw": X_test_raw,
                "X_test_proc": X_test_proc,
                "y_test": y_test
            },
            "train_data": {
                "X_train_raw": X_train_raw,
                "y_train": y_train
            }
        }

    @staticmethod
    def save_pipeline(pipeline: Pipeline, filepath: str = "models/best_product_success_pipeline.joblib") -> str:
        """Saves trained pipeline to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(pipeline, filepath)
        return filepath

    @staticmethod
    def load_pipeline(filepath: str = "models/best_product_success_pipeline.joblib") -> Pipeline:
        """Loads trained pipeline from disk."""
        return joblib.load(filepath)
