"""
Model Interpretability and What-If Sensitivity Simulator Module
Calculates feature importance, individual product prediction drivers,
and interactive what-if scenario simulations.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.pipeline import Pipeline
from src.core.pipeline import PreprocessingBuilder
from src.utils.labels import humanize_feature_name


class ModelExplainer:
    """Explains model decisions and performs what-if scenario analyses."""

    @staticmethod
    def get_global_feature_importance(
        model: Any,
        preprocessor: Any,
        top_n: int = 15,
        template: str = "plotly_dark",
        col_types: Optional[Dict[str, List[str]]] = None
    ) -> Tuple[pd.DataFrame, Optional[go.Figure]]:
        """Extracts top global feature importances and returns Plotly bar chart."""
        try:
            feature_names = PreprocessingBuilder.get_feature_names_from_preprocessor(preprocessor)
        except Exception:
            feature_names = []

        importances = None

        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            coef = model.coef_
            if coef.ndim > 1:
                importances = np.abs(coef).mean(axis=0)
            else:
                importances = np.abs(coef)

        if importances is None:
            return pd.DataFrame(), None

        # Align length if mismatch
        min_len = min(len(feature_names), len(importances))
        if min_len > 0:
            names = feature_names[:min_len]
            scores = importances[:min_len]
        else:
            names = [f"Feature_{i}" for i in range(len(importances))]
            scores = importances

        fi_df = pd.DataFrame({
            "Feature": names,
            "Importance": scores
        }).sort_values(by="Importance", ascending=False).head(top_n)

        # Convert pipeline-internal encoded names (e.g. "category_Elektronik",
        # "yorum_tfidf_harika") into plain-language labels so the chart reads
        # naturally for a non-technical viewer — the raw "Feature" column is
        # kept for anything downstream that still needs the exact name.
        fi_df["FeatureLabel"] = fi_df["Feature"].apply(lambda f: humanize_feature_name(f, col_types))

        # Plotly chart
        fig = px.bar(
            fi_df.sort_values(by="Importance", ascending=True),
            x="Importance",
            y="FeatureLabel",
            orientation="h",
            title=f"🏆 En Etkili {top_n} Ürün Başarı Faktörü (Feature Importance)",
            color="Importance",
            color_continuous_scale="Viridis",
            labels={"FeatureLabel": "Özellik", "Importance": "Etki Skoru"}
        )
        fig.update_layout(template=template, margin=dict(l=20, r=20, t=50, b=20))

        return fi_df, fig

    @staticmethod
    def explain_single_prediction(
        pipeline: Pipeline,
        input_row_df: pd.DataFrame,
        task_type: str = "classification",
        col_types: Optional[Dict[str, List[str]]] = None
    ) -> Dict[str, Any]:
        """
        Computes prediction and extracts top contributing features for a single product.
        """
        pred = pipeline.predict(input_row_df)[0]
        prob = None

        if task_type == "classification":
            if hasattr(pipeline.named_steps["model"], "predict_proba"):
                try:
                    proba_arr = pipeline.predict_proba(input_row_df)[0]
                    if len(proba_arr) == 2:
                        # proba_arr[0] = P(class=0), proba_arr[1] = P(class=1).
                        # The gauge always shows P(success=1), regardless of which
                        # class was actually predicted — that's the "success
                        # probability" the rest of the UI is built around.
                        prob = float(proba_arr[1])
                    else:
                        # Multi-class: use max class probability
                        classes = pipeline.named_steps["model"].classes_
                        pred_class_idx = list(classes).index(pred) if pred in classes else int(np.argmax(proba_arr))
                        prob = float(proba_arr[pred_class_idx])
                except Exception:
                    prob = None
            # If no predict_proba, derive from binary prediction (0 → ~25%, 1 → ~75%)
            if prob is None:
                try:
                    pred_int = int(float(pred))
                    prob = 0.75 if pred_int == 1 else 0.25
                except Exception:
                    prob = 0.5
        else:
            # Regression: probability field unused, keep None
            prob = None

        # Feature contribution heuristic
        model = pipeline.named_steps["model"]
        preprocessor = pipeline.named_steps["preprocessor"]
        
        try:
            X_proc = preprocessor.transform(input_row_df)
            feat_names = PreprocessingBuilder.get_feature_names_from_preprocessor(preprocessor)
            
            if hasattr(model, "feature_importances_"):
                weights = model.feature_importances_
            elif hasattr(model, "coef_"):
                weights = np.abs(model.coef_).flatten()
            else:
                weights = np.ones(X_proc.shape[1])

            min_len = min(len(feat_names), len(weights), X_proc.shape[1])
            row_vals = X_proc[0, :min_len]
            if hasattr(row_vals, "toarray"):
                row_vals = row_vals.toarray().flatten()

            impact = row_vals * weights[:min_len]

            impact_df = pd.DataFrame({
                "Feature": feat_names[:min_len],
                "Impact": impact
            }).sort_values(by="Impact", key=abs, ascending=False).head(8)

            # Add a human-readable label alongside the raw encoded name — the
            # raw "Feature" stays available for logic that categorizes by
            # keyword (e.g. schematics.py's Sankey grouping), while every
            # place this reaches the screen (driver chips, LLM prompt, etc.)
            # should prefer "FeatureLabel".
            impact_df["FeatureLabel"] = impact_df["Feature"].apply(lambda f: humanize_feature_name(f, col_types))

            top_drivers = impact_df.to_dict(orient="records")
        except Exception:
            top_drivers = []

        return {
            "prediction": pred,
            "probability": prob,
            "top_drivers": top_drivers
        }

    @staticmethod
    def simulate_what_if(
        pipeline: Pipeline,
        base_product_df: pd.DataFrame,
        feature_to_vary: str,
        variation_range: List[Any],
        task_type: str = "classification",
        template: str = "plotly_dark"
    ) -> Tuple[pd.DataFrame, go.Figure]:
        """
        Simulates how changing one feature alters the predicted success probability or score.
        """
        results = []
        base_row = base_product_df.copy()

        for val in variation_range:
            row_variant = base_row.copy()
            row_variant[feature_to_vary] = val
            
            pred = pipeline.predict(row_variant)[0]
            prob = 0.0
            if task_type == "classification" and hasattr(pipeline.named_steps["model"], "predict_proba"):
                try:
                    prob = float(pipeline.predict_proba(row_variant)[0][1]) * 100
                except Exception:
                    prob = float(pred) * 100
            else:
                prob = float(pred)

            results.append({
                feature_to_vary: val,
                "Success Rate / Score": round(prob, 2),
                "Prediction": pred
            })

        sim_df = pd.DataFrame(results)

        # Plotly chart
        is_num = all(isinstance(v, (int, float, np.number)) for v in variation_range)
        if is_num:
            fig = px.line(
                sim_df,
                x=feature_to_vary,
                y="Success Rate / Score",
                markers=True,
                title=f"🔮 'What-If' Simülasyonu: '{feature_to_vary}' Değişiminin Başarıya Etkisi",
                color_discrete_sequence=["#10B981"]
            )
        else:
            fig = px.bar(
                sim_df,
                x=feature_to_vary,
                y="Success Rate / Score",
                title=f"🔮 'What-If' Simülasyonu: '{feature_to_vary}' Seçeneklerinin Başarıya Etkisi",
                color="Success Rate / Score",
                color_continuous_scale="Tealgrn"
            )

        fig.update_layout(template=template, margin=dict(l=20, r=20, t=50, b=20))
        return sim_df, fig
