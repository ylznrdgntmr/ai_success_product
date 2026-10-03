"""
Automated Exploratory Data Analysis (EDA) Module
Generates statistical insights, health audits, and rich Plotly visualizations.
"""

from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


class AutoEDA:
    """Automated Exploratory Data Analysis and Visualization Engine."""

    @staticmethod
    def plot_target_distribution(df: pd.DataFrame, target_col: str, task_type: str, template: str = "plotly_dark") -> go.Figure:
        """Visualizes target variable distribution."""
        series = df[target_col].dropna()

        if task_type == "classification":
            counts = series.value_counts().reset_index()
            counts.columns = ["Class", "Count"]
            counts["Percentage"] = (counts["Count"] / counts["Count"].sum() * 100).round(1)
            counts["Label"] = counts["Class"].astype(str) + " (" + counts["Percentage"].astype(str) + "%)"

            fig = px.pie(
                counts,
                values="Count",
                names="Label",
                title=f"🎯 Hedef Değişken Dağılımı ({target_col})",
                hole=0.45,
                color_discrete_sequence=px.colors.qualitative.Prism
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
        else:
            fig = px.histogram(
                df,
                x=target_col,
                marginal="box",
                title=f"🎯 Sürekli Hedef Değişken Dağılımı ({target_col})",
                color_discrete_sequence=["#6366F1"],
                nbins=30
            )
            fig.update_layout(bargap=0.1)

        fig.update_layout(template=template, margin=dict(l=20, r=20, t=50, b=20))
        return fig

    @staticmethod
    def plot_correlation_heatmap(df: pd.DataFrame, numeric_cols: List[str], target_col: Optional[str] = None, template: str = "plotly_dark") -> Optional[go.Figure]:
        """Plots correlation matrix heatmap for numeric features."""
        cols = list(numeric_cols)
        if target_col and target_col in df.columns and pd.api.types.is_numeric_dtype(df[target_col]):
            if target_col not in cols:
                cols.append(target_col)

        if len(cols) < 2:
            return None

        corr = df[cols].corr().round(2)

        # A diverging color scale centered on zero (like a spreadsheet's
        # conditional-format color scale) makes positive vs. negative
        # correlations instantly distinguishable by color, not just by the
        # printed sign — a plain sequential palette (the old "Viridis")
        # makes a strong negative correlation look the same shade as "no
        # relationship", which reads as broken/unclear at a glance.
        fig = px.imshow(
            corr,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="RdBu_r",
            range_color=[-1, 1],
            color_continuous_midpoint=0,
            title="🔥 Sayısal Özellikler Korelasyon Isı Haritası"
        )
        fig.update_layout(template=template, margin=dict(l=20, r=20, t=50, b=20))
        return fig

    @staticmethod
    def plot_feature_vs_target(
        df: pd.DataFrame,
        feature_col: str,
        target_col: str,
        task_type: str = "classification",
        template: str = "plotly_dark"
    ) -> go.Figure:
        """Visualizes relationship between a selected feature and the target."""
        clean_df = df[[feature_col, target_col]].dropna()

        is_feat_num = pd.api.types.is_numeric_dtype(clean_df[feature_col])

        if task_type == "classification":
            if is_feat_num:
                # Box plot or Violin
                fig = px.box(
                    clean_df,
                    x=target_col,
                    y=feature_col,
                    color=target_col,
                    title=f"📊 '{feature_col}' Değerinin '{target_col}' Sınıfına Göre Dağılımı",
                    color_discrete_sequence=px.colors.qualitative.Set2
                )
            else:
                # Grouped Bar Chart
                cross_tab = pd.crosstab(clean_df[feature_col], clean_df[target_col], normalize="index") * 100
                cross_tab = cross_tab.reset_index().melt(id_vars=feature_col, var_name=target_col, value_name="Percentage")
                
                # Keep top 15 categories to avoid clutter
                top_cats = clean_df[feature_col].value_counts().head(15).index
                cross_tab = cross_tab[cross_tab[feature_col].isin(top_cats)]

                fig = px.bar(
                    cross_tab,
                    x=feature_col,
                    y="Percentage",
                    color=target_col,
                    barmode="stack",
                    title=f"📊 '{feature_col}' Kategorilerine Göre '{target_col}' Başarı Oranı (%)",
                    color_discrete_sequence=px.colors.qualitative.Set2
                )
        else:
            # Regression
            if is_feat_num:
                fig = px.scatter(
                    clean_df,
                    x=feature_col,
                    y=target_col,
                    trendline="ols",
                    title=f"📈 '{feature_col}' vs '{target_col}' Regresyon Eğilimi",
                    color_discrete_sequence=["#38BDF8"]
                )
            else:
                top_cats = clean_df[feature_col].value_counts().head(12).index
                filtered_df = clean_df[clean_df[feature_col].isin(top_cats)]
                fig = px.box(
                    filtered_df,
                    x=feature_col,
                    y=target_col,
                    title=f"📊 '{feature_col}' Kategorilerine Göre '{target_col}' Dağılımı",
                    color=feature_col
                )

        fig.update_layout(template=template, margin=dict(l=20, r=20, t=50, b=20))
        return fig

    @staticmethod
    def plot_missing_values(df: pd.DataFrame, template: str = "plotly_dark") -> Optional[go.Figure]:
        """Plots missing value percentages per column."""
        missing = (df.isnull().sum() / len(df) * 100).round(1)
        missing = missing[missing > 0].reset_index()
        missing.columns = ["Column", "MissingPct"]

        if missing.empty:
            return None

        missing = missing.sort_values(by="MissingPct", ascending=True)

        fig = px.bar(
            missing,
            x="MissingPct",
            y="Column",
            orientation="h",
            title="⚠️ Eksik Değer Oranları (%)",
            color="MissingPct",
            color_continuous_scale="Reds"
        )
        fig.update_layout(template=template, margin=dict(l=20, r=20, t=50, b=20))
        return fig

    @staticmethod
    def compute_quick_insights(df: pd.DataFrame, target_col: str, task_type: str) -> List[str]:
        """Generates quick textual analytical bullet points about the dataset."""
        insights = []
        n_rows, n_cols = df.shape
        insights.append(f"Veri setinde toplam **{n_rows:,} satır** ve **{n_cols} sütun** bulunuyor.")

        # Missing analysis
        missing_total = df.isnull().sum().sum()
        if missing_total == 0:
            insights.append("✅ Veri setinde hiçbir eksik veri bulunmuyor, veri sağlığı mükemmel.")
        else:
            missing_cols_count = (df.isnull().sum() > 0).sum()
            insights.append(f"⚠️ {missing_cols_count} sütunda toplam **{missing_total:,} eksik değer** tespit edildi.")

        # Target analysis
        if target_col in df.columns:
            if task_type == "classification":
                val_counts = df[target_col].value_counts(normalize=True)
                majority_class = val_counts.index[0]
                majority_pct = val_counts.iloc[0] * 100
                insights.append(f"🎯 Hedef sütun '{target_col}' dağılımı: En yaygın sınıf **'{majority_class}' (%{majority_pct:.1f})**.")
                if majority_pct > 75:
                    insights.append("⚖️ Sınıf dengesizliği yüksek (Imbalance). Model eğitiminde dengeleme teknikleri uygulanacaktır.")
            else:
                mean_val = df[target_col].mean()
                median_val = df[target_col].median()
                insights.append(f"🎯 Hedef sütun '{target_col}' ortalaması **{mean_val:.2f}**, medyanı **{median_val:.2f}**.")

        # Top correlated feature
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_col in num_cols and len(num_cols) > 1:
            corrs = df[num_cols].corr()[target_col].drop(target_col).abs().sort_values(ascending=False)
            if not corrs.empty:
                top_corr_feat = corrs.index[0]
                top_corr_val = corrs.iloc[0]
                insights.append(f"🔗 Hedef ile en güçlü doğrusal ilişkiye sahip özellik: **'{top_corr_feat}' (Korelasyon: {top_corr_val:.2f})**.")

        return insights
