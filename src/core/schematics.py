"""
Recommendation Schematics Module
Turns the model's top decision factors and the AI advisor's recommendation
structure into two deterministic, always-available diagrams — independent
of whatever free-form text a live LLM returns, so they render identically
whether the heuristic advisor or a live LLM produced the analysis:

1. A Sankey diagram: top decision factors → prediction outcome →
   recommendation categories (pricing / content / marketing / trust).
2. A 90-day roadmap timeline (3 phases).
"""

from typing import Any, Dict, List
import plotly.graph_objects as go

from src.utils.i18n import t

_CATEGORY_KEYWORDS = {
    "pricing": ["price", "fiyat", "discount", "indirim", "goal", "cost", "maliyet"],
    "content": ["title", "desc", "name", "başlık", "açıklama", "content", "image", "photo", "video"],
    "marketing": ["ad", "marketing", "reklam", "budget", "bütçe", "platform", "channel", "influencer"],
    "trust": ["rating", "review", "puan", "yorum", "prime", "trust", "certificate", "community"],
}


def _categorize_driver(feature_name: str) -> str:
    name_low = feature_name.lower()
    for category, keywords in _CATEGORY_KEYWORDS.items():
        if any(kw in name_low for kw in keywords):
            return category
    return "marketing"  # sensible default bucket for anything unmatched


def build_recommendation_sankey(
    top_drivers: List[Dict[str, Any]],
    is_high_success: bool,
    plotly_template: str = "plotly_dark",
    accent_gradient: tuple = ("#6366f1", "#a855f7", "#ec4899"),
) -> go.Figure:
    """
    Builds a Sankey diagram: top factors -> prediction -> recommendation
    categories, weighted by each driver's absolute impact.

    `accent_gradient` ties the diagram's driver/category node colors to the
    user's chosen accent palette (src/utils/theme.py) so it matches the rest
    of the app rather than always showing a fixed indigo/purple/pink scheme.
    """
    drivers = top_drivers[:5] if top_drivers else []

    category_keys = ["pricing", "content", "marketing", "trust"]
    category_labels = {
        "pricing": t("sankey_cat_pricing"),
        "content": t("sankey_cat_content"),
        "marketing": t("sankey_cat_marketing"),
        "trust": t("sankey_cat_trust"),
    }
    category_weights = {k: 0.0 for k in category_keys}

    driver_labels = []
    driver_values = []
    for d in drivers:
        raw_feat = str(d.get("Feature", "Feature"))
        # Categorization runs on the raw, pipeline-internal name (its
        # keywords like "price"/"fiyat" match reliably there); the diagram
        # itself shows the human-readable label so a non-technical viewer
        # never sees an encoded name like "category_Elektronik".
        display_feat = str(d.get("FeatureLabel", raw_feat))[:32]
        impact = abs(float(d.get("Impact", 0.0))) or 0.01
        driver_labels.append(display_feat)
        driver_values.append(impact)
        category_weights[_categorize_driver(raw_feat)] += impact

    if not driver_labels:
        driver_labels = [t("t5_drivers_header").strip("*: ")]
        driver_values = [1.0]
        category_weights = {k: 1.0 for k in category_keys}

    if sum(category_weights.values()) <= 0:
        category_weights = {k: 1.0 for k in category_keys}

    prediction_label = t("sankey_node_prediction_success") if is_high_success else t("sankey_node_prediction_risk")
    prediction_color = "#10b981" if is_high_success else "#f59e0b"

    node_labels = driver_labels + [prediction_label] + [category_labels[k] for k in category_keys]
    n_drivers = len(driver_labels)
    prediction_idx = n_drivers
    category_start_idx = n_drivers + 1

    # Use the 3 accent stops for the driver/prediction/category nodes, plus
    # two fixed complementary tones for the remaining category nodes —
    # tolerant of a shorter tuple just in case a caller passes fewer stops.
    g1 = accent_gradient[0]
    g2 = accent_gradient[1] if len(accent_gradient) > 1 else accent_gradient[0]
    g3 = accent_gradient[2] if len(accent_gradient) > 2 else g2

    def _to_rgb_tuple(hex_color: str) -> str:
        h = hex_color.lstrip("#")
        try:
            r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
            return f"{r},{g},{b}"
        except Exception:
            return "99,102,241"

    rgb1 = _to_rgb_tuple(g1)
    rgb2 = _to_rgb_tuple(g2)

    node_colors = (
        [g1] * n_drivers
        + [prediction_color]
        + [g2, g3, "#38bdf8", "#f59e0b"]
    )

    sources, targets, values, link_colors = [], [], [], []
    total_driver_value = sum(driver_values) or 1.0
    for i, val in enumerate(driver_values):
        sources.append(i)
        targets.append(prediction_idx)
        values.append(val)
        link_colors.append(f"rgba({rgb1},0.35)")

    total_cat_weight = sum(category_weights.values()) or 1.0
    for j, cat_key in enumerate(category_keys):
        w = category_weights[cat_key]
        if w <= 0:
            continue
        sources.append(prediction_idx)
        targets.append(category_start_idx + j)
        values.append((w / total_cat_weight) * total_driver_value)
        link_colors.append(f"rgba({rgb2},0.30)")

    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=18,
            thickness=18,
            line=dict(color="rgba(0,0,0,0.15)", width=0.5),
            label=node_labels,
            color=node_colors,
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color=link_colors,
        )
    )])
    fig.update_layout(
        template=plotly_template,
        margin=dict(l=10, r=10, t=20, b=10),
        height=340,
        font=dict(size=13),
    )
    return fig


def build_roadmap_timeline(
    plotly_template: str = "plotly_dark",
    accent_gradient: tuple = ("#6366f1", "#a855f7", "#ec4899"),
) -> go.Figure:
    """Builds a simple 3-phase, 90-day horizontal roadmap timeline (deterministic, language-aware).
    `accent_gradient` ties the 3 phase colors to the user's chosen accent palette."""
    g1 = accent_gradient[0]
    g2 = accent_gradient[1] if len(accent_gradient) > 1 else g1
    g3 = accent_gradient[2] if len(accent_gradient) > 2 else g2
    phases = [
        {"label": t("t6_roadmap_phase1"), "start": 0, "duration": 30, "color": g1},
        {"label": t("t6_roadmap_phase2"), "start": 30, "duration": 30, "color": g2},
        {"label": t("t6_roadmap_phase3"), "start": 60, "duration": 30, "color": g3},
    ]

    fig = go.Figure()
    for phase in phases:
        fig.add_trace(go.Bar(
            x=[phase["duration"]],
            y=[phase["label"]],
            base=[phase["start"]],
            orientation="h",
            marker=dict(color=phase["color"]),
            text=f"{phase['start']}–{phase['start'] + phase['duration']}d",
            textposition="inside",
            insidetextanchor="middle",
            showlegend=False,
            hovertemplate=f"{phase['label']}: %{{base}}–{phase['start'] + phase['duration']} days<extra></extra>",
        ))

    fig.update_layout(
        template=plotly_template,
        barmode="stack",
        height=220,
        margin=dict(l=10, r=10, t=20, b=20),
        xaxis_title="Gün / Days",
        showlegend=False,
    )
    return fig
