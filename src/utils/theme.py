"""
Theme Module
Provides Dark/Light surface themes AND a set of selectable accent color
palettes for the Streamlit UI, plus the Plotly template name matching each
surface theme, so charts and custom HTML cards stay visually consistent with
whatever is active.

Why this file looks the way it does:
Streamlit's own widget chrome (the app background, the sidebar, buttons,
text inputs, selectboxes, dataframes, tabs...) is NOT plain CSS you can
override by styling a handful of custom classes — it's rendered by
Streamlit's frontend with its own computed styles. The only reliable way to
re-skin it from injected CSS is to target the stable `data-testid`/
`data-baseweb` attributes Streamlit always attaches to these elements, with
`!important` so our rule wins over the computed style, and to reach every
*descendant* of an input/select (`... *`) rather than only its direct
child — the actual visible text often sits two or three `<span>`s deep
inside a baseweb component, which is exactly why a narrower selector can
still show unreadable, uncolored (effectively gray-on-gray) text even after
the box itself is themed.

Two independent choices compose into the final look:
- `theme` ("dark" / "light") — the surface: page background, sidebar,
  cards, borders, and body text color.
- `accent` (one of ACCENT_PALETTES) — the signature color: the header
  gradient, primary buttons, driver chips, active tab indicator, and card
  borders. Six curated palettes are offered instead of a full custom color
  picker so every combination still looks intentional and readable in both
  themes; a user who doesn't want to think about it can also pick "auto",
  which the app resolves to a different palette each login.
"""

from typing import Dict, Tuple

PLOTLY_TEMPLATE = {
    "dark": "plotly_dark",
    "light": "plotly_white",
}

# ---------------------------------------------------------------------------
# Accent palettes. `gradient` drives the header/button gradient (3 stops).
# `rgb` is the primary stop as "r,g,b", used to build translucent tints
# (chip backgrounds, card borders, tab highlight) at different alpha levels
# per surface theme. `chip_color_dark` / `chip_color_light` are pre-picked
# solid colors for text-on-tint contexts (driver chips, active tab label) —
# handwritten per theme rather than computed, because a hue that reads
# clearly on a near-black background (e.g. raw gold) often reads poorly on
# white and needs a darker sibling shade to stay legible there.
# ---------------------------------------------------------------------------
ACCENT_PALETTES: Dict[str, Dict[str, object]] = {
    "indigo": {
        "gradient": ("#6366f1", "#a855f7", "#ec4899"),
        "rgb": "99,102,241",
        "chip_color_dark": "#c4b5fd",
        "chip_color_light": "#4f46e5",
    },
    "emerald": {
        "gradient": ("#10b981", "#14b8a6", "#06b6d4"),
        "rgb": "16,185,129",
        "chip_color_dark": "#6ee7b7",
        "chip_color_light": "#047857",
    },
    "ocean": {
        "gradient": ("#0ea5e9", "#6366f1", "#8b5cf6"),
        "rgb": "14,165,233",
        "chip_color_dark": "#7dd3fc",
        "chip_color_light": "#0369a1",
    },
    "sunset": {
        "gradient": ("#f59e0b", "#f97316", "#ef4444"),
        "rgb": "249,115,22",
        "chip_color_dark": "#fdba74",
        "chip_color_light": "#c2410c",
    },
    "gold": {
        "gradient": ("#eab308", "#f59e0b", "#fb923c"),
        "rgb": "234,179,8",
        "chip_color_dark": "#fde047",
        "chip_color_light": "#a16207",
    },
    "berry": {
        "gradient": ("#a855f7", "#d946ef", "#f43f5e"),
        "rgb": "217,70,239",
        "chip_color_dark": "#f0abfc",
        "chip_color_light": "#a21caf",
    },
}
ACCENT_KEYS = list(ACCENT_PALETTES.keys())
DEFAULT_ACCENT = "indigo"


def get_theme_css(theme: str = "dark", accent: str = DEFAULT_ACCENT) -> str:
    theme = theme if theme in ("dark", "light") else "dark"
    accent = accent if accent in ACCENT_PALETTES else DEFAULT_ACCENT
    palette = ACCENT_PALETTES[accent]
    g1, g2, g3 = palette["gradient"]
    rgb = palette["rgb"]
    chip_color = palette["chip_color_dark" if theme == "dark" else "chip_color_light"]

    bg_alpha = 0.16 if theme == "dark" else 0.10
    border_alpha = 0.34 if theme == "dark" else 0.26
    card_border_alpha = 0.28 if theme == "dark" else 0.20
    guide_alpha = 0.12 if theme == "dark" else 0.08

    fmt_vars = dict(_THEME_VARS[theme])
    fmt_vars.update(
        gradient_1=g1, gradient_2=g2, gradient_3=g3,
        accent_solid=g1,
        driver_chip_bg=f"rgba({rgb},{bg_alpha})",
        driver_chip_border=f"rgba({rgb},{border_alpha})",
        driver_chip_color=chip_color,
        metric_card_border=f"rgba({rgb},{card_border_alpha})",
        llm_card_border=f"rgba({rgb},{border_alpha})",
        guide_tip_bg=f"rgba({rgb},{guide_alpha})",
        tab_active_border=g1,
    )
    return _BASE_CSS.format(**fmt_vars)


_BASE_CSS = """
<style>
    /* =====================================================================
       1) BASE TYPOGRAPHY — scoped to text containers, not every internal
          Streamlit classname, so widget layouts never get distorted.
       ===================================================================== */
    .stApp, .stApp p, .stApp li, .stApp label,
    [data-testid="stMarkdownContainer"], [data-testid="stMarkdownContainer"] p,
    [data-testid="stWidgetLabel"] p, [data-testid="stCaptionContainer"] {{
        font-size: 16px;
        line-height: 1.55;
    }}

    /* =====================================================================
       2) NATIVE STREAMLIT CHROME — this is the part that actually makes
          Dark/Light switch visible across the whole app, not just custom
          cards. Every rule targets a stable data-testid/data-baseweb hook,
          and reaches every descendant (` *`) of inputs/selects so a value
          rendered by a nested span never falls back to Streamlit's own
          (low-contrast, effectively gray-on-gray) default text color.
       ===================================================================== */
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {{
        background-color: {app_bg} !important;
        color: {text_color} !important;
    }}
    [data-testid="stHeader"] {{
        background-color: {app_bg} !important;
    }}
    [data-testid="stSidebar"], [data-testid="stSidebarContent"] {{
        background-color: {sidebar_bg} !important;
        border-right: 1px solid {border_color};
    }}
    [data-testid="stSidebar"] * {{
        color: {text_color} !important;
    }}

    /* Headings & body text */
    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6,
    [data-testid="stMarkdownContainer"] h1, [data-testid="stMarkdownContainer"] h2,
    [data-testid="stMarkdownContainer"] h3, [data-testid="stMarkdownContainer"] h4 {{
        color: {heading_color} !important;
    }}
    [data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li,
    [data-testid="stWidgetLabel"] p, [data-testid="stCaptionContainer"] {{
        color: {text_color} !important;
    }}
    [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] * {{
        color: {text_secondary_color} !important;
        opacity: 1 !important;
    }}

    /* Inputs: text, number, textarea, selectbox, multiselect — every
       descendant is covered so a value/placeholder span nested inside a
       baseweb component always inherits a readable color, never the
       library's own low-contrast default. */
    .stTextInput input, .stNumberInput input, .stTextArea textarea,
    [data-baseweb="input"], [data-baseweb="input"] *,
    [data-baseweb="base-input"], [data-baseweb="base-input"] *,
    [data-baseweb="select"], [data-baseweb="select"] > div, [data-baseweb="select"] *,
    [data-baseweb="textarea"] {{
        background-color: transparent;
        color: {input_text} !important;
    }}
    [data-baseweb="input"], [data-baseweb="base-input"],
    [data-baseweb="select"] > div, [data-baseweb="textarea"],
    .stTextInput input, .stNumberInput input, .stTextArea textarea {{
        background-color: {input_bg} !important;
        border-color: {input_border} !important;
    }}
    [data-testid="stMultiSelect"] span, [data-baseweb="tag"], [data-baseweb="tag"] * {{
        background-color: {chip_bg} !important;
        color: {input_text} !important;
    }}
    input::placeholder, textarea::placeholder {{
        color: {text_secondary_color} !important;
        opacity: 1 !important;
    }}

    /* Buttons (regular, form-submit, download) */
    .stButton button, .stFormSubmitButton button, .stDownloadButton button {{
        background-color: {button_bg} !important;
        color: {button_text} !important;
        border: 1px solid {button_border} !important;
    }}
    .stButton button *, .stFormSubmitButton button *, .stDownloadButton button * {{
        color: inherit !important;
    }}
    .stButton button:hover, .stFormSubmitButton button:hover, .stDownloadButton button:hover {{
        background-color: {button_hover_bg} !important;
        border-color: {accent_solid} !important;
        color: {button_hover_text} !important;
    }}
    .stButton button[kind="primary"], .stFormSubmitButton button[kind="primary"] {{
        background: linear-gradient(135deg, {gradient_1} 0%, {gradient_2} 100%) !important;
        color: #ffffff !important;
        border: none !important;
    }}

    /* Checkboxes / radio / slider labels */
    [data-testid="stCheckbox"] label p, [data-testid="stRadio"] label p,
    [data-testid="stSlider"] label p {{
        color: {text_color} !important;
    }}

    /* Expander */
    [data-testid="stExpander"] {{
        background-color: {card_bg} !important;
        border: 1px solid {border_color} !important;
        border-radius: 10px;
    }}
    [data-testid="stExpander"] summary, [data-testid="stExpander"] summary * {{
        color: {text_color} !important;
    }}

    /* DataFrame / table */
    [data-testid="stDataFrame"], [data-testid="stTable"] {{
        background-color: {card_bg} !important;
        border: 1px solid {border_color} !important;
    }}

    /* Alerts (info/success/warning/error) keep Streamlit's own accent hue
       but adopt the theme's surface + text color so they don't look like a
       mismatched light-mode box on a dark page or vice versa. */
    [data-testid="stAlert"] {{
        background-color: {card_bg} !important;
        color: {text_color} !important;
    }}
    [data-testid="stAlertContentInfo"] *, [data-testid="stAlertContentSuccess"] *,
    [data-testid="stAlertContentWarning"] *, [data-testid="stAlertContentError"] * {{
        color: {text_color} !important;
    }}

    /* Divider */
    hr {{
        border-color: {border_color} !important;
    }}

    /* Metric widget (st.metric, if used) */
    [data-testid="stMetric"] {{
        background-color: {card_bg} !important;
        border: 1px solid {border_color};
        border-radius: 10px;
        padding: 10px;
    }}
    [data-testid="stMetricLabel"] p {{ color: {text_secondary_color} !important; }}
    [data-testid="stMetricValue"] {{ color: {text_color} !important; }}

    /* File uploader dropzone */
    [data-testid="stFileUploaderDropzone"] {{
        background-color: {card_bg} !important;
        border: 1px dashed {border_color} !important;
    }}
    [data-testid="stFileUploaderDropzone"] * {{
        color: {text_color} !important;
    }}

    /* =====================================================================
       3) APP-SPECIFIC CUSTOM CLASSES — the header gradient, chips, tab
          highlight and card borders below all follow the chosen accent
          palette (gradient_1/2/3, accent_solid, driver_chip_*).
       ===================================================================== */

    /* ── Gradient başlık ── */
    .main-header {{
        font-size: 2.3rem;
        font-weight: 800;
        background: linear-gradient(135deg, {gradient_1} 0%, {gradient_2} 50%, {gradient_3} 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.15rem;
        letter-spacing: -0.5px;
    }}
    .sub-header {{
        color: {sub_header_color} !important;
        font-size: 1.08rem;
        margin-bottom: 1.5rem;
        font-style: italic;
        line-height: 1.6;
    }}

    /* ── Metrik kartı ── */
    .metric-card {{
        background: {metric_card_bg};
        border: 1px solid {metric_card_border};
        border-radius: 14px;
        padding: 18px 16px;
        text-align: center;
        box-shadow: {metric_card_shadow};
    }}
    .metric-title {{
        color: {metric_title_color} !important;
        font-size: 0.8rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }}
    .metric-value {{
        color: {metric_value_color} !important;
        font-size: 1.8rem;
        font-weight: 800;
        margin-top: 6px;
    }}

    /* ── Başarı / başarısız badge (semantic — always green/amber/red,
         independent of the chosen accent, since they signal a result) ── */
    .success-badge {{
        background: linear-gradient(135deg, rgba(16,185,129,0.18), rgba(5,150,105,0.12));
        color: #10b981 !important;
        border: 1px solid rgba(16,185,129,0.4);
        padding: 8px 20px;
        border-radius: 40px;
        font-weight: 700;
        font-size: 0.98rem;
        display: inline-block;
        box-shadow: 0 0 16px rgba(16,185,129,0.15);
        letter-spacing: 0.03em;
    }}
    .warning-badge {{
        background: linear-gradient(135deg, rgba(245,158,11,0.18), rgba(217,119,6,0.12));
        color: #d97706 !important;
        border: 1px solid rgba(217,119,6,0.4);
        padding: 8px 20px;
        border-radius: 40px;
        font-weight: 700;
        font-size: 0.98rem;
        display: inline-block;
        box-shadow: 0 0 16px rgba(251,191,36,0.12);
        letter-spacing: 0.03em;
    }}
    .danger-badge {{
        background: linear-gradient(135deg, rgba(239,68,68,0.18), rgba(220,38,38,0.12));
        color: #dc2626 !important;
        border: 1px solid rgba(220,38,38,0.4);
        padding: 8px 20px;
        border-radius: 40px;
        font-weight: 700;
        font-size: 0.98rem;
        display: inline-block;
        box-shadow: 0 0 16px rgba(248,113,113,0.12);
        letter-spacing: 0.03em;
    }}

    /* ── LLM analiz kutusu ── */
    .llm-card {{
        background: {llm_card_bg};
        border: 1px solid {llm_card_border};
        border-radius: 16px;
        padding: 26px 30px;
        box-shadow: {llm_card_shadow};
        margin-top: 8px;
        line-height: 1.65;
        color: {text_color} !important;
    }}
    .llm-card * {{
        color: {text_color} !important;
    }}

    /* ── Olasılık progress bar (semantic colors, not accent-driven) ── */
    .prob-bar-wrap {{
        background: {prob_bar_wrap_bg};
        border-radius: 8px;
        height: 14px;
        margin: 6px 0 2px 0;
        overflow: hidden;
    }}
    .prob-bar-fill-green {{
        height: 100%;
        border-radius: 8px;
        background: linear-gradient(90deg, #10b981, #34d399);
        transition: width 0.6s ease;
    }}
    .prob-bar-fill-amber {{
        height: 100%;
        border-radius: 8px;
        background: linear-gradient(90deg, #f59e0b, #fbbf24);
    }}
    .prob-bar-fill-red {{
        height: 100%;
        border-radius: 8px;
        background: linear-gradient(90deg, #ef4444, #f87171);
    }}

    /* ── Driver faktör chip (accent-driven) ── */
    .driver-chip {{
        display: inline-block;
        background: {driver_chip_bg};
        border: 1px solid {driver_chip_border};
        border-radius: 20px;
        padding: 4px 13px;
        font-size: 0.85rem;
        color: {driver_chip_color} !important;
        margin: 3px 4px 3px 0;
    }}

    /* ── Onboarding kılavuz kutusu (accent-driven) ── */
    .guide-tip {{
        background: {guide_tip_bg};
        border-left: 4px solid {accent_solid};
        border-radius: 8px;
        padding: 10px 16px;
        margin-top: 10px;
        font-size: 0.95rem;
        color: {text_color} !important;
    }}

    /* ── Tab listesi (active tab underlined in the accent color) ── */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 6px;
        background: {tab_list_bg};
        border-radius: 12px;
        padding: 4px;
    }}
    .stTabs [data-baseweb="tab"] {{
        padding: 10px 18px;
        border-radius: 9px;
        font-weight: 600;
        font-size: 0.92rem;
        color: {text_secondary_color} !important;
    }}
    .stTabs [aria-selected="true"] {{
        color: {text_color} !important;
        border-bottom: 3px solid {tab_active_border} !important;
    }}
</style>
"""

_THEME_VARS: Dict[str, Dict[str, str]] = {
    "dark": dict(
        app_bg="#0e1117",
        sidebar_bg="#161b29",
        text_color="#e2e8f0",
        text_secondary_color="#a8b5c7",
        heading_color="#f1f5f9",
        border_color="rgba(255,255,255,0.12)",
        input_bg="#1e293b",
        input_text="#f1f5f9",
        input_border="rgba(255,255,255,0.18)",
        chip_bg="#334155",
        button_bg="#1e293b",
        button_text="#e2e8f0",
        button_border="rgba(255,255,255,0.18)",
        button_hover_bg="#293548",
        button_hover_text="#ffffff",
        card_bg="#161b29",
        sub_header_color="#b7c2d4",
        metric_card_bg="linear-gradient(135deg, #1a1f35 0%, #1e293b 100%)",
        metric_card_shadow="0 4px 24px rgba(0,0,0,0.18), 0 0 0 1px rgba(255,255,255,0.03)",
        metric_title_color="#a8b5c7",
        metric_value_color="#e2e8f0",
        llm_card_bg="linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%)",
        llm_card_shadow="0 8px 32px rgba(139,92,246,0.12), 0 0 0 1px rgba(255,255,255,0.04)",
        prob_bar_wrap_bg="rgba(255,255,255,0.07)",
        tab_list_bg="rgba(255,255,255,0.03)",
    ),
    "light": dict(
        app_bg="#ffffff",
        sidebar_bg="#f8fafc",
        text_color="#1e1b3a",
        text_secondary_color="#44444b",
        heading_color="#0f172a",
        border_color="rgba(15,23,42,0.10)",
        input_bg="#ffffff",
        input_text="#1e1b3a",
        input_border="#cbd5e1",
        chip_bg="#eef2ff",
        button_bg="#ffffff",
        button_text="#1e1b3a",
        button_border="#cbd5e1",
        button_hover_bg="#f1f5f9",
        button_hover_text="#1e1b3a",
        card_bg="#ffffff",
        sub_header_color="#3f3f46",
        metric_card_bg="linear-gradient(135deg, #ffffff 0%, #f4f4f8 100%)",
        metric_card_shadow="0 2px 14px rgba(30,41,59,0.08), 0 0 0 1px rgba(15,23,42,0.02)",
        metric_title_color="#475569",
        metric_value_color="#1e1b3a",
        llm_card_bg="linear-gradient(135deg, #f5f3ff 0%, #eef2ff 50%, #eff6ff 100%)",
        llm_card_shadow="0 4px 20px rgba(99,92,246,0.10)",
        prob_bar_wrap_bg="rgba(15,23,42,0.08)",
        tab_list_bg="rgba(15,23,42,0.04)",
    ),
}
