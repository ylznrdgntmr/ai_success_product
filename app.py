"""
AI Product Success Predictor & Strategic Intelligence Studio
Universal Dataset Ingestion, AutoML Benchmark, Interactive Prediction & Multi-LLM Advisor
"""

import os
import time
import random
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.utils.sample_data import SAMPLE_DATASETS
from src.core.data_processor import DataProcessor
from src.core.eda import AutoEDA
from src.core.pipeline import PreprocessingBuilder
from src.core.models import ModelTrainer
from src.core.explainer import ModelExplainer
from src.core.schematics import build_recommendation_sankey, build_roadmap_timeline
from src.llm.advisor import AIProductAdvisor
from src.llm.providers import PROVIDERS
from src.utils.report_export import generate_markdown_report, generate_pdf_report
from src.core.image_features import ImageFeatureUnavailableError
from src.utils.i18n import t, get_lang, set_lang, LANGUAGES, EXTENDED_LANGUAGES, ALL_LANGUAGES, ensure_dynamic_language
from src.utils.theme import get_theme_css, PLOTLY_TEMPLATE, ACCENT_PALETTES, ACCENT_KEYS
from src.utils.auth import render_auth_gate, is_authenticated, current_user, logout, save_user_preferences
from src.utils.labels import metric_column_labels, prettify_column_name

# Columns whose values reflect a product's EXISTING traction (review counts,
# ratings, sales/backers to date, revenue, etc.) rather than something a
# not-yet-launched product actually has. Used both for the "New Product /
# Pre-Launch" training mode (Tab 4) and the cautionary note in Tab 5.
TRACTION_TERMS = ["review", "rating", "puan", "yorum", "sales", "satış", "satis",
                   "sold", "bought", "backers", "pledged", "mrr", "revenue"]

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="AI Product Success Predictor",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

if "lang" not in st.session_state:
    st.session_state["lang"] = "tr"
if "theme" not in st.session_state:
    st.session_state["theme"] = "dark"
if "accent_mode" not in st.session_state:
    # "auto" = the app picks a fresh palette every login ("kendini yeniler");
    # any other value pins a specific palette the user chose deliberately.
    st.session_state["accent_mode"] = "auto"
if "accent_resolved" not in st.session_state:
    st.session_state["accent_resolved"] = random.choice(ACCENT_KEYS)
# Seed each selectbox's OWN session_state key once, up front, so none of
# them ever need an `index=` argument below — Streamlit ignores `index`
# after a widget's first-ever render anyway, and warns if session_state is
# also set for that same key on top of it. Seeding here means the widget
# always finds its key already populated correctly.
if "lang_picker" not in st.session_state:
    st.session_state["lang_picker"] = st.session_state["lang"] if st.session_state["lang"] in LANGUAGES else "tr"
if "theme_picker" not in st.session_state:
    st.session_state["theme_picker"] = st.session_state["theme"]
if "accent_picker" not in st.session_state:
    st.session_state["accent_picker"] = st.session_state["accent_mode"]

# Apply a returning, already-authenticated user's saved look (theme/accent/
# language) BEFORE any widget below is instantiated. This must happen here,
# not after render_auth_gate() runs further down: `current_user()` is a
# plain session_state read (safe any time, renders nothing), but a
# `st.selectbox(..., key=...)` widget's session_state entry can only be
# pre-set BEFORE that widget is created in this run — Streamlit raises
# StreamlitWidgetAlreadyInstantiatedError if you try to set it afterward,
# and silently ignores `index=` as a way to "force" a value on any run
# after the widget's first one. Doing the restore this early keeps both
# rules satisfied.
_user_for_prefs = current_user()
if _user_for_prefs and st.session_state.get("_prefs_loaded_for") != _user_for_prefs.get("email"):
    _prefs = _user_for_prefs.get("preferences", {}) or {}
    if _prefs.get("theme") in ("dark", "light"):
        st.session_state["theme"] = _prefs["theme"]
        st.session_state["theme_picker"] = _prefs["theme"]
    if _prefs.get("accent_mode") in ACCENT_KEYS + ["auto"]:
        st.session_state["accent_mode"] = _prefs["accent_mode"]
        st.session_state["accent_picker"] = _prefs["accent_mode"]
        if _prefs["accent_mode"] == "auto":
            st.session_state["accent_resolved"] = random.choice(ACCENT_KEYS)
    if _prefs.get("lang") in ALL_LANGUAGES:
        set_lang(_prefs["lang"])
        if _prefs["lang"] in LANGUAGES:
            st.session_state["lang_picker"] = _prefs["lang"]
    st.session_state["_prefs_loaded_for"] = _user_for_prefs.get("email")

_active_accent = (
    st.session_state["accent_resolved"] if st.session_state["accent_mode"] == "auto"
    else st.session_state["accent_mode"]
)
st.markdown(get_theme_css(st.session_state["theme"], _active_accent), unsafe_allow_html=True)
PLOTLY_TPL = PLOTLY_TEMPLATE.get(st.session_state["theme"], "plotly_dark")
ACCENT_GRADIENT = ACCENT_PALETTES[_active_accent]["gradient"]


# ---------------------------------------------------------
# Sidebar: Language, Theme & Color Palette (always visible, even pre-login)
# ---------------------------------------------------------
with st.sidebar:
    lc1, lc2, lc3 = st.columns(3)
    with lc1:
        lang_keys = list(LANGUAGES.keys())
        current_lang = st.session_state["lang"]
        new_lang = st.selectbox(
            t("language_label"), options=lang_keys,
            format_func=lambda k: LANGUAGES[k],
            key="lang_picker"
        )
        if new_lang != current_lang:
            set_lang(new_lang)
            if is_authenticated():
                save_user_preferences(current_user()["email"], {"lang": new_lang})
            st.rerun()
    with lc2:
        theme_options = ["dark", "light"]
        theme_labels = {"dark": t("theme_dark"), "light": t("theme_light")}
        new_theme = st.selectbox(
            t("theme_label"), options=theme_options,
            format_func=lambda k: theme_labels[k],
            key="theme_picker"
        )
        if new_theme != st.session_state["theme"]:
            st.session_state["theme"] = new_theme
            if is_authenticated():
                save_user_preferences(current_user()["email"], {"theme": new_theme})
            st.rerun()
    with lc3:
        accent_options = ACCENT_KEYS + ["auto"]
        accent_labels = {k: t(f"accent_{k}") for k in ACCENT_KEYS}
        accent_labels["auto"] = t("accent_auto")
        current_accent_mode = st.session_state["accent_mode"]
        new_accent_mode = st.selectbox(
            t("accent_label"), options=accent_options,
            format_func=lambda k: accent_labels[k],
            key="accent_picker",
            help=t("accent_help"),
        )
        if new_accent_mode != current_accent_mode:
            st.session_state["accent_mode"] = new_accent_mode
            if new_accent_mode == "auto":
                st.session_state["accent_resolved"] = random.choice(ACCENT_KEYS)
            if is_authenticated():
                save_user_preferences(current_user()["email"], {"accent_mode": new_accent_mode})
            st.rerun()
    if current_lang not in lang_keys:
        st.caption(t("active_extended_lang_note", lang=ALL_LANGUAGES.get(current_lang, current_lang)))
    if st.session_state["accent_mode"] == "auto":
        if st.button(t("accent_shuffle_btn"), key="accent_shuffle_btn", width="stretch"):
            # Never re-pick the same palette so the click always visibly changes something.
            choices = [k for k in ACCENT_KEYS if k != st.session_state["accent_resolved"]] or ACCENT_KEYS
            st.session_state["accent_resolved"] = random.choice(choices)
            st.rerun()
    st.markdown("---")


# ---------------------------------------------------------
# Authentication Gate
# ---------------------------------------------------------
if not render_auth_gate():
    st.stop()

with st.sidebar:
    user = current_user() or {}
    st.caption(t("logged_in_as", name=user.get("name", user.get("email", "?"))))
    if st.button(t("logout_btn"), width="stretch"):
        logout()
        st.rerun()
    st.markdown("---")


# ---------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------
if "df" not in st.session_state:
    default_ds_key = list(SAMPLE_DATASETS.keys())[0]
    st.session_state["df"] = SAMPLE_DATASETS[default_ds_key]["generator"]()
    st.session_state["dataset_name"] = default_ds_key
    st.session_state["target_col"] = SAMPLE_DATASETS[default_ds_key]["default_target"]
    st.session_state["task_type"] = SAMPLE_DATASETS[default_ds_key]["task_type"]

if "col_types" not in st.session_state:
    st.session_state["col_types"] = DataProcessor.detect_column_types(st.session_state["df"], st.session_state["target_col"])

if "training_results" not in st.session_state:
    st.session_state["training_results"] = None

if "single_prediction" not in st.session_state:
    st.session_state["single_prediction"] = None

if "llm_analysis" not in st.session_state:
    st.session_state["llm_analysis"] = None

if "use_image_features" not in st.session_state:
    st.session_state["use_image_features"] = False

if "image_url_col" not in st.session_state:
    st.session_state["image_url_col"] = None

if "new_product_mode" not in st.session_state:
    st.session_state["new_product_mode"] = False

if "llm_provider" not in st.session_state:
    st.session_state["llm_provider"] = "gemini"

if "llm_api_keys" not in st.session_state:
    st.session_state["llm_api_keys"] = {}


# ---------------------------------------------------------
# Sidebar: Dataset & Model Controls
# ---------------------------------------------------------
with st.sidebar:
    st.markdown(f"### {t('sidebar_data_control_header')}")

    dataset_source = st.radio(
        t("dataset_source_label"),
        [t("dataset_source_builtin"), t("dataset_source_custom")],
        index=0
    )

    if dataset_source == t("dataset_source_builtin"):
        selected_sample = st.selectbox(t("sector_select_label"), list(SAMPLE_DATASETS.keys()))
        if st.button(t("load_dataset_btn"), width="stretch"):
            st.session_state["df"] = SAMPLE_DATASETS[selected_sample]["generator"]()
            st.session_state["dataset_name"] = selected_sample
            st.session_state["target_col"] = SAMPLE_DATASETS[selected_sample]["default_target"]
            st.session_state["task_type"] = SAMPLE_DATASETS[selected_sample]["task_type"]
            st.session_state["col_types"] = DataProcessor.detect_column_types(st.session_state["df"], st.session_state["target_col"])
            st.session_state["training_results"] = None
            st.session_state["single_prediction"] = None
            st.session_state["llm_analysis"] = None
            st.session_state["use_image_features"] = False
            st.session_state["image_url_col"] = None
            st.session_state["new_product_mode"] = False
            st.success(t("dataset_loaded_msg", name=selected_sample))
            st.rerun()

    else:
        uploaded_file = st.file_uploader(
            t("upload_file_label"),
            type=["csv", "xlsx", "xls", "json", "zip"],
            help=t("upload_file_help")
        )
        if uploaded_file is not None:
            fname = uploaded_file.name
            selected_file_in_zip = None

            if fname.lower().endswith(".zip"):
                try:
                    zip_files = DataProcessor.list_zip_contents(uploaded_file)
                    if not zip_files:
                        st.error(t("zip_no_valid_file"))
                    elif len(zip_files) == 1:
                        selected_file_in_zip = zip_files[0]["filename"]
                        st.info(t("zip_single_info", filename=selected_file_in_zip, size=zip_files[0]["size_str"]))
                    else:
                        st.markdown(t("zip_multi_found", n=len(zip_files)))
                        file_opts = [f"{item['filename']} ({item['size_str']})" for item in zip_files]
                        chosen_opt = st.selectbox(t("zip_choose_file"), file_opts)
                        selected_file_in_zip = zip_files[file_opts.index(chosen_opt)]["filename"]
                except Exception as e:
                    st.error(t("zip_read_error", error=e))

            if st.button(t("process_upload_btn"), width="stretch"):
                try:
                    df_custom = DataProcessor.load_dataset(
                        file_or_path=uploaded_file,
                        filename=uploaded_file.name,
                        file_inside_zip=selected_file_in_zip
                    )
                    st.session_state["df"] = df_custom
                    st.session_state["dataset_name"] = f"{uploaded_file.name} ➔ {selected_file_in_zip}" if selected_file_in_zip else uploaded_file.name
                    st.session_state["target_col"] = df_custom.columns[-1]
                    target_info = DataProcessor.inspect_target(df_custom, st.session_state["target_col"])
                    st.session_state["task_type"] = target_info["task_type"]
                    st.session_state["col_types"] = DataProcessor.detect_column_types(df_custom, st.session_state["target_col"])
                    st.session_state["training_results"] = None
                    st.session_state["single_prediction"] = None
                    st.session_state["llm_analysis"] = None
                    st.session_state["use_image_features"] = False
                    st.session_state["image_url_col"] = None
                    st.session_state["new_product_mode"] = False
                    st.success(t("upload_success", name=st.session_state["dataset_name"], n=len(df_custom)))
                    st.rerun()
                except Exception as e:
                    st.error(t("upload_error", error=e))

    st.markdown("---")

    # ---- LLM Provider Configuration ----
    st.markdown(f"### {t('llm_settings_header')}")
    provider_keys = list(PROVIDERS.keys())
    provider_choice = st.selectbox(
        t("llm_provider_label"),
        options=provider_keys,
        format_func=lambda k: PROVIDERS[k]["label"],
        index=provider_keys.index(st.session_state["llm_provider"]) if st.session_state["llm_provider"] in provider_keys else 0,
    )
    st.session_state["llm_provider"] = provider_choice
    provider_meta = PROVIDERS[provider_choice]

    entered_key = st.text_input(
        t("llm_api_key_label", provider=provider_meta["label"]),
        type="password",
        value=st.session_state["llm_api_keys"].get(provider_choice, os.getenv(provider_meta["env_key"], "")),
        help=t("llm_api_key_help", provider=provider_meta["label"]),
        key=f"api_key_input_{provider_choice}",
    )
    st.session_state["llm_api_keys"][provider_choice] = entered_key

    with st.expander(t("llm_model_override_expander_title"), expanded=False):
        st.caption(t(
            "llm_model_override_help",
            default_model=provider_meta["default_model"],
            example=provider_meta.get("example_model", provider_meta["default_model"]),
        ))
        model_override = st.text_input(
            t("llm_model_override_label"),
            value=st.session_state.get("llm_model_override", ""),
            label_visibility="collapsed",
            placeholder=provider_meta["default_model"],
        )
        st.session_state["llm_model_override"] = model_override.strip()

    if entered_key:
        st.caption(t("llm_active_live", provider=provider_meta["label"]))
    else:
        st.caption(t("llm_active_heuristic"))

    st.markdown("---")

    # ---- Extended (AI-translated) languages ----
    # Beyond the 3 hand-translated languages, any of these ~40 is generated
    # on demand by the provider/key configured just above and cached to
    # disk — see src/utils/i18n.py for why this scales better than hand
    # translating every string per language.
    with st.expander(f"🌍 {t('extended_lang_header')}", expanded=False):
        st.caption(t("extended_lang_help"))
        ext_lang_keys = list(EXTENDED_LANGUAGES.keys())
        ext_default = st.session_state["lang"] if st.session_state["lang"] in ext_lang_keys else ext_lang_keys[0]
        ext_choice = st.selectbox(
            t("extended_lang_select_label"),
            options=ext_lang_keys,
            format_func=lambda k: EXTENDED_LANGUAGES[k],
            index=ext_lang_keys.index(ext_default),
            key="ext_lang_choice",
        )
        if st.button(t("extended_lang_apply_btn"), width="stretch", key="ext_lang_apply_btn"):
            if not entered_key:
                st.warning(t("extended_lang_need_key", provider=provider_meta["label"]))
            else:
                with st.spinner(t("extended_lang_translating", lang=EXTENDED_LANGUAGES[ext_choice])):
                    ok = ensure_dynamic_language(
                        ext_choice, provider_choice, entered_key,
                        st.session_state.get("llm_model_override") or None,
                    )
                if ok:
                    set_lang(ext_choice)
                    if is_authenticated():
                        save_user_preferences(current_user()["email"], {"lang": ext_choice})
                    st.success(t("extended_lang_success", lang=EXTENDED_LANGUAGES[ext_choice]))
                    st.rerun()
                else:
                    st.error(t("extended_lang_fail"))
        st.caption(t("extended_lang_advisor_note"))

    st.markdown("---")
    st.markdown(t("footer_badge"))
    st.caption(t("footer_caption"))


# ---------------------------------------------------------
# Main Page Header
# ---------------------------------------------------------
st.markdown(f'<div class="main-header">{t("app_title")}</div>', unsafe_allow_html=True)
st.markdown(f'<div class="sub-header">{t("app_subtitle")}</div>', unsafe_allow_html=True)

with st.expander(t("guide_expander_title"), expanded=False):
    st.markdown(t("guide_intro"))
    for step_key in ["guide_step1", "guide_step2", "guide_step3", "guide_step4", "guide_step5", "guide_step6"]:
        st.markdown(f"- {t(step_key)}")
    st.markdown(f'<div class="guide-tip">{t("guide_tip")}</div>', unsafe_allow_html=True)

df = st.session_state["df"]
target_col = st.session_state["target_col"]
task_type = st.session_state["task_type"]
col_types = st.session_state["col_types"]

# ---------------------------------------------------------
# Navigation Tabs
# ---------------------------------------------------------
tabs = st.tabs([
    t("tab1_title"), t("tab2_title"), t("tab3_title"),
    t("tab4_title"), t("tab5_title"), t("tab6_title"),
])


# =========================================================
# TAB 1: DATASET & TARGET SELECTION
# =========================================================
with tabs[0]:
    st.subheader(t("t1_subheader"))

    col_t1, col_t2 = st.columns([2, 1])
    with col_t1:
        st.info(t("t1_active_dataset", name=st.session_state["dataset_name"], rows=len(df), cols=len(df.columns)))
    with col_t2:
        target_choice = st.selectbox(
            t("t1_target_select"),
            df.columns.tolist(),
            index=df.columns.tolist().index(target_col) if target_col in df.columns else len(df.columns) - 1,
            format_func=prettify_column_name,
            help=t("t1_target_select_help"),
        )
        if target_choice != target_col:
            st.session_state["target_col"] = target_choice
            target_info = DataProcessor.inspect_target(df, target_choice)
            st.session_state["task_type"] = target_info["task_type"]
            st.session_state["col_types"] = DataProcessor.detect_column_types(df, target_choice)
            st.session_state["training_results"] = None
            st.rerun()

    with st.expander(t("t1_derive_expander"), expanded=False):
        st.markdown(t("t1_derive_help"))
        rc1, rc2, rc3, rc4 = st.columns(4)
        with rc1:
            base_col = st.selectbox(t("t1_derive_base_col"), df.select_dtypes(include=[np.number]).columns.tolist())
        with rc2:
            cond_op = st.selectbox(t("t1_derive_condition"), [">=", ">", "<=", "<", "Top %"])
        with rc3:
            if cond_op == "Top %":
                cond_val = st.number_input(
                    t("t1_derive_threshold_pct"), value=35.0, min_value=0.0, max_value=100.0, step=1.0
                )
            else:
                default_val = float(df[base_col].median()) if base_col in df else 100.0
                cond_val = st.number_input(t("t1_derive_threshold"), value=default_val)
        with rc4:
            new_name = st.text_input(t("t1_derive_new_name"), value=f"{base_col}_success_target")

        if st.button(t("t1_derive_btn")):
            df_new = DataProcessor.create_derived_target(df, base_col, cond_op, cond_val, new_name)
            st.session_state["df"] = df_new
            st.session_state["target_col"] = new_name
            st.session_state["task_type"] = "classification"
            new_col_types = DataProcessor.detect_column_types(df_new, new_name)
            if base_col in new_col_types["numeric"]:
                new_col_types["numeric"].remove(base_col)
            if base_col not in new_col_types["id_or_url"]:
                new_col_types["id_or_url"].append(base_col)
            st.session_state["col_types"] = new_col_types
            st.session_state["training_results"] = None
            st.success(t("t1_derive_success", name=new_name))
            st.warning(t("t1_derive_leak_warning", col=base_col))
            st.rerun()

    t_info = DataProcessor.inspect_target(df, st.session_state["target_col"])
    task_label = t("t1_task_classification") if t_info["task_type"] == "classification" else t("t1_task_regression")
    st.markdown(t("t1_task_type_line", task=task_label, subtype=t_info.get("subtype", "N/A")))

    st.markdown(f"#### {t('t1_preview_header')}")
    st.dataframe(df.head(50), width="stretch")

    st.markdown(f"#### {t('t1_column_types_header')}")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"**{t('t1_numeric', n=len(col_types['numeric']))}**")
        st.caption(", ".join(col_types["numeric"]) or t("t1_none"))
    with c2:
        st.markdown(f"**{t('t1_categorical', n=len(col_types['categorical']))}**")
        st.caption(", ".join(col_types["categorical"]) or t("t1_none"))
    with c3:
        st.markdown(f"**{t('t1_text', n=len(col_types['text']))}**")
        st.caption(", ".join(col_types["text"]) or t("t1_none"))
    with c4:
        st.markdown(f"**{t('t1_boolean', n=len(col_types['boolean']))}**")
        st.caption(", ".join(col_types["boolean"]) or t("t1_none"))


# =========================================================
# TAB 2: AUTOMATED EDA & PROFILING
# =========================================================
with tabs[1]:
    st.subheader(t("t2_subheader"))

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    summary_stats = DataProcessor.get_dataset_summary(df)

    for col_ui, title_key, value in [
        (kpi1, "t2_kpi_rows", f"{summary_stats['n_rows']:,}"),
        (kpi2, "t2_kpi_cols", f"{summary_stats['n_cols']}"),
        (kpi3, "t2_kpi_missing", f"{summary_stats['total_missing']:,}"),
        (kpi4, "t2_kpi_duplicates", f"{summary_stats['duplicates']}"),
    ]:
        with col_ui:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">{t(title_key)}</div>
                <div class="metric-value">{value}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    insights = AutoEDA.compute_quick_insights(df, target_col, task_type)
    st.markdown(f"#### {t('t2_insights_header')}")
    for ins in insights:
        st.markdown(f"- {ins}")

    st.markdown("---")

    v_col1, v_col2 = st.columns(2)
    with v_col1:
        st.plotly_chart(AutoEDA.plot_target_distribution(df, target_col, task_type, template=PLOTLY_TPL), width="stretch")
    with v_col2:
        corr_fig = AutoEDA.plot_correlation_heatmap(df, col_types["numeric"], target_col, template=PLOTLY_TPL)
        if corr_fig:
            st.plotly_chart(corr_fig, width="stretch")
        else:
            st.info(t("t2_corr_insufficient"))

    st.markdown(f"#### {t('t2_feature_deep_dive')}")
    selectable_feats = [c for c in df.columns if c != target_col and c not in col_types["id_or_url"]]
    if selectable_feats:
        selected_feat = st.selectbox(t("t2_feature_select"), selectable_feats, format_func=prettify_column_name)
        feat_fig = AutoEDA.plot_feature_vs_target(df, selected_feat, target_col, task_type, template=PLOTLY_TPL)
        st.plotly_chart(feat_fig, width="stretch")

    missing_fig = AutoEDA.plot_missing_values(df, template=PLOTLY_TPL)
    if missing_fig:
        st.plotly_chart(missing_fig, width="stretch")

    # ---- Data Leakage Check ----
    st.markdown("---")
    st.markdown(f"#### {t('t2_leakage_header')}")
    leakage_risks = DataProcessor.detect_leakage_risks(df, target_col, task_type, col_types)
    st.session_state["leakage_risks"] = leakage_risks
    if not leakage_risks:
        st.success(t("t2_leakage_none"))
    else:
        st.warning(t("t2_leakage_found"))
        for risk in leakage_risks:
            reason = t(f"leak_reason_{risk['reason_key']}", **{k: v for k, v in risk.items() if k in ("corr", "value")})
            action = t(f"leak_action_{risk['action_key']}")
            st.markdown(f"- {t('t2_leakage_row', col=risk['feature'], reason=reason, action=action)}")

    # ---- Data Gaps ----
    st.markdown(f"#### {t('t2_gap_header')}")
    gaps = DataProcessor.detect_data_gaps(df, target_col)
    if not gaps:
        st.success(t("t2_gap_none"))
    else:
        for gap in gaps:
            if gap["type"] == "high_missing":
                st.markdown(f"- {t('t2_gap_high_missing', col=gap['feature'], pct=gap['pct'])}")
            elif gap["type"] == "low_variance":
                st.markdown(f"- {t('t2_gap_low_variance', col=gap['feature'])}")


# =========================================================
# TAB 3: PREPROCESSING & PIPELINE CONFIGURATION
# =========================================================
with tabs[2]:
    st.subheader(t("t3_subheader"))
    st.markdown(t("t3_intro"))

    p1, p2, p3, p4 = st.columns(4)
    with p1:
        scaler_choice = st.selectbox(t("t3_scaler"), ["standard", "robust", "minmax"], index=0)
    with p2:
        max_cats = st.slider(t("t3_max_cats"), min_value=10, max_value=60, value=30, step=5, help=t("t3_max_cats_help"))
    with p3:
        tfidf_max = st.slider(t("t3_tfidf"), min_value=10, max_value=80, value=35, step=5)
    with p4:
        test_split_ratio = st.slider(t("t3_test_split"), min_value=10, max_value=40, value=20, step=5) / 100.0

    st.markdown(f"#### {t('t3_steps_header')}")
    st.markdown(f"""
    1. **{t('t1_numeric', n=len(col_types['numeric']))}:** `SimpleImputer(strategy='median')` ➔ `{scaler_choice.title()}Scaler` *(float32)*
    2. **{t('t1_categorical', n=len(col_types['categorical']))}:** `SimpleImputer(fill_value='Unknown')` ➔ `OneHotEncoder(max_categories={max_cats})`
    3. **{t('t1_text', n=len(col_types['text']))}:** `TfidfVectorizer(max_features={tfidf_max})`
    4. **{t('t1_boolean', n=len(col_types['boolean']))}:** `BooleanTransformer(float32)`
    5. **ID/URL ({len(col_types['id_or_url'])}):** auto-excluded
    """)

    st.markdown("---")
    st.markdown(f"#### {t('t3_image_header')}")
    use_images = st.checkbox(
        t("t3_image_checkbox"),
        value=st.session_state.get("use_image_features", False),
        help=t("t3_image_help"),
    )
    st.session_state["use_image_features"] = use_images

    if use_images:
        url_like_cols = [c for c in col_types["id_or_url"] if any(k in c.lower() for k in ["img", "image", "photo", "picture"])]
        candidate_cols = url_like_cols or col_types["id_or_url"]
        if not candidate_cols:
            st.warning(t("t3_image_no_col"))
            st.session_state["image_url_col"] = None
        else:
            prev_choice = st.session_state.get("image_url_col")
            default_idx = candidate_cols.index(prev_choice) if prev_choice in candidate_cols else 0
            image_url_col = st.selectbox(t("t3_image_col_select"), candidate_cols, index=default_idx)
            img_c1, img_c2 = st.columns(2)
            with img_c1:
                image_n_components = st.slider(t("t3_image_pca_dim"), min_value=8, max_value=64, value=32, step=8)
            with img_c2:
                image_max_fit_rows = st.slider(t("t3_image_sample_n"), min_value=50, max_value=1000, value=300, step=50, help=t("t3_image_sample_help"))
            st.session_state["image_url_col"] = image_url_col
            st.session_state["image_n_components"] = image_n_components
            st.session_state["image_max_fit_rows"] = image_max_fit_rows
            st.caption(t("t3_image_note"))
    else:
        st.session_state["image_url_col"] = None


# =========================================================
# TAB 4: AUTOML MODEL TRAINING & BENCHMARK
# =========================================================
with tabs[3]:
    st.subheader(t("t4_subheader"))
    st.markdown(t("t4_intro"))

    candidate_models_dict = ModelTrainer.get_candidate_models(task_type)

    t_c1, t_c2 = st.columns([2, 1])
    with t_c1:
        selected_models = st.multiselect(
            t("t4_model_select"),
            options=list(candidate_models_dict.keys()),
            default=list(candidate_models_dict.keys())
        )
    with t_c2:
        max_samples_val = None
        if len(df) > 20000:
            sample_opt = st.selectbox(
                t("t4_sample_size_label"),
                ["50,000", "100,000", "25,000", "Full Dataset"],
                help=t("t4_sample_size_help", n=len(df))
            )
            if sample_opt == "50,000":
                max_samples_val = 50000
            elif sample_opt == "100,000":
                max_samples_val = 100000
            elif sample_opt == "25,000":
                max_samples_val = 25000
            else:
                max_samples_val = None
        else:
            st.caption(t("t4_dataset_size_info", n=len(df)))

    st.markdown("---")
    new_product_mode = st.checkbox(
        t("t4_new_product_checkbox"),
        value=st.session_state.get("new_product_mode", False),
        help=t("t4_new_product_help"),
    )
    st.session_state["new_product_mode"] = new_product_mode

    traction_num = [c for c in col_types["numeric"] if any(k in c.lower() for k in TRACTION_TERMS)] if new_product_mode else []
    traction_cat = [c for c in col_types["categorical"] if any(k in c.lower() for k in TRACTION_TERMS)] if new_product_mode else []
    traction_bool = [c for c in col_types["boolean"] if any(k in c.lower() for k in TRACTION_TERMS)] if new_product_mode else []
    traction_txt = [c for c in col_types["text"] if any(k in c.lower() for k in TRACTION_TERMS)] if new_product_mode else []
    excluded_traction_cols = traction_num + traction_cat + traction_bool + traction_txt

    if new_product_mode:
        if excluded_traction_cols:
            st.caption(t("t4_new_product_excluded", cols=", ".join(f"`{c}`" for c in excluded_traction_cols)))
        else:
            st.caption(t("t4_new_product_none_found"))

    if not selected_models:
        st.warning(t("t4_select_warning"))

    leakage_risks_t4 = st.session_state.get("leakage_risks", [])
    selected_feature_pool = (
        [c for c in col_types["numeric"] if c not in traction_num]
        + [c for c in col_types["categorical"] if c not in traction_cat]
        + [c for c in col_types["text"] if c not in traction_txt]
        + [c for c in col_types["boolean"] if c not in traction_bool]
    )
    high_risk_in_selection = [r for r in leakage_risks_t4 if r["severity"] == "high" and r["feature"] in selected_feature_pool]
    if high_risk_in_selection:
        st.warning(t("t4_leakage_train_warning"))

    if st.button(t("t4_train_btn"), type="primary", width="stretch", disabled=not selected_models):
        with st.spinner(t("t4_training_spinner")):
            time.sleep(0.5)

            try:
                image_url_col = st.session_state.get("image_url_col")

                preprocessor, input_cols = PreprocessingBuilder.create_preprocessor(
                    numeric_cols=[c for c in col_types["numeric"] if c not in traction_num],
                    categorical_cols=[c for c in col_types["categorical"] if c not in traction_cat],
                    text_cols=[c for c in col_types["text"] if c not in traction_txt],
                    boolean_cols=[c for c in col_types["boolean"] if c not in traction_bool],
                    scaler_type=scaler_choice,
                    max_tfidf_features=tfidf_max,
                    max_categories=max_cats,
                    image_url_col=image_url_col,
                    image_n_components=st.session_state.get("image_n_components", 32),
                    image_max_fit_rows=st.session_state.get("image_max_fit_rows", 300)
                )

                X = df[[c for c in df.columns if c != target_col
                        and (c not in col_types["id_or_url"] or c == image_url_col)
                        and c not in excluded_traction_cols]]
                y = df[target_col]

                results = ModelTrainer.train_and_benchmark(
                    X_df=X,
                    y_series=y,
                    preprocessor=preprocessor,
                    task_type=task_type,
                    test_size=test_split_ratio,
                    selected_model_names=selected_models,
                    max_train_samples=max_samples_val
                )

                st.session_state["training_results"] = results
                st.session_state["input_cols"] = input_cols
                st.success(t("t4_train_success", name=results["best_model_name"]))
            except ImageFeatureUnavailableError as e:
                st.error(t("t4_train_image_error", error=e))
            except ValueError as e:
                st.error(t("t4_train_value_error", error=e))

    if st.session_state["training_results"] is not None:
        res = st.session_state["training_results"]

        st.markdown("---")
        st.markdown(f"### {t('t4_leaderboard_header')}")
        # Display-only rename: raw statistics jargon (F1-Score, ROC-AUC, RMSE...)
        # becomes a plain-language column header in the active UI language.
        # Sorting/logic elsewhere still uses the untouched benchmark_table.
        display_table = res["benchmark_table"].rename(columns=metric_column_labels(get_lang()))
        st.dataframe(display_table, width="stretch")

        best_name = res["best_model_name"]
        st.markdown(f"#### {t('t4_best_model_header', name=best_name)}")

        m_col1, m_col2 = st.columns(2)

        with m_col1:
            if task_type == "classification" and best_name in res["predictions_dict"]:
                cm = res["predictions_dict"][best_name]["cm"]
                cm_fig = px.imshow(
                    cm, text_auto=True,
                    labels=dict(x="Predicted", y="Actual", color="Count"),
                    x=[f"Pred {c}" for c in range(cm.shape[1])],
                    y=[f"True {c}" for c in range(cm.shape[0])],
                    title=f"🎯 Confusion Matrix — {best_name}",
                    color_continuous_scale="Blues"
                )
                cm_fig.update_layout(template=PLOTLY_TPL, margin=dict(l=20, r=20, t=50, b=20))
                st.plotly_chart(cm_fig, width="stretch")
            elif task_type == "regression" and best_name in res["predictions_dict"]:
                residuals = res["predictions_dict"][best_name]["residuals"]
                res_fig = px.histogram(
                    residuals, nbins=25,
                    title=f"📈 Residuals — {best_name}",
                    color_discrete_sequence=["#38BDF8"]
                )
                res_fig.update_layout(template=PLOTLY_TPL, margin=dict(l=20, r=20, t=50, b=20))
                st.plotly_chart(res_fig, width="stretch")

        with m_col2:
            fi_df, fi_fig = ModelExplainer.get_global_feature_importance(
                model=res["best_estimator"],
                preprocessor=res["full_pipeline"].named_steps["preprocessor"],
                top_n=12,
                template=PLOTLY_TPL,
                col_types=col_types,
            )
            if fi_fig:
                st.plotly_chart(fi_fig, width="stretch")
            else:
                st.info(t("t4_fi_missing"))


# =========================================================
# TAB 5: INTERACTIVE PREDICTION & WHAT-IF SIMULATION
# =========================================================
with tabs[4]:
    st.subheader(t("t5_subheader"))

    if st.session_state["training_results"] is None:
        st.warning(t("t5_need_training"))
    else:
        pipeline = st.session_state["training_results"]["full_pipeline"]
        best_name = st.session_state["training_results"]["best_model_name"]

        pred_subtabs = st.tabs([t("t5_subtab_single"), t("t5_subtab_batch")])

        # Subtab 1: Single Product
        with pred_subtabs[0]:
            st.markdown(t("t5_form_intro"))

            image_url_col = st.session_state.get("image_url_col")
            feature_cols = st.session_state.get("input_cols") or [
                c for c in df.columns if c != target_col and (c not in col_types["id_or_url"] or c == image_url_col)
            ]

            traction_cols = [c for c in feature_cols if any(k in c.lower() for k in TRACTION_TERMS)]
            if traction_cols:
                st.caption(t("t5_traction_note", cols=", ".join(f"`{c}`" for c in traction_cols)))

            if st.button(t("t5_random_fill_btn")):
                sample_row = df.sample(1).iloc[0]
                st.session_state["sample_inputs"] = sample_row.to_dict()
            else:
                if "sample_inputs" not in st.session_state:
                    st.session_state["sample_inputs"] = df.iloc[0].to_dict()

            user_inputs = {}
            cols_per_row = 3
            form_cols = st.columns(cols_per_row)

            for idx, col_name in enumerate(feature_cols):
                with form_cols[idx % cols_per_row]:
                    default_val = st.session_state["sample_inputs"].get(col_name, df[col_name].dropna().iloc[0] if len(df[col_name].dropna()) > 0 else "")
                    # Field labels show a readable version of the dataset's own
                    # column name (spaced/capitalized) — not a translation, just
                    # cleanup, since the underlying column can be anything the
                    # user's own file defines (see t1_target_select_help).
                    col_label = prettify_column_name(col_name)

                    if col_name == image_url_col:
                        # --- Simplified photo input: upload is the primary, obvious
                        # path; a URL is available but tucked away in an expander so
                        # first-time users see one clear action, not a fork. ---
                        st.markdown(f"🖼️ **{col_label}**")
                        uploaded_img = st.file_uploader(
                            t("t5_photo_upload_label"), type=["jpg", "jpeg", "png", "webp"],
                            key=f"img_upload_{col_name}",
                            help=t("t5_photo_upload_help"),
                            label_visibility="visible",
                        )
                        photo_value = ""
                        if uploaded_img is not None:
                            os.makedirs(".uploaded_images", exist_ok=True)
                            save_path = os.path.join(".uploaded_images", f"{id(uploaded_img)}_{uploaded_img.name}")
                            with open(save_path, "wb") as f:
                                f.write(uploaded_img.getbuffer())
                            photo_value = save_path
                            st.image(uploaded_img, caption=t("t5_photo_preview_caption"), width=140)
                        else:
                            with st.expander(t("t5_photo_url_expander"), expanded=False):
                                curr_txt = str(default_val) if str(default_val).lower().startswith(("http://", "https://")) else ""
                                url_val = st.text_input(t("t5_photo_url_label"), value=curr_txt, key=f"img_url_{col_name}", label_visibility="collapsed")
                                if url_val:
                                    photo_value = url_val
                            if not photo_value:
                                st.caption(t("t5_photo_none_caption"))
                        user_inputs[col_name] = photo_value

                    elif col_name in col_types["numeric"]:
                        min_v = float(df[col_name].min())
                        max_v = float(df[col_name].max())
                        curr_v = float(default_val) if isinstance(default_val, (int, float, np.number)) else min_v
                        curr_v = max(min(curr_v, max_v), min_v)
                        step = (max_v - min_v) / 100.0 if max_v > min_v else 1.0
                        user_inputs[col_name] = st.number_input(f"🔢 {col_label}", value=curr_v, min_value=min_v, max_value=max_v, step=round(step, 2))

                    elif col_name in col_types["boolean"]:
                        curr_bool = bool(default_val) if isinstance(default_val, (bool, np.bool_)) else str(default_val).lower() == "true"
                        user_inputs[col_name] = st.checkbox(f"🔘 {col_label}", value=curr_bool)

                    elif col_name in col_types["text"]:
                        curr_txt = str(default_val)
                        user_inputs[col_name] = st.text_input(f"📝 {col_label}", value=curr_txt)

                    elif col_name in col_types["categorical"]:
                        unique_cats = [str(x) for x in df[col_name].dropna().unique().tolist()]
                        curr_cat = str(default_val)
                        cat_idx = unique_cats.index(curr_cat) if curr_cat in unique_cats else 0
                        user_inputs[col_name] = st.selectbox(f"🏷️ {col_label}", unique_cats, index=cat_idx)
                    else:
                        user_inputs[col_name] = st.text_input(f"📌 {col_label}", value=str(default_val))

            st.markdown("<br>", unsafe_allow_html=True)

            if st.button(t("t5_predict_btn"), type="primary", width="stretch"):
                input_df = pd.DataFrame([user_inputs])

                exp_res = ModelExplainer.explain_single_prediction(
                    pipeline=pipeline, input_row_df=input_df, task_type=task_type, col_types=col_types
                )

                st.session_state["single_prediction"] = {
                    "product_dict": user_inputs,
                    "prediction": exp_res["prediction"],
                    "probability": exp_res["probability"],
                    "top_drivers": exp_res["top_drivers"],
                    "best_model_name": best_name
                }

                provider = st.session_state["llm_provider"]
                advisor = AIProductAdvisor(
                    provider=provider,
                    api_key=st.session_state["llm_api_keys"].get(provider),
                    model_name=st.session_state.get("llm_model_override"),
                    lang=get_lang(),
                )
                model_metric_str = "F1 / Accuracy" if task_type == "classification" else "R2 Score"
                pred_label = "SUCCESS" if (str(exp_res["prediction"]) in ["1", "True", "1.0", "success"]) else str(exp_res["prediction"])

                ai_res = advisor.analyze_product_success(
                    product_dict=user_inputs,
                    prediction_label=pred_label,
                    probability=exp_res["probability"] if exp_res["probability"] is not None else 0.5,
                    best_model_name=best_name,
                    model_metric=model_metric_str,
                    top_drivers=exp_res["top_drivers"]
                )
                st.session_state["llm_analysis"] = ai_res
                st.rerun()

            if st.session_state["single_prediction"] is not None:
                p_res = st.session_state["single_prediction"]
                st.markdown("---")

                def resolve_probability(raw_prob, raw_pred, task):
                    """Returns (prob_pct: float 0-100, display_mode: 'percent'|'score'|'binary')."""
                    if task == "classification":
                        if raw_prob is not None:
                            p = float(raw_prob)
                            p = max(0.0, min(1.0, p))
                            return round(p * 100, 1), "percent"
                        else:
                            try:
                                pred_int = int(float(raw_pred))
                                return (75.0 if pred_int == 1 else 25.0), "binary"
                            except Exception:
                                return 50.0, "binary"
                    else:
                        try:
                            v = float(raw_pred)
                            return round(v, 3), "score"
                        except Exception:
                            return 0.0, "score"

                prob_val, prob_mode = resolve_probability(p_res["probability"], p_res["prediction"], task_type)

                if prob_mode == "score":
                    is_succ = True
                    level = "neutral"
                elif prob_val >= 65:
                    is_succ, level = True, "high"
                elif prob_val >= 40:
                    is_succ, level = False, "mid"
                else:
                    is_succ, level = False, "low"

                level_cfg = {
                    "high":    {"color": "#10b981", "bar_cls": "prob-bar-fill-green", "badge_cls": "success-badge", "label": t("t5_level_high")},
                    "mid":     {"color": "#f59e0b", "bar_cls": "prob-bar-fill-amber", "badge_cls": "warning-badge", "label": t("t5_level_mid")},
                    "low":     {"color": "#ef4444", "bar_cls": "prob-bar-fill-red",   "badge_cls": "danger-badge", "label": t("t5_level_low")},
                    "neutral": {"color": "#6366f1", "bar_cls": "prob-bar-fill-green", "badge_cls": "success-badge", "label": t("t5_level_neutral")},
                }[level]

                st.markdown(f"### {t('t5_result_header')}")
                res_c1, res_c2 = st.columns([1, 1])

                with res_c1:
                    gauge_color = level_cfg["color"]
                    gauge_fig = go.Figure(go.Indicator(
                        mode="gauge+number+delta" if prob_mode == "percent" else "gauge+number",
                        value=prob_val,
                        number={"suffix": "%" if prob_mode == "percent" else "", "font": {"size": 34}, "valueformat": ".1f"},
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': t("t5_gauge_title_pct") if prob_mode == "percent" else t("t5_gauge_title_score"), 'font': {'size': 16}},
                        delta={'reference': 50, 'relative': False, 'valueformat': '.1f',
                               'increasing': {'color': '#10b981'}, 'decreasing': {'color': '#ef4444'}} if prob_mode == "percent" else None,
                        gauge={
                            'axis': {'range': [0, 100 if prob_mode in ("percent", "binary") else max(prob_val * 1.5, 1)], 'tickwidth': 1},
                            'bar': {'color': gauge_color, 'thickness': 0.28},
                            'steps': [
                                {'range': [0, 40], 'color': 'rgba(239,68,68,0.10)'},
                                {'range': [40, 65], 'color': 'rgba(245,158,11,0.10)'},
                                {'range': [65, 100], 'color': 'rgba(16,185,129,0.10)'},
                            ],
                            'threshold': {'line': {'color': gauge_color, 'width': 3}, 'thickness': 0.82, 'value': prob_val}
                        }
                    ))
                    gauge_fig.update_layout(template=PLOTLY_TPL, height=300, margin=dict(l=16, r=16, t=28, b=16), paper_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(gauge_fig, width="stretch")

                with res_c2:
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown(f'<div class="{level_cfg["badge_cls"]}">{level_cfg["label"]}</div>', unsafe_allow_html=True)
                    st.markdown("<br>", unsafe_allow_html=True)

                    if prob_mode in ("percent", "binary"):
                        bar_cls = level_cfg["bar_cls"]
                        st.markdown(
                            f"""<div style="font-size:0.85rem;margin-bottom:2px;">{t('t5_prob_label')}</div>
                            <div class="prob-bar-wrap"><div class="{bar_cls}" style="width:{min(prob_val,100):.1f}%"></div></div>
                            <div style="font-size:0.9rem;color:{level_cfg['color']};font-weight:700;margin-top:4px;">%{prob_val:.1f}</div>""",
                            unsafe_allow_html=True
                        )
                        st.markdown("<br>", unsafe_allow_html=True)

                    pred_display = str(p_res["prediction"])
                    if pred_display in ["1", "1.0", "True", "success"]:
                        pred_display = t("t5_success_label")
                    elif pred_display in ["0", "0.0", "False"]:
                        pred_display = t("t5_failure_label")

                    st.markdown(t("t5_prediction_label", value=pred_display))
                    st.markdown(t("t5_model_label", value=p_res["best_model_name"]))
                    if prob_mode == "binary":
                        st.caption(t("t5_proba_note"))

                    if p_res["top_drivers"]:
                        st.markdown(t("t5_drivers_header"))
                        # Prefer the humanized "FeatureLabel" (e.g. "Kategori:
                        # Elektronik") over the raw pipeline-internal "Feature"
                        # name (e.g. "category_Elektronik") so these chips read
                        # as plain language, not encoded column names.
                        chips_html = "".join([
                            f'<span class="driver-chip">📌 {d.get("FeatureLabel", d.get("Feature","?"))[:40]} <b>{abs(d.get("Impact",0.0)):.3f}</b></span>'
                            for d in p_res["top_drivers"][:6]
                        ])
                        st.markdown(chips_html, unsafe_allow_html=True)

                st.markdown("---")
                st.markdown(f"### {t('t5_whatif_header')}")
                st.markdown(t("t5_whatif_intro"))

                sim_c1, sim_c2 = st.columns([1, 2])
                with sim_c1:
                    sim_feat = st.selectbox(t("t5_whatif_var_select"), feature_cols, format_func=prettify_column_name)

                    if sim_feat in col_types["numeric"]:
                        f_min = float(df[sim_feat].min())
                        f_max = float(df[sim_feat].max())
                        var_range = np.linspace(f_min, f_max, 15).round(2).tolist()
                    elif sim_feat in col_types["boolean"]:
                        var_range = [True, False]
                    else:
                        var_range = df[sim_feat].dropna().unique().tolist()[:10]

                    run_sim = st.button(t("t5_whatif_run_btn"))

                with sim_c2:
                    if run_sim:
                        input_base_df = pd.DataFrame([p_res["product_dict"]])
                        sim_df, sim_fig = ModelExplainer.simulate_what_if(
                            pipeline=pipeline, base_product_df=input_base_df,
                            feature_to_vary=sim_feat, variation_range=var_range,
                            task_type=task_type, template=PLOTLY_TPL,
                        )
                        st.plotly_chart(sim_fig, width="stretch")
                        st.dataframe(sim_df, width="stretch")

        # Subtab 2: Batch CSV Prediction
        with pred_subtabs[1]:
            st.markdown(f"#### {t('t5_batch_header')}")
            if st.session_state.get("image_url_col"):
                st.caption(t("t5_batch_image_note"))
            batch_file = st.file_uploader(t("t5_batch_upload_label"), type=["csv", "xlsx", "xls", "json", "zip"])

            if batch_file is not None:
                df_batch = DataProcessor.load_dataset(batch_file, filename=batch_file.name)
                st.dataframe(df_batch.head(10), width="stretch")

                if st.button(t("t5_batch_score_btn")):
                    with st.spinner(t("t5_batch_spinner")):
                        preds = pipeline.predict(df_batch)
                        df_batch["predicted_success"] = preds

                        if task_type == "classification" and hasattr(pipeline.named_steps["model"], "predict_proba"):
                            try:
                                probas = pipeline.predict_proba(df_batch)[:, 1]
                                df_batch["success_probability_pct"] = (probas * 100).round(1)
                            except Exception:
                                pass

                        st.success(t("t5_batch_success", n=len(df_batch)))
                        st.dataframe(df_batch.head(30), width="stretch")

                        csv_data = df_batch.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label=t("t5_batch_download_btn"), data=csv_data,
                            file_name="predicted_product_success_results.csv", mime="text/csv"
                        )


# =========================================================
# TAB 6: LLM STRATEGIC ADVISOR & EXPORT
# =========================================================
with tabs[5]:
    st.subheader(t("t6_subheader"))

    active_provider_meta = PROVIDERS[st.session_state["llm_provider"]]
    api_key_set = bool(st.session_state["llm_api_keys"].get(st.session_state["llm_provider"]))
    if api_key_set:
        st.success(t("t6_live_active", provider=active_provider_meta["label"]))
    else:
        with st.expander(t("t6_heuristic_expander"), expanded=False):
            st.markdown(t("t6_heuristic_body"))

    if st.session_state["llm_analysis"] is None:
        if st.session_state.get("single_prediction") is not None:
            p_res_t6 = st.session_state["single_prediction"]
            prob_t6 = p_res_t6.get("probability")
            prob_pct_t6 = round(float(prob_t6) * 100, 1) if prob_t6 is not None else None

            st.markdown(f"#### {t('t6_product_summary_header')}")
            st.markdown(t(
                "t6_product_summary_line",
                model=p_res_t6['best_model_name'], pred=p_res_t6['prediction'],
                prob=(f"{t('t5_prob_label')}: **%{prob_pct_t6:.1f}**" if prob_pct_t6 is not None else "")
            ))
            st.markdown("---")
            if st.button(t("t6_analyze_btn"), type="primary", width="stretch"):
                with st.spinner(t("t6_analyze_spinner")):
                    provider = st.session_state["llm_provider"]
                    advisor_t6 = AIProductAdvisor(
                        provider=provider,
                        api_key=st.session_state["llm_api_keys"].get(provider),
                        model_name=st.session_state.get("llm_model_override"),
                        lang=get_lang(),
                    )
                    pred_label_t6 = (
                        "SUCCESS" if str(p_res_t6["prediction"]) in ["1", "True", "1.0", "success"]
                        else f"NOT SUCCESSFUL (Prediction: {p_res_t6['prediction']})"
                    )
                    ai_res_t6 = advisor_t6.analyze_product_success(
                        product_dict=p_res_t6["product_dict"],
                        prediction_label=pred_label_t6,
                        probability=float(p_res_t6["probability"]) if p_res_t6["probability"] is not None else 0.5,
                        best_model_name=p_res_t6["best_model_name"],
                        model_metric="AutoML Best Model",
                        top_drivers=p_res_t6["top_drivers"]
                    )
                    st.session_state["llm_analysis"] = ai_res_t6
                    st.rerun()
        else:
            st.info(t("t6_need_prediction"))

    else:
        ai_res = st.session_state["llm_analysis"]
        p_res = st.session_state["single_prediction"]

        hdr_c1, hdr_c2, hdr_c3 = st.columns([3, 1, 1])
        with hdr_c1:
            st.markdown(t("t6_engine_label", source=ai_res["source"]))
        with hdr_c2:
            if st.button(t("t6_reanalyze_btn")):
                st.session_state["llm_analysis"] = None
                st.rerun()
        with hdr_c3:
            if st.button(t("t6_reset_btn")):
                st.session_state["llm_analysis"] = None
                st.session_state["single_prediction"] = None
                st.rerun()

        prob_display = p_res.get("probability")
        prob_pct_display = round(float(prob_display) * 100, 1) if prob_display is not None else None
        pred_display_t6 = str(p_res["prediction"])
        if pred_display_t6 in ["1", "1.0", "True", "success"]:
            pred_display_t6 = t("t5_success_label")
        elif pred_display_t6 in ["0", "0.0", "False"]:
            pred_display_t6 = t("t5_failure_label")

        st.markdown(t(
            "t6_summary_quote", pred=pred_display_t6, model=p_res['best_model_name'],
            prob=(f"**{t('t5_prob_label')}:** %{prob_pct_display:.1f} &nbsp;|&nbsp;" if prob_pct_display else "")
        ))

        st.markdown('<div class="llm-card">', unsafe_allow_html=True)
        st.markdown(ai_res["analysis_markdown"])
        st.markdown('</div>', unsafe_allow_html=True)

        # ---- Schematic visualization of the recommendations ----
        st.markdown("---")
        st.markdown(f"### {t('t6_schematic_header')}")
        st.caption(t("t6_schematic_caption"))
        is_high_success_viz = bool(prob_pct_display and prob_pct_display >= 60.0)
        sankey_fig = build_recommendation_sankey(p_res.get("top_drivers", []), is_high_success_viz, plotly_template=PLOTLY_TPL, accent_gradient=ACCENT_GRADIENT)
        st.plotly_chart(sankey_fig, width="stretch")

        st.markdown(f"### {t('t6_roadmap_header')}")
        roadmap_fig = build_roadmap_timeline(plotly_template=PLOTLY_TPL, accent_gradient=ACCENT_GRADIENT)
        st.plotly_chart(roadmap_fig, width="stretch")

        st.markdown("---")
        st.markdown(f"### {t('t6_export_header')}")
        rep_c1, rep_c2 = st.columns(2)

        with rep_c1:
            md_text = generate_markdown_report(
                product_dict=p_res["product_dict"], prediction_label=str(p_res["prediction"]),
                probability=p_res["probability"], best_model_name=p_res["best_model_name"],
                model_metric="High Confidence", analysis_markdown=ai_res["analysis_markdown"], source=ai_res["source"]
            )
            st.download_button(
                label=t("t6_export_md_btn"), data=md_text,
                file_name="AI_Product_Success_Report.md", mime="text/markdown", width="stretch"
            )

        with rep_c2:
            pdf_bytes = generate_pdf_report(
                product_dict=p_res["product_dict"], prediction_label=str(p_res["prediction"]),
                probability=p_res["probability"], best_model_name=p_res["best_model_name"],
                model_metric="High Confidence", analysis_text=ai_res["analysis_markdown"]
            )
            st.download_button(
                label=t("t6_export_pdf_btn"), data=pdf_bytes,
                file_name="AI_Product_Success_Report.pdf", mime="application/pdf", width="stretch"
            )
