"""
Human-Readable Label Helpers
Machine learning pipelines produce technical, encoded feature names
(one-hot "category_Electronics", TF-IDF "title_tfidf_ultra", image PCA
"photo_img_pca_3", etc.). Showing these raw names directly to a business
user is confusing and reads as "robotic" — this module turns them back
into plain sentences wherever a feature/driver name reaches the UI.

It also provides friendlier display labels for the AutoML leaderboard's
metric column headers, which are otherwise raw statistics jargon
(F1-Score, ROC-AUC, RMSE...).
"""

from typing import Dict, List, Optional
import re


def _prettify_plain(name: str) -> str:
    """Turns a raw column/token name into a spaced, capitalized phrase."""
    cleaned = re.sub(r"[_\-]+", " ", str(name)).strip()
    if not cleaned:
        return str(name)
    return " ".join(w[:1].upper() + w[1:] if w else w for w in cleaned.split(" "))


def prettify_column_name(name: str) -> str:
    """
    Public wrapper around the plain-text prettifier, for display of a
    dataset's OWN raw column names (e.g. in the target-column picker or a
    prediction form's field labels).

    Important distinction from `humanize_feature_name`: this does NOT
    translate anything — a column called "unit_price" in an uploaded CSV
    stays about unit price, we just show "Unit Price" instead of the raw
    snake_case. There is no reliable way to guess a Turkish/German name for
    a column we've never seen defined, so a dataset with English headers
    (e.g. Yeliz's Trendyol export) will still show English words — just
    spaced and capitalized instead of a raw, hard-to-scan identifier.
    """
    return _prettify_plain(name)


def humanize_feature_name(raw_name: str, col_types: Optional[Dict[str, List[str]]] = None) -> str:
    """
    Converts a pipeline-internal feature name into a phrase a non-technical
    person can read, using the known original column types to decode the
    encoding pattern precisely (rather than guessing from the string alone):

    - One-hot encoded category: "category_Electronics" -> "Category: Electronics"
    - TF-IDF text token:        "title_tfidf_ultra"     -> "Title — mentions 'ultra'"
    - Image PCA component:      "photo_img_pca_3"       -> "Photo (visual feature #3)"
    - Anything else (numeric/boolean columns, or an unrecognized pattern):
      just spaced and capitalized.
    """
    col_types = col_types or {}
    categorical_cols = col_types.get("categorical", [])
    text_cols = col_types.get("text", [])

    # TF-IDF: "{col}_tfidf_{word}" — match the longest column prefix first so
    # a column name that itself contains "_tfidf_" style substrings doesn't
    # get mis-split.
    for col in sorted(text_cols, key=len, reverse=True):
        prefix = f"{col}_tfidf_"
        if raw_name.startswith(prefix):
            word = raw_name[len(prefix):]
            return f"{_prettify_plain(col)} — ‘{word}’ kelimesi geçiyor"

    # Image PCA component: "{col}_img_pca_{i}"
    m = re.match(r"^(.*)_img_pca_(\d+)$", raw_name)
    if m:
        col, idx = m.group(1), m.group(2)
        return f"{_prettify_plain(col)} (görsel özellik #{idx})"

    # One-hot encoded category: "{col}_{value}"
    for col in sorted(categorical_cols, key=len, reverse=True):
        prefix = f"{col}_"
        if raw_name.startswith(prefix) and len(raw_name) > len(prefix):
            value = raw_name[len(prefix):]
            return f"{_prettify_plain(col)}: {_prettify_plain(value)}"

    return _prettify_plain(raw_name)


# ---------------------------------------------------------------------------
# Leaderboard metric column labels (display-only rename; safe because the
# benchmark table is only ever shown, never re-keyed elsewhere by column name
# after this rename is applied at the point of display).
# ---------------------------------------------------------------------------
_METRIC_LABELS = {
    "tr": {
        "Model": "Model",
        "Accuracy": "Doğruluk",
        "F1-Score": "F1 (Genel Denge Skoru)",
        "F1 (Başarı Sınıfı)": "F1 (Yalnızca 'Başarılı' Sınıfı İçin)",
        "Precision": "Kesinlik",
        "Recall": "Duyarlılık (Yakalama Oranı)",
        "ROC-AUC": "Ayırt Etme Gücü (ROC-AUC)",
        "R2 Score": "Açıklanan Varyans (R²)",
        "RMSE": "Ortalama Hata (RMSE)",
        "MAE": "Ortalama Mutlak Hata (MAE)",
        "Training Time (s)": "Eğitim Süresi (sn)",
    },
    "en": {
        "Model": "Model",
        "Accuracy": "Accuracy",
        "F1-Score": "F1 (Overall Balance)",
        "F1 (Başarı Sınıfı)": "F1 (Success Class Only)",
        "Precision": "Precision",
        "Recall": "Recall (Catch Rate)",
        "ROC-AUC": "Separation Power (ROC-AUC)",
        "R2 Score": "Explained Variance (R²)",
        "RMSE": "Average Error (RMSE)",
        "MAE": "Average Absolute Error (MAE)",
        "Training Time (s)": "Training Time (s)",
    },
    "de": {
        "Model": "Modell",
        "Accuracy": "Genauigkeit",
        "F1-Score": "F1 (Gesamtbalance)",
        "F1 (Başarı Sınıfı)": "F1 (nur Erfolgsklasse)",
        "Precision": "Präzision",
        "Recall": "Trefferquote (Recall)",
        "ROC-AUC": "Trennschärfe (ROC-AUC)",
        "R2 Score": "Erklärte Varianz (R²)",
        "RMSE": "Durchschnittsfehler (RMSE)",
        "MAE": "Mittlerer absoluter Fehler (MAE)",
        "Training Time (s)": "Trainingszeit (s)",
    },
}


def metric_column_labels(lang: str = "tr") -> Dict[str, str]:
    """Returns {technical_column_name: friendly_label} for the given language.
    Only known columns are mapped; unknown columns are left untouched by
    the caller (pandas .rename ignores keys that don't match)."""
    return _METRIC_LABELS.get(lang, _METRIC_LABELS["tr"])
