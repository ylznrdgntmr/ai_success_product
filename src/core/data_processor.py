"""
Data Processor Module
Handles dataset ingestion, automatic column type detection, data cleaning,
and target variable inspection/derivation.
"""

from typing import Dict, List, Optional, Tuple, Any
import io
import zipfile
import pandas as pd
import numpy as np


class DataProcessor:
    """Universal Dataset Processor for Product Success Prediction."""

    @staticmethod
    def list_zip_contents(file_or_path: Any) -> List[Dict[str, Any]]:
        """
        Lists all compatible tabular dataset files inside a ZIP archive.
        """
        results = []
        if hasattr(file_or_path, "seek"):
            file_or_path.seek(0)

        with zipfile.ZipFile(file_or_path, 'r') as z:
            for info in z.infolist():
                # Ignore macOS __MACOSX metadata and hidden directories
                if info.is_dir() or info.filename.startswith("__MACOSX/") or info.filename.startswith("."):
                    continue

                fn_low = info.filename.lower()
                if fn_low.endswith((".csv", ".xlsx", ".xls", ".json", ".tsv")):
                    size_kb = info.file_size / 1024
                    size_str = f"{size_kb / 1024:.2f} MB" if size_kb > 1024 else f"{size_kb:.1f} KB"
                    results.append({
                        "filename": info.filename,
                        "size_bytes": info.file_size,
                        "size_str": size_str
                    })

        return sorted(results, key=lambda x: x["size_bytes"], reverse=True)

    @classmethod
    def load_dataset(
        cls,
        file_or_path: Any,
        filename: Optional[str] = None,
        file_inside_zip: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Loads dataset from file path, uploaded file buffer, BytesIO, or ZIP archive.
        Supports CSV (auto delimiter), Excel (.xlsx, .xls), JSON, and ZIP (.zip).
        """
        if isinstance(file_or_path, pd.DataFrame):
            return file_or_path.copy()

        # Handle Streamlit UploadedFile or file path
        fname = filename or getattr(file_or_path, "name", str(file_or_path)).lower()

        # Handle ZIP files
        if fname.endswith(".zip"):
            if hasattr(file_or_path, "seek"):
                file_or_path.seek(0)
            
            with zipfile.ZipFile(file_or_path, 'r') as z:
                # If specific file inside zip requested
                target_name = file_inside_zip
                if not target_name:
                    zip_entries = cls.list_zip_contents(file_or_path)
                    if not zip_entries:
                        raise ValueError("ZIP arşivi içinde geçerli bir veri dosyası (CSV/Excel/JSON) bulunamadı.")
                    target_name = zip_entries[0]["filename"]

                with z.open(target_name) as f:
                    file_bytes = io.BytesIO(f.read())
                    return cls.load_dataset(file_bytes, filename=target_name)

        elif fname.endswith((".csv", ".tsv")):
            try:
                # Try standard comma separator
                df = pd.read_csv(file_or_path)
            except Exception:
                if hasattr(file_or_path, "seek"):
                    file_or_path.seek(0)
                # Try semicolon or tab delimiter with python engine
                df = pd.read_csv(file_or_path, sep=None, engine="python")
        elif fname.endswith((".xlsx", ".xls")):
            df = pd.read_excel(file_or_path)
        elif fname.endswith(".json"):
            df = pd.read_json(file_or_path)
        else:
            # Default fallback try csv
            try:
                df = pd.read_csv(file_or_path)
            except Exception:
                if hasattr(file_or_path, "seek"):
                    file_or_path.seek(0)
                df = pd.read_excel(file_or_path)

        # Standardize column names (strip whitespace)
        df.columns = [str(c).strip() for c in df.columns]
        return df

    @staticmethod
    def detect_column_types(df: pd.DataFrame, target_col: Optional[str] = None) -> Dict[str, List[str]]:
        """
        Automatically classifies DataFrame columns into:
        - numeric
        - categorical
        - text (longer descriptions, titles)
        - boolean
        - id_or_url (candidates to drop)
        - datetime
        """
        numeric_cols = []
        categorical_cols = []
        text_cols = []
        boolean_cols = []
        id_or_url_cols = []
        datetime_cols = []

        n_rows = len(df)
        cols_to_check = [c for c in df.columns if c != target_col]

        for col in cols_to_check:
            series = df[col].dropna()
            if len(series) == 0:
                id_or_url_cols.append(col)
                continue

            col_lower = str(col).lower()
            n_unique = series.nunique()

            # Check for URL / Image links (excluding count/numeric features)
            is_url_or_link = any(term in col_lower for term in ["url", "link", "image_url", "img_url", "photo_url", "imgurl", "producturl"]) or (
                (pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series)) and series.astype(str).str.startswith(("http://", "https://")).mean() > 0.25
            )
            if is_url_or_link and not any(term in col_lower for term in ["count", "num", "total", "rate", "pct"]):
                id_or_url_cols.append(col)
                continue

            # Check for ID / SKU / ASIN columns
            is_id_name = any(id_term == col_lower or id_term in col_lower.split("_") or id_term in col_lower for id_term in ["asin", "sku", "uuid", "isbn", "hash", "item_id", "product_id", "prod_id"])
            if is_id_name and not any(term in col_lower for term in ["count", "num", "price", "sales", "rating", "review"]):
                id_or_url_cols.append(col)
                continue

            # Check for generic high-uniqueness ID columns (> 80% unique rows)
            if (pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series) or pd.api.types.is_integer_dtype(series)) and (n_unique >= n_rows * 0.8) and any(id_term in col_lower for id_term in ["id", "code", "key", "number"]):
                id_or_url_cols.append(col)
                continue

            # Boolean check
            if pd.api.types.is_bool_dtype(series) or (
                n_unique == 2 and set(series.unique()).issubset({0, 1, 0.0, 1.0, "0", "1", "True", "False", "true", "false", True, False})
            ):
                boolean_cols.append(col)
                continue

            # Datetime check
            if pd.api.types.is_datetime64_any_dtype(series):
                datetime_cols.append(col)
                continue
            elif series.dtype == "object" and any(date_term in col_lower for date_term in ["date", "time", "created", "launched", "timestamp"]):
                try:
                    pd.to_datetime(series.head(20))
                    datetime_cols.append(col)
                    continue
                except Exception:
                    pass

            # Numeric check
            if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
                if n_unique <= 5 and not any(m in col_lower for m in ["price", "cost", "rating", "score", "amount", "sales", "revenue", "percent", "discount", "count", "num"]):
                    categorical_cols.append(col)
                else:
                    numeric_cols.append(col)
                continue

            # Text vs Categorical check for string / object / categorical types
            if pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series) or isinstance(series.dtype, pd.CategoricalDtype):
                avg_length = series.astype(str).str.len().mean()
                avg_words = series.astype(str).str.split().str.len().mean()

                # If text has title/desc naming, or high word count / length with high cardinality -> text
                if "title" in col_lower or "name" in col_lower or "desc" in col_lower or "headline" in col_lower or "summary" in col_lower or "body" in col_lower:
                    text_cols.append(col)
                elif (avg_length > 30 or avg_words > 4) and n_unique > 25:
                    text_cols.append(col)
                elif n_unique > 200: # High cardinality non-text is treated as text or capped categorical
                    if avg_words >= 2:
                        text_cols.append(col)
                    else:
                        categorical_cols.append(col)
                else:
                    categorical_cols.append(col)
                continue
            else:
                # Default fallback
                categorical_cols.append(col)

        return {
            "numeric": numeric_cols,
            "categorical": categorical_cols,
            "text": text_cols,
            "boolean": boolean_cols,
            "datetime": datetime_cols,
            "id_or_url": id_or_url_cols
        }

    @staticmethod
    def inspect_target(df: pd.DataFrame, target_col: str) -> Dict[str, Any]:
        """
        Analyzes the selected target variable and determines task type
        (classification vs regression) and class distribution.
        """
        if target_col not in df.columns:
            raise ValueError(f"Hedef sütun '{target_col}' veri setinde bulunamadı.")

        series = df[target_col].dropna()
        n_unique = series.nunique()
        is_numeric = pd.api.types.is_numeric_dtype(series)

        # Boolean or 2 unique values -> Binary Classification
        if n_unique == 2:
            unique_vals = list(series.unique())
            val_counts = series.value_counts(normalize=True).to_dict()
            return {
                "task_type": "classification",
                "subtype": "binary",
                "classes": unique_vals,
                "n_classes": 2,
                "distribution": val_counts,
                "is_balanced": min(val_counts.values()) >= 0.25
            }

        # Small number of unique non-float values -> Multi-class Classification
        elif n_unique <= 10 and not (is_numeric and not pd.api.types.is_integer_dtype(series)):
            unique_vals = list(series.unique())
            val_counts = series.value_counts(normalize=True).to_dict()
            return {
                "task_type": "classification",
                "subtype": "multiclass",
                "classes": unique_vals,
                "n_classes": n_unique,
                "distribution": val_counts,
                "is_balanced": min(val_counts.values()) >= (1.0 / n_unique * 0.4)
            }

        # Otherwise continuous / numeric -> Regression
        elif is_numeric:
            return {
                "task_type": "regression",
                "subtype": "continuous",
                "min": float(series.min()),
                "max": float(series.max()),
                "mean": float(series.mean()),
                "median": float(series.median()),
                "std": float(series.std())
            }
        else:
            # Fallback high cardinality string treated as multi-class
            unique_vals = list(series.unique())[:10]
            return {
                "task_type": "classification",
                "subtype": "multiclass",
                "classes": unique_vals,
                "n_classes": n_unique,
                "distribution": series.value_counts(normalize=True).to_dict()
            }

    @staticmethod
    def create_derived_target(
        df: pd.DataFrame,
        base_col: str,
        condition: str,
        threshold: float,
        target_name: str = "custom_success_target"
    ) -> pd.DataFrame:
        """
        Creates a custom binary success target based on rule.
        E.g. sales > 500, rating >= 4.2
        """
        df_copy = df.copy()
        if condition == ">=":
            df_copy[target_name] = (df_copy[base_col] >= threshold).astype(int)
        elif condition == ">":
            df_copy[target_name] = (df_copy[base_col] > threshold).astype(int)
        elif condition == "<=":
            df_copy[target_name] = (df_copy[base_col] <= threshold).astype(int)
        elif condition == "<":
            df_copy[target_name] = (df_copy[base_col] < threshold).astype(int)
        elif condition == "==":
            df_copy[target_name] = (df_copy[base_col] == threshold).astype(int)
        elif condition == "Top %":
            # threshold is a percentage; clamp defensively so a stray UI value
            # (e.g. a raw column value left over from switching conditions)
            # can never push the percentile argument outside [0, 100] and crash.
            safe_pct = max(0.0, min(100.0, float(threshold)))
            pct_val = np.percentile(df_copy[base_col].dropna(), 100 - safe_pct)
            df_copy[target_name] = (df_copy[base_col] >= pct_val).astype(int)
        return df_copy

    @staticmethod
    def detect_leakage_risks(
        df: pd.DataFrame,
        target_col: str,
        task_type: str,
        col_types: Dict[str, List[str]]
    ) -> List[Dict[str, str]]:
        """
        Heuristically flags features that are suspiciously predictive of the
        target — the classic symptom of data leakage (a feature that is
        itself derived from, or a near-duplicate of, the target, so the
        model "cheats" during training and then fails in the real world).

        Returns a list of {"feature", "reason", "action", "severity"} dicts,
        sorted by severity (high risk first). This is intentionally simple —
        correlation / purity thresholds, not a formal causal test — but it
        catches the most common real-world mistakes (e.g. accidentally
        leaving in a column the target was derived from, or an ID-like
        column that happens to perfectly separate classes).
        """
        risks: List[Dict[str, str]] = []
        if target_col not in df.columns:
            return risks

        candidate_cols = [c for c in df.columns if c != target_col and c not in col_types.get("id_or_url", [])]

        # 1. Exact-duplicate-of-target check (works for any dtype)
        target_series = df[target_col]
        for col in candidate_cols:
            try:
                if df[col].equals(target_series):
                    risks.append({
                        "feature": col, "severity": "high",
                        "reason_key": "duplicate_of_target", "action_key": "remove_column"
                    })
            except Exception:
                pass

        # 2. Numeric correlation with target (binary/continuous targets only)
        if task_type in ("classification", "regression"):
            try:
                if task_type == "classification":
                    uniq = target_series.dropna().unique()
                    if len(uniq) == 2:
                        y_numeric = target_series.map({uniq[0]: 0, uniq[1]: 1}).astype(float)
                    else:
                        y_numeric = None
                else:
                    y_numeric = pd.to_numeric(target_series, errors="coerce")

                if y_numeric is not None:
                    for col in col_types.get("numeric", []):
                        if col == target_col or col not in df.columns:
                            continue
                        try:
                            x_numeric = pd.to_numeric(df[col], errors="coerce")
                            corr = x_numeric.corr(y_numeric)
                            if pd.isna(corr):
                                continue
                            abs_corr = abs(corr)
                            if abs_corr >= 0.95:
                                risks.append({
                                    "feature": col, "severity": "high",
                                    "reason_key": "high_correlation", "corr": f"{corr:.2f}",
                                    "action_key": "remove_or_check"
                                })
                            elif abs_corr >= 0.85:
                                risks.append({
                                    "feature": col, "severity": "medium",
                                    "reason_key": "moderate_correlation", "corr": f"{corr:.2f}",
                                    "action_key": "review_column"
                                })
                        except Exception:
                            continue
            except Exception:
                pass

        # 3. Near-perfect categorical purity (a category value that almost
        #    always maps to a single class, with enough support to be real)
        if task_type == "classification":
            for col in col_types.get("categorical", []) + col_types.get("boolean", []):
                if col == target_col or col not in df.columns:
                    continue
                try:
                    sub = df[[col, target_col]].dropna()
                    if len(sub) < 20:
                        continue
                    purity = sub.groupby(col)[target_col].value_counts(normalize=True).groupby(level=0).max()
                    support = sub[col].value_counts()
                    for cat_val, p in purity.items():
                        if p >= 0.98 and support.get(cat_val, 0) >= max(10, int(0.03 * len(sub))):
                            risks.append({
                                "feature": col, "severity": "medium",
                                "reason_key": "categorical_purity", "value": str(cat_val),
                                "action_key": "review_column"
                            })
                            break
                except Exception:
                    continue

        severity_order = {"high": 0, "medium": 1, "low": 2}
        risks.sort(key=lambda r: severity_order.get(r["severity"], 3))
        # De-duplicate by feature, keeping the highest-severity entry
        seen = set()
        deduped = []
        for r in risks:
            if r["feature"] in seen:
                continue
            seen.add(r["feature"])
            deduped.append(r)
        return deduped

    @staticmethod
    def detect_data_gaps(df: pd.DataFrame, target_col: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Flags columns with data-quality gaps: heavy missingness or almost no
        variance (a single value dominating the column). Both silently hurt
        a model without showing up as an error, so they're surfaced explicitly
        rather than left for the user to discover after training.
        """
        gaps: List[Dict[str, Any]] = []
        n_rows = len(df)
        if n_rows == 0:
            return gaps

        for col in df.columns:
            if col == target_col:
                continue
            series = df[col]
            missing_pct = float(series.isnull().sum() / n_rows * 100)
            if missing_pct >= 30.0:
                gaps.append({"type": "high_missing", "feature": col, "pct": missing_pct})

            non_null = series.dropna()
            if len(non_null) > 10:
                top_freq = non_null.value_counts(normalize=True).iloc[0]
                if top_freq >= 0.98:
                    gaps.append({"type": "low_variance", "feature": col})

        return gaps

    @staticmethod
    def get_dataset_summary(df: pd.DataFrame) -> Dict[str, Any]:
        """Returns comprehensive metadata and health statistics of dataset."""
        n_rows, n_cols = df.shape
        missing_counts = df.isnull().sum()
        missing_pcts = (missing_counts / n_rows * 100).round(2)
        duplicates = df.duplicated().sum()

        cols_info = []
        for col in df.columns:
            dtype_str = str(df[col].dtype)
            missing = int(missing_counts[col])
            missing_pct = float(missing_pcts[col])
            n_unique = int(df[col].nunique())
            sample_val = str(df[col].dropna().iloc[0]) if len(df[col].dropna()) > 0 else "N/A"
            cols_info.append({
                "Column": col,
                "Type": dtype_str,
                "Missing": missing,
                "Missing (%)": f"{missing_pct}%",
                "Unique": n_unique,
                "Sample": sample_val[:30] + "..." if len(sample_val) > 30 else sample_val
            })

        return {
            "n_rows": n_rows,
            "n_cols": n_cols,
            "duplicates": int(duplicates),
            "total_missing": int(missing_counts.sum()),
            "columns_info": pd.DataFrame(cols_info)
        }
