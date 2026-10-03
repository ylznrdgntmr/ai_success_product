"""
Preprocessing Pipeline Module
Builds dynamic Sklearn ColumnTransformer pipelines handling numerical,
categorical, and text feature transformations.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

from src.core.image_features import ImageFeatureExtractor


class TextColumnTransformer(BaseEstimator, TransformerMixin):
    """Custom transformer to handle single text column with TF-IDF within ColumnTransformer."""
    def __init__(self, max_features: int = 40):
        self.max_features = max_features
        self.vectorizer = TfidfVectorizer(max_features=self.max_features, stop_words="english", dtype=np.float32)

    def fit(self, X, y=None):
        # Flatten input 1D series or array
        if isinstance(X, pd.DataFrame):
            text_series = X.iloc[:, 0].fillna("").astype(str)
        elif isinstance(X, pd.Series):
            text_series = X.fillna("").astype(str)
        else:
            text_series = pd.Series(X.ravel()).fillna("").astype(str)
        self.vectorizer.fit(text_series)
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            text_series = X.iloc[:, 0].fillna("").astype(str)
        elif isinstance(X, pd.Series):
            text_series = X.fillna("").astype(str)
        else:
            text_series = pd.Series(X.ravel()).fillna("").astype(str)
        return self.vectorizer.transform(text_series).toarray().astype(np.float32)

    def get_feature_names_out(self, input_features=None):
        prefix = input_features[0] if input_features is not None and len(input_features) > 0 else "text"
        return [f"{prefix}_tfidf_{feat}" for feat in self.vectorizer.get_feature_names_out()]


class BooleanTransformer(BaseEstimator, TransformerMixin):
    """Custom transformer to safely convert boolean/binary columns to float32 0.0/1.0 with imputation."""
    def __init__(self):
        self.imputer = SimpleImputer(strategy="most_frequent")

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            X_cast = X.fillna(False).astype(np.float32).values
        elif isinstance(X, pd.Series):
            X_cast = X.fillna(False).astype(np.float32).values.reshape(-1, 1)
        else:
            X_cast = np.asarray(X, dtype=np.float32)
            X_cast = np.nan_to_num(X_cast, nan=0.0)
        self.imputer.fit(X_cast)
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            X_cast = X.fillna(False).astype(np.float32).values
        elif isinstance(X, pd.Series):
            X_cast = X.fillna(False).astype(np.float32).values.reshape(-1, 1)
        else:
            X_cast = np.asarray(X, dtype=np.float32)
            X_cast = np.nan_to_num(X_cast, nan=0.0)
        return self.imputer.transform(X_cast).astype(np.float32)

    def get_feature_names_out(self, input_features=None):
        if input_features is not None:
            return list(input_features)
        return ["bool_feat"]


class PreprocessingBuilder:
    """Constructs dynamic Preprocessing Pipelines for arbitrary datasets."""

    @staticmethod
    def create_preprocessor(
        numeric_cols: List[str],
        categorical_cols: List[str],
        text_cols: List[str],
        boolean_cols: List[str],
        scaler_type: str = "standard",
        max_tfidf_features: int = 40,
        max_categories: int = 30,
        image_url_col: Optional[str] = None,
        image_n_components: int = 32,
        image_max_fit_rows: Optional[int] = 300,
        image_cache_dir: str = ".image_feature_cache"
    ) -> Tuple[ColumnTransformer, List[str]]:
        """
        Builds a memory-efficient Sklearn ColumnTransformer that scales to massive datasets.

        image_url_col is optional and off by default: when set, a ResNet50+PCA
        image branch (see src.core.image_features.ImageFeatureExtractor) is
        added to the SAME ColumnTransformer used for training and inference,
        so a live prediction's photo is always processed for real — never
        swapped for a placeholder.
        """
        transformers = []

        # 1. Numeric pipeline
        if numeric_cols:
            if scaler_type == "robust":
                scaler = RobustScaler()
            elif scaler_type == "minmax":
                scaler = MinMaxScaler()
            else:
                scaler = StandardScaler()

            num_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", scaler)
            ])
            transformers.append(("num", num_pipeline, numeric_cols))

        # 2. Boolean pipeline
        if boolean_cols:
            transformers.append(("bool", BooleanTransformer(), boolean_cols))

        # 3. Categorical pipeline (with max_categories cap to prevent memory blowup)
        if categorical_cols:
            cat_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
                ("onehot", OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                    max_categories=max_categories,
                    min_frequency=0.002,
                    dtype=np.float32
                ))
            ])
            transformers.append(("cat", cat_pipeline, categorical_cols))

        # 4. Text pipelines (one per text column)
        for idx, txt_col in enumerate(text_cols):
            txt_trans = TextColumnTransformer(max_features=max_tfidf_features)
            transformers.append((f"txt_{idx}", txt_trans, [txt_col]))

        # 5. Image pipeline (optional, opt-in — see image_features.py)
        image_cols: List[str] = []
        if image_url_col:
            img_trans = ImageFeatureExtractor(
                n_components=image_n_components,
                max_fit_rows=image_max_fit_rows,
                cache_dir=image_cache_dir
            )
            transformers.append(("img", img_trans, [image_url_col]))
            image_cols = [image_url_col]

        preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder="drop",
            verbose_feature_names_out=False
        )

        all_input_cols = numeric_cols + boolean_cols + categorical_cols + text_cols + image_cols
        return preprocessor, all_input_cols

    @staticmethod
    def get_feature_names_from_preprocessor(preprocessor: ColumnTransformer) -> List[str]:
        """Extracts output feature names after ColumnTransformer is fitted."""
        try:
            return list(preprocessor.get_feature_names_out())
        except Exception:
            feature_names = []
            for name, trans, cols in preprocessor.transformers_:
                if name == "remainder" and trans == "drop":
                    continue
                if hasattr(trans, "get_feature_names_out"):
                    try:
                        feature_names.extend(trans.get_feature_names_out(cols))
                    except Exception:
                        feature_names.extend([f"{cols[0]}_{i}" for i in range(10)])
                elif hasattr(trans, "named_steps"):
                    last_step = trans.steps[-1][1]
                    if hasattr(last_step, "get_feature_names_out"):
                        feature_names.extend(last_step.get_feature_names_out(cols))
                    else:
                        feature_names.extend(cols)
                else:
                    feature_names.extend(cols)
            return feature_names
