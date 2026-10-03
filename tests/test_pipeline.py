"""
Comprehensive Unit and Integration Tests for AI Product Success Predictor
"""

import os
import sys
import tempfile
import unittest
from unittest.mock import patch
import pandas as pd
import numpy as np

# Ensure root directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# The app now requires sign-in. Point the local user store at an isolated
# temp file for the whole test run *before* src.utils.auth is imported by
# anything (directly here, or indirectly when AppTest executes app.py) —
# its storage path is read once at import time. This keeps tests from
# reading/writing the real .local_users.json on disk.
_TEST_USER_STORE = os.path.join(tempfile.gettempdir(), "app_test_users.json")
if os.path.exists(_TEST_USER_STORE):
    os.remove(_TEST_USER_STORE)
os.environ["APP_USER_STORE_PATH"] = _TEST_USER_STORE

from src.utils.sample_data import generate_ecommerce_dataset, generate_saas_dataset, generate_crowdfunding_dataset, SAMPLE_DATASETS
from src.core.data_processor import DataProcessor
from src.core.eda import AutoEDA
from src.core.pipeline import PreprocessingBuilder
from src.core.models import ModelTrainer
from src.core.explainer import ModelExplainer
from src.llm.advisor import AIProductAdvisor
from src.utils.report_export import generate_markdown_report, generate_pdf_report
import src.core.image_features as image_features


class TestAIProductSuccessPredictor(unittest.TestCase):
    """Test Suite for the end-to-end ML & AI pipeline."""

    def test_01_sample_data_generation(self):
        """Tests that all sample datasets are properly generated with expected shapes."""
        df_ecom = generate_ecommerce_dataset(n_samples=100)
        self.assertEqual(len(df_ecom), 100)
        self.assertIn("is_best_seller", df_ecom.columns)
        self.assertIn("price", df_ecom.columns)

        df_saas = generate_saas_dataset(n_samples=100)
        self.assertEqual(len(df_saas), 100)
        self.assertIn("is_successful_launch", df_saas.columns)

        df_crowd = generate_crowdfunding_dataset(n_samples=100)
        self.assertEqual(len(df_crowd), 100)
        self.assertIn("is_successfully_funded", df_crowd.columns)

    def test_02_data_processor_column_detection(self):
        """Tests automatic column classification (numeric, categorical, text, boolean, id)."""
        df = generate_ecommerce_dataset(n_samples=50)
        col_types = DataProcessor.detect_column_types(df, target_col="is_best_seller")
        
        self.assertIn("price", col_types["numeric"])
        self.assertIn("category", col_types["categorical"])
        self.assertIn("has_prime_badge", col_types["boolean"])
        self.assertIn("product_id", col_types["id_or_url"])

        target_info = DataProcessor.inspect_target(df, "is_best_seller")
        self.assertEqual(target_info["task_type"], "classification")
        self.assertEqual(target_info["subtype"], "binary")

    def test_03_derived_target_creation(self):
        """Tests creating dynamic rule-based target columns."""
        df = generate_ecommerce_dataset(n_samples=50)
        df_derived = DataProcessor.create_derived_target(
            df=df,
            base_col="monthly_sales_units",
            condition=">=",
            threshold=500,
            target_name="custom_sales_success"
        )
        self.assertIn("custom_sales_success", df_derived.columns)
        self.assertTrue(set(df_derived["custom_sales_success"].unique()).issubset({0, 1}))

    def test_04_pipeline_and_model_training(self):
        """Tests building dynamic preprocessor and running AutoML model training."""
        df = generate_ecommerce_dataset(n_samples=150, random_state=42)
        col_types = DataProcessor.detect_column_types(df, target_col="is_best_seller")

        preprocessor, input_cols = PreprocessingBuilder.create_preprocessor(
            numeric_cols=col_types["numeric"],
            categorical_cols=col_types["categorical"],
            text_cols=col_types["text"],
            boolean_cols=col_types["boolean"],
            scaler_type="standard",
            max_tfidf_features=20
        )

        X = df[input_cols]
        y = df["is_best_seller"]

        # Run benchmark with subset of models
        results = ModelTrainer.train_and_benchmark(
            X_df=X,
            y_series=y,
            preprocessor=preprocessor,
            task_type="classification",
            test_size=0.25,
            selected_model_names=["Random Forest", "Logistic Regression"]
        )

        self.assertIsNotNone(results["best_model_name"])
        self.assertIsNotNone(results["full_pipeline"])
        self.assertFalse(results["benchmark_table"].empty)

        # Test single prediction
        sample_input = X.iloc[[0]]
        single_pred = ModelExplainer.explain_single_prediction(
            pipeline=results["full_pipeline"],
            input_row_df=sample_input,
            task_type="classification"
        )
        self.assertIn(single_pred["prediction"], [0, 1])
        self.assertIsNotNone(single_pred["probability"])

    def test_05_what_if_simulation(self):
        """Tests What-If parameter sensitivity simulation."""
        df = generate_ecommerce_dataset(n_samples=120, random_state=42)
        col_types = DataProcessor.detect_column_types(df, target_col="is_best_seller")

        preprocessor, input_cols = PreprocessingBuilder.create_preprocessor(
            numeric_cols=col_types["numeric"],
            categorical_cols=col_types["categorical"],
            text_cols=col_types["text"],
            boolean_cols=col_types["boolean"]
        )

        results = ModelTrainer.train_and_benchmark(
            X_df=df[input_cols],
            y_series=df["is_best_seller"],
            preprocessor=preprocessor,
            task_type="classification",
            selected_model_names=["Logistic Regression"]
        )

        sim_df, sim_fig = ModelExplainer.simulate_what_if(
            pipeline=results["full_pipeline"],
            base_product_df=df[input_cols].iloc[[0]],
            feature_to_vary="price",
            variation_range=[19.99, 49.99, 99.99, 149.99],
            task_type="classification"
        )
        self.assertEqual(len(sim_df), 4)
        self.assertIn("Success Rate / Score", sim_df.columns)

    def test_06_llm_advisor_heuristic_fallback(self):
        """Tests AIProductAdvisor fallback generation with rich SWOT and recommendations."""
        advisor = AIProductAdvisor(api_key=None) # Tests heuristic engine
        
        sample_prod = {
            "title": "Ultra Ergonomic Mechanical Keyboard",
            "price": 89.99,
            "rating": 4.6,
            "reviews_count": 450,
            "has_prime_badge": True,
            "has_product_video": True
        }

        res = advisor.analyze_product_success(
            product_dict=sample_prod,
            prediction_label="BAŞARILI (1)",
            probability=0.82,
            best_model_name="XGBoost",
            model_metric="F1: 0.88",
            top_drivers=[{"Feature": "rating", "Impact": 0.45}]
        )

        self.assertIn("SWOT Analizi", res["analysis_markdown"])
        self.assertIn("Güçlü Yönler", res["analysis_markdown"])
        self.assertIn("82.0", res["analysis_markdown"])

    def test_07_report_export(self):
        """Tests Markdown and PDF report generation."""
        sample_prod = {"price": 99.99, "rating": 4.5}
        analysis_text = "### Yönetici Özeti\nÜrün başarı oranı yüksektir."

        md_rep = generate_markdown_report(
            product_dict=sample_prod,
            prediction_label="1",
            probability=0.78,
            best_model_name="Random Forest",
            model_metric="0.85",
            analysis_markdown=analysis_text,
            source="Test Heuristic"
        )
        self.assertIn("# 🚀 AI Product Success Predictor", md_rep)

        pdf_bytes = generate_pdf_report(
            product_dict=sample_prod,
            prediction_label="1",
            probability=0.78,
            best_model_name="Random Forest",
            model_metric="0.85",
            analysis_text=analysis_text
        )
        self.assertGreater(len(pdf_bytes), 100)

    def test_08_zip_file_extraction_and_loading(self):
        """Tests reading, listing and extracting datasets inside a ZIP archive."""
        import zipfile
        import io

        # Create in-memory zip archive with 2 CSV datasets
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as z:
            df_products = pd.DataFrame({
                "product_name": ["Prod A", "Prod B", "Prod C"],
                "price": [10.0, 20.0, 30.0],
                "success": [1, 0, 1]
            })
            df_cats = pd.DataFrame({
                "cat_id": [1, 2],
                "cat_name": ["Tech", "Home"]
            })
            z.writestr("amazon_products.csv", df_products.to_csv(index=False))
            z.writestr("amazon_categories.csv", df_cats.to_csv(index=False))

        zip_buffer.seek(0)

        # 1. List contents
        entries = DataProcessor.list_zip_contents(zip_buffer)
        self.assertEqual(len(entries), 2)
        filenames = [e["filename"] for e in entries]
        self.assertIn("amazon_products.csv", filenames)
        self.assertIn("amazon_categories.csv", filenames)

        # 2. Extract and load specific file
        zip_buffer.seek(0)
        loaded_df = DataProcessor.load_dataset(zip_buffer, filename="archive.zip", file_inside_zip="amazon_products.csv")
        self.assertEqual(len(loaded_df), 3)
        self.assertIn("product_name", loaded_df.columns)
        self.assertIn("price", loaded_df.columns)

    def test_09_high_cardinality_and_large_dataset_memory_safety(self):
        """Tests that datasets with 10,000+ rows and 2,000+ unique categories train without memory blowup."""
        n_rows = 5000
        df_large = pd.DataFrame({
            "high_card_cat": [f"Category_{i % 1500}" for i in range(n_rows)],
            "title": [f"Product Title with words {i % 300}" for i in range(n_rows)],
            "price": np.random.uniform(10, 500, size=n_rows),
            "is_success": np.random.choice([0, 1], size=n_rows)
        })

        col_types = DataProcessor.detect_column_types(df_large, target_col="is_success")

        preprocessor, input_cols = PreprocessingBuilder.create_preprocessor(
            numeric_cols=col_types["numeric"],
            categorical_cols=col_types["categorical"],
            text_cols=col_types["text"],
            boolean_cols=col_types["boolean"],
            max_categories=25
        )

        results = ModelTrainer.train_and_benchmark(
            X_df=df_large[input_cols],
            y_series=df_large["is_success"],
            preprocessor=preprocessor,
            task_type="classification",
            selected_model_names=["Logistic Regression"],
            max_train_samples=2500
        )

        self.assertIsNotNone(results["best_model_name"])
        self.assertFalse(results["benchmark_table"].empty)

    def test_10_no_models_selected_raises_clear_error(self):
        """Regression test: selecting zero models must raise a clear error, not crash
        deep inside pandas with a KeyError on an empty benchmark table."""
        df = generate_ecommerce_dataset(n_samples=60, random_state=42)
        col_types = DataProcessor.detect_column_types(df, target_col="is_best_seller")
        preprocessor, input_cols = PreprocessingBuilder.create_preprocessor(
            numeric_cols=col_types["numeric"],
            categorical_cols=col_types["categorical"],
            text_cols=col_types["text"],
            boolean_cols=col_types["boolean"],
        )
        with self.assertRaises(ValueError):
            ModelTrainer.train_and_benchmark(
                X_df=df[input_cols],
                y_series=df["is_best_seller"],
                preprocessor=preprocessor,
                task_type="classification",
                selected_model_names=[],
            )

    def test_11_binary_benchmark_includes_positive_class_f1_and_selects_by_it(self):
        """Best-model selection should rank by positive-class ('success') F1 for
        binary classification, not just weighted F1 which can hide poor recall
        on the minority class the whole app is meant to be predicting."""
        df = generate_ecommerce_dataset(n_samples=200, random_state=1)
        col_types = DataProcessor.detect_column_types(df, target_col="is_best_seller")
        preprocessor, input_cols = PreprocessingBuilder.create_preprocessor(
            numeric_cols=col_types["numeric"],
            categorical_cols=col_types["categorical"],
            text_cols=col_types["text"],
            boolean_cols=col_types["boolean"],
        )
        results = ModelTrainer.train_and_benchmark(
            X_df=df[input_cols],
            y_series=df["is_best_seller"],
            preprocessor=preprocessor,
            task_type="classification",
            selected_model_names=["Random Forest", "Logistic Regression"],
        )
        self.assertIn("F1 (Başarı Sınıfı)", results["benchmark_table"].columns)
        self.assertIsNotNone(results["best_model_name"])

    def test_13_leakage_detection_catches_common_mistakes(self):
        """Regression test for the new data-leakage checker: it must flag an
        exact duplicate of the target, a near-perfectly-correlated numeric
        column, and a categorical value that almost always maps to one
        class — the three most common real-world leakage mistakes."""
        np.random.seed(0)
        n = 200
        target = np.random.choice([0, 1], size=n, p=[0.6, 0.4])
        df = pd.DataFrame({
            "price": np.random.uniform(10, 200, n),
            "leaky_copy": target,
            "almost_leaky": target * 10 + np.random.normal(0, 0.05, n),
            "category": np.random.choice(["A", "B", "C"], n),
            "rare_signal": np.where(target == 1, "special", np.random.choice(["x", "y"], n)),
            "is_success": target,
        })
        col_types = DataProcessor.detect_column_types(df, target_col="is_success")
        risks = DataProcessor.detect_leakage_risks(df, "is_success", "classification", col_types)
        flagged = {r["feature"]: r for r in risks}

        self.assertIn("leaky_copy", flagged)
        self.assertEqual(flagged["leaky_copy"]["severity"], "high")
        self.assertIn("almost_leaky", flagged)
        self.assertEqual(flagged["almost_leaky"]["severity"], "high")
        self.assertIn("rare_signal", flagged)
        # A column with no real relationship to the target must not be flagged.
        self.assertNotIn("price", flagged)
        self.assertNotIn("category", flagged)

    def test_14_data_gap_detection_flags_missingness_and_low_variance(self):
        """Regression test: heavy missingness and near-constant columns should
        both be surfaced as data-quality gaps, independent of leakage checks."""
        df = pd.DataFrame({
            "mostly_missing": [np.nan] * 150 + list(np.random.uniform(0, 1, 50)),
            "constant_col": ["same_value"] * 200,
            "healthy_col": np.random.uniform(0, 1, 200),
            "target": np.random.choice([0, 1], size=200),
        })
        gaps = DataProcessor.detect_data_gaps(df, target_col="target")
        gap_types = {g["feature"]: g["type"] for g in gaps}

        self.assertEqual(gap_types.get("mostly_missing"), "high_missing")
        self.assertEqual(gap_types.get("constant_col"), "low_variance")
        self.assertNotIn("healthy_col", gap_types)
        self.assertNotIn("target", gap_types)  # target itself must never be flagged as its own gap

    def test_15_i18n_translations_cover_all_three_languages(self):
        """Every key defined for Turkish (the default) must also exist for
        English and German, and t() must gracefully fall back rather than
        crash on an unknown key."""
        from src.utils import i18n

        tr_keys = set(i18n.TRANSLATIONS["tr"].keys())
        en_keys = set(i18n.TRANSLATIONS["en"].keys())
        de_keys = set(i18n.TRANSLATIONS["de"].keys())

        missing_en = tr_keys - en_keys
        missing_de = tr_keys - de_keys
        self.assertEqual(missing_en, set(), f"English translation missing keys: {missing_en}")
        self.assertEqual(missing_de, set(), f"German translation missing keys: {missing_de}")

    def test_16_auth_register_login_and_password_hashing(self):
        """Unit-level test of the auth module: registration validation,
        successful login, wrong-password rejection, and that the password
        is never stored anywhere in plain text."""
        import json
        import tempfile
        from src.utils import auth as auth_module

        with tempfile.TemporaryDirectory() as tmp:
            store_path = os.path.join(tmp, "users.json")
            store = auth_module.JSONUserStore(path=store_path)
            original_store = auth_module._store
            auth_module._store = store
            try:
                # Too-short password is rejected
                err = auth_module.register_user("Test User", "test@example.com", "123")
                self.assertIsNotNone(err)

                # Valid registration succeeds
                err = auth_module.register_user("Test User", "test@example.com", "securepass1")
                self.assertIsNone(err)

                # Duplicate email is rejected
                err = auth_module.register_user("Other", "test@example.com", "securepass2")
                self.assertIsNotNone(err)

                # Correct credentials authenticate
                user = auth_module.authenticate_user("test@example.com", "securepass1")
                self.assertIsNotNone(user)
                self.assertEqual(user["email"], "test@example.com")

                # Wrong password is rejected
                self.assertIsNone(auth_module.authenticate_user("test@example.com", "wrongpassword"))

                # The password must never appear in plain text on disk
                with open(store_path, "r", encoding="utf-8") as f:
                    raw = f.read()
                self.assertNotIn("securepass1", raw)
            finally:
                auth_module._store = original_store

    def test_17_llm_provider_registry_and_model_resolution(self):
        """All four required providers must be registered, and an explicit
        model override must take priority over the env-var/default model."""
        from src.llm.providers import PROVIDERS, resolve_model_name

        for key in ["gemini", "claude", "deepseek", "qwen"]:
            self.assertIn(key, PROVIDERS)
            self.assertIn("label", PROVIDERS[key])
            self.assertIn("default_model", PROVIDERS[key])

        self.assertEqual(resolve_model_name("claude", override="custom-model"), "custom-model")
        self.assertEqual(resolve_model_name("claude", override=None), PROVIDERS["claude"]["default_model"])

    def test_18_heuristic_advisor_supports_all_three_languages(self):
        """The no-API-key fallback advisor must produce a real report in
        Turkish, English and German, not just Turkish."""
        sample_prod = {"price": 89.99, "rating": 4.6, "reviews_count": 450}
        for lang in ["tr", "en", "de"]:
            advisor = AIProductAdvisor(provider="gemini", api_key=None, lang=lang)
            res = advisor.analyze_product_success(
                product_dict=sample_prod,
                prediction_label="SUCCESS",
                probability=0.82,
                best_model_name="XGBoost",
                model_metric="F1: 0.88",
                top_drivers=[{"Feature": "rating", "Impact": 0.45}]
            )
            self.assertGreater(len(res["analysis_markdown"]), 200)
            self.assertIn("SWOT", res["analysis_markdown"])

    def test_12_derived_target_top_percent_clamps_out_of_range_threshold(self):
        """Regression test: a 'Top %' threshold outside [0, 100] (e.g. a leftover
        raw column value from switching conditions in the UI) must not crash
        np.percentile — it should be clamped instead."""
        df = generate_ecommerce_dataset(n_samples=100, random_state=3)
        # 500 is not a valid percentage; this used to raise
        # "ValueError: Percentiles must be in the range [0, 100]".
        df_new = DataProcessor.create_derived_target(
            df=df,
            base_col="price",
            condition="Top %",
            threshold=500,
            target_name="price_top_target",
        )
        self.assertIn("price_top_target", df_new.columns)
        self.assertTrue(set(df_new["price_top_target"].unique()).issubset({0, 1}))


def _tensorflow_available():
    try:
        import tensorflow  # noqa: F401
        return True
    except ImportError:
        return False


def _fake_download_and_preprocess(url, timeout=6.0):
    """Deterministic stand-in for a real ResNet50 preprocessing step, keyed by URL
    so different URLs produce different (but reproducible) pixel arrays. URLs
    containing 'broken' simulate an unreachable image."""
    if "broken" in url:
        return None
    rng = np.random.RandomState(abs(hash(url)) % (2**32))
    return rng.rand(224, 224, 3).astype(np.float32)


class _FakeResNet:
    """Stand-in for the real Keras ResNet50 model: turns each preprocessed image
    array into a deterministic, distinguishable 2048-d vector without needing
    TensorFlow or network access in tests."""
    def predict(self, batch_arr, verbose=0):
        n = batch_arr.shape[0]
        flat_means = batch_arr.reshape(n, -1).mean(axis=1)
        return np.tile(flat_means[:, None], (1, 2048)).astype(np.float32)


class TestImageFeatureExtractor(unittest.TestCase):
    """Tests for the optional image-feature pipeline (mocked network + model calls
    so these run fast and offline). Focus: PCA output shape, that a failed
    download falls back to the training-set MEAN embedding (not a zero vector —
    that was the exact bug in the original prototype), and that the SAME
    pipeline object re-processes a product's real photo at prediction time."""

    def _patched(self):
        return patch.multiple(
            image_features,
            _download_and_preprocess=_fake_download_and_preprocess,
            _get_resnet_model=lambda: _FakeResNet(),
        )

    def test_fit_transform_shape_and_broken_url_fallback(self):
        import tempfile
        with self._patched(), tempfile.TemporaryDirectory() as tmp:
            extractor = image_features.ImageFeatureExtractor(n_components=4, max_fit_rows=10, cache_dir=tmp)
            # PCA needs at least as many successfully-embedded samples as the
            # requested component count, so fit on a few extra distinct URLs.
            fit_urls = pd.DataFrame({"img": [f"https://example.com/fit_{i}.jpg" for i in range(6)]})
            extractor.fit(fit_urls)

            urls = pd.DataFrame({"img": [
                "https://example.com/a.jpg",
                "https://example.com/b.jpg",
                "https://example.com/broken.jpg",
                "https://example.com/c.jpg",
            ]})
            out = extractor.transform(urls)

            self.assertEqual(out.shape, (4, 4))
            # The broken URL must fall back to the fitted mean embedding, which
            # is virtually never the zero vector — this is the regression check
            # for the "dummy zero-vector at inference" bug from the prototype.
            self.assertFalse(np.allclose(out[2], 0.0))

            # Repeated transform on the same URLs must hit the on-disk cache and
            # return identical output (no re-download/re-embed drift).
            out2 = extractor.transform(urls)
            np.testing.assert_allclose(out, out2, atol=1e-6)

    @unittest.skipUnless(_tensorflow_available(), "TensorFlow not installed (optional image feature)")
    def test_local_uploaded_file_path_is_processed_like_a_url(self):
        """A user-uploaded photo (no public URL yet, e.g. an unlaunched product)
        is saved to a local path and fed to the transformer as that path
        instead of a URL — this must be preprocessed for real, not silently
        dropped. Runs the actual Pillow/Keras preprocessing (no network
        needed), only mocking the final ResNet50 forward pass."""
        import tempfile
        from PIL import Image

        with tempfile.TemporaryDirectory() as tmp:
            local_path = f"{tmp}/uploaded.png"
            Image.new("RGB", (64, 64), color=(10, 200, 30)).save(local_path)

            arr = image_features._download_and_preprocess(local_path)
            self.assertIsNotNone(arr)
            self.assertEqual(arr.shape, (224, 224, 3))

            # A path that doesn't exist and isn't a URL must fail cleanly (None),
            # not raise.
            self.assertIsNone(image_features._download_and_preprocess(f"{tmp}/missing.png"))

    def test_full_pipeline_with_image_column_trains_and_predicts(self):
        """End-to-end: a tabular dataset + an image URL column trains through
        ModelTrainer, and the resulting full_pipeline can score a brand-new
        single-row product — exercising the exact code path the Streamlit
        single-prediction / What-If / batch screens use."""
        import tempfile
        df = generate_ecommerce_dataset(n_samples=80, random_state=7)
        df["image_url"] = [f"https://example.com/prod_{i}.jpg" for i in range(len(df))]
        col_types = DataProcessor.detect_column_types(df, target_col="is_best_seller")

        with self._patched(), tempfile.TemporaryDirectory() as tmp:
            preprocessor, input_cols = PreprocessingBuilder.create_preprocessor(
                numeric_cols=col_types["numeric"],
                categorical_cols=col_types["categorical"],
                text_cols=col_types["text"],
                boolean_cols=col_types["boolean"],
                image_url_col="image_url",
                image_n_components=4,
                image_max_fit_rows=20,
                image_cache_dir=tmp,
            )
            self.assertIn("image_url", input_cols)

            X = df[input_cols]
            y = df["is_best_seller"]
            results = ModelTrainer.train_and_benchmark(
                X_df=X,
                y_series=y,
                preprocessor=preprocessor,
                task_type="classification",
                selected_model_names=["Logistic Regression"],
            )

            pipeline = results["full_pipeline"]
            new_product = X.iloc[[0]].copy()
            new_product["image_url"] = "https://example.com/brand_new_product.jpg"
            pred = pipeline.predict(new_product)
            self.assertIn(pred[0], [0, 1])
            proba = pipeline.predict_proba(new_product)
            self.assertEqual(proba.shape, (1, 2))


class TestNewProductModeAppFlow(unittest.TestCase):
    """Drives the actual Streamlit app (via AppTest) end-to-end to verify the
    '🆕 Yeni Ürün / Lansman Öncesi' training mode really excludes traction
    columns (reviews, etc.) from both the trained pipeline's inputs and the
    Tab 5 prediction form — and that leaving it off still behaves as before.
    This exercises real widget/session-state wiring that a plain unit test
    calling the ML functions directly would not catch."""

    TEST_EMAIL = "apptest@example.com"
    TEST_PASSWORD = "apptest123"

    @classmethod
    def setUpClass(cls):
        from src.utils.auth import register_user
        # Idempotent: ignore "already exists" if this class runs more than once.
        register_user(name="AppTest User", email=cls.TEST_EMAIL, password=cls.TEST_PASSWORD)

    def _login(self, at):
        """Drives the sign-in form so the rest of the app (behind the auth gate) renders.
        All widget interactions are queued and only take effect on the next
        at.run(), which is exactly how a real browser submits a st.form: fill
        both fields, then click submit, then rerun once."""
        at.text_input(key="login_email").set_value(self.TEST_EMAIL)
        at.text_input(key="login_password").set_value(self.TEST_PASSWORD)
        login_btn = [b for b in at.button if b.label == "🔓 Giriş Yap"][0]
        login_btn.click()
        at.run()

    def _train(self, enable_new_product_mode: bool):
        from streamlit.testing.v1 import AppTest
        app_path = os.path.join(os.path.dirname(__file__), "..", "app.py")
        at = AppTest.from_file(app_path, default_timeout=120)
        at.run()
        self._login(at)
        if enable_new_product_mode:
            cb = [c for c in at.checkbox if "piyasaya sürülmemiş" in c.label][0]
            cb.set_value(True)
            at.run()
        train_btn = [b for b in at.button if "AutoML Model Eğitimini Başlat" in b.label][0]
        train_btn.click()
        at.run()
        self.assertEqual(len(at.exception), 0, msg=f"App raised an exception: {list(at.exception)}")
        return at

    def test_new_product_mode_excludes_traction_columns_everywhere(self):
        at = self._train(enable_new_product_mode=True)
        input_cols = at.session_state["input_cols"]
        self.assertNotIn("reviews_count", input_cols)
        self.assertFalse(any("review" in c.lower() or "rating" in c.lower() for c in input_cols))

        # Tab 5's prediction form must not ask for a column the model was
        # never trained on.
        num_labels = [ni.label for ni in at.number_input]
        self.assertFalse(any("reviews_count" in l for l in num_labels))

        # The "enter 0 for a new product" caution note has nothing left to
        # warn about, so it should no longer render.
        captions = [c.value for c in at.caption]
        self.assertFalse(any("mevcut" in c and "performans" in c for c in captions))

    def test_default_mode_still_includes_traction_columns(self):
        at = self._train(enable_new_product_mode=False)
        input_cols = at.session_state["input_cols"]
        self.assertIn("reviews_count", input_cols)

        captions = [c.value for c in at.caption]
        self.assertTrue(any("mevcut" in c and "performans" in c for c in captions))


if __name__ == "__main__":
    unittest.main()
