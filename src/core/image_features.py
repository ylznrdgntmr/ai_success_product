"""
Image Feature Extraction Module
Downloads product images from a URL column and extracts visual embeddings
via a pretrained ResNet50 CNN (ImageNet weights), reduced with PCA.

Design note (why this file exists / what bug it fixes):
The original prototype notebook trained its model on real ResNet50 image
embeddings but then, at prediction time (single product + the Gradio demo),
substituted an all-zero vector instead of actually processing the product's
photo — the model never saw an all-zero input during training, so every
"live" prediction was silently running on out-of-distribution input, and
the resulting "your photo is hurting your score" advice wasn't really based
on the photo at all.

ImageFeatureExtractor is built to make that mistake structurally impossible:
it's a single scikit-learn transformer used inside the SAME Pipeline object
for training, single-product prediction, the What-If simulator, and batch
CSV scoring alike. Whatever code path a training row's image went through,
a live prediction's image goes through identically — there is no separate
"inference-time shortcut" to accidentally take. A row whose image can't be
fetched (broken link, timeout, missing URL) falls back to the *mean*
embedding observed during fit(), not a zero vector, so it stays inside the
distribution the model actually trained on.
"""

from typing import Optional
import hashlib
import io
import os
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from concurrent.futures import ThreadPoolExecutor, as_completed

_RESNET_MODEL = None  # process-wide singleton so weights load once, not per call


class ImageFeatureUnavailableError(RuntimeError):
    """Raised when the optional deep-learning dependencies aren't installed."""
    pass


def _get_resnet_model():
    global _RESNET_MODEL
    if _RESNET_MODEL is None:
        try:
            from tensorflow.keras.applications.resnet50 import ResNet50
        except ImportError as e:
            raise ImageFeatureUnavailableError(
                "Görsel özellik çıkarımı için TensorFlow gerekli ama kurulu değil. "
                "Kurmak için: pip install tensorflow"
            ) from e
        _RESNET_MODEL = ResNet50(weights="imagenet", include_top=False, pooling="avg")
    return _RESNET_MODEL


def _cache_path(cache_dir: Optional[str], url: str) -> Optional[str]:
    if not cache_dir:
        return None
    os.makedirs(cache_dir, exist_ok=True)
    key = hashlib.md5(url.encode("utf-8")).hexdigest()
    return os.path.join(cache_dir, f"{key}.npy")


def _download_and_preprocess(url: str, timeout: float = 6.0):
    """Fetches one image — from an http(s) URL OR a local filesystem path (used
    when a user uploads a photo directly instead of pasting a link, e.g. for a
    not-yet-launched product with no public URL yet) — and returns a
    preprocessed (224,224,3) float array, or None on any failure. Pure I/O +
    CPU array work — no model call here, so this is safe to run concurrently
    across threads."""
    if not isinstance(url, str):
        return None
    ref = url.strip()
    if not ref:
        return None
    try:
        from PIL import Image
        from tensorflow.keras.applications.resnet50 import preprocess_input

        if ref.lower().startswith(("http://", "https://")):
            import requests
            resp = requests.get(ref, timeout=timeout)
            resp.raise_for_status()
            img = Image.open(io.BytesIO(resp.content)).convert("RGB").resize((224, 224))
        elif os.path.isfile(ref):
            img = Image.open(ref).convert("RGB").resize((224, 224))
        else:
            return None

        arr = np.array(img, dtype=np.float32)
        return preprocess_input(arr)
    except Exception:
        return None


class ImageFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Sklearn-compatible transformer: a column of image references (each value
    an http(s) URL, or a local filesystem path — e.g. from a user-uploaded
    photo that has no public URL yet) -> fixed-size numeric embedding
    (ResNet50 -> PCA). Meant to be used as one branch of a ColumnTransformer
    inside a Pipeline shared by training and inference.

    Parameters
    ----------
    n_components : PCA output dimensionality (kept modest relative to the
        tabular/text feature count so images don't dominate the model).
    max_fit_rows : caps how many images are downloaded to *establish* the
        PCA basis and the fallback mean embedding — a representative sample
        is enough for that. This does NOT limit transform(): every row
        passed to transform() has its real image processed, always.
    cache_dir : on-disk cache (by URL hash) so the same photo is never
        re-downloaded/re-embedded across repeated training runs, single
        predictions, or What-If simulations.
    """

    def __init__(self, n_components: int = 32, max_fit_rows: Optional[int] = 300,
                 cache_dir: str = ".image_feature_cache", n_workers: int = 8,
                 batch_size: int = 32, random_state: int = 42):
        self.n_components = n_components
        self.max_fit_rows = max_fit_rows
        self.cache_dir = cache_dir
        self.n_workers = n_workers
        self.batch_size = batch_size
        self.random_state = random_state

    # -- internals ---------------------------------------------------------

    @staticmethod
    def _as_list(X):
        if isinstance(X, pd.DataFrame):
            return X.iloc[:, 0].fillna("").astype(str).tolist()
        if isinstance(X, pd.Series):
            return X.fillna("").astype(str).tolist()
        return pd.Series(np.asarray(X).ravel()).fillna("").astype(str).tolist()

    def _embed_urls(self, urls):
        """Returns a dict {index -> 2048-d embedding or None}, batching the
        actual model.predict() calls (fast + avoids calling the shared Keras
        model concurrently from multiple threads, which isn't safe)."""
        n = len(urls)
        results = {i: None for i in range(n)}

        # Serve whatever's already cached without touching the network/model.
        pending = []
        for i, url in enumerate(urls):
            cpath = _cache_path(self.cache_dir, url) if url else None
            if cpath and os.path.exists(cpath):
                try:
                    results[i] = np.load(cpath)
                    continue
                except Exception:
                    pass
            pending.append(i)

        if not pending:
            return results

        # Download + preprocess the rest in parallel (I/O-bound).
        preprocessed = {}
        with ThreadPoolExecutor(max_workers=self.n_workers) as ex:
            futures = {ex.submit(_download_and_preprocess, urls[i]): i for i in pending}
            for fut in as_completed(futures):
                i = futures[fut]
                arr = fut.result()
                if arr is not None:
                    preprocessed[i] = arr

        if not preprocessed:
            return results

        # Single batched model call (thread-safe, and much faster than
        # one-image-at-a-time inference).
        model = _get_resnet_model()
        idxs = list(preprocessed.keys())
        for start in range(0, len(idxs), self.batch_size):
            batch_idxs = idxs[start:start + self.batch_size]
            batch_arr = np.stack([preprocessed[i] for i in batch_idxs], axis=0)
            batch_emb = model.predict(batch_arr, verbose=0)
            for j, i in enumerate(batch_idxs):
                emb = batch_emb[j]
                results[i] = emb
                cpath = _cache_path(self.cache_dir, urls[i])
                if cpath:
                    try:
                        np.save(cpath, emb)
                    except Exception:
                        pass

        return results

    # -- sklearn API ---------------------------------------------------------

    def fit(self, X, y=None):
        urls = self._as_list(X)
        # Bound *fit* cost only (mirrors the app's existing max_train_samples
        # pattern) — a representative sample is enough to establish the PCA
        # basis and fallback mean; transform() is never bounded like this.
        fit_urls = urls if self.max_fit_rows is None else urls[: self.max_fit_rows]

        try:
            raw = self._embed_urls(fit_urls)
        except ImageFeatureUnavailableError:
            raise

        embeddings = [v for v in raw.values() if v is not None]
        self._raw_dim = embeddings[0].shape[0] if embeddings else 2048

        if not embeddings:
            # Nothing downloadable at fit time — degrade to a harmless
            # constant (zero-width) feature block instead of crashing
            # training. transform() will do the same for every row.
            self.mean_embedding_ = np.zeros(self._raw_dim, dtype=np.float32)
            self.pca_ = None
            self.n_components_ = 0
            return self

        emb_arr = np.vstack(embeddings)
        self.mean_embedding_ = emb_arr.mean(axis=0)

        n_comp = max(0, min(self.n_components, emb_arr.shape[0], emb_arr.shape[1]))
        if n_comp < 1:
            self.pca_ = None
            self.n_components_ = 0
        else:
            self.pca_ = PCA(n_components=n_comp, random_state=self.random_state)
            self.pca_.fit(emb_arr)
            self.n_components_ = n_comp
        return self

    def transform(self, X):
        urls = self._as_list(X)
        raw = self._embed_urls(urls)  # no row limit: every row's real photo is processed

        mean_emb = getattr(self, "mean_embedding_", np.zeros(getattr(self, "_raw_dim", 2048), dtype=np.float32))
        rows = [raw.get(i) if raw.get(i) is not None else mean_emb for i in range(len(urls))]
        emb_arr = np.vstack(rows) if rows else np.zeros((0, mean_emb.shape[0]), dtype=np.float32)

        pca = getattr(self, "pca_", None)
        if pca is None or getattr(self, "n_components_", 0) == 0:
            return np.zeros((emb_arr.shape[0], 0), dtype=np.float32)
        return pca.transform(emb_arr).astype(np.float32)

    def get_feature_names_out(self, input_features=None):
        prefix = input_features[0] if input_features is not None and len(input_features) > 0 else "image"
        n = getattr(self, "n_components_", 0)
        return [f"{prefix}_img_pca_{i}" for i in range(n)]
