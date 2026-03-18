"""
data_pipeline.py — Data loading, UMAP projection, and k-means clustering
=========================================================================
Reads the artefacts produced by the training notebook:
  - data/predictions.csv   (from va_export/predictions.csv)
  - data/embeddings.npy    (from va_export/embeddings.npy)
  - data/images/           (symlink or copy of APTOS train_images/)

The notebook exports ResNet-50 embeddings of shape (N, 4096) via
AdaptiveConcatPool2d. UMAP handles any dimensionality so this works
identically to 2048-D embeddings.

Setup:
  1. Run the training notebook to generate va_export/
  2. Copy va_export/predictions.csv  → dr_prototype/data/predictions.csv
  3. Copy va_export/embeddings.npy   → dr_prototype/data/embeddings.npy
  4. Symlink your APTOS train_images → dr_prototype/data/images/
     (or copy the folder)
"""

import os
import numpy as np
import pandas as pd
from functools import lru_cache

import umap
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "va_export")
PREDICTIONS_PATH = os.path.join(DATA_DIR, "predictions.csv")
EMBEDDINGS_PATH = os.path.join(DATA_DIR, "embeddings.npy")
IMAGES_DIR = os.path.join(BASE_DIR, "aptos2019-blindness-detection", "train_images")


# ═══════════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════

@lru_cache(maxsize=1)
def load_data() -> pd.DataFrame:
    """Load predictions.csv produced by the training notebook.

    Expected columns (all produced by the notebook export cells):
        image_id          : str   (e.g. "000c1434d8d7")
        true_grade        : int   (0–4)
        pred_grade        : int   (0–4)
        raw_prediction    : float (regression output before rounding)
        confidence        : float (max pseudo-probability)
        class_confidences : str   (JSON array of 5 floats)

    Returns DataFrame with an added 'image_path' column.
    """
    if not os.path.exists(PREDICTIONS_PATH):
        raise FileNotFoundError(
            f"predictions.csv not found at {PREDICTIONS_PATH}\n"
            f"Run the training notebook first, then copy va_export/predictions.csv here."
        )

    df = pd.read_csv(PREDICTIONS_PATH)

    required = {"image_id", "true_grade", "pred_grade", "confidence", "class_confidences"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"predictions.csv is missing columns: {missing}")

    # Attach image paths — detect .png or .jpg
    def _find_image(image_id):
        for ext in [".png", ".jpg", ".jpeg"]:
            p = os.path.join(IMAGES_DIR, f"{image_id}{ext}")
            if os.path.exists(p):
                return p
        return os.path.join(IMAGES_DIR, f"{image_id}.png")  # default fallback

    df["image_path"] = df["image_id"].apply(_find_image)

    # Verify at least some images exist
    sample_path = df["image_path"].iloc[0]
    if not os.path.exists(sample_path):
        print(f"WARNING: Image not found at {sample_path}")
        print(f"Make sure your APTOS train_images/ folder is at: {IMAGES_DIR}/")
        # Check what extensions actually exist in the folder
        if os.path.isdir(IMAGES_DIR):
            files = os.listdir(IMAGES_DIR)[:5]
            print(f"  Files found in images/: {files}")

    return df


@lru_cache(maxsize=1)
def load_embeddings() -> np.ndarray:
    """Load the embedding matrix produced by the notebook.

    Shape is (N, 4096) from ResNet-50 AdaptiveConcatPool2d (avg + max pool).
    """
    if not os.path.exists(EMBEDDINGS_PATH):
        raise FileNotFoundError(
            f"embeddings.npy not found at {EMBEDDINGS_PATH}\n"
            f"Run the training notebook first, then copy va_export/embeddings.npy here."
        )

    emb = np.load(EMBEDDINGS_PATH)
    print(f"Loaded embeddings: shape {emb.shape}")
    return emb


# ═══════════════════════════════════════════════════════════════════════════
#  FILTERING
# ═══════════════════════════════════════════════════════════════════════════

def get_misclassified(df: pd.DataFrame) -> pd.DataFrame:
    """Return only rows where predicted grade != true grade."""
    return df[df["pred_grade"] != df["true_grade"]].copy()


def filter_by_confidence(
    df: pd.DataFrame, min_conf: float = 0.0, max_conf: float = 1.0
) -> pd.DataFrame:
    """Filter to images whose confidence falls within [min_conf, max_conf]."""
    return df[(df["confidence"] >= min_conf) & (df["confidence"] <= max_conf)].copy()


def filter_by_grades(df: pd.DataFrame, grades: list) -> pd.DataFrame:
    """Keep only images whose true grade is in the provided list."""
    return df[df["true_grade"].isin(grades)].copy()


# ═══════════════════════════════════════════════════════════════════════════
#  PCA → UMAP PROJECTION
# ═══════════════════════════════════════════════════════════════════════════

_projection_cache = {}


def get_umap_embeddings(
    df: pd.DataFrame,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    pca_components: int = 50,
    random_state: int = 42,
) -> np.ndarray:
    """Project embeddings into 2-D via PCA followed by UMAP.

    Pipeline: 4096-D embeddings → StandardScaler → PCA (to pca_components)
              → UMAP (to 2-D)

    PCA first reduces the 4096-D space to a manageable size (default 50),
    which makes UMAP much faster and more stable. This is standard practice
    for high-dimensional embeddings.

    Returns np.ndarray of shape (len(df), 2).
    """
    full_df = load_data()
    full_embeddings = load_embeddings()

    # Map image_id → row index
    idx_map = {img_id: i for i, img_id in enumerate(full_df["image_id"])}
    indices = [idx_map[img_id] for img_id in df["image_id"]]
    subset_embeddings = full_embeddings[indices]

    # Cache key includes PCA components
    cache_key = (tuple(sorted(indices)), n_neighbors, min_dist, pca_components)
    if cache_key in _projection_cache:
        return _projection_cache[cache_key]

    # Step 1: Standardise
    scaler = StandardScaler()
    scaled = scaler.fit_transform(subset_embeddings)

    # Step 2: PCA — reduce from 4096-D to pca_components
    n_components = min(pca_components, scaled.shape[0] - 1, scaled.shape[1])
    pca = PCA(n_components=n_components, random_state=random_state)
    pca_result = pca.fit_transform(scaled)
    print(f"PCA: {scaled.shape[1]}D → {n_components}D "
          f"(explained variance: {pca.explained_variance_ratio_.sum():.1%})")

    # Step 3: UMAP — reduce from pca_components to 2-D
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=min(n_neighbors, len(pca_result) - 1),
        min_dist=min_dist,
        metric="euclidean",  # euclidean is better after PCA (not cosine)
        random_state=random_state,
    )
    coords = reducer.fit_transform(pca_result)

    _projection_cache[cache_key] = coords
    return coords


# ═══════════════════════════════════════════════════════════════════════════
#  K-MEANS CLUSTERING
# ═══════════════════════════════════════════════════════════════════════════

def run_kmeans(coords: np.ndarray, k: int = 5, random_state: int = 42) -> np.ndarray:
    """Run k-means on the 2-D UMAP coordinates. Returns cluster labels (0 to k-1)."""
    k = min(k, len(coords))  # guard against k > n_samples
    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    return kmeans.fit_predict(coords)


# ═══════════════════════════════════════════════════════════════════════════
#  SETUP HELPER
# ═══════════════════════════════════════════════════════════════════════════

def check_setup():
    """Print a status report of which data files are present."""
    print("=" * 50)
    print("DR-VA Data Pipeline — Setup Check")
    print("=" * 50)

    files = {
        "predictions.csv": PREDICTIONS_PATH,
        "embeddings.npy": EMBEDDINGS_PATH,
        "images/ folder": IMAGES_DIR,
        "export.pkl (LIME)": os.path.join(DATA_DIR, "export.pkl"),
        "coefficients.json": os.path.join(DATA_DIR, "optimized_rounder_coefficients.json"),
    }

    all_ok = True
    for name, path in files.items():
        exists = os.path.exists(path)
        status = "OK" if exists else "MISSING"
        if not exists:
            all_ok = False
        print(f"  {status:8s}  {name:20s}  {path}")

    if os.path.exists(PREDICTIONS_PATH):
        df = pd.read_csv(PREDICTIONS_PATH)
        print(f"\n  predictions.csv: {len(df)} rows, columns: {list(df.columns)}")

    if os.path.exists(EMBEDDINGS_PATH):
        emb = np.load(EMBEDDINGS_PATH)
        print(f"  embeddings.npy:  shape {emb.shape}")

    if os.path.isdir(IMAGES_DIR):
        imgs = [f for f in os.listdir(IMAGES_DIR) if f.endswith(".png")]
        print(f"  images/:         {len(imgs)} .png files")

    if all_ok:
        print("\nAll files present. Ready to run: python app.py")
    else:
        print("\nSome files missing. See README.md for setup instructions.")

    return all_ok


if __name__ == "__main__":
    check_setup()
