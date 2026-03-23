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

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import umap

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


def clear_cache():
    """Clear all cached data so next load_data() reads fresh from disk.
    Call this after retraining updates predictions.csv and model weights."""
    load_data.cache_clear()
    load_embeddings.cache_clear()
    _projection_cache.clear()
    _hidden_layer_cache.clear()
    print("Data cache cleared (including hidden layer cache).")


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
#  MODEL HIDDEN LAYER → PCA → UMAP PROJECTION
# ═══════════════════════════════════════════════════════════════════════════

_projection_cache = {}
_hidden_layer_cache = {}  # image_id → hidden layer vector


def _extract_hidden_layers(image_ids: list) -> np.ndarray:
    """Run the model and extract the last hidden layer activations.

    This captures how the network 'sees' each image — the representation
    just before the final classification head. Uses the same hook approach
    as the embedding extraction in the notebook, but runs live so it
    always reflects the current model weights (including after retraining).

    Returns np.ndarray of shape (len(image_ids), D) where D is the
    hidden layer dimension (4096 for fastai ResNet-50 with concat pool).
    """
    # Check cache first — only extract missing ones
    missing_ids = [iid for iid in image_ids if iid not in _hidden_layer_cache]

    if missing_ids:
        import torch
        from torchvision import transforms
        from PIL import Image as PILImage
        from lime_explainer import get_learner

        learn = get_learner()
        model = learn.model
        device = next(model.parameters()).device

        tfm = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        # Hook into the last hidden layer (AdaptiveConcatPool2d output)
        hidden_outputs = []
        def hook(module, input, output):
            out = output.detach().cpu()
            if out.dim() == 4:
                out = out.squeeze(-1).squeeze(-1)
            elif out.dim() == 3:
                out = out.squeeze(-1)
            hidden_outputs.append(out.numpy())

        pool_layer = model[1][0]  # AdaptiveConcatPool2d
        handle = pool_layer.register_forward_hook(hook)

        model.eval()
        batch_size = 32
        for start in range(0, len(missing_ids), batch_size):
            batch_ids = missing_ids[start:start+batch_size]
            tensors = []
            for img_id in batch_ids:
                path = _find_image_path(img_id)
                img = PILImage.open(path).convert("RGB")
                tensors.append(tfm(img))

            batch = torch.stack(tensors).to(device)
            with torch.no_grad():
                _ = model(batch)

        handle.remove()

        # Flatten and cache per image
        all_hidden = np.concatenate(hidden_outputs, axis=0)
        for i, iid in enumerate(missing_ids):
            _hidden_layer_cache[iid] = all_hidden[i]

        print(f"Extracted hidden layers for {len(missing_ids)} images "
              f"(dim: {all_hidden.shape[1]})")

    # Return in the requested order
    return np.array([_hidden_layer_cache[iid] for iid in image_ids])


def _find_image_path(image_id: str) -> str:
    """Find image file, trying .png then .jpg."""
    for ext in [".png", ".jpg", ".jpeg"]:
        p = os.path.join(IMAGES_DIR, f"{image_id}{ext}")
        if os.path.exists(p):
            return p
    return os.path.join(IMAGES_DIR, f"{image_id}.png")


def get_pca_embeddings(
    df: pd.DataFrame,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
) -> np.ndarray:
    """Project the model's last hidden layer into 2-D via PCA → UMAP.

    Pipeline:
      1. Run model on all images, capture last hidden layer (4096-D)
      2. StandardScaler → PCA (to 50-D) for noise reduction
      3. UMAP (to 2-D) for nonlinear structure preservation

    This visualizes how the NETWORK sees the data, not just raw pixel features.
    After retraining, the hidden representations change, so the scatter plot
    will reflect the updated model's internal representation.

    Returns np.ndarray of shape (len(df), 2).
    """
    image_ids = df["image_id"].tolist()
    cache_key = (tuple(sorted(image_ids)), n_neighbors, min_dist)
    if cache_key in _projection_cache:
        return _projection_cache[cache_key]

    # Step 1: Extract hidden layer from the live model
    hidden = _extract_hidden_layers(image_ids)

    # Step 2: PCA for dimensionality reduction + noise removal
    scaler = StandardScaler()
    scaled = scaler.fit_transform(hidden)

    n_pca = min(50, scaled.shape[0] - 1, scaled.shape[1])
    pca = PCA(n_components=n_pca, random_state=random_state)
    pca_result = pca.fit_transform(scaled)
    print(f"PCA: {scaled.shape[1]}D → {n_pca}D "
          f"(explained variance: {pca.explained_variance_ratio_.sum():.1%})")

    # Step 3: UMAP for 2D projection
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=min(n_neighbors, len(pca_result) - 1),
        min_dist=min_dist,
        metric="euclidean",
        random_state=random_state,
    )
    coords = reducer.fit_transform(pca_result)
    print(f"UMAP: {n_pca}D → 2D ({len(coords)} points)")

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
