"""
data_pipeline_DCP.py — Data loading, UMAP projection, and k-means clustering
==============================================================================
Reads predictions.csv produced by the training notebook.
Hidden layer embeddings are extracted live from the model for UMAP projection.

Dataset: Dog / Cat / Panda (3 classes, ~3000 images)
Model: ResNet-50 (fastai, classification with softmax)
"""

import os
import json
import numpy as np
import pandas as pd
from functools import lru_cache

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import umap

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "va_export")
PREDICTIONS_PATH = os.path.join(DATA_DIR, "predictions.csv")
# Images are in subdirectories: data/cats/, data/dogs/, data/panda/
IMAGES_BASE = os.path.join(BASE_DIR, "data")

# Class mapping — must match fastai's class order (folder names, alphabetical)
CLASS_NAMES = {0: "cats", 1: "dogs", 2: "panda"}
CLASS_DISPLAY = {0: "Cat", 1: "Dog", 2: "Panda"}  # Pretty names for UI
CLASS_FOLDERS = {0: "cats", 1: "dogs", 2: "panda"}


# ═══════════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════

@lru_cache(maxsize=1)
def load_data() -> pd.DataFrame:
    """Load predictions.csv produced by the training notebook.

    Expected columns:
        image_id          : str
        true_class        : int   (0=Cat, 1=Dog, 2=Panda)
        pred_class        : int   (0=Cat, 1=Dog, 2=Panda)
        confidence        : float (max softmax probability)
        class_confidences : str   (JSON array of 3 floats)

    Returns DataFrame with an added 'image_path' column.
    """
    if not os.path.exists(PREDICTIONS_PATH):
        raise FileNotFoundError(
            f"predictions.csv not found at {PREDICTIONS_PATH}\n"
            f"Run the training notebook first, then copy va_export/ here."
        )

    df = pd.read_csv(PREDICTIONS_PATH)

    required = {"image_id", "true_class", "pred_class", "confidence", "class_confidences"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"predictions.csv is missing columns: {missing}")

    df["image_path"] = df.apply(
        lambda r: _find_image(r["image_id"], int(r["true_class"])), axis=1
    )

    sample = df["image_path"].iloc[0]
    if not os.path.exists(sample):
        print(f"WARNING: Image not found at {sample}")
        print(f"Make sure cats/, dogs/, panda/ folders are in: {BASE_DIR}")

    return df


def _find_image(image_id: str, true_class: int = None) -> str:
    """Find image file. Searches in class subfolder first, then all folders."""
    # Try the known class folder first
    if true_class is not None:
        folder = CLASS_FOLDERS.get(true_class, "")
        for ext in [".jpg", ".jpeg", ".png"]:
            p = os.path.join(IMAGES_BASE, folder, f"{image_id}{ext}")
            if os.path.exists(p):
                return p

    # Search all class folders
    for folder in CLASS_FOLDERS.values():
        for ext in [".jpg", ".jpeg", ".png"]:
            p = os.path.join(IMAGES_BASE, folder, f"{image_id}{ext}")
            if os.path.exists(p):
                return p

    return os.path.join(IMAGES_BASE, "cats", f"{image_id}.jpg")


def clear_cache():
    """Clear all cached data. Call after retraining."""
    load_data.cache_clear()
    _projection_cache.clear()
    _hidden_layer_cache.clear()
    print("Data cache cleared (including hidden layer cache).")


# ═══════════════════════════════════════════════════════════════════════════
#  FILTERING
# ═══════════════════════════════════════════════════════════════════════════

def get_misclassified(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["pred_class"] != df["true_class"]].copy()


def filter_by_confidence(
    df: pd.DataFrame, min_conf: float = 0.0, max_conf: float = 1.0
) -> pd.DataFrame:
    return df[(df["confidence"] >= min_conf) & (df["confidence"] <= max_conf)].copy()


def filter_by_classes(df: pd.DataFrame, classes: list) -> pd.DataFrame:
    return df[df["true_class"].isin(classes)].copy()


# ═══════════════════════════════════════════════════════════════════════════
#  MODEL HIDDEN LAYER → UMAP PROJECTION
# ═══════════════════════════════════════════════════════════════════════════

_projection_cache = {}
_hidden_layer_cache = {}


def _extract_hidden_layers(image_ids: list, true_classes: list = None) -> np.ndarray:
    """Run model, extract last hidden layer (AdaptiveConcatPool2d output).

    Returns np.ndarray of shape (len(image_ids), 4096).
    """
    missing_ids = [iid for iid in image_ids if iid not in _hidden_layer_cache]

    if missing_ids:
        import torch
        from torchvision import transforms
        from PIL import Image as PILImage
        from lime_explainer_DCP import get_learner

        learn = get_learner()
        model = learn.model
        device = next(model.parameters()).device

        tfm = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

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

        # Build a lookup for missing IDs
        df = load_data()
        id_to_class = dict(zip(df["image_id"], df["true_class"]))

        model.eval()
        batch_size = 32
        for start in range(0, len(missing_ids), batch_size):
            batch_ids = missing_ids[start:start + batch_size]
            tensors = []
            for img_id in batch_ids:
                tc = id_to_class.get(img_id)
                path = _find_image(img_id, tc)
                img = PILImage.open(path).convert("RGB")
                tensors.append(tfm(img))

            batch = torch.stack(tensors).to(device)
            with torch.no_grad():
                _ = model(batch)

            if (start // batch_size) % 10 == 0:
                print(f"  Hidden layers: {min(start + batch_size, len(missing_ids))}/{len(missing_ids)}")

        handle.remove()

        all_hidden = np.concatenate(hidden_outputs, axis=0)
        for i, iid in enumerate(missing_ids):
            _hidden_layer_cache[iid] = all_hidden[i]

        print(f"Extracted hidden layers for {len(missing_ids)} images "
              f"(dim: {all_hidden.shape[1]})")

    return np.array([_hidden_layer_cache[iid] for iid in image_ids])


def get_umap_embeddings(
    df: pd.DataFrame,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
) -> np.ndarray:
    """Project model's last hidden layer into 2-D via UMAP.

    Pipeline: last hidden layer (4096-D) → StandardScaler → UMAP (2-D)

    Returns np.ndarray of shape (len(df), 2).
    """
    image_ids = df["image_id"].tolist()
    cache_key = (tuple(sorted(image_ids)), n_neighbors, min_dist)
    if cache_key in _projection_cache:
        return _projection_cache[cache_key]

    hidden = _extract_hidden_layers(image_ids)

    scaler = StandardScaler()
    scaled = scaler.fit_transform(hidden)

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=min(n_neighbors, len(scaled) - 1),
        min_dist=min_dist,
        metric="cosine",
        random_state=random_state,
    )
    coords = reducer.fit_transform(scaled)
    print(f"UMAP: {scaled.shape[1]}D → 2D ({len(coords)} points)")

    _projection_cache[cache_key] = coords
    return coords


# ═══════════════════════════════════════════════════════════════════════════
#  K-MEANS CLUSTERING
# ═══════════════════════════════════════════════════════════════════════════

def run_kmeans(coords: np.ndarray, k: int = 3, random_state: int = 42) -> np.ndarray:
    k = min(k, len(coords))
    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    return kmeans.fit_predict(coords)


# ═══════════════════════════════════════════════════════════════════════════
#  SETUP HELPER
# ═══════════════════════════════════════════════════════════════════════════

def check_setup():
    print("=" * 50)
    print("DCP-VA Data Pipeline — Setup Check")
    print("=" * 50)

    files = {
        "predictions.csv": PREDICTIONS_PATH,
        "images/ base": IMAGES_BASE,
        "export.pkl": os.path.join(BASE_DIR, "export.pkl"),
        "class_info.json": os.path.join(DATA_DIR, "class_info.json"),
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

    for folder in CLASS_FOLDERS.values():
        p = os.path.join(IMAGES_BASE, folder)
        if os.path.isdir(p):
            n = len([f for f in os.listdir(p) if f.endswith((".jpg", ".png"))])
            print(f"  {folder}/: {n} images")

    if all_ok:
        print("\nAll files present. Ready to run: python app_DCP.py")
    else:
        print("\nSome files missing. Run the notebook first.")

    return all_ok


if __name__ == "__main__":
    check_setup()
