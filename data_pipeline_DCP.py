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

# Paths 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "va_export")
PREDICTIONS_PATH = os.path.join(DATA_DIR, "predictions.csv")
# Images are in subdirectories: data/cats/, data/dogs/, data/panda/
IMAGES_BASE = os.path.join(BASE_DIR, "data")

# Class mapping — must match fastai's class order (folder names, alphabetical)
CLASS_NAMES = {0: "cats", 1: "dogs", 2: "panda"}
CLASS_DISPLAY = {0: "Cat", 1: "Dog", 2: "Panda"}  # Pretty names for UI
CLASS_FOLDERS = {0: "cats", 1: "dogs", 2: "panda"}


#  Data loading and processing
@lru_cache(maxsize=1)
def load_data(require_umap=True) -> pd.DataFrame:
    """Load predictions.csv produced by the training notebook.

    If require_umap=False, u1/u2 are NOT required (used by precompute_UMAP.py).
    """

    if not os.path.exists(PREDICTIONS_PATH):
        raise FileNotFoundError(
            f"predictions.csv not found at {PREDICTIONS_PATH}\n"
            f"Run the training notebook first, then copy va_export/ here."
        )

    df = pd.read_csv(PREDICTIONS_PATH)

    # Columns that must ALWAYS exist
    required = {
        "image_id", "true_class", "pred_class",
        "confidence", "class_confidences", "split"
    }

    # Only require UMAP columns when running the dashboard
    if require_umap:
        required |= {"u1", "u2"}

    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"predictions.csv is missing columns: {missing}")

    df["image_path"] = df.apply(
        lambda r: _find_image(r["image_id"], int(r["true_class"])), axis=1
    )

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
    print("Data cache cleared.")


#  Filtering functions
def get_misclassified(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["pred_class"] != df["true_class"]].copy()


def filter_by_confidence(
    df: pd.DataFrame, min_conf: float = 0.0, max_conf: float = 1.0
) -> pd.DataFrame:
    return df[(df["confidence"] >= min_conf) & (df["confidence"] <= max_conf)].copy()


def filter_by_classes(df: pd.DataFrame, classes: list) -> pd.DataFrame:
    return df[df["true_class"].isin(classes)].copy()


#  K-means clustering on UMAP coordinates
def run_kmeans(coords: np.ndarray, k: int = 3, random_state: int = 42) -> np.ndarray:
    k = min(k, len(coords))
    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    return kmeans.fit_predict(coords)


#  Setup check and helper functions
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
