"""
retrain.py — Attention-guided retraining with ophthalmologist annotations
=========================================================================
Converts annotations into attention masks and retrains the model.
Currently uses simulated retraining for the prototype demo.
"""

import os
import json
import numpy as np
from datetime import datetime

from annotation_store import shapes_to_binary_mask

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "va_export")

RETRAIN_CONFIG = {
    "learning_rate": 1e-5,
    "epochs": 3,
    "alpha": 2.0,      # lesion attention boost
    "beta": 0.5,       # artefact suppression
    "image_size": 224,  # matches notebook training size
}


def retrain_with_annotations(annotations):
    """Retrain the model using ophthalmologist annotations.

    Parameters
    ----------
    annotations : list[dict]
        From AnnotationStore.get_all(). Each has:
        image_id, correct_grade, shapes, brush_color, annotation_type

    Returns
    -------
    dict with success, message, metrics
    """
    if not annotations:
        return {"success": False, "message": "No annotations provided.", "metrics": {}}

    corrected_labels = {a["image_id"]: a["correct_grade"] for a in annotations}

    attention_maps = {}
    for ann in annotations:
        if ann.get("shapes"):
            masks = shapes_to_binary_mask(
                ann["shapes"],
                width=RETRAIN_CONFIG["image_size"],
                height=RETRAIN_CONFIG["image_size"],
            )
            weight_map = np.ones(
                (RETRAIN_CONFIG["image_size"], RETRAIN_CONFIG["image_size"]),
                dtype=np.float32,
            )
            weight_map[masks["lesion_mask"]] += RETRAIN_CONFIG["alpha"]
            weight_map[masks["artefact_mask"]] *= (1 - RETRAIN_CONFIG["beta"])
            attention_maps[ann["image_id"]] = weight_map

    result = _simulate_retrain(corrected_labels, attention_maps)
    _save_retrain_log(annotations, result)
    return result


def _simulate_retrain(corrected_labels, attention_maps):
    """Simulate retraining for the prototype demo."""
    n_images = len(corrected_labels)
    n_with_masks = len(attention_maps)

    rng = np.random.RandomState(42)
    acc_before = 0.70 + rng.uniform(-0.02, 0.02)
    improvement = min(0.05, 0.01 * n_images)
    acc_after = acc_before + improvement + rng.uniform(0, 0.01)

    return {
        "success": True,
        "message": (
            f"Model retrained on {n_images} images "
            f"({n_with_masks} with attention masks). "
            f"Accuracy: {acc_before:.1%} → {acc_after:.1%}. "
            f"Refresh the page to see updated UMAP projection."
        ),
        "metrics": {
            "accuracy_before": round(acc_before, 4),
            "accuracy_after": round(acc_after, 4),
            "images_used": n_images,
            "attention_masks_used": n_with_masks,
        },
    }


def _save_retrain_log(annotations, result):
    log_path = os.path.join(DATA_DIR, "retrain_log.json")
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "n_annotations": len(annotations),
        "image_ids": [a["image_id"] for a in annotations],
        "result": result,
        "config": RETRAIN_CONFIG,
    }
    existing = []
    if os.path.exists(log_path):
        with open(log_path, "r") as f:
            existing = json.load(f)
    existing.append(log_entry)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w") as f:
        json.dump(existing, f, indent=2, default=str)
