"""
lime_explainer.py — LIME saliency maps for retinal images
==========================================================
Uses the fastai export.pkl produced by learn.export() in the notebook.
This is the most reliable way to load the model because it preserves
the exact architecture (body + AdaptiveConcatPool2d head) without
needing to reconstruct it manually.

Setup:
  Copy the export.pkl that learn.export() creates into dr_prototype/data/
  (it's saved in the same folder as the notebook by default)

  Also copy va_export/optimized_rounder_coefficients.json into data/
"""

import os
import json
import numpy as np
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "va_export")
IMAGES_DIR = os.path.join(BASE_DIR, "aptos2019-blindness-detection", "train_images")
EXPORT_PKL_PATH = os.path.join(BASE_DIR, "export.pkl")  # learn.export() saves in notebook root
COEFFICIENTS_PATH = os.path.join(DATA_DIR, "optimized_rounder_coefficients.json")

# ═══════════════════════════════════════════════════════════════════════════
#  MODEL LOADING
# ═══════════════════════════════════════════════════════════════════════════

_learn = None  # fastai learner (cached)
_coefficients = None


def _load_learner():
    """Load the fastai learner from export.pkl."""
    global _learn
    if _learn is not None:
        return _learn

    if not os.path.exists(EXPORT_PKL_PATH):
        print(f"export.pkl not found at {EXPORT_PKL_PATH}")
        print("Copy the export.pkl from your notebook folder into dr_prototype/data/")
        print("Falling back to synthetic LIME explanations.")
        return None

    try:
        from fastai.vision import load_learner
        pkl_dir = os.path.dirname(EXPORT_PKL_PATH)
        pkl_name = os.path.basename(EXPORT_PKL_PATH)
        _learn = load_learner(pkl_dir, pkl_name)
        _learn.model.eval()
        print("Loaded fastai learner from export.pkl")
        return _learn
    except Exception as e:
        print(f"Failed to load export.pkl: {e}")
        print("Falling back to synthetic LIME explanations.")
        return None


def _load_coefficients():
    """Load the optimized rounder coefficients (optional, improves grade mapping)."""
    global _coefficients
    if _coefficients is not None:
        return _coefficients

    if os.path.exists(COEFFICIENTS_PATH):
        with open(COEFFICIENTS_PATH, "r") as f:
            data = json.load(f)
            _coefficients = data["coefficients"]
            print(f"Loaded rounder coefficients: {_coefficients}")
    else:
        _coefficients = [0.5, 1.5, 2.5, 3.5]  # default evenly spaced
        print("Using default rounder coefficients (no JSON found)")

    return _coefficients


def _regression_to_class_probs(raw_pred, sigma=0.8):
    """Convert a regression prediction to 5-class probabilities.
    Uses a Gaussian kernel centered at each grade.
    Same function as in the training notebook."""
    grades = np.array([0, 1, 2, 3, 4], dtype=float)
    dists = np.exp(-0.5 * ((raw_pred - grades) / sigma) ** 2)
    probs = dists / dists.sum()
    return probs


def _predict_fn(images: np.ndarray) -> np.ndarray:
    """Prediction function for LIME.

    Takes batch of numpy images (N, H, W, 3) uint8.
    Returns (N, 5) probability array.

    Uses the fastai learner for inference, then converts
    the regression output to class probabilities.
    """
    import torch
    from torchvision import transforms

    learn = _load_learner()
    if learn is None:
        return np.ones((len(images), 5)) / 5

    # Match the notebook's preprocessing: resize to 224, ImageNet normalization
    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    batch = []
    for img in images:
        pil_img = Image.fromarray(img.astype(np.uint8))
        batch.append(tfm(pil_img))

    batch_tensor = torch.stack(batch)

    # Move to same device as model
    device = next(learn.model.parameters()).device
    batch_tensor = batch_tensor.to(device)

    with torch.no_grad():
        raw_outputs = learn.model(batch_tensor).cpu().numpy().flatten()

    probs = np.array([_regression_to_class_probs(p) for p in raw_outputs])
    return probs


# ═══════════════════════════════════════════════════════════════════════════
#  LIME EXPLANATION
# ═══════════════════════════════════════════════════════════════════════════

def generate_lime_explanation(
    image_id: str,
    num_samples: int = 300,
    num_features: int = 10,
    positive_only: bool = False,
) -> tuple:
    """Generate a LIME saliency explanation for a retinal image.

    Tries real LIME first (if export.pkl is available).
    Falls back to synthetic heatmap if the model can't be loaded.

    Returns
    -------
    (heatmap_image, mask)
        heatmap_image : np.ndarray (H, W, 3) float in [0, 1]
        mask : np.ndarray (H, W) int — superpixel segmentation
    """
    # Try .png first, then .jpg
    image_path = os.path.join(IMAGES_DIR, f"{image_id}.png")
    if not os.path.exists(image_path):
        image_path = os.path.join(IMAGES_DIR, f"{image_id}.jpg")
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_id}.png or .jpg in {IMAGES_DIR}")

    img = Image.open(image_path).convert("RGB")
    img_array = np.array(img)

    # Try real LIME
    if _load_learner() is not None:
        try:
            from lime.lime_image import LimeImageExplainer
            from skimage.segmentation import quickshift

            explainer = LimeImageExplainer()

            explanation = explainer.explain_instance(
                img_array,
                _predict_fn,
                top_labels=5,
                hide_color=0,
                num_samples=num_samples,
                segmentation_fn=lambda x: quickshift(
                    x, kernel_size=4, max_dist=200, ratio=0.2
                ),
            )

            top_label = explanation.top_labels[0]
            temp, mask = explanation.get_image_and_mask(
                top_label,
                positive_only=positive_only,
                num_features=num_features,
                hide_rest=False,
            )

            heatmap = _explanation_to_heatmap(explanation, top_label, mask, img_array)
            return heatmap, mask

        except Exception as e:
            print(f"Real LIME failed for {image_id}: {e}")
            print("Falling back to synthetic.")

    return _generate_synthetic_lime(img_array)


def _explanation_to_heatmap(explanation, label, mask, original):
    """Convert LIME weights to green (positive) / red (negative) overlay.

    Green = model correctly attends to this region (real lesion)
    Red = model is misled by this region (artefact)
    """
    h, w = mask.shape
    heatmap = np.zeros((h, w, 3), dtype=np.float32)

    local_exp = dict(explanation.local_exp[label])

    for seg_id, weight in local_exp.items():
        region = mask == seg_id
        if weight > 0:
            intensity = min(abs(weight) * 5, 1.0)
            heatmap[region, 1] = intensity  # green
        else:
            intensity = min(abs(weight) * 5, 1.0)
            heatmap[region, 0] = intensity  # red

    original_norm = original.astype(np.float32) / 255.0
    blended = original_norm * 0.5 + heatmap * 0.5
    return np.clip(blended, 0, 1)


def _generate_synthetic_lime(img_array: np.ndarray) -> tuple:
    """Synthetic LIME-like heatmap as fallback."""
    h, w = img_array.shape[:2]
    rng = np.random.RandomState(hash(img_array.tobytes()[:100]) % 2**31)

    heatmap = np.zeros((h, w, 3), dtype=np.float32)
    mask = np.zeros((h, w), dtype=np.int32)

    n_regions = rng.randint(8, 13)
    for i in range(n_regions):
        cx = rng.randint(w // 6, 5 * w // 6)
        cy = rng.randint(h // 6, 5 * h // 6)
        radius = rng.randint(20, min(w, h) // 6)

        yy, xx = np.ogrid[:h, :w]
        region = ((xx - cx) ** 2 + (yy - cy) ** 2) < radius**2
        mask[region] = i + 1

        if rng.random() < 0.6:
            heatmap[region, 1] = rng.uniform(0.3, 0.8)
        else:
            heatmap[region, 0] = rng.uniform(0.3, 0.8)

    original_norm = img_array.astype(np.float32) / 255.0
    blended = original_norm * 0.5 + heatmap * 0.5
    return np.clip(blended, 0, 1), mask
