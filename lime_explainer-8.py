"""
lime_explainer.py — Real LIME saliency maps using the trained model
====================================================================
Loads the fastai learner from export.pkl and runs actual LIME
perturbation-based explanations. No synthetic fallback.

Required files:
  - export.pkl in the project root (from learn.export())
  - APTOS train_images in aptos2019-blindness-detection/train_images/
"""

import os
import json
import numpy as np
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(BASE_DIR, "aptos2019-blindness-detection", "train_images")
EXPORT_PKL_PATH = os.path.join(BASE_DIR, "export.pkl")
COEFFICIENTS_PATH = os.path.join(BASE_DIR, "va_export", "optimized_rounder_coefficients.json")

# ═══════════════════════════════════════════════════════════════════════════
#  MODEL LOADING
# ═══════════════════════════════════════════════════════════════════════════

_learn = None


def get_learner():
    """Load and return the fastai learner. Cached after first call.
    Only called when user clicks a dot (LIME) or presses retrain — NOT on app startup.
    This is also used by retrain.py to access the model for fine-tuning."""
    global _learn
    if _learn is not None:
        return _learn

    if not os.path.exists(EXPORT_PKL_PATH):
        raise FileNotFoundError(
            f"export.pkl not found at {EXPORT_PKL_PATH}\n"
            f"Copy the export.pkl from your notebook folder to the project root."
        )

    import torch

    # PyTorch 2.6+ changed torch.load default to weights_only=True,
    # which breaks fastai's pickle-based load_learner.
    # Patch it to use weights_only=False for the fastai load.
    _original_torch_load = torch.load
    def _patched_load(*args, **kwargs):
        if "weights_only" not in kwargs:
            kwargs["weights_only"] = False
        return _original_torch_load(*args, **kwargs)
    torch.load = _patched_load

    try:
        from fastai.vision import load_learner
        pkl_dir = os.path.dirname(EXPORT_PKL_PATH)
        pkl_name = os.path.basename(EXPORT_PKL_PATH)
        _learn = load_learner(pkl_dir, pkl_name)
        _learn.model.eval()
        print(f"Loaded fastai learner from {EXPORT_PKL_PATH}")
    finally:
        # Restore original torch.load
        torch.load = _original_torch_load

    return _learn


def reload_learner():
    """Force reload the learner (after retraining saves new weights)."""
    global _learn
    _learn = None
    return get_learner()


def _regression_to_class_probs(raw_pred, sigma=0.8):
    """Convert regression output to 5-class probabilities via Gaussian kernel."""
    grades = np.array([0, 1, 2, 3, 4], dtype=float)
    dists = np.exp(-0.5 * ((raw_pred - grades) / sigma) ** 2)
    probs = dists / dists.sum()
    return probs


def _predict_fn(images: np.ndarray) -> np.ndarray:
    """Prediction function for LIME.
    Takes (N, H, W, 3) uint8 images, returns (N, 5) probability array."""
    import torch
    from torchvision import transforms

    learn = get_learner()

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
    device = next(learn.model.parameters()).device
    batch_tensor = batch_tensor.to(device)

    learn.model.eval()
    with torch.no_grad():
        raw_outputs = learn.model(batch_tensor).cpu().numpy().flatten()

    probs = np.array([_regression_to_class_probs(p) for p in raw_outputs])
    return probs


def predict_single(image_id: str) -> dict:
    """Run the model on a single image and return full prediction info.
    Used by retrain.py to get updated predictions after fine-tuning."""
    import torch
    from torchvision import transforms

    image_path = _find_image(image_id)
    img = Image.open(image_path).convert("RGB")

    learn = get_learner()
    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    tensor = tfm(img).unsqueeze(0)
    device = next(learn.model.parameters()).device
    tensor = tensor.to(device)

    learn.model.eval()
    with torch.no_grad():
        raw = learn.model(tensor).cpu().item()

    # Load optimized rounder coefficients
    coefficients = [0.5, 1.5, 2.5, 3.5]
    if os.path.exists(COEFFICIENTS_PATH):
        with open(COEFFICIENTS_PATH) as f:
            coefficients = json.load(f)["coefficients"]

    # Convert regression to grade
    if raw < coefficients[0]:
        grade = 0
    elif raw < coefficients[1]:
        grade = 1
    elif raw < coefficients[2]:
        grade = 2
    elif raw < coefficients[3]:
        grade = 3
    else:
        grade = 4

    probs = _regression_to_class_probs(raw).tolist()

    return {
        "raw_prediction": round(raw, 4),
        "pred_grade": grade,
        "confidence": round(max(probs), 4),
        "class_confidences": json.dumps([round(p, 4) for p in probs]),
    }


def extract_embedding(image_id: str) -> np.ndarray:
    """Run the model on a single image and capture the embedding vector.
    Used by retrain.py to update the embedding after fine-tuning."""
    import torch
    from torchvision import transforms

    image_path = _find_image(image_id)
    img = Image.open(image_path).convert("RGB")

    learn = get_learner()
    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    tensor = tfm(img).unsqueeze(0)
    device = next(learn.model.parameters()).device
    tensor = tensor.to(device)

    # Hook into the pooling layer to capture embedding
    embedding = []
    def hook(module, input, output):
        out = output.detach().cpu()
        if out.dim() == 4:
            out = out.squeeze(-1).squeeze(-1)
        elif out.dim() == 3:
            out = out.squeeze(-1)
        embedding.append(out.numpy().flatten())

    pool_layer = learn.model[1][0]  # AdaptiveConcatPool2d
    handle = pool_layer.register_forward_hook(hook)

    learn.model.eval()
    with torch.no_grad():
        _ = learn.model(tensor)

    handle.remove()
    return embedding[0]


# ═══════════════════════════════════════════════════════════════════════════
#  LIME EXPLANATION
# ═══════════════════════════════════════════════════════════════════════════

def generate_lime_explanation(
    image_id: str,
    num_samples: int = 300,
    num_features: int = 10,
    positive_only: bool = False,
) -> tuple:
    """Generate a real LIME saliency explanation for a retinal image.

    Returns (overlay_rgba, mask) where overlay is (H, W, 4) uint8 RGBA.
    """
    image_path = _find_image(image_id)
    img = Image.open(image_path).convert("RGB")
    img_array = np.array(img)

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

    overlay = _explanation_to_overlay(explanation, top_label, mask, img_array)
    return overlay, mask


def _explanation_to_overlay(explanation, label, mask, original):
    """Convert LIME weights to a clean RGBA overlay with white outlines.

    Cyan (#00FFEE) = supports prediction
    Yellow (#FFD600) = opposes prediction (artefact)
    """
    h, w = mask.shape
    overlay = np.zeros((h, w, 4), dtype=np.uint8)

    local_exp = dict(explanation.local_exp[label])
    weights = np.array(list(local_exp.values()))
    max_abs = max(abs(weights.max()), abs(weights.min()), 1e-8)

    for seg_id, weight in local_exp.items():
        region = mask == seg_id
        intensity = min(abs(weight) / max_abs, 1.0)
        alpha = int(80 + 140 * intensity)

        if weight > 0:
            overlay[region] = [0, 255, 238, alpha]
        else:
            overlay[region] = [255, 214, 0, alpha]

        _draw_region_outline(overlay, region)

    return overlay


def _draw_region_outline(overlay, region, color=(255, 255, 255, 200)):
    """Draw a thin white outline around a region boundary."""
    from scipy.ndimage import binary_erosion
    eroded = binary_erosion(region, iterations=2)
    boundary = region & ~eroded
    overlay[boundary] = color


def _find_image(image_id: str) -> str:
    """Find the image file, trying .png then .jpg."""
    for ext in [".png", ".jpg", ".jpeg"]:
        p = os.path.join(IMAGES_DIR, f"{image_id}{ext}")
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"Image not found: {image_id} in {IMAGES_DIR}")
