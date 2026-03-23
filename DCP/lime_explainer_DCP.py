"""
lime_explainer_DCP.py — Real LIME saliency maps for Dog/Cat/Panda classifier
==============================================================================
Loads the fastai learner from export.pkl and runs LIME perturbation-based
explanations. 3-class classification with softmax (no regression conversion).
"""

import os
import json
import numpy as np
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_BASE = BASE_DIR
EXPORT_PKL_PATH = os.path.join(BASE_DIR, "export.pkl")

CLASS_NAMES = {0: "Cat", 1: "Dog", 2: "Panda"}
CLASS_FOLDERS = {0: "cats", 1: "dogs", 2: "panda"}

# ═══════════════════════════════════════════════════════════════════════════
#  MODEL LOADING
# ═══════════════════════════════════════════════════════════════════════════

_learn = None
_fastai_classes = None  # e.g. ['Cat', 'Dog', 'Panda']


def get_learner():
    """Load and return the fastai learner. Cached after first call."""
    global _learn, _fastai_classes
    if _learn is not None:
        return _learn

    if not os.path.exists(EXPORT_PKL_PATH):
        raise FileNotFoundError(
            f"export.pkl not found at {EXPORT_PKL_PATH}\n"
            f"Copy export.pkl from the notebook folder to the project root."
        )

    import torch

    # PyTorch 2.6+ fix: force weights_only=False for fastai pickle
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

        # Move to GPU if available
        if torch.cuda.is_available():
            _learn.model = _learn.model.cuda()
            print(f"Model moved to GPU: {torch.cuda.get_device_name(0)}")

        _learn.model.eval()
        _fastai_classes = _learn.data.classes  # e.g. ['Cat', 'Dog', 'Panda']
        print(f"Loaded fastai learner from {EXPORT_PKL_PATH}")
        print(f"Classes: {_fastai_classes}")
    finally:
        torch.load = _original_torch_load

    return _learn


def get_fastai_classes():
    """Return the class order used by fastai (alphabetically sorted)."""
    global _fastai_classes
    if _fastai_classes is None:
        get_learner()
    return _fastai_classes


def reload_learner():
    """Reload model weights after retraining."""
    global _learn
    import torch

    weights_path = os.path.join(BASE_DIR, "retrained_weights.pth")

    if _learn is not None and os.path.exists(weights_path):
        _original = torch.load
        def _patched(*a, **kw):
            if "weights_only" not in kw:
                kw["weights_only"] = False
            return _original(*a, **kw)
        torch.load = _patched
        try:
            _learn.model.load_state_dict(torch.load(weights_path))
            _learn.model.eval()
            print("Reloaded retrained weights into existing model.")
        finally:
            torch.load = _original
        return _learn

    _learn = None
    return get_learner()


def _predict_fn(images: np.ndarray) -> np.ndarray:
    """Prediction function for LIME.
    Takes (N, H, W, 3) uint8 images, returns (N, 3) probability array.

    The model outputs softmax probabilities in fastai class order.
    We reorder to our label order: 0=Cat, 1=Dog, 2=Panda.
    """
    import torch
    from torchvision import transforms

    learn = get_learner()
    fastai_cls = get_fastai_classes()

    # Build reorder mapping: fastai index → our label index
    name_to_label = {v: k for k, v in CLASS_NAMES.items()}
    reorder = [name_to_label[fastai_cls[i]] for i in range(len(fastai_cls))]

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
        logits = learn.model(batch_tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()

    # Reorder columns from fastai order to our label order
    reordered = np.zeros_like(probs)
    for fastai_idx, our_label in enumerate(reorder):
        reordered[:, our_label] = probs[:, fastai_idx]

    return reordered


def predict_single(image_id: str, true_class: int = None) -> dict:
    """Run model on a single image, return prediction info."""
    import torch
    from torchvision import transforms

    image_path = _find_image(image_id, true_class)
    img = Image.open(image_path).convert("RGB")

    learn = get_learner()
    fastai_cls = get_fastai_classes()

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
        logits = learn.model(tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    # Reorder to our label order
    name_to_label = {v: k for k, v in CLASS_NAMES.items()}
    ordered_probs = [float(probs[fastai_cls.index(CLASS_NAMES[j])]) for j in range(3)]

    pred_class = int(np.argmax(ordered_probs))

    return {
        "pred_class": pred_class,
        "confidence": round(float(max(ordered_probs)), 4),
        "class_confidences": json.dumps([round(p, 4) for p in ordered_probs]),
    }


# ═══════════════════════════════════════════════════════════════════════════
#  LIME EXPLANATION
# ═══════════════════════════════════════════════════════════════════════════

def generate_lime_explanation(
    image_id: str,
    true_class: int = None,
    num_samples: int = 50,
    num_features: int = 5,
    positive_only: bool = False,
) -> tuple:
    """Generate a LIME saliency explanation.

    Returns (overlay_rgba, mask) where overlay is (H, W, 4) uint8 RGBA.
    """
    image_path = _find_image(image_id, true_class)
    img = Image.open(image_path).convert("RGB")

    img_resized = img.resize((224, 224), Image.LANCZOS)
    img_array = np.array(img_resized)

    from lime.lime_image import LimeImageExplainer
    from skimage.segmentation import quickshift

    explainer = LimeImageExplainer()

    explanation = explainer.explain_instance(
        img_array,
        _predict_fn,
        top_labels=1,
        hide_color=0,
        num_samples=num_samples,
        batch_size=10,
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
    """Convert LIME weights to RGBA overlay.
    Cyan = supports prediction, Yellow = opposes prediction."""
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
            overlay[region] = [0, 255, 238, alpha]  # Cyan
        else:
            overlay[region] = [255, 214, 0, alpha]   # Yellow

        _draw_region_outline(overlay, region)

    return overlay


def _draw_region_outline(overlay, region, color=(255, 255, 255, 200)):
    from scipy.ndimage import binary_erosion
    eroded = binary_erosion(region, iterations=2)
    boundary = region & ~eroded
    overlay[boundary] = color


def _find_image(image_id: str, true_class: int = None) -> str:
    """Find image file in class subfolders."""
    if true_class is not None:
        folder = CLASS_FOLDERS.get(true_class, "")
        for ext in [".jpg", ".jpeg", ".png"]:
            p = os.path.join(IMAGES_BASE, folder, f"{image_id}{ext}")
            if os.path.exists(p):
                return p

    for folder in CLASS_FOLDERS.values():
        for ext in [".jpg", ".jpeg", ".png"]:
            p = os.path.join(IMAGES_BASE, folder, f"{image_id}{ext}")
            if os.path.exists(p):
                return p

    raise FileNotFoundError(f"Image not found: {image_id} in {IMAGES_BASE}")
