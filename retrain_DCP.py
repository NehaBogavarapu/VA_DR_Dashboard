"""
retrain_DCP.py — Attention-guided model retraining with user annotations
=========================================================================
3-class classification (Dog/Cat/Panda) using CrossEntropyLoss.

When the user draws annotation shapes on the image:
  - "Relevant" regions (red)     → model should focus here
  - "Not relevant" regions (blue) → model should ignore these

How shapes are used during retraining:
  1. Shapes are converted to a binary attention mask at 224×224
  2. Pixels in "not relevant" regions are blurred/suppressed
  3. This forces the model to base its prediction on the relevant
     features (e.g. the animal's face) rather than shortcuts
     (e.g. background, watermarks, image borders)
  4. The corrected label tells the model WHAT the answer is,
     the attention mask tells it WHERE to look

This implements a form of human-in-the-loop attention guidance,
similar to the Husky vs Wolf study (Ribeiro et al., 2016) where
LIME revealed the model was using snow as a feature. Here, the
user can explicitly mark which regions matter.
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "va_export")
PREDICTIONS_PATH = os.path.join(DATA_DIR, "predictions.csv")
IMAGES_BASE = os.path.join(BASE_DIR, "data")

CLASS_NAMES = {0: "cats", 1: "dogs", 2: "panda"}
CLASS_DISPLAY = {0: "Cat", 1: "Dog", 2: "Panda"}
CLASS_FOLDERS = {0: "cats", 1: "dogs", 2: "panda"}

RETRAIN_CONFIG = {
    "learning_rate": 1e-4,
    "epochs": 2,
    "image_size": 224,
    "num_classes": 3,
    "mask_blur_sigma": 10.0,   # Gaussian blur sigma for suppressed regions
    "mask_suppress_alpha": 0.1, # How much of the original pixel to keep in suppressed regions (0=black, 1=no effect)
}


# ═══════════════════════════════════════════════════════════════════════════
#  SHAPE → ATTENTION MASK CONVERSION
# ═══════════════════════════════════════════════════════════════════════════

def _shapes_to_attention_mask(shapes, orig_w, orig_h, target_size=224):
    """Convert Plotly annotation shapes to a float attention mask.

    Returns a (target_size, target_size) float32 array where:
      - 1.0 = relevant (model should focus)
      - 0.0 = not relevant (model should ignore)
      - default (no annotation) = 1.0 (keep everything)

    Shape coordinates are in original image space, so we scale to target_size.
    """
    if not shapes:
        return None  # No shapes → no masking, use image as-is

    # Start with "everything relevant" (1.0)
    relevant_mask = np.ones((target_size, target_size), dtype=np.float32)
    not_relevant_mask = np.zeros((target_size, target_size), dtype=np.float32)

    has_relevant = False
    has_not_relevant = False

    sx = target_size / max(orig_w, 1)
    sy = target_size / max(orig_h, 1)

    for shape in shapes:
        if not isinstance(shape, dict):
            continue

        fill = shape.get("fillcolor", "")
        is_relevant = "255,0,0" in fill or "255, 0, 0" in fill  # red
        is_not_relevant = "0,100,255" in fill or "0, 100, 255" in fill  # blue

        if not is_relevant and not is_not_relevant:
            continue

        # Rasterise the shape into a binary region
        region = _rasterise_shape(shape, orig_w, orig_h, target_size, sx, sy)

        if is_relevant:
            relevant_mask = np.maximum(relevant_mask, region)
            has_relevant = True
        elif is_not_relevant:
            not_relevant_mask = np.maximum(not_relevant_mask, region)
            has_not_relevant = True

    # Build final attention mask:
    # If user drew "relevant" regions → only those are 1.0, everything else is suppressed
    # If user drew "not relevant" regions → those become 0.0, everything else stays 1.0
    # If both → relevant=1.0, not_relevant=0.0, rest=0.5 (partially suppressed)

    if has_relevant and not has_not_relevant:
        # Only relevant regions drawn → everything outside is suppressed
        mask = relevant_mask * 0.0  # start with all suppressed
        # Re-fill the relevant regions
        for shape in shapes:
            fill = shape.get("fillcolor", "")
            if "255,0,0" in fill or "255, 0, 0" in fill:
                region = _rasterise_shape(shape, orig_w, orig_h, target_size, sx, sy)
                mask = np.maximum(mask, region)
    elif has_not_relevant and not has_relevant:
        # Only "not relevant" drawn → suppress those, keep everything else
        mask = 1.0 - not_relevant_mask
    else:
        # Both types drawn
        mask = np.ones((target_size, target_size), dtype=np.float32)
        mask = mask - not_relevant_mask  # suppress blue regions
        # Ensure relevant regions stay at 1.0
        for shape in shapes:
            fill = shape.get("fillcolor", "")
            if "255,0,0" in fill or "255, 0, 0" in fill:
                region = _rasterise_shape(shape, orig_w, orig_h, target_size, sx, sy)
                mask = np.maximum(mask, region)
        mask = np.clip(mask, 0.0, 1.0)

    return mask


def _rasterise_shape(shape, orig_w, orig_h, target_size, sx, sy):
    """Rasterise a single Plotly shape into a binary mask at target_size."""
    region = np.zeros((target_size, target_size), dtype=np.float32)
    shape_type = shape.get("type", "")

    if shape_type == "rect":
        x0 = int(shape["x0"] * sx)
        y0 = int((orig_h - shape["y1"]) * sy)  # Plotly y is inverted
        x1 = int(shape["x1"] * sx)
        y1 = int((orig_h - shape["y0"]) * sy)
        x0, x1 = max(0, min(x0, x1)), min(target_size, max(x0, x1))
        y0, y1 = max(0, min(y0, y1)), min(target_size, max(y0, y1))
        region[y0:y1, x0:x1] = 1.0

    elif shape_type == "circle":
        cx = (shape["x0"] + shape["x1"]) / 2 * sx
        cy = (orig_h - (shape["y0"] + shape["y1"]) / 2) * sy
        rx = abs(shape["x1"] - shape["x0"]) / 2 * sx
        ry = abs(shape["y1"] - shape["y0"]) / 2 * sy
        yy, xx = np.ogrid[:target_size, :target_size]
        ellipse = ((xx - cx) / max(rx, 1)) ** 2 + ((yy - cy) / max(ry, 1)) ** 2
        region[ellipse <= 1] = 1.0

    elif shape_type == "path":
        import re
        coords = re.findall(r"[ML]\s*([\d.]+)[,\s]+([\d.]+)", shape.get("path", ""))
        if len(coords) >= 3:
            try:
                from PIL import Image as PILImage, ImageDraw
                polygon = [(float(x) * sx, (orig_h - float(y)) * sy) for x, y in coords]
                temp = PILImage.new("L", (target_size, target_size), 0)
                ImageDraw.Draw(temp).polygon(polygon, fill=255)
                region = np.array(temp).astype(np.float32) / 255.0
            except Exception:
                xs = [float(x) * sx for x, y in coords]
                ys = [(orig_h - float(y)) * sy for x, y in coords]
                x0, x1 = int(min(xs)), int(max(xs))
                y0, y1 = int(min(ys)), int(max(ys))
                region[max(0,y0):min(target_size,y1), max(0,x0):min(target_size,x1)] = 1.0

    return region


def _apply_attention_mask(image_tensor, mask, suppress_alpha=0.1):
    """Apply attention mask to a (C, H, W) image tensor.

    Where mask=1.0: keep original pixel
    Where mask=0.0: suppress to suppress_alpha * original (near black)

    This creates a smooth transition using the mask as a weight,
    so the model sees the relevant regions clearly and the
    suppressed regions as darkened/blurred.
    """
    import torch

    # mask is (H, W), expand to (1, H, W) for broadcasting with (C, H, W)
    mask_t = torch.from_numpy(mask).unsqueeze(0).to(image_tensor.device)

    # Blend: pixel = mask * original + (1 - mask) * suppressed
    # suppressed = suppress_alpha * original (darkened)
    weight = mask_t * (1.0 - suppress_alpha) + suppress_alpha
    return image_tensor * weight


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN RETRAIN FUNCTION
# ═══════════════════════════════════════════════════════════════════════════

def retrain_with_annotations(annotations):
    if not annotations:
        return {"success": False, "message": "No annotations provided.", "metrics": {}}

    import torch
    from lime_explainer_DCP import get_learner, reload_learner

    try:
        learn = get_learner()
    except FileNotFoundError as e:
        return {"success": False, "message": str(e), "metrics": {}}

    # Extract labels AND shapes from annotations
    corrected_labels = {}
    annotation_shapes = {}
    for a in annotations:
        iid = a["image_id"]
        corrected_labels[iid] = int(a["correct_class"])
        annotation_shapes[iid] = a.get("shapes", [])

    n_with_shapes = sum(1 for s in annotation_shapes.values() if s)

    print(f"\n{'='*50}")
    print(f"RETRAINING on {len(corrected_labels)} annotated images")
    print(f"  All {len(corrected_labels)} have corrected labels")
    print(f"  {n_with_shapes} also have attention masks (drawn regions)")
    print(f"{'='*50}")

    acc_before = _compute_accuracy()

    _finetune(learn, corrected_labels, annotation_shapes)

    # Save weights
    weights_path = os.path.join(BASE_DIR, "retrained_weights.pth")
    torch.save(learn.model.state_dict(), weights_path)
    print(f"Saved updated weights to {weights_path}")

    # Re-predict all images
    print("Re-predicting all images...")
    df = pd.read_csv(PREDICTIONS_PATH)
    _repredict_all(learn, df)
    df.to_csv(PREDICTIONS_PATH, index=False)
    print(f"Updated predictions.csv ({len(df)} rows)")

    # Reload and clear cache
    reload_learner()
    from data_pipeline_DCP import clear_cache
    clear_cache()

    acc_after = _compute_accuracy()

    result = {
        "success": True,
        "message": (
            f"Retrained on {len(corrected_labels)} images "
            f"(all with corrected labels, {n_with_shapes} with attention masks). "
            f"Accuracy: {acc_before:.1%} → {acc_after:.1%}."
        ),
        "metrics": {
            "accuracy_before": round(acc_before, 4),
            "accuracy_after": round(acc_after, 4),
            "images_retrained": len(corrected_labels),
            "images_with_shapes": n_with_shapes,
        },
    }

    _save_retrain_log(annotations, result)
    print(f"Retrain complete: {result['message']}")
    return result


# ═══════════════════════════════════════════════════════════════════════════
#  FINE-TUNING WITH ATTENTION MASKS
# ═══════════════════════════════════════════════════════════════════════════

def _finetune(learn, corrected_labels: dict, annotation_shapes: dict):
    """Fine-tune head using CrossEntropyLoss with attention-masked images.

    For images WITH annotation shapes:
      - Convert shapes to an attention mask
      - Suppress "not relevant" regions in the image
      - Train on the masked image with the corrected label
      → Forces the model to use relevant features only

    For images WITHOUT shapes:
      - Train on the original image with the corrected label
      → Standard label correction
    """
    import torch
    from torch.utils.data import Dataset, DataLoader
    from torchvision import transforms
    from PIL import Image

    class AttentionDataset(Dataset):
        """Dataset that applies attention masks from annotation shapes."""

        def __init__(self, image_ids, labels, true_classes, shapes_map, transform):
            self.image_ids = image_ids
            self.labels = labels
            self.true_classes = true_classes
            self.shapes_map = shapes_map
            self.transform = transform

        def __len__(self):
            return len(self.image_ids)

        def __getitem__(self, idx):
            img_id = self.image_ids[idx]
            label = self.labels[idx]
            tc = self.true_classes[idx]

            path = _find_image(img_id, tc)
            img = Image.open(path).convert("RGB")
            orig_w, orig_h = img.size

            # Apply standard transforms (resize, normalize, etc.)
            tensor = self.transform(img)

            # If this image has annotation shapes, create and apply attention mask
            shapes = self.shapes_map.get(img_id, [])
            if shapes:
                mask = _shapes_to_attention_mask(
                    shapes, orig_w, orig_h,
                    target_size=RETRAIN_CONFIG["image_size"]
                )
                if mask is not None:
                    tensor = _apply_attention_mask(
                        tensor, mask,
                        suppress_alpha=RETRAIN_CONFIG["mask_suppress_alpha"]
                    )

            return tensor, torch.tensor(label, dtype=torch.long)

    tfm = transforms.Compose([
        transforms.Resize((RETRAIN_CONFIG["image_size"], RETRAIN_CONFIG["image_size"])),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Map our label order to fastai class order
    from lime_explainer_DCP import get_fastai_classes, CLASS_DISPLAY
    fastai_cls = get_fastai_classes()
    # Build flexible mapping that handles Cat/cats/Cats etc.
    name_to_fastai_idx = {}
    for i, name in enumerate(fastai_cls):
        name_to_fastai_idx[name] = i
        name_to_fastai_idx[name.lower()] = i

    image_ids = list(corrected_labels.keys())
    # Use CLASS_DISPLAY (Cat/Dog/Panda) which matches fastai class names
    fastai_labels = [name_to_fastai_idx[CLASS_NAMES[corrected_labels[i]]] for i in image_ids]

    df = pd.read_csv(PREDICTIONS_PATH)
    id_to_tc = dict(zip(df["image_id"], df["true_class"]))
    true_classes = [id_to_tc.get(iid, 0) for iid in image_ids]

    # BatchNorm needs batch_size > 1
    if len(image_ids) == 1:
        image_ids = image_ids * 2
        fastai_labels = fastai_labels * 2
        true_classes = true_classes * 2

    dataset = AttentionDataset(image_ids, fastai_labels, true_classes, annotation_shapes, tfm)
    loader = DataLoader(dataset, batch_size=min(8, len(dataset)), shuffle=True, drop_last=True)

    model = learn.model
    for param in model[0].parameters():
        param.requires_grad = False
    for param in model[1].parameters():
        param.requires_grad = True

    device = next(model.parameters()).device
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=RETRAIN_CONFIG["learning_rate"],
    )
    criterion = torch.nn.CrossEntropyLoss()

    model.train()
    for epoch in range(RETRAIN_CONFIG["epochs"]):
        total_loss = 0
        n_batches = 0
        for batch_x, batch_y in loader:
            if batch_x.size(0) < 2:
                continue
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        print(f"  Epoch {epoch+1}/{RETRAIN_CONFIG['epochs']}: loss = {avg_loss:.4f}")

    model.eval()
    for param in model.parameters():
        param.requires_grad = True
    print("  Fine-tuning complete.")


# ═══════════════════════════════════════════════════════════════════════════
#  RE-PREDICTION
# ═══════════════════════════════════════════════════════════════════════════

def _repredict_all(learn, df: pd.DataFrame):
    """Re-run inference on all images with updated model."""
    import torch
    from torchvision import transforms
    from PIL import Image
    from lime_explainer_DCP import get_fastai_classes

    fastai_cls = get_fastai_classes()

    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    model = learn.model
    device = next(model.parameters()).device
    model.eval()

    batch_size = 32
    all_probs = []

    for start in range(0, len(df), batch_size):
        batch_rows = df.iloc[start:start + batch_size]
        tensors = []
        for _, row in batch_rows.iterrows():
            path = _find_image(row["image_id"], int(row["true_class"]))
            img = Image.open(path).convert("RGB")
            tensors.append(tfm(img))

        batch = torch.stack(tensors).to(device)
        with torch.no_grad():
            logits = model(batch)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        all_probs.append(probs)

        if (start // batch_size) % 20 == 0:
            print(f"  Predicted {min(start + batch_size, len(df))}/{len(df)} images...")

    all_probs = np.concatenate(all_probs, axis=0)

    for i in range(len(df)):
        probs = all_probs[i]
        fastai_lower = [c.lower() for c in fastai_cls]
        ordered = [float(probs[fastai_lower.index(CLASS_DISPLAY[j].lower())]) for j in range(3)]
        pred_class = int(np.argmax(ordered))

        df.at[df.index[i], "pred_class"] = pred_class
        df.at[df.index[i], "confidence"] = round(float(max(ordered)), 4)
        df.at[df.index[i], "class_confidences"] = json.dumps([round(p, 4) for p in ordered])


# ═══════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _find_image(image_id: str, true_class: int = None) -> str:
    if true_class is not None:
        folder = CLASS_FOLDERS.get(int(true_class), "")
        for ext in [".jpg", ".jpeg", ".png"]:
            p = os.path.join(IMAGES_BASE, folder, f"{image_id}{ext}")
            if os.path.exists(p):
                return p
    for folder in CLASS_FOLDERS.values():
        for ext in [".jpg", ".jpeg", ".png"]:
            p = os.path.join(IMAGES_BASE, folder, f"{image_id}{ext}")
            if os.path.exists(p):
                return p
    return os.path.join(IMAGES_BASE, "cats", f"{image_id}.jpg")


def _compute_accuracy() -> float:
    df = pd.read_csv(PREDICTIONS_PATH)
    return (df["pred_class"] == df["true_class"]).mean()


def _save_retrain_log(annotations, result):
    log_path = os.path.join(DATA_DIR, "retrain_log.json")
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "n_annotations": len(annotations),
        "image_ids": [a["image_id"] for a in annotations],
        "n_with_shapes": sum(1 for a in annotations if a.get("shapes")),
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
