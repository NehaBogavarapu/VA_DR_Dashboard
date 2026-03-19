"""
retrain.py — Real model retraining with ophthalmologist annotations
====================================================================
When the user clicks "Retrain":
  1. Load the current fastai learner
  2. Fine-tune last layers using corrected labels (actual backprop)
  3. Re-run inference on ALL images to get new predictions
  4. Re-extract embeddings for all images
  5. Save updated predictions.csv, embeddings.npy, and model weights
  6. Clear cache so dashboard refreshes with real new data
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime

from annotation_store import shapes_to_binary_mask

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "va_export")
PREDICTIONS_PATH = os.path.join(DATA_DIR, "predictions.csv")
EMBEDDINGS_PATH = os.path.join(DATA_DIR, "embeddings.npy")
IMAGES_DIR = os.path.join(BASE_DIR, "aptos2019-blindness-detection", "train_images")
COEFFICIENTS_PATH = os.path.join(DATA_DIR, "optimized_rounder_coefficients.json")

RETRAIN_CONFIG = {
    "learning_rate": 1e-4,
    "epochs": 2,
    "image_size": 224,
}


def retrain_with_annotations(annotations):
    """Actually fine-tune the model and recompute everything.

    Steps:
      1. Fine-tune model on corrected labels
      2. Re-predict all images with updated model
      3. Re-extract all embeddings
      4. Save predictions.csv + embeddings.npy + model weights
      5. Clear cache
    """
    if not annotations:
        return {"success": False, "message": "No annotations provided.", "metrics": {}}

    import torch
    from lime_explainer import get_learner, reload_learner

    try:
        learn = get_learner()
    except FileNotFoundError as e:
        return {"success": False, "message": str(e), "metrics": {}}

    corrected_labels = {a["image_id"]: int(a["correct_grade"]) for a in annotations}

    # ── Step 1: Fine-tune the model on corrected images ────────────────
    print(f"\n{'='*50}")
    print(f"RETRAINING on {len(corrected_labels)} annotated images")
    print(f"{'='*50}")

    acc_before = _compute_accuracy()

    _finetune(learn, corrected_labels)

    # Save updated model weights (state dict only — avoids pickle issues)
    import torch
    weights_path = os.path.join(BASE_DIR, "retrained_weights.pth")
    torch.save(learn.model.state_dict(), weights_path)
    print(f"Saved updated weights to {weights_path}")

    # ── Step 2: Re-predict ALL images with updated model ───────────────
    print("Re-predicting all images...")
    df = pd.read_csv(PREDICTIONS_PATH)
    _repredict_all(learn, df)
    df.to_csv(PREDICTIONS_PATH, index=False)
    print(f"Updated predictions.csv ({len(df)} rows)")

    # ── Step 3: Re-extract ALL embeddings ──────────────────────────────
    print("Re-extracting embeddings...")
    _reextract_embeddings(learn, df)
    print(f"Updated embeddings.npy")

    # ── Step 4: Clear cache ────────────────────────────────────────────
    # Reload learner to pick up the new weights
    reload_learner()

    from data_pipeline import clear_cache
    clear_cache()

    acc_after = _compute_accuracy()

    n_changed = sum(1 for a in annotations
                    if corrected_labels.get(a["image_id"]) is not None)

    result = {
        "success": True,
        "message": (
            f"Model retrained on {len(corrected_labels)} images. "
            f"Accuracy: {acc_before:.1%} → {acc_after:.1%}. "
            f"All predictions and embeddings recomputed."
        ),
        "metrics": {
            "accuracy_before": round(acc_before, 4),
            "accuracy_after": round(acc_after, 4),
            "images_retrained": len(corrected_labels),
        },
    }

    _save_retrain_log(annotations, result)
    print(f"Retrain complete: {result['message']}")
    return result


def _finetune(learn, corrected_labels: dict):
    """Fine-tune the model's last layers on the corrected images.

    Creates a small dataset from the annotated images with their
    corrected labels and runs a few epochs of training.
    """
    import torch
    from torch.utils.data import Dataset, DataLoader
    from torchvision import transforms
    from PIL import Image

    class CorrectedDataset(Dataset):
        def __init__(self, image_ids, labels, images_dir, transform):
            self.image_ids = image_ids
            self.labels = labels
            self.images_dir = images_dir
            self.transform = transform

        def __len__(self):
            return len(self.image_ids)

        def __getitem__(self, idx):
            img_id = self.image_ids[idx]
            label = self.labels[idx]
            # Find image
            for ext in [".png", ".jpg", ".jpeg"]:
                path = os.path.join(self.images_dir, f"{img_id}{ext}")
                if os.path.exists(path):
                    break
            img = Image.open(path).convert("RGB")
            tensor = self.transform(img)
            return tensor, torch.tensor(float(label), dtype=torch.float32)

    tfm = transforms.Compose([
        transforms.Resize((RETRAIN_CONFIG["image_size"], RETRAIN_CONFIG["image_size"])),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image_ids = list(corrected_labels.keys())
    labels = [corrected_labels[i] for i in image_ids]

    dataset = CorrectedDataset(image_ids, labels, IMAGES_DIR, tfm)
    loader = DataLoader(dataset, batch_size=min(8, len(dataset)), shuffle=True)

    # Freeze everything except the head (last layers)
    model = learn.model
    # In fastai cnn_learner: model[0] = body, model[1] = head
    for param in model[0].parameters():
        param.requires_grad = False
    for param in model[1].parameters():
        param.requires_grad = True

    device = next(model.parameters()).device
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=RETRAIN_CONFIG["learning_rate"],
    )
    criterion = torch.nn.MSELoss()

    model.train()
    for epoch in range(RETRAIN_CONFIG["epochs"]):
        total_loss = 0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device).unsqueeze(1)

            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"  Epoch {epoch+1}/{RETRAIN_CONFIG['epochs']}: loss = {avg_loss:.4f}")

    model.eval()

    # Unfreeze for future use
    for param in model.parameters():
        param.requires_grad = True

    print(f"  Fine-tuning complete.")


def _repredict_all(learn, df: pd.DataFrame):
    """Run inference on all images with the updated model and update the DataFrame."""
    import torch
    from torchvision import transforms
    from PIL import Image

    # Load coefficients
    coefficients = [0.5, 1.5, 2.5, 3.5]
    if os.path.exists(COEFFICIENTS_PATH):
        with open(COEFFICIENTS_PATH) as f:
            coefficients = json.load(f)["coefficients"]

    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    model = learn.model
    device = next(model.parameters()).device
    model.eval()

    batch_size = 32
    all_raw = []

    for start in range(0, len(df), batch_size):
        batch_ids = df["image_id"].iloc[start:start+batch_size].tolist()
        tensors = []
        for img_id in batch_ids:
            for ext in [".png", ".jpg", ".jpeg"]:
                path = os.path.join(IMAGES_DIR, f"{img_id}{ext}")
                if os.path.exists(path):
                    break
            img = Image.open(path).convert("RGB")
            tensors.append(tfm(img))

        batch = torch.stack(tensors).to(device)
        with torch.no_grad():
            raw = model(batch).cpu().numpy().flatten()
        all_raw.extend(raw.tolist())

        if (start // batch_size) % 20 == 0:
            print(f"  Predicted {min(start+batch_size, len(df))}/{len(df)} images...")

    # Update DataFrame columns
    all_raw = np.array(all_raw)
    for i, raw in enumerate(all_raw):
        # Apply optimized rounder
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

        grades_arr = np.array([0, 1, 2, 3, 4], dtype=float)
        dists = np.exp(-0.5 * ((raw - grades_arr) / 0.8) ** 2)
        probs = dists / dists.sum()

        df.at[i, "pred_grade"] = grade
        df.at[i, "confidence"] = round(float(probs.max()), 4)
        df.at[i, "class_confidences"] = json.dumps([round(float(p), 4) for p in probs])
        if "raw_prediction" in df.columns:
            df.at[i, "raw_prediction"] = round(float(raw), 4)


def _reextract_embeddings(learn, df: pd.DataFrame):
    """Re-extract embeddings for all images using the updated model."""
    import torch
    from torchvision import transforms
    from PIL import Image

    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    model = learn.model
    device = next(model.parameters()).device
    model.eval()

    all_embeddings = []

    def hook(module, input, output):
        out = output.detach().cpu()
        if out.dim() == 4:
            out = out.squeeze(-1).squeeze(-1)
        elif out.dim() == 3:
            out = out.squeeze(-1)
        all_embeddings.append(out.numpy())

    pool_layer = model[1][0]  # AdaptiveConcatPool2d
    handle = pool_layer.register_forward_hook(hook)

    batch_size = 32
    for start in range(0, len(df), batch_size):
        batch_ids = df["image_id"].iloc[start:start+batch_size].tolist()
        tensors = []
        for img_id in batch_ids:
            for ext in [".png", ".jpg", ".jpeg"]:
                path = os.path.join(IMAGES_DIR, f"{img_id}{ext}")
                if os.path.exists(path):
                    break
            img = Image.open(path).convert("RGB")
            tensors.append(tfm(img))

        batch = torch.stack(tensors).to(device)
        with torch.no_grad():
            _ = model(batch)

        if (start // batch_size) % 20 == 0:
            print(f"  Extracted {min(start+batch_size, len(df))}/{len(df)} embeddings...")

    handle.remove()

    embeddings = np.concatenate(all_embeddings, axis=0)
    np.save(EMBEDDINGS_PATH, embeddings)
    print(f"  Saved embeddings: {embeddings.shape}")


def _compute_accuracy() -> float:
    """Read current predictions.csv and compute accuracy."""
    df = pd.read_csv(PREDICTIONS_PATH)
    return (df["pred_grade"] == df["true_grade"]).mean()


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
