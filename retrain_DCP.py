"""
retrain_DCP.py — Real model retraining with user annotations
==============================================================
3-class classification (Dog/Cat/Panda) using CrossEntropyLoss.
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "va_export")
PREDICTIONS_PATH = os.path.join(DATA_DIR, "predictions.csv")
IMAGES_BASE = BASE_DIR

CLASS_NAMES = {0: "Cat", 1: "Dog", 2: "Panda"}
CLASS_FOLDERS = {0: "cats", 1: "dogs", 2: "panda"}

RETRAIN_CONFIG = {
    "learning_rate": 1e-4,
    "epochs": 2,
    "image_size": 224,
    "num_classes": 3,
}


def retrain_with_annotations(annotations):
    if not annotations:
        return {"success": False, "message": "No annotations provided.", "metrics": {}}

    import torch
    from lime_explainer_DCP import get_learner, reload_learner

    try:
        learn = get_learner()
    except FileNotFoundError as e:
        return {"success": False, "message": str(e), "metrics": {}}

    corrected_labels = {a["image_id"]: int(a["correct_class"]) for a in annotations}

    print(f"\n{'='*50}")
    print(f"RETRAINING on {len(corrected_labels)} annotated images")
    print(f"{'='*50}")

    acc_before = _compute_accuracy()

    _finetune(learn, corrected_labels)

    # Save weights (state dict only)
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
            f"Model retrained on {len(corrected_labels)} images. "
            f"Accuracy: {acc_before:.1%} → {acc_after:.1%}. "
            f"All predictions recomputed."
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
    """Fine-tune head using CrossEntropyLoss (3-class classification)."""
    import torch
    from torch.utils.data import Dataset, DataLoader
    from torchvision import transforms
    from PIL import Image

    class CorrectedDataset(Dataset):
        def __init__(self, image_ids, labels, true_classes, transform):
            self.image_ids = image_ids
            self.labels = labels
            self.true_classes = true_classes
            self.transform = transform

        def __len__(self):
            return len(self.image_ids)

        def __getitem__(self, idx):
            img_id = self.image_ids[idx]
            label = self.labels[idx]
            tc = self.true_classes[idx]
            path = _find_image(img_id, tc)
            img = Image.open(path).convert("RGB")
            tensor = self.transform(img)
            return tensor, torch.tensor(label, dtype=torch.long)

    tfm = transforms.Compose([
        transforms.Resize((RETRAIN_CONFIG["image_size"], RETRAIN_CONFIG["image_size"])),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Map our label order to fastai class order for the model head
    from lime_explainer_DCP import get_fastai_classes
    fastai_cls = get_fastai_classes()
    name_to_fastai_idx = {name: i for i, name in enumerate(fastai_cls)}

    image_ids = list(corrected_labels.keys())
    # Convert our labels to fastai indices
    fastai_labels = [name_to_fastai_idx[CLASS_NAMES[corrected_labels[i]]] for i in image_ids]

    # Look up original true_class for finding images
    df = pd.read_csv(PREDICTIONS_PATH)
    id_to_tc = dict(zip(df["image_id"], df["true_class"]))
    true_classes = [id_to_tc.get(iid, 0) for iid in image_ids]

    # BatchNorm needs batch_size > 1
    if len(image_ids) == 1:
        image_ids = image_ids * 2
        fastai_labels = fastai_labels * 2
        true_classes = true_classes * 2

    dataset = CorrectedDataset(image_ids, fastai_labels, true_classes, tfm)
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


def _repredict_all(learn, df: pd.DataFrame):
    """Re-run inference on all images with updated model."""
    import torch
    from torchvision import transforms
    from PIL import Image
    from lime_explainer_DCP import get_fastai_classes

    fastai_cls = get_fastai_classes()
    name_to_label = {v: k for k, v in CLASS_NAMES.items()}

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
        # Reorder from fastai to our label order
        ordered = [float(probs[fastai_cls.index(CLASS_NAMES[j])]) for j in range(3)]
        pred_class = int(np.argmax(ordered))

        df.at[df.index[i], "pred_class"] = pred_class
        df.at[df.index[i], "confidence"] = round(float(max(ordered)), 4)
        df.at[df.index[i], "class_confidences"] = json.dumps([round(p, 4) for p in ordered])


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
