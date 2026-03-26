"""
precompute_umap.py — Extract hidden layers once and compute UMAP.
Run this BEFORE launching the dashboard.
"""

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
import umap
from PIL import Image as PILImage
import torch
from torchvision import transforms

from data_pipeline_DCP import load_data, _find_image, CLASS_FOLDERS
from lime_explainer_DCP import get_learner

# ───────────────────────────────────────────────────────────────
# Hidden layer extraction (moved here from old pipeline)
# ───────────────────────────────────────────────────────────────

def extract_hidden_layers_for_all_images(df):
    """Extract last hidden layer (AdaptiveConcatPool2d output) for all images."""
    learn = get_learner()
    model = learn.model
    model = model.float()
    device = next(model.parameters()).device

    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    hidden_outputs = []

    def hook(module, input, output):
        out = output.detach().cpu()
        if out.dim() == 4:
            out = out.squeeze(-1).squeeze(-1)
        elif out.dim() == 3:
            out = out.squeeze(-1)
        hidden_outputs.append(out.numpy())

    # AdaptiveConcatPool2d is model[1][0] in your architecture
    pool_layer = model[1][0]
    handle = pool_layer.register_forward_hook(hook)

    image_ids = df["image_id"].tolist()
    id_to_class = dict(zip(df["image_id"], df["true_class"]))

    model.eval()
    batch_size = 32

    print("Extracting hidden layers...")

    for start in range(0, len(image_ids), batch_size):
        batch_ids = image_ids[start:start + batch_size]
        tensors = []

        for img_id in batch_ids:
            tc = id_to_class[img_id]
            path = _find_image(img_id, tc)
            img = PILImage.open(path).convert("RGB")
            tensors.append(tfm(img))

        batch = torch.stack(tensors).to(device)

        with torch.no_grad():
            _ = model(batch)

        print(f"  {min(start + batch_size, len(image_ids))}/{len(image_ids)} extracted")

    handle.remove()

    hidden = np.concatenate(hidden_outputs, axis=0)
    print(f"Hidden layer shape: {hidden.shape}")

    return hidden


# ───────────────────────────────────────────────────────────────
# Main UMAP preprocessing
# ───────────────────────────────────────────────────────────────

print("Loading predictions.csv...")
df = load_data(require_umap=False)

# Extract hidden layers
hidden = extract_hidden_layers_for_all_images(df)

print("Scaling...")
scaler = StandardScaler()
scaled = scaler.fit_transform(hidden)

print("Running UMAP...")
reducer = umap.UMAP(
    n_components=2,
    n_neighbors=15,
    min_dist=0.1,
    metric="cosine",
    random_state=42
)
coords = reducer.fit_transform(scaled)

df["u1"] = coords[:, 0]
df["u2"] = coords[:, 1]

print("Saving updated predictions.csv...")
df.to_csv("va_export/predictions.csv", index=False)

print("Done! UMAP coordinates added.")
