# Dog / Cat / Panda — Visual Analytics Dashboard

Interactive dashboard for exploring and correcting a ResNet-50 image classifier (Dog / Cat / Panda) with LIME explanations, UMAP projections, and a human-in-the-loop retraining pipeline.

**Course:** AMV10 Visual Analytics — TU/e, Group 14

---

## Prerequisites

- **Python 3.11 or higher**
- pip (included with Python)
- (Optional) NVIDIA GPU with CUDA for faster inference

---

## Dataset

This project uses the **Animal Image Dataset (Dog, Cat, and Panda)** from Kaggle:

https://www.kaggle.com/datasets/ashishsaxena2209/animal-image-datasetdog-cat-and-panda

### Download instructions

1. Download the dataset from the link above.
2. Extract the contents into the `data/` folder so the structure looks like:

```
data/
├── cats/
│   ├── cats_00001.jpg
│   ├── ...
├── dogs/
│   ├── dogs_00001.jpg
│   ├── ...
└── panda/
    ├── panda_00001.jpg
    ├── ...
```

Each subfolder should contain the corresponding animal images directly (no extra nesting).

---

## Installation

### 1. Extract the repository

Unzip the provided archive:

```bash
unzip VA_DR_DASHBOARD.zip
cd VA_DR_DASHBOARD
```

### 2. Create a virtual environment

Make sure you are using **Python 3.11+** when creating the environment.

```bash
python3.11 -m venv .venv
```

Activate it:

- **Linux / macOS:** `source .venv/bin/activate`
- **Windows:** `.venv\Scripts\activate`

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the model (if not done yet)

Open and run **`DCP_animal_classifier_resnet50.ipynb`** in Jupyter. This produces:

- `export.pkl` — the trained fastai learner
- `va_export/predictions.csv` — per-image predictions and metadata

Make sure both files are present in the project root / `va_export/` folder before continuing.

---

## Execution

**The two steps below must be executed in order.**

### Step 1 — Precompute UMAP embeddings

```bash
python precompute_UMAP.py
```

This extracts hidden-layer features from the model, runs UMAP, and writes the `u1` / `u2` columns back into `va_export/predictions.csv`.

### Step 2 — Launch the dashboard

```bash
python app_DCP.py
```

The dashboard will be available at **http://127.0.0.1:8050** (or the port shown in the terminal).

---

## Dependencies

Key external libraries used:

- **Dash** + **Dash Bootstrap Components** — interactive web dashboard
- **Plotly** — interactive charts and scatter plots
- **fastai** / **PyTorch** / **torchvision** — model training, inference, and retraining
- **LIME** — perturbation-based saliency explanations
- **umap-learn** — UMAP dimensionality reduction
- **scikit-learn** — k-means clustering and preprocessing
- **scikit-image** — image segmentation (quickshift for LIME)
- **Pandas** / **NumPy** / **SciPy** — data manipulation and numerical operations
- **Pillow** / **OpenCV** — image loading and processing

All dependencies with pinned versions are listed in `requirements.txt`. Install them with:

```bash
pip install -r requirements.txt
```

---

## What We Built vs. What We Reused

The base model originates from a publicly available Kaggle notebook for diabetic retinopathy classification:

https://www.kaggle.com/code/tanlikesmath/intro-aptos-diabetic-retinopathy-eda-starter

We adapted the pretrained ResNet-50 architecture from that notebook to work with the Dog / Cat / Panda dataset (3-class classification with softmax instead of regression). The ImageNet-pretrained backbone weights come from PyTorch / torchvision.

**Implemented by the students (Group 14):**

- Full interactive Dash dashboard (`app_DCP.py`) — layout, callbacks, linked views
- Data pipeline with live hidden-layer extraction, UMAP projection, and k-means clustering (`data_pipeline_DCP.py`, `precompute_UMAP.py`)
- LIME explainability integration with custom overlay rendering (`lime_explainer_DCP.py`)
- Annotation store for human-in-the-loop label corrections (`annotation_store_DCP.py`)
- Retraining pipeline with frozen backbone fine-tuning (`retrain_DCP.py`)
- Adapted training notebook for the new dataset and classification task (`DCP_animal_classifier_resnet50.ipynb`)

**Tools:** Claude AI (Anthropic) was used for debugging purposes during development.

---

## Project Structure

```
VA_DR_DASHBOARD/
├── .venv/                        # Python virtual environment
├── assets/                       # Dash CSS / static assets
├── data/                         # Image dataset (download from Kaggle)
│   ├── cats/
│   ├── dogs/
│   └── panda/
├── models/                       # Saved model checkpoints
├── va_export/                    # Generated outputs
│   ├── predictions.csv           # Model predictions + UMAP coords
│   └── annotations.json          # Saved user annotations
├── .gitignore
├── annotation_store_DCP.py       # Annotation persistence (JSON)
├── app_DCP.py                    # Dash dashboard (main entry point)
├── data_pipeline_DCP.py          # Data loading, filtering, k-means
├── DCP_animal_classifier_resnet50.ipynb  # Model training notebook
├── export.pkl                    # Trained model (generated by notebook)
├── lime_explainer_DCP.py         # LIME saliency explanations
├── precompute_UMAP.py            # Hidden-layer extraction + UMAP
├── README.md                     # This file
├── requirements.txt              # Python dependencies
├── retrain_DCP.py                # Human-in-the-loop retraining
└── retrained_weights.pth         # Weights after retraining (generated)
```
