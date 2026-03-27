# Dog / Cat / Panda — Visual Analytics Dashboard

Interactive dashboard for exploring and correcting a ResNet-50 image classifier (Dog / Cat / Panda) with LIME explanations, UMAP projections, and a human-in-the-loop retraining pipeline.

**Course:** AMV10 Visual Analytics — TU/e, Group 14

---

## Prerequisites

- **Python 3.11 or higher**
- pip (included with Python)
- (Optional) NVIDIA GPU with CUDA for faster inference

---

## Setup

### 1. Clone the repository

```bash
git clone <repo-url>
cd <repo-folder>
```

### 2. Create a virtual environment

Make sure you are using **Python 3.11+** when creating the environment.

```bash
python3.11 -m venv venv
```

Activate it:

- **Linux / macOS:** `source venv/bin/activate`
- **Windows:** `venv\Scripts\activate`

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

## Running the Dashboard

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

