# Project Setup Guide

This guide will help you set up the project from scratch.

## Prerequisites

- Python 3.9+
- Node.js 16+
- Git
- Kaggle account (for dataset download)

## Step 1: Clone the Repository

```bash
git clone <your-repo-url>
cd assignment
```

## Step 2: Download the Dataset

**Important:** The dataset is NOT included in the repository (best practice).

### Option A: Automatic Download (Recommended)

```bash
# Install kagglehub
pip install kagglehub

# Download dataset
python scripts/download_dataset.py
```

### Option B: Manual Download

1. Visit: https://www.kaggle.com/datasets/kundanbedmutha/healthcare-symptomsdisease-classification-dataset
2. Click "Download"
3. Extract and copy `Symptoms_Disease_Classification.csv` to `data/` directory

## Step 3: Set Up Backend

```bash
cd backend
pip install -r requirements.txt

# Ensure models are trained (see notebook)
# Models should be in backend/models/ directory
```

## Step 4: Set Up Frontend

```bash
cd frontend
npm install
```

## Step 5: Train Models (Optional)

If you want to retrain the models:

```bash
cd notebook
pip install -r requirements.txt

# Open Jupyter notebook
jupyter notebook healthcare_classification.ipynb

# Run all cells to train models
# Models will be saved to backend/models/
```

## Step 6: Run Locally

### Backend
```bash
cd backend
python app.py
```

### Frontend
```bash
cd frontend
npm start
```

## Why Dataset is Not in Git?

✅ **Best Practice:** Large files should not be in version control
✅ **Performance:** Keeps repository fast and lightweight
✅ **Storage:** Saves GitHub storage quota
✅ **Professional:** Industry standard for ML projects
✅ **Reproducible:** Download script ensures everyone gets the same data

## Troubleshooting

### Dataset Not Found
- Run `python scripts/download_dataset.py`
- Check file is in `data/Symptoms_Disease_Classification.csv`
- Verify file size (~2 MB)

### Models Not Found
- Train models using the Jupyter notebook
- Or download pre-trained models (if provided separately)
- Models should be in `backend/models/` directory

