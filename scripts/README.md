# Scripts Directory

This directory contains utility scripts for project setup and data management.

## Dataset Download Script

**File:** `download_dataset.py`

**Purpose:** Downloads the dataset from Kaggle without committing it to git (best practice).

### Usage

```bash
# From project root
python scripts/download_dataset.py
```

### Requirements

**Option 1: Using kagglehub (Recommended)**
```bash
pip install kagglehub
python scripts/download_dataset.py
```

**Option 2: Using Kaggle API**
```bash
pip install kaggle

# Setup Kaggle API credentials
# 1. Go to https://www.kaggle.com/settings
# 2. Create API token
# 3. Save to ~/.kaggle/kaggle.json

python scripts/download_dataset.py
```

### Why This Approach?

✅ **Best Practice:** Large datasets should not be committed to git repositories
✅ **Reproducible:** Anyone can download the exact same dataset
✅ **Version Control:** Dataset source is documented, not the data itself
✅ **Storage Efficient:** Git repositories stay small and fast
✅ **Professional:** Follows industry standards for ML projects

### Manual Download

If the script fails, download manually from:
https://www.kaggle.com/datasets/kundanbedmutha/healthcare-symptomsdisease-classification-dataset

Place the CSV file in: `data/Symptoms_Disease_Classification.csv`

