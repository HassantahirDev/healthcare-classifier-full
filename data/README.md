# Dataset Directory

This directory contains the dataset for the Healthcare Symptoms-Disease Classification project.

## ⚠️ Important: Dataset Not in Git

**The dataset CSV file is NOT committed to this repository** (following best practices).

This is the correct approach because:
- ✅ Keeps repository size small
- ✅ Avoids version control issues with large files
- ✅ Follows industry best practices
- ✅ Makes cloning and collaboration faster

## 📥 Downloading the Dataset

### Automatic Download (Recommended)

Run the download script from the project root:

```bash
python scripts/download_dataset.py
```

This will automatically download the dataset from Kaggle and place it in this directory.

### Manual Download

1. Visit the Kaggle dataset page:
   https://www.kaggle.com/datasets/kundanbedmutha/healthcare-symptomsdisease-classification-dataset

2. Click the "Download" button (requires Kaggle account)

3. Extract the downloaded ZIP file

4. Copy `Symptoms_Disease_Classification.csv` to this directory:
   ```
   data/Symptoms_Disease_Classification.csv
   ```

## 📊 Dataset Information

- **Name:** Healthcare Symptoms-Disease Classification Dataset
- **Source:** Kaggle
- **Author:** kundanbedmutha
- **Size:** ~2 MB
- **Rows:** 25,000
- **Columns:** 6 (Patient_ID, Age, Gender, Symptoms, Symptom_Count, Disease)

## 🔍 Dataset Structure

```
Patient_ID | Age | Gender | Symptoms | Symptom_Count | Disease
```

- **Symptoms:** Comma-separated list of symptoms
- **Disease:** Target variable (30 disease classes)

## ✅ Verification

After downloading, verify the file exists:

```bash
ls -lh data/Symptoms_Disease_Classification.csv
```

The file should be approximately 2 MB in size.
