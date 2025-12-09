#!/usr/bin/env python3
"""
Dataset Download Script
Downloads the Healthcare Symptoms-Disease Classification dataset from Kaggle.

This script follows best practices by:
1. Not committing large datasets to git
2. Downloading from the original source (Kaggle)
3. Providing clear instructions for setup

Usage:
    python scripts/download_dataset.py

Requirements:
    - kagglehub package: pip install kagglehub
    - Or Kaggle API credentials in ~/.kaggle/kaggle.json
"""

import os
import sys
import shutil
from pathlib import Path

def download_with_kagglehub():
    """Download dataset using kagglehub (recommended method)"""
    try:
        import kagglehub
        
        print("Downloading dataset using kagglehub...")
        print("Dataset: kundanbedmutha/healthcare-symptomsdisease-classification-dataset")
        
        # Download the dataset
        dataset_path = kagglehub.dataset_download("kundanbedmutha/healthcare-symptomsdisease-classification-dataset")
        
        print(f"Dataset downloaded to: {dataset_path}")
        
        # Find the CSV file
        csv_files = list(Path(dataset_path).glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError("No CSV file found in downloaded dataset")
        
        csv_file = csv_files[0]
        print(f"Found CSV file: {csv_file.name}")
        
        # Copy to data directory
        data_dir = Path(__file__).parent.parent / "data"
        data_dir.mkdir(exist_ok=True)
        
        dest_path = data_dir / "Symptoms_Disease_Classification.csv"
        shutil.copy2(csv_file, dest_path)
        
        print(f"✅ Dataset copied to: {dest_path}")
        return str(dest_path)
        
    except ImportError:
        print("❌ kagglehub not installed. Install with: pip install kagglehub")
        return None
    except Exception as e:
        print(f"❌ Error downloading with kagglehub: {e}")
        return None

def download_with_kaggle_api():
    """Download dataset using Kaggle API (alternative method)"""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        
        print("Downloading dataset using Kaggle API...")
        
        # Initialize API
        api = KaggleApi()
        api.authenticate()
        
        # Download dataset
        dataset_name = "kundanbedmutha/healthcare-symptomsdisease-classification-dataset"
        data_dir = Path(__file__).parent.parent / "data"
        data_dir.mkdir(exist_ok=True)
        
        print(f"Downloading to: {data_dir}")
        api.dataset_download_files(
            dataset_name,
            path=str(data_dir),
            unzip=True
        )
        
        # Find the CSV file
        csv_files = list(data_dir.glob("*.csv"))
        if csv_files:
            print(f"✅ Dataset downloaded: {csv_files[0]}")
            return str(csv_files[0])
        else:
            print("❌ No CSV file found after download")
            return None
            
    except ImportError:
        print("❌ kaggle package not installed. Install with: pip install kaggle")
        return None
    except Exception as e:
        print(f"❌ Error downloading with Kaggle API: {e}")
        print("\nMake sure you have:")
        print("1. Kaggle account")
        print("2. API token at https://www.kaggle.com/settings")
        print("3. Token saved to ~/.kaggle/kaggle.json")
        return None

def check_existing_dataset():
    """Check if dataset already exists"""
    data_dir = Path(__file__).parent.parent / "data"
    csv_path = data_dir / "Symptoms_Disease_Classification.csv"
    
    if csv_path.exists():
        size_mb = csv_path.stat().st_size / (1024 * 1024)
        print(f"📁 Dataset already exists: {csv_path}")
        print(f"   Size: {size_mb:.2f} MB")
        response = input("   Download again? (y/N): ").strip().lower()
        if response != 'y':
            return str(csv_path)
    
    return None

def main():
    """Main function to download dataset"""
    print("=" * 60)
    print("Healthcare Symptoms-Disease Classification Dataset Downloader")
    print("=" * 60)
    print()
    
    # Check if dataset already exists
    existing = check_existing_dataset()
    if existing:
        print(f"✅ Using existing dataset: {existing}")
        return
    
    # Try kagglehub first (recommended)
    print("Attempting to download with kagglehub...")
    result = download_with_kagglehub()
    
    if result:
        print("\n✅ Dataset download successful!")
        return
    
    # Fallback to Kaggle API
    print("\nTrying alternative method (Kaggle API)...")
    result = download_with_kaggle_api()
    
    if result:
        print("\n✅ Dataset download successful!")
        return
    
    # Manual instructions
    print("\n" + "=" * 60)
    print("Manual Download Instructions")
    print("=" * 60)
    print("""
If automatic download failed, please download manually:

1. Visit: https://www.kaggle.com/datasets/kundanbedmutha/healthcare-symptomsdisease-classification-dataset

2. Click "Download" button

3. Extract the ZIP file

4. Copy "Symptoms_Disease_Classification.csv" to:
   {}/data/

5. Ensure the file is named exactly:
   Symptoms_Disease_Classification.csv
    """.format(Path(__file__).parent.parent))
    
    sys.exit(1)

if __name__ == "__main__":
    main()

