# Model Training Notebook

This is my Jupyter notebook for training and comparing all four classification models for the healthcare symptom-disease classification assignment.

## Overview

I implemented and compared four different approaches:

1. **TF-IDF + XGBoost** - Classic ML with text embeddings
2. **Feed-Forward Neural Network** - Simple neural network on embeddings
3. **RNN** - Recurrent neural network
4. **LSTM** - Long short-term memory network

## Setup

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Required Packages

I used the following packages:
- pandas - Data manipulation
- numpy - Numerical computing
- matplotlib - Visualization
- seaborn - Statistical plots
- scikit-learn - ML algorithms and preprocessing
- tensorflow - Deep learning framework
- xgboost - Gradient boosting classifier
- joblib - Model serialization
- jupyter - Notebook environment

## Prerequisites

Before running the notebook:

1. **Dataset**: Place `Symptoms_Disease_Classification.csv` in the `../data/` directory
2. **Models Directory**: The `../models/` directory will be created automatically when saving models

## Notebook Structure

### Section 1: Imports and Setup
- Import all required libraries
- Set up environment
- Check TensorFlow and XGBoost versions

### Section 2: Data Loading and Exploration
- Load dataset from CSV
- Display dataset statistics
- Visualize class distribution
- Check for missing values

### Section 3: Data Preprocessing
- Clean and preprocess text
- Create binary symptom features
- Extract patient metadata (Age, Gender, Symptom Count)
- Encode target labels
- Split into train/test sets (80/20)

### Section 4: Model 1 - TF-IDF + XGBoost
- Create TF-IDF vectorizer
- Combine with binary features and metadata
- Train XGBoost classifier with regularization
- Evaluate and display results

### Section 5: Prepare Data for Neural Networks
- Tokenize text
- Convert to sequences
- Pad sequences
- Prepare tabular features for multi-input models
- One-hot encode labels

### Section 6: Model 2 - Feed-Forward Neural Network
- Build multi-input architecture (text + tabular)
- Train with early stopping
- Evaluate performance

### Section 7: Model 3 - RNN
- Build bidirectional RNN with multi-input
- Train and evaluate
- Compare with previous models

### Section 8: Model 4 - LSTM
- Build bidirectional LSTM with multi-input
- Train and evaluate
- Final model comparison

### Section 9: Model Comparison
- Compare all accuracies
- Training vs test accuracy analysis
- Generate comparison bar chart
- Plot training histories
- Create ensemble model

### Section 10: Save Models
- Save all trained models
- Save preprocessors (vectorizer, tokenizer, label encoder)
- Save configuration file
- Ready for backend deployment

### Section 11: Test Predictions
- Test with sample symptoms
- Show predictions from all models
- Verify model consistency

## Running the Notebook

### Option 1: Jupyter Notebook

```bash
jupyter notebook healthcare_classification.ipynb
```

### Option 2: Jupyter Lab

```bash
jupyter lab healthcare_classification.ipynb
```

### Option 3: VS Code

Open the `.ipynb` file in VS Code with Jupyter extension

## Expected Results

### Training Output

Each model will display:
- Training progress (epochs, loss, accuracy)
- Validation metrics
- Final test accuracy
- Classification report

### Model Comparison

My results showed:
- TF-IDF + XGBoost: ~3.3%
- Feed-Forward NN: ~3.2%
- RNN: ~4.0%
- LSTM: ~3.5%
- Ensemble: ~3.6%

Note: The low accuracy indicates the dataset has limited learnable patterns, which I analyzed in the training vs test accuracy comparison section.

### Generated Files

After running, the following files are created in `../models/`:

**Traditional ML:**
- `tfidf_lr_model.pkl` - XGBoost model
- `tfidf_vectorizer.pkl` - TF-IDF vectorizer

**Deep Learning:**
- `simple_nn_model.h5` - Feed-Forward NN model
- `rnn_model.h5` - RNN model
- `lstm_model.h5` - LSTM model

**Preprocessors:**
- `tokenizer.pkl` - Text tokenizer
- `label_encoder.pkl` - Label encoder
- `config.pkl` - Configuration and metadata

**Visualizations:**
- `model_comparison.png` - Accuracy comparison chart
- `training_history.png` - Training curves

## Hyperparameters

### TF-IDF + XGBoost
```python
max_features = None  # Use all features
ngram_range = (1, 3)  # unigrams, bigrams, trigrams
n_estimators = 50
max_depth = 3
learning_rate = 0.05
```

### Neural Networks
```python
MAX_LEN = 50         # Sequence length
embedding_dim = 128  # Embedding dimension
epochs = 50          # Training epochs
batch_size = 128     # Batch size
validation_split = 0.2
```

## Model Architectures

**Feed-Forward NN:**
- Multi-input: Text embeddings + Tabular features
- Embedding → Dense layers → Output

**RNN:**
- Multi-input: Text sequences + Tabular features
- Bidirectional SimpleRNN layers

**LSTM:**
- Multi-input: Text sequences + Tabular features
- Bidirectional LSTM layers

## Customization

### Adjust Hyperparameters

```python
# Change sequence length
MAX_LEN = 100

# Change training epochs
epochs = 100

# Change batch size
batch_size = 64
```

### Modify Architecture

You can experiment with:
- Different embedding dimensions
- More/fewer layers
- Different dropout rates
- Different activation functions

## Visualizations

The notebook generates several visualizations:

1. **Class Distribution**: Bar chart of disease frequencies
2. **Model Comparison**: Bar chart comparing accuracies
3. **Training History**: Line plots of training/validation curves

## Troubleshooting

### Common Issues

**1. ModuleNotFoundError**
```bash
pip install -r requirements.txt
```

**2. Dataset not found**
- Ensure CSV file is in `../data/` directory
- Check file name: `Symptoms_Disease_Classification.csv`

**3. Out of memory**
- Reduce batch size
- Reduce MAX_LEN
- Use CPU instead of GPU

**4. Low accuracy**
- This is expected given the dataset characteristics
- I analyzed this in the training vs test accuracy section

**5. Slow training**
- Use GPU if available
- Reduce model complexity
- Decrease MAX_LEN

## Tips

### For Better Understanding

1. **Read each section carefully** - Each cell has markdown explanations
2. **Run cells sequentially** - Don't skip ahead
3. **Check outputs** - Verify each step works before proceeding
4. **Experiment** - Try changing hyperparameters to see effects

### For Faster Training

1. **GPU**: Use GPU if available
2. **Batch Size**: Increase batch size
3. **Epochs**: Start with fewer epochs
4. **Sequence Length**: Reduce MAX_LEN

## Code Quality

I followed:
- **PEP 8**: Python style guidelines
- **Clear Comments**: Explanatory comments throughout
- **Modular Structure**: Organized into logical sections
- **Error Handling**: Try-catch where appropriate
- **Reproducibility**: Set random seeds

## Learning Objectives

After completing this notebook, I learned:

1. **Text Preprocessing**: TF-IDF vs embeddings
2. **Classical ML**: XGBoost for text classification
3. **Neural Networks**: Feed-forward architecture
4. **RNNs**: Sequential data processing
5. **LSTMs**: Long-term dependencies
6. **Model Comparison**: Evaluating different approaches
7. **Deployment**: Saving models for production
8. **Data Quality Analysis**: Understanding when models can't learn

## Notes

Training time varies based on:
- Dataset size
- Hardware (CPU vs GPU)
- Model complexity
- Number of epochs

Expected total runtime: 15-20 minutes on CPU
