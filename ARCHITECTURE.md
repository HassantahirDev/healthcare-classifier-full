# System Architecture

This document describes the architecture of my Healthcare Disease Classification system.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER INTERFACE                          │
│                    (React Web Application)                      │
│                      http://localhost:3000                      │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             │ HTTP/REST API
                             │ (JSON)
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                       BACKEND API                               │
│                    (Flask REST Server)                          │
│                    http://localhost:5000                        │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              API Endpoints                               │  │
│  │  - POST /predict                                         │  │
│  │  - POST /predict/<model>                                 │  │
│  │  - GET /models                                           │  │
│  │  - GET /health                                           │  │
│  └──────────────────────────────────────────────────────────┘  │
│                             │                                   │
│  ┌──────────────────────────▼───────────────────────────────┐  │
│  │         Model Loader & Inference Engine                  │  │
│  │  - Load saved models                                     │  │
│  │  - Text preprocessing                                    │  │
│  │  - Batch predictions                                     │  │
│  └──────────────────────────────────────────────────────────┘  │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             │ Load Models
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                      MODELS DIRECTORY                           │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  TF-IDF + XGBoost                                        │  │
│  │  - tfidf_lr_model.pkl                                    │  │
│  │  - tfidf_vectorizer.pkl                                  │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Feed-Forward Neural Network                             │  │
│  │  - simple_nn_model.h5                                    │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  RNN Model                                               │  │
│  │  - rnn_model.h5                                          │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  LSTM Model                                              │  │
│  │  - lstm_model.h5                                         │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Preprocessors                                           │  │
│  │  - tokenizer.pkl                                         │  │
│  │  - label_encoder.pkl                                     │  │
│  │  - config.pkl                                            │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                             ▲
                             │
                             │ Train & Save
                             │
┌────────────────────────────┴────────────────────────────────────┐
│                  TRAINING PIPELINE                              │
│                 (Jupyter Notebook)                              │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Data Loading & Preprocessing                            │  │
│  │  - Load CSV dataset                                      │  │
│  │  - Clean and prepare data                                │  │
│  │  - Feature engineering                                   │  │
│  │  - Train/test split                                      │  │
│  └──────────────────────────────────────────────────────────┘  │
│                             │                                   │
│  ┌──────────────────────────▼───────────────────────────────┐  │
│  │  Model Training                                          │  │
│  │  1. TF-IDF + XGBoost                                     │  │
│  │  2. Feed-Forward Neural Network                          │  │
│  │  3. RNN                                                   │  │
│  │  4. LSTM                                                  │  │
│  └──────────────────────────────────────────────────────────┘  │
│                             │                                   │
│  ┌──────────────────────────▼───────────────────────────────┐  │
│  │  Model Evaluation & Comparison                           │  │
│  │  - Accuracy scores                                       │  │
│  │  - Classification reports                                │  │
│  │  - Training vs test analysis                             │  │
│  │  - Visualizations                                        │  │
│  └──────────────────────────────────────────────────────────┘  │
│                             │                                   │
│  ┌──────────────────────────▼───────────────────────────────┐  │
│  │  Save Models                                             │  │
│  │  - Serialize models                                      │  │
│  │  - Save preprocessors                                    │  │
│  │  - Save configuration                                    │  │
│  └──────────────────────────────────────────────────────────┘  │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             │ Read Data
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                      DATA DIRECTORY                             │
│                                                                 │
│  Symptoms_Disease_Classification.csv                           │
│  - Symptoms column (text)                                      │
│  - Disease column (labels)                                     │
│  - Age, Gender, Symptom_Count columns                          │
└─────────────────────────────────────────────────────────────────┘
```

## Data Flow

### 1. Training Phase

I trained all models in the Jupyter notebook:

1. Load the CSV dataset
2. Preprocess and engineer features (text, binary symptoms, metadata)
3. Split into train/test sets
4. Train each model:
   - TF-IDF + XGBoost
   - Feed-Forward NN
   - RNN
   - LSTM
5. Evaluate and compare models
6. Save all models and preprocessors

### 2. Inference Phase

When a user makes a prediction:

1. User enters symptoms in the React frontend
2. Frontend sends POST request to Flask backend
3. Backend loads all models (if not already loaded)
4. Backend preprocesses input:
   - TF-IDF vectorization for XGBoost
   - Tokenization and padding for neural networks
5. All models make predictions
6. Backend aggregates results
7. Returns JSON response to frontend
8. Frontend displays predictions with confidence scores

## Component Details

### Frontend (React)

I built the frontend using:
- React 18.2.0
- Axios for HTTP requests
- CSS3 for styling

**Components:**
- `App.js` - Main application logic
- `PredictionResults.js` - Display predictions
- `ModelInfo.js` - Show model information

**Services:**
- `api.js` - API communication layer

### Backend (Flask)

I built the backend using:
- Flask 2.3.3
- Flask-CORS for cross-origin requests
- TensorFlow 2.13.0 for neural networks
- Scikit-learn for TF-IDF
- XGBoost for gradient boosting

**Key Functions:**
- `load_all_models()` - Load models on startup
- `predict_tfidf_lr()` - TF-IDF + XGBoost predictions
- `predict_neural_network()` - NN/RNN/LSTM predictions

**API Routes:**
- `/` - Home/info
- `/health` - Health check
- `/models` - Model list
- `/classes` - Disease classes
- `/predict` - All models prediction
- `/predict/<model>` - Single model prediction

### Models

#### 1. TF-IDF + XGBoost

I used XGBoost as my classic ML classifier with:
- TF-IDF text embeddings
- Binary symptom features
- Patient metadata (Age, Gender, Symptom Count)
- Regularization to prevent overfitting

#### 2. Feed-Forward Neural Network

I built a multi-input architecture:
- Text input branch with embeddings
- Tabular input branch for binary symptoms and metadata
- Combined and passed through dense layers

#### 3. RNN

I implemented a bidirectional RNN:
- Processes text sequences in both directions
- Combined with tabular features
- Captures sequential patterns

#### 4. LSTM

I implemented a bidirectional LSTM:
- Better at capturing long-term dependencies
- Combined with tabular features
- Most complex architecture

## File Structure

```
assignment/
├── ARCHITECTURE.md              # This file
├── .gitignore                   # Git ignore rules
│
├── data/                        # Dataset directory
│   └── Symptoms_Disease_Classification.csv
│
├── notebook/                    # Training environment
│   ├── healthcare_classification.ipynb
│   ├── requirements.txt
│   └── README.md
│
├── models/                      # Saved models
│   ├── tfidf_lr_model.pkl
│   ├── tfidf_vectorizer.pkl
│   ├── simple_nn_model.h5
│   ├── rnn_model.h5
│   ├── lstm_model.h5
│   ├── tokenizer.pkl
│   ├── label_encoder.pkl
│   └── config.pkl
│
├── backend/                     # API server
│   ├── app.py
│   ├── requirements.txt
│   └── README.md
│
└── frontend/                    # Web interface
    ├── public/
    │   └── index.html
    ├── src/
    │   ├── components/
    │   │   ├── PredictionResults.js
    │   │   └── ModelInfo.js
    │   ├── services/
    │   │   └── api.js
    │   ├── App.js
    │   ├── App.css
    │   └── index.js
    ├── package.json
    └── README.md
```

## Workflow

### Development Workflow

1. **Data Preparation**
   - Load dataset
   - Preprocess and engineer features
   - Split train/test

2. **Model Training**
   - Train all four models in notebook
   - Evaluate performance
   - Save models

3. **Backend Development**
   - Load saved models
   - Implement API endpoints
   - Test with cURL

4. **Frontend Development**
   - Create React components
   - Implement API calls
   - Style interface

5. **Integration Testing**
   - Start backend
   - Start frontend
   - Test end-to-end

## Design Decisions

### Why Multiple Models?

I implemented four different models to:
- Compare different approaches
- Demonstrate understanding of various techniques
- Allow users to see how different models perform
- Meet assignment requirements

### Why Flask?

I chose Flask because:
- Lightweight and easy to use
- Good Python ecosystem integration
- Easy to deploy
- Perfect for ML model serving

### Why React?

I chose React because:
- Modern UI library
- Component-based architecture
- Good ecosystem
- Easy to build responsive interfaces

### Why Separate Projects?

I separated the projects to:
- Maintain clear separation of concerns
- Allow independent development
- Make it easier to maintain
- Enable flexible deployment

## Performance Characteristics

### Model Inference Times (CPU)
- TF-IDF + XGBoost: ~10-20ms
- Feed-Forward NN: ~50-100ms
- RNN: ~100-150ms
- LSTM: ~150-200ms

### Memory Usage
- TF-IDF model: ~50MB
- Neural Network models: ~100-200MB each
- Total: ~500MB

### API Response Times
- `/health`: <10ms
- `/models`: <10ms
- `/predict`: 200-400ms (all models)
- `/predict/<model>`: 50-200ms (single model)

## Notes

This architecture supports the assignment requirements while providing a solid foundation. I designed it to be modular, maintainable, and easy to understand.
