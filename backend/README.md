# Backend API - Healthcare Disease Classifier

This is my Flask-based REST API for disease prediction using the trained ML models I developed.

## Quick Start

### Installation

```bash
cd backend
pip install -r requirements.txt
```

### Running the Server

```bash
python app.py
```

The server will start at: `http://localhost:5000`

## API Endpoints

### 1. Home
```
GET /
```
Returns API information and available endpoints.

### 2. Health Check
```
GET /health
```
Check if the API is running and models are loaded.

**Response:**
```json
{
  "status": "healthy",
  "models_loaded": true,
  "available_models": ["tfidf_lr", "simple_nn", "rnn", "lstm"]
}
```

### 3. Get Available Models
```
GET /models
```
List all models and their performance metrics.

**Response:**
```json
{
  "models": {
    "Model": ["TF-IDF + XGBoost", "Simple NN", "RNN", "LSTM"],
    "Accuracy": [0.0334, 0.0318, 0.0396, 0.0354]
  },
  "total_models": 4,
  "available": ["tfidf_lr", "simple_nn", "rnn", "lstm"]
}
```

### 4. Get Disease Classes
```
GET /classes
```
List all possible disease classifications.

**Response:**
```json
{
  "classes": ["Disease1", "Disease2", ...],
  "num_classes": 30
}
```

### 5. Predict with All Models
```
POST /predict
Content-Type: application/json
```

**Request Body:**
```json
{
  "symptoms": "fever, cough, difficulty breathing"
}
```

**Response:**
```json
{
  "symptoms": "fever, cough, difficulty breathing",
  "predictions": {
    "tfidf_lr": {
      "disease": "Asthma",
      "confidence": 0.12,
      "model_name": "TF-IDF + XGBoost"
    },
    "simple_nn": {
      "disease": "Arthritis",
      "confidence": 0.08,
      "model_name": "Simple Neural Network"
    },
    "rnn": {
      "disease": "Allergy",
      "confidence": 0.10,
      "model_name": "RNN"
    },
    "lstm": {
      "disease": "Thyroid Disorder",
      "confidence": 0.09,
      "model_name": "LSTM"
    }
  }
}
```

### 6. Predict with Specific Model
```
POST /predict/<model_name>
Content-Type: application/json
```

**Available model names:**
- `tfidf_lr` - TF-IDF + XGBoost
- `simple_nn` - Simple Neural Network
- `rnn` - RNN
- `lstm` - LSTM

**Request Body:**
```json
{
  "symptoms": "severe headache, nausea"
}
```

**Response:**
```json
{
  "symptoms": "severe headache, nausea",
  "model": "LSTM",
  "disease": "Diabetes",
  "confidence": 0.11
}
```

## Testing with cURL

### Test Health
```bash
curl http://localhost:5000/health
```

### Get Models
```bash
curl http://localhost:5000/models
```

### Make Prediction
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"symptoms": "fever, cough, difficulty breathing"}'
```

### Predict with Specific Model
```bash
curl -X POST http://localhost:5000/predict/lstm \
  -H "Content-Type: application/json" \
  -d '{"symptoms": "headache, nausea, dizziness"}'
```

## Dependencies

I used the following libraries:
- **Flask**: Web framework
- **Flask-CORS**: Handle cross-origin requests
- **TensorFlow**: Deep learning models
- **Scikit-learn**: ML utilities and TF-IDF model
- **XGBoost**: Gradient boosting classifier
- **NumPy**: Numerical operations
- **Joblib**: Model serialization

## Configuration

### Environment Variables

- `PORT`: Server port (default: 5000)
- Models are loaded from: `../models/` directory

### Required Model Files

I saved the following files in the `models/` directory:
- `tfidf_lr_model.pkl`
- `tfidf_vectorizer.pkl`
- `simple_nn_model.h5`
- `rnn_model.h5`
- `lstm_model.h5`
- `tokenizer.pkl`
- `label_encoder.pkl`
- `config.pkl`

## Error Handling

I implemented error handling for:

### 400 Bad Request
- Missing symptoms in request
- Empty symptoms string

### 404 Not Found
- Invalid endpoint
- Model not found

### 500 Internal Server Error
- Model loading failure
- Prediction error

## CORS

I enabled CORS for all origins to allow frontend access.

For production, you should configure specific origins:
```python
CORS(app, resources={r"/*": {"origins": "https://yourdomain.com"}})
```

## Production Deployment

### Using Gunicorn

```bash
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Using Docker

```dockerfile
FROM python:3.9
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
```

## Model Loading

Models are loaded on server startup:
1. TF-IDF model and vectorizer
2. Neural network models (h5 format)
3. Tokenizer and label encoder
4. Configuration file

If models fail to load, check:
- Models exist in `../models/` directory
- Models were trained and saved successfully
- File permissions are correct

## Code Structure

```
app.py
├── load_all_models()      # Load models on startup
├── predict_tfidf_lr()     # TF-IDF predictions
├── predict_neural_network() # NN/RNN/LSTM predictions
├── Routes:
│   ├── /                  # Home
│   ├── /health            # Health check
│   ├── /models            # Model list
│   ├── /classes           # Disease classes
│   ├── /predict           # All models
│   └── /predict/<model>   # Specific model
└── Error Handlers
```

## Integration

### With Frontend
```javascript
const response = await fetch('http://localhost:5000/predict', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ symptoms: 'fever, cough' })
});
const data = await response.json();
```

### With Python
```python
import requests
response = requests.post(
    'http://localhost:5000/predict',
    json={'symptoms': 'fever, cough'}
)
predictions = response.json()
```

## Troubleshooting

If you encounter issues:
1. Check model files exist
2. Verify dependencies installed
3. Review server logs
4. Test with cURL first
5. Check CORS configuration
