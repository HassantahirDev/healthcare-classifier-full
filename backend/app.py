"""
Healthcare Symptoms-Disease Classification Backend API
This Flask API loads trained models and provides prediction endpoints
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pickle
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from scipy.sparse import hstack, csr_matrix
import os

app = Flask(__name__)

# Configure CORS - allow all origins for production (Vercel uses dynamic URLs)
# Set CORS_ORIGINS environment variable to restrict to specific domains
# Example: CORS_ORIGINS=https://your-app.vercel.app,http://localhost:3000
CORS_ORIGINS = os.environ.get('CORS_ORIGINS', '*')
if CORS_ORIGINS == '*':
    CORS(app)  # Allow all origins
else:
    CORS(app, resources={
        r"/*": {
            "origins": CORS_ORIGINS.split(','),
            "methods": ["GET", "POST", "OPTIONS"],
            "allow_headers": ["Content-Type"]
        }
    })

# Global variables for models and preprocessors
models = {}
preprocessors = {}
config = {}

# Model directory (models folder is now inside backend/)
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')


def load_all_models():
    """Load all trained models and preprocessors on startup"""
    global models, preprocessors, config
    
    try:
        print("Loading models and preprocessors...")
        
        # Load TF-IDF model and vectorizer
        models['tfidf_lr'] = joblib.load(os.path.join(MODEL_DIR, 'tfidf_lr_model.pkl'))
        preprocessors['tfidf_vectorizer'] = joblib.load(os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl'))
        
        # Load neural network models
        models['simple_nn'] = load_model(os.path.join(MODEL_DIR, 'simple_nn_model.h5'))
        models['rnn'] = load_model(os.path.join(MODEL_DIR, 'rnn_model.h5'))
        models['lstm'] = load_model(os.path.join(MODEL_DIR, 'lstm_model.h5'))
        
        # Load tokenizer and label encoder
        with open(os.path.join(MODEL_DIR, 'tokenizer.pkl'), 'rb') as f:
            preprocessors['tokenizer'] = pickle.load(f)
        
        with open(os.path.join(MODEL_DIR, 'label_encoder.pkl'), 'rb') as f:
            preprocessors['label_encoder'] = pickle.load(f)
        
        # Load configuration
        with open(os.path.join(MODEL_DIR, 'config.pkl'), 'rb') as f:
            config.update(pickle.load(f))
        
        print("All models loaded successfully!")
        print(f"Available models: {list(models.keys())}")
        print(f"Number of classes: {config.get('num_classes', 'Unknown')}")
        
    except FileNotFoundError as e:
        print(f"❌ Model file not found: {str(e)}")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Model directory path: {MODEL_DIR}")
        print(f"Model directory exists: {os.path.exists(MODEL_DIR)}")
        if os.path.exists(MODEL_DIR):
            print(f"Files in model directory: {os.listdir(MODEL_DIR)}")
        raise
    except Exception as e:
        print(f"❌ Error loading models: {str(e)}")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Model directory path: {MODEL_DIR}")
        raise


def predict_tfidf_lr(symptoms):
    """Predict using TF-IDF + XGBoost model"""
    # 1. TF-IDF features
    symptom_tfidf = preprocessors['tfidf_vectorizer'].transform([symptoms])
    
    # 2. Binary symptom features (all zeros for new input)
    # Since we only have text input, we can't extract exact binary features
    # Use zeros as placeholder (model was trained with this structure)
    symptom_list_size = config.get('symptom_list_size', 28)
    binary_features = np.zeros((1, symptom_list_size))
    
    # 3. Dummy metadata (Age=50, Gender=Male=0, Count=3) - normalized
    metadata = np.array([[50/100, 0/2, 3/10]])
    
    # 4. Combine all features: TF-IDF + Binary + Metadata
    all_tabular = np.column_stack([binary_features, metadata])
    combined_features = hstack([symptom_tfidf, csr_matrix(all_tabular)])
    
    # Make prediction
    prediction = models['tfidf_lr'].predict(combined_features)[0]
    probabilities = models['tfidf_lr'].predict_proba(combined_features)[0]
    disease = preprocessors['label_encoder'].inverse_transform([prediction])[0]
    confidence = float(np.max(probabilities))
    return disease, confidence


def predict_neural_network(symptoms, model_name):
    """Predict using neural network models (Simple NN, RNN, or LSTM)"""
    # Tokenize and pad the symptoms
    symptom_seq = preprocessors['tokenizer'].texts_to_sequences([symptoms])
    symptom_pad = pad_sequences(symptom_seq, maxlen=config['MAX_LEN'], padding='post')
    
    # Prepare tabular features (binary symptoms + metadata)
    # Since we only have text input, use dummy values for tabular features
    symptom_list_size = config.get('symptom_list_size', 28)
    binary_features = np.zeros((1, symptom_list_size))
    
    # Dummy metadata (Age=50, Gender=Male=0, Count=3) - normalized
    metadata = np.array([[50/100, 0/2, 3/10]])
    
    # Combine binary features and metadata
    tabular_features = np.column_stack([binary_features, metadata])
    
    # Make prediction with multi-input
    prediction = models[model_name].predict([symptom_pad, tabular_features], verbose=0)
    predicted_class = np.argmax(prediction[0])
    confidence = float(np.max(prediction[0]))
    disease = preprocessors['label_encoder'].inverse_transform([predicted_class])[0]
    
    return disease, confidence


@app.route('/')
def home():
    """Home endpoint with API information"""
    return jsonify({
        'message': 'Healthcare Symptoms-Disease Classification API',
        'version': '1.0',
        'endpoints': {
            '/predict': 'POST - Make predictions with all models',
            '/predict/<model_name>': 'POST - Make prediction with specific model',
            '/models': 'GET - List available models',
            '/classes': 'GET - List all disease classes',
            '/health': 'GET - Check API health status'
        }
    })


@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': len(models) > 0,
        'available_models': list(models.keys())
    })


@app.route('/models', methods=['GET'])
def get_models():
    """Get list of available models and their performance"""
    return jsonify({
        'models': config.get('results', {}),
        'total_models': len(models),
        'available': list(models.keys())
    })


@app.route('/classes', methods=['GET'])
def get_classes():
    """Get list of all disease classes"""
    return jsonify({
        'classes': config.get('classes', []),
        'num_classes': config.get('num_classes', 0)
    })


@app.route('/predict', methods=['POST'])
def predict_all():
    """Predict disease using all models"""
    try:
        data = request.get_json()
        
        if not data or 'symptoms' not in data:
            return jsonify({'error': 'No symptoms provided'}), 400
        
        symptoms = data['symptoms']
        
        if not symptoms or symptoms.strip() == '':
            return jsonify({'error': 'Symptoms cannot be empty'}), 400
        
        # Make predictions with all models
        predictions = {}
        
        # TF-IDF + XGBoost
        disease_lr, conf_lr = predict_tfidf_lr(symptoms)
        predictions['tfidf_lr'] = {
            'disease': disease_lr,
            'confidence': conf_lr,
            'model_name': 'TF-IDF + XGBoost'
        }
        
        # Feed-Forward Neural Network
        disease_nn, conf_nn = predict_neural_network(symptoms, 'simple_nn')
        predictions['simple_nn'] = {
            'disease': disease_nn,
            'confidence': conf_nn,
            'model_name': 'Feed-Forward NN'
        }
        
        # RNN
        disease_rnn, conf_rnn = predict_neural_network(symptoms, 'rnn')
        predictions['rnn'] = {
            'disease': disease_rnn,
            'confidence': conf_rnn,
            'model_name': 'RNN'
        }
        
        # LSTM
        disease_lstm, conf_lstm = predict_neural_network(symptoms, 'lstm')
        predictions['lstm'] = {
            'disease': disease_lstm,
            'confidence': conf_lstm,
            'model_name': 'LSTM'
        }
        
        return jsonify({
            'symptoms': symptoms,
            'predictions': predictions
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/predict/<model_name>', methods=['POST'])
def predict_single(model_name):
    """Predict disease using a specific model"""
    try:
        data = request.get_json()
        
        if not data or 'symptoms' not in data:
            return jsonify({'error': 'No symptoms provided'}), 400
        
        symptoms = data['symptoms']
        
        if not symptoms or symptoms.strip() == '':
            return jsonify({'error': 'Symptoms cannot be empty'}), 400
        
        # Check if model exists
        if model_name not in models:
            return jsonify({
                'error': f'Model {model_name} not found',
                'available_models': list(models.keys())
            }), 404
        
        # Make prediction based on model type
        if model_name == 'tfidf_lr':
            disease, confidence = predict_tfidf_lr(symptoms)
            model_full_name = 'TF-IDF + XGBoost'
        else:
            disease, confidence = predict_neural_network(symptoms, model_name)
            model_full_name = {
                'simple_nn': 'Feed-Forward NN',
                'rnn': 'RNN',
                'lstm': 'LSTM'
            }.get(model_name, model_name)
        
        return jsonify({
            'symptoms': symptoms,
            'model': model_full_name,
            'disease': disease,
            'confidence': confidence
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    # Load models on startup
    load_all_models()
    
    # Run the Flask app
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)

