import React, { useState, useEffect } from 'react';
import './App.css';
import PredictionResults from './components/PredictionResults';
import ModelInfo from './components/ModelInfo';
import { predictAll, checkHealth } from './services/api';

function App() {
  const [symptoms, setSymptoms] = useState('');
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState('');
  const [apiStatus, setApiStatus] = useState('checking');

  // Example symptoms for quick testing
  const exampleSymptoms = [
    "fever, cough, and difficulty breathing",
    "severe headache, nausea, and sensitivity to light",
    "chest pain, shortness of breath",
    "joint pain, fatigue, and muscle weakness"
  ];

  useEffect(() => {
    // Check API health on mount
    const checkApiHealth = async () => {
      try {
        await checkHealth();
        setApiStatus('healthy');
      } catch (err) {
        setApiStatus('error');
        setError('Unable to connect to the API. Please ensure the backend server is running on http://localhost:5000');
      }
    };

    checkApiHealth();
  }, []);

  const handlePredict = async () => {
    if (!symptoms.trim()) {
      setError('Please enter symptoms');
      return;
    }

    setLoading(true);
    setError('');
    setResults(null);

    try {
      const data = await predictAll(symptoms);
      setResults(data);
    } catch (err) {
      setError(err.response?.data?.error || 'An error occurred while making predictions. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleClear = () => {
    setSymptoms('');
    setResults(null);
    setError('');
  };

  const loadExample = (example) => {
    setSymptoms(example);
    setResults(null);
    setError('');
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handlePredict();
    }
  };

  return (
    <div className="App">
      <div className="container">
        <header className="header">
          <h1>🏥 Healthcare Disease Classifier</h1>
          <p>Multi-Class Text Classification using Machine Learning</p>
          <p style={{ fontSize: '0.9rem', marginTop: '10px' }}>
            Comparing TF-IDF+XGBoost, Feed-Forward NN, RNN, and LSTM Models
          </p>
        </header>

        <main>
          <div className="main-card">
            {apiStatus === 'error' && (
              <div className="error" style={{ marginBottom: '20px' }}>
                ⚠️ {error}
              </div>
            )}

            <div className="input-section">
              <label htmlFor="symptoms">
                Enter Symptoms:
              </label>
              <textarea
                id="symptoms"
                className="symptom-input"
                rows="5"
                placeholder="Describe the symptoms (e.g., fever, cough, headache, difficulty breathing...)"
                value={symptoms}
                onChange={(e) => setSymptoms(e.target.value)}
                onKeyPress={handleKeyPress}
                disabled={loading || apiStatus === 'error'}
              />
            </div>

            <div style={{ marginBottom: '20px' }}>
              <p style={{ 
                fontSize: '0.9rem', 
                color: '#666', 
                marginBottom: '10px' 
              }}>
                Quick Examples:
              </p>
              <div style={{ 
                display: 'flex', 
                gap: '10px', 
                flexWrap: 'wrap' 
              }}>
                {exampleSymptoms.map((example, idx) => (
                  <button
                    key={idx}
                    onClick={() => loadExample(example)}
                    disabled={loading || apiStatus === 'error'}
                    style={{
                      padding: '8px 15px',
                      fontSize: '0.85rem',
                      background: 'white',
                      border: '2px solid #e0e0e0',
                      borderRadius: '20px',
                      cursor: 'pointer',
                      transition: 'all 0.3s ease'
                    }}
                    onMouseOver={(e) => {
                      if (!loading && apiStatus !== 'error') {
                        e.target.style.borderColor = '#667eea';
                        e.target.style.color = '#667eea';
                      }
                    }}
                    onMouseOut={(e) => {
                      e.target.style.borderColor = '#e0e0e0';
                      e.target.style.color = 'inherit';
                    }}
                  >
                    {example.length > 40 ? example.substring(0, 40) + '...' : example}
                  </button>
                ))}
              </div>
            </div>

            <div className="button-group">
              <button
                className="btn btn-primary"
                onClick={handlePredict}
                disabled={loading || !symptoms.trim() || apiStatus === 'error'}
              >
                {loading ? 'Analyzing...' : '🔍 Predict Disease'}
              </button>
              <button
                className="btn btn-secondary"
                onClick={handleClear}
                disabled={loading}
              >
                🗑️ Clear
              </button>
            </div>

            {error && apiStatus !== 'error' && (
              <div className="error">
                {error}
              </div>
            )}

            {loading && (
              <div className="loading">
                <div className="loading-spinner"></div>
                <div>Analyzing symptoms with all models...</div>
              </div>
            )}

            {results && <PredictionResults results={results} />}
          </div>

          {apiStatus === 'healthy' && <ModelInfo />}
        </main>

        <footer className="footer">
          <p>Healthcare Symptoms-Disease Multi-Class Classification</p>
          <p style={{ fontSize: '0.9rem', marginTop: '5px' }}>
            Built with React, Flask, TensorFlow & Scikit-learn
          </p>
        </footer>
      </div>
    </div>
  );
}

export default App;

