import React from 'react';

const PredictionResults = ({ results }) => {
  if (!results || !results.predictions) {
    return null;
  }

  const { symptoms, predictions } = results;

  const getConfidenceColor = (confidence) => {
    if (confidence >= 0.8) return '#4caf50';
    if (confidence >= 0.6) return '#ff9800';
    return '#f44336';
  };

  return (
    <div>
      <div style={{ 
        background: '#f0f4ff', 
        padding: '15px', 
        borderRadius: '10px', 
        marginBottom: '25px' 
      }}>
        <strong>Analyzed Symptoms:</strong>
        <p style={{ marginTop: '8px', color: '#555' }}>{symptoms}</p>
      </div>

      <h2 style={{ 
        color: '#333', 
        marginBottom: '20px', 
        fontSize: '1.5rem' 
      }}>
        Prediction Results
      </h2>

      <div className="results-grid">
        {Object.entries(predictions).map(([key, pred]) => (
          <div key={key} className="result-card">
            <h3>{pred.model_name}</h3>
            <div className="disease-name">
              {pred.disease}
            </div>
            <div className="confidence">
              <div style={{ 
                display: 'flex', 
                justifyContent: 'space-between', 
                alignItems: 'center',
                marginBottom: '5px'
              }}>
                <span>Confidence:</span>
                <strong style={{ 
                  color: getConfidenceColor(pred.confidence) 
                }}>
                  {(pred.confidence * 100).toFixed(2)}%
                </strong>
              </div>
              <div className="confidence-bar">
                <div 
                  className="confidence-fill" 
                  style={{ 
                    width: `${pred.confidence * 100}%`,
                    background: getConfidenceColor(pred.confidence)
                  }}
                />
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default PredictionResults;

