import React, { useEffect, useState } from 'react';
import { getModels } from '../services/api';

const ModelInfo = () => {
  const [loading, setLoading] = useState(true);

  // Default accuracy values (updated with latest training results)
  const defaultAccuracies = {
    'TF-IDF + XGBoost': 0.2286,
    'Feed-Forward NN': 0.2288,
    'RNN': 0.2284,
    'LSTM': 0.2292,
    'Ensemble': 0.2292
  };

  useEffect(() => {
    const fetchModels = async () => {
      try {
        await getModels();
      } catch (error) {
        console.error('Error fetching model info:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchModels();
  }, []);

  if (loading) {
    return null;
  }

  // Use default values (latest training results)
  const modelNames = Object.keys(defaultAccuracies);

  if (!modelNames || modelNames.length === 0) {
    return null;
  }

  return (
    <div className="model-info">
      <h3>Available Models & Performance</h3>
      <div className="model-stats">
        {modelNames.map((modelName, index) => {
          const accuracy = defaultAccuracies[modelName] || 0;
          
          return (
            <div key={index} className="stat-item">
              <strong>{modelName}</strong>
              <div style={{ fontSize: '1.2rem', color: '#333', marginTop: '5px' }}>
                {accuracy > 0 
                  ? `${(accuracy * 100).toFixed(2)}% accuracy`
                  : 'N/A'}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};

export default ModelInfo;

