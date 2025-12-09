import React, { useEffect, useState } from 'react';
import { getModels } from '../services/api';

const ModelInfo = () => {
  const [modelInfo, setModelInfo] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchModels = async () => {
      try {
        const data = await getModels();
        setModelInfo(data);
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

  if (!modelInfo || !modelInfo.models) {
    return null;
  }

  const { Model, Accuracy } = modelInfo.models;

  return (
    <div className="model-info">
      <h3>Available Models & Performance</h3>
      <div className="model-stats">
        {Model && Model.map((modelName, index) => (
          <div key={index} className="stat-item">
            <strong>{modelName}</strong>
            <div style={{ fontSize: '1.2rem', color: '#333', marginTop: '5px' }}>
              {Accuracy && Accuracy[index] 
                ? `${(Accuracy[index] * 100).toFixed(2)}% accuracy`
                : 'N/A'}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default ModelInfo;

