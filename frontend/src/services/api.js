import axios from 'axios';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000';

const api = axios.create({
  baseURL: API_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

/**
 * Get list of available models
 */
export const getModels = async () => {
  try {
    const response = await api.get('/models');
    return response.data;
  } catch (error) {
    console.error('Error fetching models:', error);
    throw error;
  }
};

/**
 * Get list of disease classes
 */
export const getClasses = async () => {
  try {
    const response = await api.get('/classes');
    return response.data;
  } catch (error) {
    console.error('Error fetching classes:', error);
    throw error;
  }
};

/**
 * Predict disease using all models
 */
export const predictAll = async (symptoms) => {
  try {
    const response = await api.post('/predict', { symptoms });
    return response.data;
  } catch (error) {
    console.error('Error predicting:', error);
    throw error;
  }
};

/**
 * Predict disease using a specific model
 */
export const predictSingle = async (symptoms, modelName) => {
  try {
    const response = await api.post(`/predict/${modelName}`, { symptoms });
    return response.data;
  } catch (error) {
    console.error('Error predicting with model:', error);
    throw error;
  }
};

/**
 * Check API health
 */
export const checkHealth = async () => {
  try {
    const response = await api.get('/health');
    return response.data;
  } catch (error) {
    console.error('Error checking health:', error);
    throw error;
  }
};

export default api;

