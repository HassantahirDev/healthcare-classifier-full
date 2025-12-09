# Healthcare Symptoms-Disease Classification

A machine learning project that classifies diseases based on patient symptoms using multiple ML and deep learning models.

## 🏗️ Project Structure

```
assignment/
├── backend/              # Flask API server
│   ├── models/          # Trained model files
│   ├── app.py           # Main Flask application
│   ├── Dockerfile       # Docker configuration
│   └── requirements.txt # Python dependencies
├── frontend/            # React frontend application
│   ├── src/            # React source code
│   ├── Dockerfile      # Docker configuration
│   └── package.json    # Node dependencies
├── notebook/            # Jupyter notebook for model training
│   └── healthcare_classification.ipynb
└── data/               # Dataset files
```

## 🚀 Quick Start

### Initial Setup

1. **Download Dataset** (Required - dataset not in git):
```bash
pip install kagglehub
python scripts/download_dataset.py
```

2. **Backend:**
```bash
cd backend
pip install -r requirements.txt
python app.py
```

3. **Frontend:**
```bash
cd frontend
npm install
npm start
```

> **Note:** The dataset is NOT committed to git (best practice). Use the download script to get it.

### Docker (Recommended)

```bash
# Build and run all services
docker-compose up --build
```

Access:
- Frontend: http://localhost:3000
- Backend: http://localhost:5000

## 📦 Deployment

### Backend on GCP Cloud Run

See [QUICK_DEPLOY.md](QUICK_DEPLOY.md) for quick deployment or [DEPLOYMENT.md](DEPLOYMENT.md) for detailed instructions.

```bash
cd backend
./deploy.sh YOUR_PROJECT_ID
```

### Frontend on Vercel

```bash
cd frontend
vercel --prod
```

Set environment variable: `REACT_APP_API_URL` = your GCP backend URL

## 🧪 Models

1. **TF-IDF + XGBoost** - Classic ML with text embeddings
2. **Feed-Forward NN** - Simple neural network
3. **RNN** - Recurrent Neural Network
4. **LSTM** - Long Short-Term Memory

## 📚 Documentation

- [SETUP.md](SETUP.md) - Complete setup guide
- [DEPLOYMENT.md](DEPLOYMENT.md) - Complete deployment guide
- [QUICK_DEPLOY.md](QUICK_DEPLOY.md) - Quick deployment steps
- [ARCHITECTURE.md](ARCHITECTURE.md) - System architecture
- [DATASET_BEST_PRACTICES.md](DATASET_BEST_PRACTICES.md) - Why datasets aren't in git

## 🛠️ Tech Stack

- **Backend:** Flask, TensorFlow, XGBoost, scikit-learn
- **Frontend:** React, Axios
- **Deployment:** Docker, GCP Cloud Run, Vercel

## 📝 License

This project is for educational purposes.

