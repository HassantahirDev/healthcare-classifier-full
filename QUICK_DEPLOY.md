# Quick Deployment Guide

## Backend on GCP Cloud Run

### Prerequisites
```bash
# Install Google Cloud SDK
# macOS: brew install google-cloud-sdk
# Or download from: https://cloud.google.com/sdk/docs/install

# Authenticate
gcloud auth login
```

### Deploy

```bash
cd backend

# Option 1: Use deployment script
./deploy.sh YOUR_PROJECT_ID

# Option 2: Manual deployment
gcloud config set project YOUR_PROJECT_ID
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/healthcare-classifier-api
gcloud run deploy healthcare-classifier-api \
  --image gcr.io/YOUR_PROJECT_ID/healthcare-classifier-api \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --port 8080 \
  --memory 2Gi
```

**Save the backend URL** - you'll need it for frontend!

## Frontend on Vercel

### Option 1: Vercel CLI

```bash
cd frontend

# Install Vercel CLI
npm install -g vercel

# Login
vercel login

# Deploy
vercel --prod
```

When prompted:
- Set environment variable: `REACT_APP_API_URL` = your GCP backend URL

### Option 2: GitHub + Vercel Dashboard

1. Push code to GitHub
2. Go to https://vercel.com
3. Click "New Project" → Import repository
4. Add environment variable:
   - `REACT_APP_API_URL` = your GCP backend URL
5. Click "Deploy"

## Update CORS

After getting your Vercel URL, update backend CORS:

In `backend/app.py`, add your Vercel URL to origins:
```python
"origins": [
    "http://localhost:3000",
    "https://your-app.vercel.app"
]
```

Then redeploy backend.

## Test

```bash
# Test backend
curl https://your-backend-url.run.app/health

# Visit frontend URL
# https://your-app.vercel.app
```

Done! 🚀

