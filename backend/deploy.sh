#!/bin/bash

# GCP Deployment Script for Healthcare Classifier Backend
# Usage: ./deploy.sh [PROJECT_ID]

set -e

PROJECT_ID=${1:-"your-project-id"}

if [ "$PROJECT_ID" == "your-project-id" ]; then
    echo "Error: Please provide your GCP Project ID"
    echo "Usage: ./deploy.sh YOUR_PROJECT_ID"
    exit 1
fi

echo "Deploying to GCP Project: $PROJECT_ID"

# Set the project
gcloud config set project $PROJECT_ID

# Enable required APIs
echo "Enabling required APIs..."
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com

# Build and push Docker image
echo "Building and pushing Docker image..."
gcloud builds submit --tag gcr.io/$PROJECT_ID/healthcare-classifier-api

# Deploy to Cloud Run
echo "Deploying to Cloud Run..."
gcloud run deploy healthcare-classifier-api \
  --image gcr.io/$PROJECT_ID/healthcare-classifier-api \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --port 8080 \
  --memory 2Gi \
  --cpu 2 \
  --timeout 300 \
  --max-instances 10

# Get the service URL
SERVICE_URL=$(gcloud run services describe healthcare-classifier-api \
  --platform managed \
  --region us-central1 \
  --format 'value(status.url)')

echo ""
echo "✅ Deployment complete!"
echo "Backend URL: $SERVICE_URL"
echo ""
echo "Update your frontend .env.production with:"
echo "REACT_APP_API_URL=$SERVICE_URL"

